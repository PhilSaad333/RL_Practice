"""Entropy influence experiments built on top of SampleGenerator batches."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from entropy_experiments.fisher_kernel import BatchRequest, BatchRequestKind, WorkspaceSpec
from entropy_experiments.delta_entropy_true import DeltaEntropyTrue
from entropy_experiments.update_vector import compute_update_vector_adamw
from entropy_experiments.utils.sample_generator import GeneratedBatch, SampleGenerator


@dataclass
class EntropyInfluencePlan:
    """Describe an entropy influence experiment run."""

    workspace: WorkspaceSpec
    evaluation_requests: List[BatchRequest] = field(default_factory=list)
    etas: List[float] = field(default_factory=list)
    per_sequence_etas: Optional[List[float]] = None
    microbatch_size: int = 4
    auto_scale: bool = False
    auto_scale_target: float = 1e-6


@dataclass
class AggregateEntropyResult:
    """Aggregate entropy change diagnostics for a single η."""

    eta: float
    delta_h: Optional[float] = None
    per_sequence_delta: Optional[List[float]] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerSequenceEntropyResult:
    """Per-sequence entropy deltas for the evaluation set."""

    eta_reference: float
    delta_matrix: List[List[float]]
    sequence_ids: List[str]
    eta_per_sequence: List[float]
    diagnostics: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class WorkspaceResult:
    """Summary of the update batch and associated vectors."""

    batch: GeneratedBatch
    update_vector: Dict[str, torch.Tensor]
    update_stats: Dict[str, Any]
    per_sequence_metadata: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class EvaluationEntropyResult:
    """Entropy diagnostics for a single evaluation batch."""

    request: BatchRequest
    batch: GeneratedBatch
    aggregate: List[AggregateEntropyResult] = field(default_factory=list)
    per_sequence: Optional[PerSequenceEntropyResult] = None


@dataclass
class EntropyInfluenceResults:
    """Container for the outputs of an entropy influence run."""

    plan: EntropyInfluencePlan
    workspace: WorkspaceResult
    evaluations: List[EvaluationEntropyResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def _max_gen_lengths(gen_lens: Sequence[Sequence[int]]) -> List[int]:
    lengths: List[int] = []
    for lens in gen_lens:
        if not lens:
            lengths.append(1)
        else:
            lengths.append(max(int(x) for x in lens))
    return lengths


def _batch_to_update_inputs(batch: GeneratedBatch) -> Dict[str, Any]:
    """Convert a GeneratedBatch into the dict expected by update_vector helpers."""

    sequences = batch.full_sequence_tensor
    attention = batch.attention_mask
    advantages = batch.advantages

    if advantages is None:
        advantages = torch.zeros(
            (sequences.shape[0], sequences.shape[1]),
            dtype=torch.float32,
        )

    return {
        "sequences": sequences,
        "attention_masks": attention,
        "prompt_lens": list(batch.prompt_lens),
        "advantages": advantages,
        "max_lengths": _max_gen_lengths(batch.gen_lens),
        "num_prompts": int(sequences.shape[0]),
        "num_responses_per_prompt": int(sequences.shape[1]) if sequences.ndim >= 2 else 1,
    }


def _batch_to_evaluation_inputs(batch: GeneratedBatch, device: torch.device) -> Dict[str, Any]:
    """Prepare evaluation batch payload for DeltaEntropyTrue."""

    sequences = batch.full_sequence_tensor.to(device)
    attention = batch.attention_mask.to(device)

    return {
        "sequences": sequences,
        "attention_masks": attention,
        "prompt_lens": list(batch.prompt_lens),
        "gen_lens": [list(row) for row in batch.gen_lens],
        "max_lengths": _max_gen_lengths(batch.gen_lens),
        "num_prompts": int(sequences.shape[0]),
        "num_responses_per_prompt": int(sequences.shape[1]) if sequences.ndim >= 2 else 1,
    }


def _sequence_ids_from_batch(batch: GeneratedBatch) -> List[str]:
    return [record.sequence_id for record in batch.sequences]


class EntropyInfluenceRunner:
    """Coordinate entropy influence experiments."""

    def __init__(self, config: Dict[str, Any], *, logger: Optional[Any] = None) -> None:
        self.config = config
        self.logger = logger
        self._generator: Optional[SampleGenerator] = None
        self._delta_entropy_true: Optional[DeltaEntropyTrue] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, plan: EntropyInfluencePlan) -> EntropyInfluenceResults:
        generator = self._ensure_generator()
        workspace_batch = self._materialise_workspace(plan.workspace)

        evaluation_pairs = [
            (request, self._materialise_batch(request))
            for request in plan.evaluation_requests
        ]

        delta_true = self._ensure_delta_entropy_true()
        model_device = next(generator._model.parameters()).device  # type: ignore[attr-defined]

        total_sequences = len(workspace_batch.sequences)

        eval_contexts = [
            {
                "request": request,
                "batch": batch,
                "data": _batch_to_evaluation_inputs(batch, model_device),
                "delta_rows": [
                    [0.0 for _ in range(total_sequences)]
                    for _ in range(len(batch.sequences))
                ],
                "aggregate": [],
                "per_sequence_details": [],
                "sequence_ids": _sequence_ids_from_batch(batch),
            }
            for request, batch in evaluation_pairs
        ]

        workspace_result, eta_per_sequence = self._compute_workspace_results(
            plan,
            workspace_batch,
            eval_contexts,
            delta_true,
        )

        evaluation_results = self._finalise_evaluations(
            plan,
            eval_contexts,
            workspace_result.update_vector,
            eta_per_sequence,
            delta_true,
        )

        metadata = {
            "num_update_sequences": len(workspace_batch.sequences),
            "num_evaluation_batches": len(evaluation_results),
        }

        return EntropyInfluenceResults(
            plan=plan,
            workspace=workspace_result,
            evaluations=evaluation_results,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Workspace + evaluation preparation
    # ------------------------------------------------------------------

    def _ensure_generator(self) -> SampleGenerator:
        if self._generator is None:
            self._generator = SampleGenerator(self.config, logger=self.logger)
        return self._generator

    def _ensure_delta_entropy_true(self) -> DeltaEntropyTrue:
        if self._delta_entropy_true is None:
            generator = self._ensure_generator()
            generator._lazy_load_resources()  # type: ignore[attr-defined]
            model = getattr(generator, "_model", None)
            sequence_processor = getattr(generator, "_sequence_processor", None)
            if model is None or sequence_processor is None:
                raise RuntimeError("SampleGenerator did not expose model/sequence processor")
            self._delta_entropy_true = DeltaEntropyTrue(
                model=model,
                sequence_processor=sequence_processor,
                config=self.config,
                logger=self.logger,
            )
        return self._delta_entropy_true

    def _materialise_workspace(self, spec: WorkspaceSpec) -> GeneratedBatch:
        if spec.load_path is not None:
            raise NotImplementedError("Loading workspaces from disk is not yet implemented.")
        request = BatchRequest(kind=spec.kind, params=spec.params)
        return self._materialise_batch(request)

    def _materialise_batch(self, request: BatchRequest) -> GeneratedBatch:
        generator = self._ensure_generator()
        params = dict(request.params or {})

        if request.kind == BatchRequestKind.UPDATE:
            return generator.generate_update_batch(**params)
        if request.kind == BatchRequestKind.EVALUATION:
            return generator.generate_evaluation_batch(**params)
        if request.kind == BatchRequestKind.PROMPT_IDS:
            return generator.generate_from_prompt_ids(**params)
        if request.kind == BatchRequestKind.CUSTOM_PROMPTS:
            return generator.generate_from_custom_prompts(**params)
        if request.kind == BatchRequestKind.CUSTOM_SEQUENCE:
            return generator.build_custom_sequence(**params)

        raise ValueError(f"Unsupported batch request kind: {request.kind}")

    # ------------------------------------------------------------------
    # Core computations
    # ------------------------------------------------------------------

    def _compute_workspace_results(
        self,
        plan: EntropyInfluencePlan,
        workspace_batch: GeneratedBatch,
        eval_contexts: List[Dict[str, Any]],
        delta_true: DeltaEntropyTrue,
    ) -> Tuple[WorkspaceResult, List[float]]:
        generator = self._ensure_generator()
        generator._lazy_load_resources()  # type: ignore[attr-defined]
        model = getattr(generator, "_model", None)
        optimizer = getattr(generator, "_optimizer", None)
        if model is None or optimizer is None:
            raise RuntimeError("SampleGenerator did not expose optimizer state for update vector computation")

        total_sequences = len(workspace_batch.sequences)
        per_sequence_metadata: List[Dict[str, Any]] = []
        eta_per_sequence: List[float] = []

        cfg_override = dict(self.config)
        cfg_override.setdefault("true_delta_h", {})
        cfg_override["true_delta_h"] = dict(cfg_override["true_delta_h"], microbatch_size=plan.microbatch_size)

        U_payload = _batch_to_update_inputs(workspace_batch)
        sequence_records = workspace_batch.sequences

        true_cfg = self.config.get("true_delta_h", {})

        def _run_direction(eta: float, direction: Dict[str, torch.Tensor]) -> List[Dict[str, Any]]:
            results: List[Dict[str, Any]] = []
            for ctx in eval_contexts:
                details = delta_true.compute_delta_h_true(
                    ctx["data"],
                    direction,
                    eta,
                    cfg=true_cfg,
                    return_details=True,
                )
                results.append(details)
            return results

        def _auto_scale_eta(
            base_eta: float,
            direction: Dict[str, torch.Tensor],
        ) -> Tuple[float, List[Dict[str, Any]], int]:
            if not eval_contexts:
                return base_eta, [], 0

            eta = base_eta
            details = _run_direction(eta, direction)
            iterations = 0

            if not plan.auto_scale:
                return eta, details, iterations

            target = plan.auto_scale_target
            max_abs = max(abs(info.get("delta_h_true", 0.0)) for info in details)
            while max_abs < target and iterations < 6:
                eta *= 2.0
                details = _run_direction(eta, direction)
                max_abs = max(abs(info.get("delta_h_true", 0.0)) for info in details)
                iterations += 1

            return eta, details, iterations

        def _sequence_callback(index: int, direction: Dict[str, torch.Tensor], meta: Dict[str, Any]) -> None:
            base_eta = self._choose_base_eta(plan, index, total_sequences)
            eta_used, detail_list, num_scales = _auto_scale_eta(base_eta, direction)

            if eval_contexts:
                for ctx, details in zip(eval_contexts, detail_list):
                    seq_deltas = details.get("delta_h_seq", [])
                    for row_idx, delta_val in enumerate(seq_deltas):
                        ctx["delta_rows"][row_idx][index] = float(delta_val)
                    ctx["per_sequence_details"].append({
                        "delta_h_true": float(details.get("delta_h_true", 0.0)),
                        "ess": float(details.get("ess", 0.0)),
                        "logweight_stats": details.get("logweight_stats", {}),
                        "clip_fraction": details.get("clip_fraction", 0.0),
                        "weights_sum": details.get("weights_sum", 0.0),
                        "eta_used": eta_used,
                    })

            seq_meta = dict(meta)
            seq_meta.update(
                {
                    "eta_base": base_eta,
                    "eta_used": eta_used,
                    "auto_scale_steps": num_scales,
                }
            )
            if detail_list:
                primary = detail_list[0]
                seq_meta["delta_h_primary"] = float(primary.get("delta_h_true", 0.0))
                seq_meta["ess_primary"] = float(primary.get("ess", 0.0))

            eta_per_sequence.append(eta_used)
            per_sequence_metadata.append(seq_meta)

        update_vector, update_stats = compute_update_vector_adamw(
            model=model,
            optimizer=optimizer,
            U_batch=U_payload,
            config=cfg_override,
            logger=self.logger,
            sequence_callback=_sequence_callback,
            sequence_records=sequence_records,
        )

        update_stats.setdefault("per_sequence_eta", eta_per_sequence)

        workspace_result = WorkspaceResult(
            batch=workspace_batch,
            update_vector=update_vector,
            update_stats=update_stats,
            per_sequence_metadata=per_sequence_metadata,
        )

        return workspace_result, eta_per_sequence

    def _finalise_evaluations(
        self,
        plan: EntropyInfluencePlan,
        eval_contexts: List[Dict[str, Any]],
        update_vector: Dict[str, torch.Tensor],
        eta_per_sequence: List[float],
        delta_true: DeltaEntropyTrue,
    ) -> List[EvaluationEntropyResult]:
        if not eval_contexts:
            return []

        true_cfg = self.config.get("true_delta_h", {})
        eta_reference = plan.etas[0] if plan.etas else (eta_per_sequence[0] if eta_per_sequence else 0.0)

        evaluation_results: List[EvaluationEntropyResult] = []

        for ctx in eval_contexts:
            request: BatchRequest = ctx["request"]
            batch: GeneratedBatch = ctx["batch"]

            aggregate_results: List[AggregateEntropyResult] = []
            for eta in plan.etas:
                details = delta_true.compute_delta_h_true(
                    ctx["data"],
                    update_vector,
                    eta,
                    cfg=true_cfg,
                    return_details=True,
                )
                aggregate_results.append(
                    AggregateEntropyResult(
                        eta=eta,
                        delta_h=float(details.get("delta_h_true", 0.0)),
                        per_sequence_delta=[float(x) for x in details.get("delta_h_seq", [])],
                        diagnostics={
                            "ess": float(details.get("ess", 0.0)),
                            "logweight_stats": details.get("logweight_stats", {}),
                            "clip_fraction": details.get("clip_fraction", 0.0),
                        },
                    )
                )

            per_seq_details = ctx.get("per_sequence_details", [])
            diagnostics_per_sequence: List[Dict[str, Any]] = []
            for idx, seq_detail in enumerate(per_seq_details):
                diag = dict(seq_detail)
                diag.setdefault("sequence_index", idx)
                diagnostics_per_sequence.append(diag)

            per_sequence = PerSequenceEntropyResult(
                eta_reference=eta_reference,
                delta_matrix=ctx["delta_rows"],
                sequence_ids=ctx["sequence_ids"],
                eta_per_sequence=eta_per_sequence,
                diagnostics=diagnostics_per_sequence,
            )

            evaluation_results.append(
                EvaluationEntropyResult(
                    request=request,
                    batch=batch,
                    aggregate=aggregate_results,
                    per_sequence=per_sequence,
                )
            )

        return evaluation_results

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------

    def _choose_base_eta(
        self,
        plan: EntropyInfluencePlan,
        seq_index: int,
        total_sequences: int,
    ) -> float:
        if plan.per_sequence_etas:
            if len(plan.per_sequence_etas) == 1:
                return float(plan.per_sequence_etas[0])
            if len(plan.per_sequence_etas) == total_sequences:
                return float(plan.per_sequence_etas[seq_index])
            raise ValueError(
                "per_sequence_etas must be length 1 or match number of update sequences"
            )

        if plan.etas:
            return float(plan.etas[0])

        opt_cfg = self.config.get("optimizer", {})
        return float(opt_cfg.get("lr", 1e-6))
