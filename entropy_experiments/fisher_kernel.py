"""Fisher kernel computation runner.

This module focuses on the numerical ingredients required to study Fisher-kernel
interactions. It delegates sequence sampling to ``SampleGenerator`` and records
rich metadata so downstream analysis can inspect kernel structure, influence
vectors, and attribution diagnostics without repeating heavy gradient work.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from entropy_experiments.utils.param_registry import (
    dot_named,
    get_optimizer_named_params,
    get_trainable_named,
)
from entropy_experiments.utils.sample_generator import GeneratedBatch, SampleGenerator
from entropy_experiments.utils.sequence_processor import BatchedSequences


# ---------------------------------------------------------------------------
# Plan + request descriptors
# ---------------------------------------------------------------------------


class BatchRequestKind(str, Enum):
    """Kinds of batch sources supported by SampleGenerator."""

    UPDATE = "update"
    EVALUATION = "evaluation"
    PROMPT_IDS = "prompt_ids"
    CUSTOM_PROMPTS = "custom_prompts"
    CUSTOM_SEQUENCE = "custom_sequence"
    FROM_FILE = "from_file"


@dataclass
class BatchRequest:
    """Describe a batch to evaluate against the Fisher workspace."""

    kind: BatchRequestKind
    params: Dict[str, Any] = field(default_factory=dict)
    sequence_filter: Optional[Sequence[str]] = None
    capture_full_kernel: Optional[bool] = None
    topk_contributors: Optional[int] = None


@dataclass
class WorkspaceSpec:
    """Instructions for creating (or loading) the update workspace."""

    kind: BatchRequestKind = BatchRequestKind.UPDATE
    params: Dict[str, Any] = field(default_factory=dict)
    load_path: Optional[str] = None
    save_path: Optional[str] = None
    capture_self_kernel: bool = False


@dataclass
class FisherKernelPlan:
    """High-level plan describing a Fisher-kernel computation run."""

    workspace: WorkspaceSpec
    evaluation_requests: List[BatchRequest] = field(default_factory=list)
    microbatch_size: int = 8
    capture_full_kernel: bool = False
    topk_contributors: Optional[int] = None
    store_gradient_norms: bool = True
    store_preconditioned: bool = True


# ---------------------------------------------------------------------------
# Gradient caches + workspaces
# ---------------------------------------------------------------------------


@dataclass
class GradientCache:
    """Per-sequence gradient storage."""

    sequence_id: str
    raw_gradient: Optional[Dict[str, torch.Tensor]] = None
    preconditioned_gradient: Optional[Dict[str, torch.Tensor]] = None
    gradient_norm: Optional[float] = None
    preconditioned_norm: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KernelWorkspace:
    """Update batch context cached for repeated Fisher-kernel queries."""

    batch: GeneratedBatch
    gradient_caches: List[GradientCache]
    adam_metadata: Dict[str, Any] = field(default_factory=dict)
    sequence_id_to_index: Dict[str, int] = field(default_factory=dict)
    self_kernel: Optional[torch.Tensor] = None
    self_influence: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# Results containers
# ---------------------------------------------------------------------------


@dataclass
class KernelBlock:
    """Dense kernel block plus ancillary statistics."""

    matrix: Optional[torch.Tensor]
    row_sequence_ids: List[str]
    col_sequence_ids: List[str]
    top_contributors: Optional[List[List[Tuple[str, float]]]] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InfluenceResult:
    """Influence vector computed for an evaluation batch."""

    delta_logprobs: torch.Tensor
    sequence_ids: List[str]
    attribution: Optional[List[List[Tuple[str, float]]]] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Bundle kernel + influence outputs for a single request."""

    request: BatchRequest
    batch: GeneratedBatch
    kernel_block: Optional[KernelBlock] = None
    influence: Optional[InfluenceResult] = None
    gradient_summary: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FisherKernelResults:
    """Aggregate of all computations performed during a run."""

    workspace: KernelWorkspace
    evaluations: List[EvaluationResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Runner implementation
# ---------------------------------------------------------------------------


class FisherKernelRunner:
    """Coordinate Fisher-kernel computations using SampleGenerator batches."""

    def __init__(self, config: Dict[str, Any], *, logger: Optional[Any] = None) -> None:
        self.config = config
        self.logger = logger
        self._sample_generator: Optional[SampleGenerator] = None
        self._named_params: Optional[OrderedDict[str, torch.nn.Parameter]] = None
        self._param_name_by_id: Dict[int, str] = {}
        self._adam_preconditioner: Optional[Dict[str, torch.Tensor]] = None
        self._preconditioner_metadata: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Resource helpers
    # ------------------------------------------------------------------

    def _ensure_generator(self) -> SampleGenerator:
        if self._sample_generator is None:
            self._sample_generator = SampleGenerator(self.config, logger=self.logger)
        return self._sample_generator

    def _ensure_named_params(self) -> None:
        if self._named_params is not None:
            return
        generator = self._ensure_generator()
        model = getattr(generator, "_model", None)
        if model is None:
            raise RuntimeError("SampleGenerator has not loaded a model.")
        optimizer = getattr(generator, "_optimizer", None)
        if optimizer is not None:
            named = get_optimizer_named_params(model, optimizer)
            if not named:
                named = get_trainable_named(model)
        else:
            named = get_trainable_named(model)
        self._named_params = named
        self._param_name_by_id = {id(param): name for name, param in named.items()}

    def _compute_preconditioner(self) -> None:
        if self._adam_preconditioner is not None:
            return
        generator = self._ensure_generator()
        optimizer = getattr(generator, "_optimizer", None)
        if optimizer is None:
            self._adam_preconditioner = {}
            self._preconditioner_metadata = {"available": False}
            return
        self._ensure_named_params()
        diagonal: Dict[str, torch.Tensor] = {}
        eps_tracker: Dict[str, float] = {}
        for group in optimizer.param_groups:
            eps = float(group.get("eps", 1e-8))
            for param in group.get("params", []):
                if param is None:
                    continue
                name = self._param_name_by_id.get(id(param))
                if name is None:
                    continue
                state = optimizer.state.get(param, {})
                exp_avg_sq = state.get("exp_avg_sq")
                if exp_avg_sq is None:
                    continue
                denom = exp_avg_sq.detach().to(torch.float32).sqrt().add_(eps)
                diagonal[name] = denom.to("cpu", torch.float32).clone()
                eps_tracker[name] = eps
        self._adam_preconditioner = diagonal
        self._preconditioner_metadata = {
            "available": bool(diagonal),
            "optimizer": optimizer.__class__.__name__,
            "eps_per_param": eps_tracker,
        }

    # ------------------------------------------------------------------
    # Workspace construction
    # ------------------------------------------------------------------

    def _prepare_workspace(self, spec: WorkspaceSpec, *, microbatch_size: int) -> KernelWorkspace:
        if spec.load_path is not None:
            raise NotImplementedError("Loading a precomputed workspace is not implemented yet.")

        request = BatchRequest(kind=spec.kind, params=spec.params)
        batch = self._materialise_request_batch(request)

        self._ensure_named_params()
        gradient_caches = self._collect_gradients(
            batch,
            microbatch_size=microbatch_size,
            apply_preconditioner=False,
            sequence_filter=None,
        )

        self._compute_preconditioner()
        if self._adam_preconditioner and gradient_caches:
            self._apply_adam_preconditioner(
                gradient_caches,
                preconditioner=self._adam_preconditioner,
            )

        sequence_id_to_index = {
            cache.sequence_id: idx for idx, cache in enumerate(gradient_caches)
        }
        workspace = KernelWorkspace(
            batch=batch,
            gradient_caches=gradient_caches,
            adam_metadata=dict(self._preconditioner_metadata),
            sequence_id_to_index=sequence_id_to_index,
        )

        if spec.capture_self_kernel:
            self._maybe_compute_self_kernel(workspace, capture_full=True)

        if spec.save_path is not None:
            raise NotImplementedError("Saving a workspace to disk is not implemented yet.")

        return workspace

    def _collect_gradients(
        self,
        batch: GeneratedBatch,
        *,
        microbatch_size: int,
        apply_preconditioner: bool,
        sequence_filter: Optional[Sequence[str]],
    ) -> List[GradientCache]:
        generator = self._ensure_generator()
        model = getattr(generator, "_model", None)
        sequence_processor = getattr(generator, "_sequence_processor", None)
        tokenizer = getattr(generator, "_tokenizer", None)
        if model is None or sequence_processor is None or tokenizer is None:
            raise RuntimeError("SampleGenerator is missing model, tokenizer, or sequence processor.")
        self._ensure_named_params()
        assert self._named_params is not None

        pad_token_id = getattr(tokenizer, "pad_token_id", 0)
        if pad_token_id is None:
            pad_token_id = 0
        device = next(model.parameters()).device

        selected: List[Tuple[int, Any]] = []
        if sequence_filter is None:
            for idx, record in enumerate(batch.sequences):
                selected.append((idx, record))
        else:
            wanted = set(sequence_filter)
            for idx, record in enumerate(batch.sequences):
                if record.sequence_id in wanted:
                    selected.append((idx, record))
        if not selected:
            return []

        model_mode = model.training
        model.eval()

        grad_mode = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        caches: List[GradientCache] = []

        full_tokens_tensor = batch.full_sequence_tensor
        prompt_lens = batch.prompt_lens or []
        gen_lens = batch.gen_lens or []

        chunk_size = max(1, microbatch_size)
        for start in range(0, len(selected), chunk_size):
            chunk = selected[start : start + chunk_size]
            chunk_tokens: List[torch.Tensor] = []
            chunk_prompt_lens: List[int] = []
            chunk_gen_lens: List[int] = []
            chunk_responses: List[List[str]] = []

            for _, record in chunk:
                pb = record.prompt_batch_idx if record.prompt_batch_idx is not None else 0
                cb = record.completion_idx if record.completion_idx is not None else 0
                prompt_len = prompt_lens[pb] if pb < len(prompt_lens) else len(record.prompt_tokens)
                if pb < len(gen_lens) and cb < len(gen_lens[pb]):
                    gen_len = gen_lens[pb][cb]
                else:
                    gen_len = len(record.response_tokens)
                total_len = prompt_len + gen_len
                if full_tokens_tensor is not None:
                    seq_tokens = full_tokens_tensor[pb, cb, :total_len]
                else:
                    seq_tokens = torch.tensor(record.prompt_tokens + record.response_tokens, dtype=torch.long)
                chunk_tokens.append(seq_tokens.to(device, non_blocking=True))
                chunk_prompt_lens.append(prompt_len)
                chunk_gen_lens.append(gen_len)
                chunk_responses.append([record.response_text])

            max_len = max(t.size(0) for t in chunk_tokens)
            padded = torch.full(
                (len(chunk_tokens), 1, max_len),
                pad_token_id,
                dtype=torch.long,
                device=device,
            )
            attention = torch.zeros(
                (len(chunk_tokens), 1, max_len),
                dtype=torch.bool,
                device=device,
            )
            for row, tokens in enumerate(chunk_tokens):
                padded[row, 0, : tokens.size(0)] = tokens
                attention[row, 0, : tokens.size(0)] = True

            batched = BatchedSequences(
                sequences=padded,
                prompt_lens=chunk_prompt_lens,
                gen_lens=[[g] for g in chunk_gen_lens],
                attention_masks=attention,
                responses_text=chunk_responses,
            )

            logprob_results, _ = sequence_processor.teacher_force_logprobs_with_diagnostics(
                batched,
                with_grad=True,
                tf_batch_size=len(chunk_tokens),
                compute_rb=False,
            )

            flat_logprobs = [logprob_results.logprobs[row][0] for row in range(len(chunk_tokens))]
            for local_idx, (global_idx, record) in enumerate(chunk):
                logprob_tensor = flat_logprobs[local_idx]
                seq_objective = logprob_tensor.sum()

                retain = local_idx != len(chunk) - 1
                model.zero_grad(set_to_none=True)
                seq_objective.backward(retain_graph=retain)

                named_grad: Dict[str, torch.Tensor] = {}
                grad_sq_sum = 0.0
                for name, param in self._named_params.items():
                    grad_tensor = param.grad
                    if grad_tensor is None:
                        grad_cpu = torch.zeros_like(param, dtype=torch.float32).cpu()
                    else:
                        grad_cpu = grad_tensor.detach().to("cpu", torch.float32).clone()
                    named_grad[name] = grad_cpu
                    grad_sq_sum += float((grad_cpu.to(torch.float64) ** 2).sum().item())

                cache = GradientCache(
                    sequence_id=record.sequence_id,
                    raw_gradient=named_grad,
                    gradient_norm=math.sqrt(grad_sq_sum),
                    metadata={
                        "prompt_batch_idx": record.prompt_batch_idx,
                        "completion_idx": record.completion_idx,
                        "prompt_length": chunk_prompt_lens[local_idx],
                        "generation_length": chunk_gen_lens[local_idx],
                        "logprob_sum": float(seq_objective.detach().to(torch.float64).item()),
                        "record_index": global_idx,
                    },
                )
                caches.append(cache)
                model.zero_grad(set_to_none=True)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        torch.set_grad_enabled(grad_mode)
        model.train(model_mode)

        if apply_preconditioner:
            self._compute_preconditioner()
            self._apply_adam_preconditioner(
                caches,
                preconditioner=self._adam_preconditioner,
            )

        return caches

    def _apply_adam_preconditioner(
        self,
        gradient_caches: List[GradientCache],
        *,
        preconditioner: Optional[Dict[str, torch.Tensor]],
    ) -> None:
        if not preconditioner:
            for cache in gradient_caches:
                cache.preconditioned_gradient = cache.raw_gradient
                cache.preconditioned_norm = cache.gradient_norm
            return

        for cache in gradient_caches:
            raw = cache.raw_gradient or {}
            precond: Dict[str, torch.Tensor] = {}
            norm_sq = 0.0
            for name, grad in raw.items():
                scale = preconditioner.get(name)
                if scale is None:
                    adjusted = grad.clone()
                else:
                    adjusted = grad / scale
                precond[name] = adjusted
                norm_sq += float((adjusted.to(torch.float64) ** 2).sum().item())
            cache.preconditioned_gradient = precond
            cache.preconditioned_norm = math.sqrt(norm_sq)

    def _maybe_compute_self_kernel(
        self,
        workspace: KernelWorkspace,
        *,
        capture_full: bool,
    ) -> None:
        if not capture_full:
            return
        if not workspace.gradient_caches:
            workspace.self_kernel = torch.zeros((0, 0), dtype=torch.float64)
            workspace.self_influence = torch.zeros(0, dtype=torch.float64)
            return

        gradients = [
            cache.preconditioned_gradient or cache.raw_gradient or {}
            for cache in workspace.gradient_caches
        ]
        size = len(gradients)
        matrix = torch.zeros((size, size), dtype=torch.float64)
        for i in range(size):
            gi = gradients[i]
            for j in range(i, size):
                val = dot_named(gi, gradients[j])
                matrix[i, j] = val
                if j != i:
                    matrix[j, i] = val
        workspace.self_kernel = matrix
        adv = self._flatten_advantages(workspace.batch, workspace.gradient_caches)
        workspace.self_influence = matrix @ adv

    # ------------------------------------------------------------------
    # Evaluation path
    # ------------------------------------------------------------------

    def _materialise_request_batch(self, request: BatchRequest) -> GeneratedBatch:
        """Use SampleGenerator (or file loaders) to obtain the requested batch."""

        generator = self._ensure_generator()
        kind = request.kind
        params = dict(request.params or {})

        if kind == BatchRequestKind.UPDATE:
            return generator.generate_update_batch(**params)
        if kind == BatchRequestKind.EVALUATION:
            return generator.generate_evaluation_batch(**params)
        if kind == BatchRequestKind.PROMPT_IDS:
            return generator.generate_from_prompt_ids(**params)
        if kind == BatchRequestKind.CUSTOM_PROMPTS:
            return generator.generate_from_custom_prompts(**params)
        if kind == BatchRequestKind.CUSTOM_SEQUENCE:
            return generator.build_custom_sequence(**params)
        if kind == BatchRequestKind.FROM_FILE:
            raise NotImplementedError("Loading batches from disk is not implemented yet")

        raise ValueError(f"Unsupported batch request kind: {kind}")

    def _compute_kernel_and_influence(
        self,
        workspace: KernelWorkspace,
        batch: GeneratedBatch,
        *,
        capture_full_kernel: bool,
        topk_contributors: Optional[int],
        microbatch_size: int,
        sequence_filter: Optional[Sequence[str]],
    ) -> Tuple[Optional[KernelBlock], Optional[InfluenceResult], Dict[str, Any]]:
        eval_caches = self._collect_gradients(
            batch,
            microbatch_size=microbatch_size,
            apply_preconditioner=True,
            sequence_filter=sequence_filter,
        )
        if not eval_caches:
            return None, None, {"count": 0}

        workspace_gradients = [
            cache.preconditioned_gradient or cache.raw_gradient or {}
            for cache in workspace.gradient_caches
        ]
        eval_gradients = [
            cache.preconditioned_gradient or cache.raw_gradient or {}
            for cache in eval_caches
        ]
        col_ids = [cache.sequence_id for cache in workspace.gradient_caches]
        row_ids = [cache.sequence_id for cache in eval_caches]

        rows: List[torch.Tensor] = []
        for grad_eval in eval_gradients:
            row = torch.zeros(len(workspace_gradients), dtype=torch.float64)
            for idx, grad_workspace in enumerate(workspace_gradients):
                row[idx] = dot_named(grad_eval, grad_workspace)
            rows.append(row)
        kernel_matrix = torch.stack(rows) if rows else torch.zeros((0, len(workspace_gradients)), dtype=torch.float64)

        adv_vector = self._flatten_advantages(workspace.batch, workspace.gradient_caches)
        influence_values = kernel_matrix @ adv_vector

        kernel_top: Optional[List[List[Tuple[str, float]]]] = None
        if topk_contributors is not None and topk_contributors > 0:
            kernel_top = []
            for row in kernel_matrix:
                scores = torch.abs(row)
                if scores.numel() == 0:
                    kernel_top.append([])
                    continue
                top_indices = torch.argsort(scores, descending=True)[:topk_contributors]
                kernel_top.append([
                    (col_ids[idx], float(row[idx].item())) for idx in top_indices.tolist()
                ])

        influence_top: Optional[List[List[Tuple[str, float]]]] = None
        if topk_contributors is not None and topk_contributors > 0:
            influence_top = []
            for row in kernel_matrix:
                contributions = row * adv_vector
                if contributions.numel() == 0:
                    influence_top.append([])
                    continue
                scores = torch.abs(contributions)
                top_indices = torch.argsort(scores, descending=True)[:topk_contributors]
                influence_top.append([
                    (col_ids[idx], float(contributions[idx].item())) for idx in top_indices.tolist()
                ])

        kernel_block = KernelBlock(
            matrix=kernel_matrix if capture_full_kernel else None,
            row_sequence_ids=row_ids,
            col_sequence_ids=col_ids,
            top_contributors=kernel_top,
            diagnostics={},
        )
        influence_result = InfluenceResult(
            delta_logprobs=influence_values,
            sequence_ids=row_ids,
            attribution=influence_top,
            diagnostics={},
        )

        gradient_summary = {
            "count": len(eval_caches),
            "gradient_norms": [cache.gradient_norm for cache in eval_caches],
            "preconditioned_norms": [cache.preconditioned_norm for cache in eval_caches],
            "logprob_sums": [cache.metadata.get("logprob_sum") for cache in eval_caches],
        }

        return kernel_block, influence_result, gradient_summary

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _flatten_advantages(
        self,
        batch: GeneratedBatch,
        gradient_caches: List[GradientCache],
    ) -> torch.Tensor:
        if batch.advantages is None:
            return torch.zeros(len(gradient_caches), dtype=torch.float64)
        advantage_matrix = batch.advantages.detach().to(torch.float64)
        record_by_id = {record.sequence_id: record for record in batch.sequences}
        values: List[float] = []
        for cache in gradient_caches:
            record = record_by_id.get(cache.sequence_id)
            if record is None:
                values.append(0.0)
                continue
            pb = record.prompt_batch_idx if record.prompt_batch_idx is not None else 0
            cb = record.completion_idx if record.completion_idx is not None else 0
            value = float(advantage_matrix[pb, cb].item())
            values.append(value)
        return torch.tensor(values, dtype=torch.float64)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, plan: FisherKernelPlan) -> FisherKernelResults:
        self._ensure_generator()
        workspace = self._prepare_workspace(plan.workspace, microbatch_size=plan.microbatch_size)

        evaluations: List[EvaluationResult] = []
        for request in plan.evaluation_requests:
            batch = self._materialise_request_batch(request)
            kernel_block, influence, grad_summary = self._compute_kernel_and_influence(
                workspace,
                batch,
                capture_full_kernel=request.capture_full_kernel
                if request.capture_full_kernel is not None
                else plan.capture_full_kernel,
                topk_contributors=request.topk_contributors
                if request.topk_contributors is not None
                else plan.topk_contributors,
                microbatch_size=plan.microbatch_size,
                sequence_filter=request.sequence_filter,
            )
            evaluations.append(
                EvaluationResult(
                    request=request,
                    batch=batch,
                    kernel_block=kernel_block,
                    influence=influence,
                    gradient_summary=grad_summary,
                )
            )

        metadata = {
            "workspace_sequences": len(workspace.gradient_caches),
            "evaluation_requests": len(evaluations),
            "preconditioner_available": bool(self._adam_preconditioner),
        }

        return FisherKernelResults(workspace=workspace, evaluations=evaluations, metadata=metadata)
