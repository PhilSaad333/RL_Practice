"""Fisher kernel diagnostics scaffolding.

This file sketches the structure for computing per-sequence Fisher-kernel
interactions between an update batch ``U`` and an evaluation batch ``E``.
Actual computation will be filled in future passes; for now the classes provide
clearly documented entry points and data structures so implementation work can
focus on individual pieces without re-reading the entire entropy probe stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch

from entropy_experiments.entropy_experiment_runner import EntropyMeasurements


@dataclass
class FisherKernelPlan:
    """Configuration knobs for the Fisher kernel runner.

    Attributes:
        capture_full_kernel: Whether to materialize and store the full |E|x|U|
            Fisher kernel matrix. Disable for very large batches when only
            aggregate stats are needed.
        microbatch_size: Number of sequences per gradient microbatch.
        max_e_sequences: Optional cap on the number of evaluation sequences to
            process (after sampling).
        max_u_sequences: Optional cap on the number of update sequences to use
            when forming the kernel.
        store_topk: If set, store only the top-k contributors per E sequence in
            addition to aggregates; ignored when ``capture_full_kernel`` is True.
        tokenizer_device: Optional override for where to stage token tensors
            (defaults to model device).
    """

    capture_full_kernel: bool = True
    microbatch_size: int = 8
    max_e_sequences: Optional[int] = None
    max_u_sequences: Optional[int] = None
    store_topk: Optional[int] = None
    tokenizer_device: Optional[str] = None


@dataclass
class SequenceMetadata:
    """Per-sequence context saved for later qualitative analysis."""

    prompt_tokens: List[int]
    response_tokens: List[int]
    prompt_text: str
    response_text: str
    logprobs: List[float]
    reward: Optional[float] = None
    advantage: Optional[float] = None


@dataclass
class FisherKernelOutputs:
    """Container for serialized diagnostics.

    The fields are intentionally straightforward (nested lists / dicts) so they
    can be passed through the existing ``to_serializable`` helper used by the
    entropy experiments CLI.
    """

    delta_logprobs_e: List[float] = field(default_factory=list)
    fisher_kernel: Optional[List[List[float]]] = None
    e_metadata: List[SequenceMetadata] = field(default_factory=list)
    u_metadata: List[SequenceMetadata] = field(default_factory=list)
    top_contributors: Optional[List[List[int]]] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class FisherKernelRunner:
    """Skeleton runner for Fisher kernel analysis.

    Responsibilities (to be implemented):
        * Load model + optimizer states identical to ``EntropyMeasurements``.
        * Sample / prepare E and U batches via ``SequenceProcessor``.
        * Collect per-sequence gradients and apply Adam diagonal preconditioner.
        * Form Fisher kernel matrix blocks and aggregate Δ log π for E.
        * Emit rich metadata (token-level logprobs, text, rewards, advantages).
    """

    def __init__(self, config: Dict[str, Any], *, logger: Optional[Any] = None) -> None:
        self.config = config
        self.logger = logger

        # Reuse the proven EntropyMeasurements stack for model loading,
        # optimizer handling, and batch preparation. FisherKernelRunner will
        # delegate to this "probe" instance for those responsibilities.
        self._probe = EntropyMeasurements(config)
        if logger is not None:
            # Align probe logger with external caller preferences when provided.
            self._probe.logger = logger

    @property
    def model(self) -> Optional[torch.nn.Module]:
        return getattr(self._probe, "model", None)

    @property
    def tokenizer(self) -> Any:
        return getattr(self._probe, "tokenizer", None)

    @property
    def optimizer(self) -> Optional[Any]:
        return getattr(self._probe, "optimizer", None)

    @property
    def sequence_processor(self) -> Optional[Any]:
        return getattr(self._probe, "_sequence_processor", None)

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def _ensure_model_loaded(self) -> None:
        """Load model, tokenizer, optimizer, and sequence processor.

        Reuse ``model_loader`` and ``SequenceProcessor`` utilities; cache the
        results on the runner instance. This mirrors ``EntropyMeasurements`` so
        the same PEFT + precision rules apply.
        """
        if not getattr(self._probe, "checkpoint_loaded", False):
            checkpoint_path = (self.config.get("checkpoint") or {}).get("checkpoint_path")
            optimizer_path = (self.config.get("checkpoint") or {}).get("optimizer_path")
            self._probe.load_checkpoint(checkpoint_path, optimizer_path)
        # Ensure sequence processor is instantiated (same guard as run_experiments).
        self._probe._ensure_sequence_processor()

    def _prepare_batches(
        self,
        plan: FisherKernelPlan,
        *,
        e_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Sample (or load) the evaluation (E) and update (U) batches.

        Returns a dictionary describing the batches (token tensors, rewards,
        advantages, metadata needed for serialization).

        The default implementation mirrors ``EntropyMeasurements._prepare_batches``
        and exposes the resulting E and U payloads, but leaves hooks for
        overriding E with custom sequences once that feature is added.
        """
        batches = self._probe._prepare_batches()
        E_batch = e_override if e_override is not None else batches.E

        return {
            "E": E_batch,
            "U": batches.U,
            "sampling_info": batches.info,
            "sampling_sec": batches.sampling_sec,
        }

    # ------------------------------------------------------------------
    # Gradient collection and preconditioning
    # ------------------------------------------------------------------
    def _collect_gradients(
        self,
        batch: Dict[str, Any],
        *,
        microbatch_size: int,
        apply_preconditioner: bool,
    ) -> Dict[str, Any]:
        """Compute per-sequence gradients of log-prob terms.

        Args:
            batch: Payload describing prompts/responses plus reward data.
            microbatch_size: Maximum number of sequences per gradient pass.
            apply_preconditioner: If True, scale gradients using Adam
                ``exp_avg_sq`` statistics directly during collection.

        Returns:
            Structure containing gradient tensors (flattened or sharded),
            per-sequence metadata, and any auxiliary information required to
            reconstruct kernel contributions.
        """

        raise NotImplementedError

    def _apply_adam_preconditioner(
        self,
        named_grads: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Scale gradients using Adam's diagonal preconditioner.

        The simplified setting keeps only the instantaneous t'=t contribution
        (no momentum). This function should pull ``exp_avg_sq`` from the loaded
        optimizer state, add eps, and divide gradients in place.
        """

        raise NotImplementedError

    # ------------------------------------------------------------------
    # Kernel contraction + result assembly
    # ------------------------------------------------------------------
    def _compute_kernel_and_influences(
        self,
        e_grads: Dict[str, Any],
        u_grads: Dict[str, Any],
        *,
        advantages_u: Sequence[float],
        plan: FisherKernelPlan,
    ) -> Dict[str, Any]:
        """Contract E and U gradient sets to produce kernel + Δ log π.

        Should support two modes:
            * ``capture_full_kernel=True``: return dense |E|x|U| blocks
              (potentially chunked internally).
            * ``capture_full_kernel=False``: stream contributions to build
              Δ log π_E and store only top-k indices per sequence if requested.
        """

        raise NotImplementedError

    def _assemble_outputs(
        self,
        kernel_payload: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> FisherKernelOutputs:
        """Package outputs into a ``FisherKernelOutputs`` dataclass."""

        raise NotImplementedError

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run(
        self,
        plan: FisherKernelPlan,
        *,
        e_override: Optional[Dict[str, Any]] = None,
    ) -> FisherKernelOutputs:
        """High-level orchestration for Fisher kernel analysis."""

        self._ensure_model_loaded()
        batches = self._prepare_batches(plan, e_override=e_override)

        u_grads = self._collect_gradients(
            batches["U"],
            microbatch_size=plan.microbatch_size,
            apply_preconditioner=True,
        )
        e_grads = self._collect_gradients(
            batches["E"],
            microbatch_size=plan.microbatch_size,
            apply_preconditioner=False,
        )

        kernel_payload = self._compute_kernel_and_influences(
            e_grads,
            u_grads,
            advantages_u=batches["U"].get("advantages", []),
            plan=plan,
        )

        outputs = self._assemble_outputs(kernel_payload, metadata=batches)
        return outputs
