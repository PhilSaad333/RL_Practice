# delta_entropy_approx.py
"""
DeltaEntropyApprox: linearized entropy change estimator

Computes the first-order (per-learning-rate) change in sequence entropy,
    ΔH/η ≈ ⟨∇_θ H(θ), v⟩,
given:
  • a LoRA/PEFT-wrapped language model `model`,
  • a `SequenceProcessor` instance providing teacher-forced with-grad logprobs and
    (optionally) differentiable Rao–Blackwell (RB) per-token entropies,
  • an E-batch of prompts+generations with G=1 per prompt,
  • a name-keyed update direction `v_named` normalized by learning rate (Δθ/η).

The class forms a REINFORCE-style surrogate loss whose gradient equals an estimator of ∇H,
accumulates gradients over microbatches of the E-batch, and finally contracts the named
gradient tensors with `v_named` using a stable float64 dot-product.

Key properties
--------------
• Precision-stable: forward runs in fp32 via the SequenceProcessor's with-grad path.
• Name coherence: gradients are gathered only for the intersection of model trainables
  and the keys present in `v_named` (LoRA adapters), ensuring ΔH/η is well-defined.
• Microbatching: supports accumulation over E with configurable microbatch size.
• RB estimator by default: uses differentiable RB entropies H^RB_k when available.

Config knobs (expected keys)
----------------------------
approx_delta_h:
  microbatch_size: 8
  normalize: "per_token"   # "per_token" | "per_sequence" | "none"
  baseline:
    kind: "Hk"             # "none" | "Hk"  (EMA/learned can be added later)
estimator:
  use_simple_entropy_for_x: false  # false => use RB estimator (recommended)

Returned dict (from `compute_delta_h_approx`)
---------------------------------------------
{
  "delta_h_per_lr": float,             # ⟨∇H, v⟩   (units: entropy per unit-LR)
  "num_sequences": int,
  "num_tokens": int,                   # total generated tokens used
  "norms": {
     "grad_l2": float,
     "v_l2": float,
     "cosine": float                  # (∇H·v)/(||∇H||·||v||)
  },
  "intersection": {
     "matched": int, "v_named": int, "trainable": int,
     "missing_in_model": list[str], "missing_in_v": list[str]
  },
  "estimator": "rb" | "simple",
  "baseline": {"kind": str, "mean_b": float, "mean_G": float},
}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple, OrderedDict as TOrderedDict

import math
import torch

# --- Tolerant imports for registry & sequence types ----------------------------------------------
# param registry helpers (name alignment, stable dot, etc.)
try:
    # project-local typical layout
    from .param_registry import (
        get_trainable_named,
        get_optimizer_named_params,
        to_cpu_fp32_named,
        dot_named,
        flatten_named,
    )
except Exception:
    try:
        # alternative layout used elsewhere in the repo
        from .utils.param_registry import (
            get_trainable_named,
            get_optimizer_named_params,
            to_cpu_fp32_named,
            dot_named,
            flatten_named,
        )
    except Exception:
        # flat import fallback (useful in notebooks / ad-hoc runs)
        from param_registry import (  # type: ignore
            get_trainable_named,
            get_optimizer_named_params,
            to_cpu_fp32_named,
            dot_named,
            flatten_named,
        )

# sequence processor dataclasses & API
try:
    from .utils.sequence_processor import BatchedSequences, LogprobResults, DiagnosticsResults
except Exception:
    try:
        from sequence_processor import BatchedSequences, LogprobResults, DiagnosticsResults  # type: ignore
    except Exception:
        BatchedSequences = None  # type: ignore
        LogprobResults = None  # type: ignore
        DiagnosticsResults = None  # type: ignore


@dataclass
class _Intersections:
    matched: int
    v_named_total: int
    trainable_total: int
    missing_in_model: List[str]
    missing_in_v: List[str]


class DeltaEntropyApprox:
    """
    Compute the linearized entropy change per unit learning rate:
        delta_h_per_lr = ⟨∇_θ H(θ), v⟩,
    using a RB-based surrogate by default.

    Workflow
    --------
    1) Iterate over E-batch in microbatches and call:
         sequence_processor.teacher_force_logprobs_with_diagnostics(
             sequences=mb_seqs, tf_batch_size=..., compute_rb=True,
             with_grad=True, return_baseline_features=False
         )
       which returns log-probs on realized tokens (graph-carrying) and, if enabled,
       differentiable RB entropies per token.

    2) Form surrogate scalar for the microbatch:
         L_sur_mb = Σ_k (G_k - b_k).detach() * logπ_k  +  𝟙_RB Σ_k H^RB_k
       where G_k is the reverse-cumsum of H^RB over generated tokens.

    3) Accumulate scaled microbatch loss so the final gradient equals the
       mean over the E-batch (per-token or per-sequence, per config).

    4) After backward accumulation, gather grads for the intersection of model
       trainables and v_named keys, move to CPU/fp32, and compute a stable
       float64 dot-product with v_named (also CPU/fp32).

    Notes
    -----
    • This class does not mutate model parameters.
    • It assumes E_batch has G=1 per prompt.
    • If RB entropies with gradients are not available (config.rb_requires_grad=False),
      and estimator is "rb", the method raises a ValueError to avoid silent bias.
    """

    def __init__(self, model, sequence_processor, config, logger):
        self.model = model
        self.sp = sequence_processor
        self.cfg = config or {}
        self.logger = logger

        approx_cfg = (self.cfg.get("approx_delta_h", {}) or {})
        self.mb = int(approx_cfg.get("microbatch_size", 8))
        self.normalize = str(approx_cfg.get("normalize", "per_sequence")).lower()
        baseline_cfg = (approx_cfg.get("baseline", {}) or {})
        self.baseline_kind = str(baseline_cfg.get("kind", "Hk")).lower()

        est_cfg = (self.cfg.get("estimator", {}) or {})
        self.use_rb = not bool(est_cfg.get("use_simple_entropy_for_x", False))

        self.debug = bool(self.cfg.get("debug", False)) or bool(approx_cfg.get("debug", False))

    # ---------------------------------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------------------------------
    @torch.no_grad()
    def dry_run_check(self, v_named: Dict[str, torch.Tensor]) -> None:
        """
        Sanity check that the intersection with model.named_parameters() is non-empty
        and surface mismatches. No gradients are computed.
        """
        name_to_param = self._select_params_intersecting(v_named)
        if len(name_to_param) == 0:
            trainable = list(n for n, p in self.model.named_parameters() if p.requires_grad)
            missing_in_model = sorted(list(k for k in v_named.keys() if k not in trainable))
            msg = (
                f"[delta-h approx] No intersection between v_named (|K|={len(v_named)}) "
                f"and model trainables (|T|={len(trainable)}). "
                f"First missing v-names (in v but not trainable): {missing_in_model[:5]}"
            )
            raise ValueError(msg)
        if self.debug and self.logger:
            inter = self._intersections_report(name_to_param, v_named)
            self.logger.info(
                f"[delta-h approx] dry-run: matched={inter.matched} / v_named={inter.v_named_total} / "
                f"trainable={inter.trainable_total}; "
                f"missing_in_model={inter.missing_in_model[:3]} ...; missing_in_v={inter.missing_in_v[:3]} ..."
            )

    def compute_delta_h_approx(
        self,
        *,
        E_batch: Dict[str, Any],
        v_named: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        """
        Compute ⟨∇H, v⟩ over the provided E-batch.

        Parameters
        ----------
        E_batch : dict
            A dict produced by your `_pack_E_from_sequences`, with keys:
              - 'sequences': [B, G, L] tensor,
              - 'attention_masks': [B, G, L] tensor,
              - 'prompt_lens': list[int] length B,
              - 'gen_lens': list[list[int]] shape [B][G], with G==1,
              - plus auxiliary fields (unused here).
        v_named : Dict[str, Tensor]
            Name-keyed update direction (CPU fp32) normalized by the LR used to produce it.

        Returns
        -------
        Dict[str, Any] with keys:
            - 'delta_h_per_lr' : float
            - 'num_sequences'  : int
            - 'num_tokens'     : int
            - 'norms'          : dict (grad_l2, v_l2, cosine)
            - 'intersection'   : dict (matched, v_named, trainable, missing lists)
            - 'estimator'      : 'rb' or 'simple'
            - 'baseline'       : dict (kind, mean_b, mean_G)
        """
        device = next(self.model.parameters()).device

        # Preflight: intersection / mismatches
        name_to_param = self._select_params_intersecting(v_named)
        inter = self._intersections_report(name_to_param, v_named)
        if inter.matched == 0:
            raise ValueError(
                "[delta-h approx] No named-parameter intersection between model trainables and v_named."
            )
        if self.debug and self.logger:
            self.logger.info(
                f"[delta-h approx] compute: matched={inter.matched} / v_named={inter.v_named_total} / "
                f"trainable={inter.trainable_total}; "
                f"missing_in_model={inter.missing_in_model[:3]} ...; missing_in_v={inter.missing_in_v[:3]} ..."
            )

        # Count totals for normalization
        B_total = int(E_batch["sequences"].shape[0])
        T_total = self._count_total_gen_tokens(E_batch)

        if self.use_rb and not bool(getattr(self.sp.config, "rb_requires_grad", True)):
            # We need differentiable RB entropies if RB estimator is requested.
            raise ValueError(
                "[delta-h approx] RB estimator requested but sequence_processor.config.rb_requires_grad=False. "
                "Enable RB entropies with grad (e.g., GenerationConfig.rb_requires_grad=True)."
            )

        # Accumulate gradients over microbatches
        self.model.zero_grad(set_to_none=True)
        total_tokens_used = 0
        baseline_means = []
        G_means = []

        for mb_E in self._iter_microbatches(E_batch, self.mb):
            # Teacher-forced with-grad forward
            # Note: tf_batch_size can be ≤ microbatch size; cap by current B_mb
            B_mb = int(mb_E.sequences.shape[0])
            tf_bs = min(self.mb, B_mb)

            res, _diag = self.sp.teacher_force_logprobs_with_diagnostics(
                sequences=mb_E,
                tf_batch_size=tf_bs,
                compute_rb=True,
                with_grad=True,
                return_baseline_features=False,
            )

            # Build surrogate for this microbatch
            sur = torch.zeros((), device=device, dtype=torch.float32)
            T_mb = 0
            b_vals = []
            G_vals = []

            for b in range(len(res.logprobs)):
                # G==1 is enforced at E construction time
                lp: torch.Tensor = res.logprobs[b][0]  # [T], graph-carrying
                T = int(lp.numel())
                if T == 0:
                    continue
                T_mb += T

                if self.use_rb:
                    H_rb: Optional[torch.Tensor] = None
                    if res.rb_entropies_torch is None or len(res.rb_entropies_torch[b]) == 0:
                        raise RuntimeError(
                            "[delta-h approx] RB estimator active but rb_entropies_torch is missing."
                        )
                    H_rb = res.rb_entropies_torch[b][0]  # [T], graph-carrying
                    if H_rb is None or H_rb.numel() != T:
                        raise RuntimeError("[delta-h approx] RB tensor shape mismatch.")

                    # Entropy-to-go G_k (detached weight)
                    G = torch.flip(torch.cumsum(torch.flip(H_rb.detach(), dims=[0]), dim=0), dims=[0])  # [T]

                    # Baseline b_k
                    if self.baseline_kind == "hk":
                        b_k = H_rb.detach()
                    elif self.baseline_kind == "none":
                        b_k = torch.zeros_like(G)
                    else:
                        # fallback: Hk
                        b_k = H_rb.detach()

                    # Accumulate statistics for diagnostics
                    b_vals.append(b_k.mean().item())
                    G_vals.append(G.mean().item())

                    # Score term + RB term
                    sur = sur + ((G - b_k) * lp).sum() + H_rb.sum()

                else:
                    # Simple estimator: treat naive entropy as -logπ, no RB term
                    # (kept as an option; rarely used)
                    # Entropy-to-go over surprisal
                    G = torch.flip(torch.cumsum(torch.flip((-lp).detach(), dims=[0]), dim=0), dims=[0])
                    if self.baseline_kind == "hk":
                        b_k = (-lp).detach()
                    elif self.baseline_kind == "none":
                        b_k = torch.zeros_like(G)
                    else:
                        b_k = (-lp).detach()

                    b_vals.append(b_k.mean().item())
                    G_vals.append(G.mean().item())

                    sur = sur + ((G - b_k) * lp).sum()

            # Normalize this microbatch contribution
            scale = self._scale_for_average(B_total, T_total, B_mb, T_mb)
            total_tokens_used += T_mb
            baseline_means.append(float(sum(b_vals) / max(len(b_vals), 1)) if b_vals else 0.0)
            G_means.append(float(sum(G_vals) / max(len(G_vals), 1)) if G_vals else 0.0)

            (sur * float(scale)).backward()

        # Collect grads on intersection (CPU/fp32) and contract with v
        grads_named = {}
        for n, p in name_to_param.items():
            if p.grad is None:
                grads_named[n] = torch.zeros_like(p.detach()).to("cpu", torch.float32)
            else:
                grads_named[n] = p.grad.detach().to("cpu", torch.float32).clone()

        # Ensure v_named is CPU/fp32 as well
        v_named_cpu = {k: (v.detach().to("cpu", torch.float32) if isinstance(v, torch.Tensor) else v)
                       for k, v in v_named.items()}

        delta_h_per_lr = float(dot_named(grads_named, v_named_cpu).item())

        # Diagnostics: norms & cosine
        grad_vec = flatten_named(grads_named)
        v_vec = flatten_named(v_named_cpu)
        grad_l2 = float(torch.linalg.norm(grad_vec).item())
        v_l2 = float(torch.linalg.norm(v_vec).item())
        cosine = float(delta_h_per_lr / (max(grad_l2, 1e-12) * max(v_l2, 1e-12)))

        out = {
            "delta_h_per_lr": delta_h_per_lr,
            "num_sequences": int(E_batch["sequences"].shape[0]),
            "num_tokens": int(T_total),
            "norms": {"grad_l2": grad_l2, "v_l2": v_l2, "cosine": cosine},
            "intersection": {
                "matched": inter.matched,
                "v_named": inter.v_named_total,
                "trainable": inter.trainable_total,
                "missing_in_model": inter.missing_in_model,
                "missing_in_v": inter.missing_in_v,
            },
            "estimator": ("rb" if self.use_rb else "simple"),
            "baseline": {
                "kind": self.baseline_kind,
                "mean_b": float(sum(baseline_means) / max(len(baseline_means), 1)) if baseline_means else 0.0,
                "mean_G": float(sum(G_means) / max(len(G_means), 1)) if G_means else 0.0,
            },
        }
        if self.logger:
            self.logger.info(
                f"[delta-h approx] ⟨∇H, v⟩={out['delta_h_per_lr']:.6e} | "
                f"B={out['num_sequences']} T={out['num_tokens']} | "
                f"||g||={grad_l2:.3e} ||v||={v_l2:.3e} cos={cosine:.3f} | "
                f"est={out['estimator']} baseline={self.baseline_kind}"
            )
        return out

    # ---------------------------------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------------------------------
    def _select_params_intersecting(self, v_named: Dict[str, torch.Tensor]) -> "TOrderedDict[str, torch.nn.Parameter]":
        """
        Build an ordered mapping of model parameters that:
          (1) require grad,
          (2) their names are present in v_named.

        This ensures all later contractions are name-aligned and cover only the
        intended LoRA trainables (or optimizer params).
        """
        params: "TOrderedDict[str, torch.nn.Parameter]" = TOrderedDict()
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if name in v_named:
                params[name] = p
        return params

    def _intersections_report(
        self,
        name_to_param: "TOrderedDict[str, torch.nn.Parameter]",
        v_named: Dict[str, torch.Tensor],
    ) -> _Intersections:
        trainable_names = [n for n, p in self.model.named_parameters() if p.requires_grad]
        v_names = set(v_named.keys())
        model_names = set(trainable_names)
        inter = set(name_to_param.keys())

        missing_in_model = sorted(list(v_names - model_names))
        missing_in_v = sorted(list(model_names - v_names))[:16]  # cap log size

        return _Intersections(
            matched=len(inter),
            v_named_total=len(v_names),
            trainable_total=len(model_names),
            missing_in_model=missing_in_model,
            missing_in_v=missing_in_v,
        )

    def _scale_for_average(self, B_total: int, T_total: int, B_mb: int, T_mb: int) -> float:
        """
        Compute the scale factor so that Σ_mb scale_mb * L_sur_mb
        produces the desired expectation over the E-batch.

        - 'per_sequence': average over sequences ⇒ scale_mb = B_mb / B_total
        - 'per_token'   : average over tokens    ⇒ scale_mb = T_mb / T_total
        - 'none'        : leave as sum
        """
        if self.normalize == "per_sequence":
            return float(B_mb) / max(B_total, 1)
        elif self.normalize == "per_token":
            return float(T_mb) / max(T_total, 1)
        else:
            return 1.0

    def _count_total_gen_tokens(self, E_batch: Dict[str, Any]) -> int:
        """Sum of generated token counts across the entire E-batch (assumes G==1)."""
        gen_lens = E_batch.get("gen_lens", None)
        if gen_lens is None:
            # Fallback: try to infer from attention mask minus prompt length
            B = int(E_batch["sequences"].shape[0])
            G = int(E_batch["sequences"].shape[1])
            assert G == 1, f"E-batch expected G=1, got G={G}"
            total = 0
            for b in range(B):
                total += int(E_batch["attention_masks"][b, 0].sum().item()) - int(E_batch["prompt_lens"][b])
            return int(total)
        # gen_lens is [B][G] with G==1
        total = 0
        for bl in gen_lens:
            total += int(bl[0] if isinstance(bl, (list, tuple)) else bl)
        return int(total)

    def _iter_microbatches(self, E_batch: Dict[str, Any], mb_size: int) -> Generator["BatchedSequences", None, None]:
        """
        Yield BatchedSequences for slices of size at most `mb_size`.
        Assumes G==1 in E_batch.
        """
        seqs: torch.Tensor = E_batch["sequences"]
        atts: torch.Tensor = E_batch["attention_masks"]
        prompt_lens: List[int] = list(map(int, E_batch["prompt_lens"]))
        gen_lens: List[List[int]] = [[int(x[0])] if isinstance(x, (list, tuple)) else [int(x)]
                                     for x in E_batch["gen_lens"]]

        B, G, L = seqs.shape
        assert G == 1, f"E-batch expected G=1, got G={G}"

        for start in range(0, B, max(1, mb_size)):
            end = min(B, start + max(1, mb_size))
            # Slice tensors
            s_seqs = seqs[start:end, :, :].contiguous()
            s_atts = atts[start:end, :, :].contiguous()
            s_prompts = prompt_lens[start:end]
            s_gens = gen_lens[start:end]
            # responses_text is not used in TF; fill with empty strings
            s_text = [[ "" ] for _ in range(end - start)]
            yield BatchedSequences(
                sequences=s_seqs,
                prompt_lens=s_prompts,
                gen_lens=s_gens,
                attention_masks=s_atts,
                responses_text=s_text,
            )
