# delta_entropy_true.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from entropy_experiments.utils.sequence_processor import SequenceProcessor
from entropy_experiments.utils.param_overrides import build_functional_params_named


@dataclass
class _SeqStats:
    """
    Per-sequence stats extracted from SequenceProcessor TF no-grad result.
    All arrays are 1-D and aligned by sequence index.
    """
    seq_logprob: np.ndarray          # sum of token log-probs under the *current* model [N_seq]
    integrand_seq: np.ndarray        # sequence integrand for entropy estimate (SUM over generated tokens) [N_seq]
    T_tokens: np.ndarray             # token count per sequence [N_seq]


class DeltaEntropyTrue:
    """
    Ground-truth ΔH(η) via self-normalized importance sampling (SNIS) at the *sequence* level.

    - Forward passes are delegated to SequenceProcessor (SP) in runtime fp32.
    - Estimator matches your config:
        * estimator.use_simple_entropy_for_x = True  -> integrand = mean_t (-log p_token)
        * estimator.use_simple_entropy_for_x = False -> integrand = mean_t (RB entropy_t)
      (top_p is forced to 1.0 in SP config elsewhere; we rely on that.)

    - IS weights use sequence log-prob differences:
         lw_i = seq_logprob_new_i - seq_logprob_base_i
         w_i  = exp( clip(lw_i, -clip_c, +clip_c) - max_i(lw_i_clipped) )

    - We cache the η=0 baseline per E-batch (keyed by id(E_batch)) to avoid recomputation
      across η sweeps.

    Precision policy:
      * SP forward: fp32 (enforced upstream by model loader + SP init)
      * Entropy/log-softmax: controlled by SP (fp32 by default; you can toggle fp64 there)
      * IS math: float64 on CPU
    """

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        sequence_processor: SequenceProcessor,
        config: Dict[str, Any],
        logger: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.sp = sequence_processor
        self.config = config or {}
        self.logger = logger

        # Simple cache for baseline (η=0) per E-batch object
        # key -> dict with 'seq_stats' (_SeqStats) and scalar 'H_base_mean'
        self._base_cache: Dict[int, Dict[str, Any]] = {}

    # ---------- Public API ----------

    @torch.no_grad()
    def compute_delta_h_true(
        self,
        E_batch: Dict[str, Any],
        v_named: Dict[str, torch.Tensor],
        eta: float,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Compute ΔH_true(η) on the given E-batch using sequence-level SNIS.

        Args:
            E_batch: dict with BatchedSequences data (sequences, prompt_lens, gen_lens, attention_masks)
            v_named: named update direction (per-parameter tensors) such that Δθ(η) = η * v_named
            eta:     scalar step size
            cfg:     optional overrides, supports:
                     - clip_c (float, default=10.0)
                     - report_per_token (bool, default=False) [reserved]
                     - is_mode (str, default='snis') [reserved for future]

        Returns:
            ΔH_true(η) as a Python float.
        """
        cfg = cfg or {}
        clip_c = float(cfg.get("clip_c", 10.0))

        key = self._batch_key(E_batch)
        base_info = self._base_cache.get(key, None)
        if base_info is None:
            # --- Baseline pass (η = 0) ---
            base_stats, H_base_mean = self._score_batch_base(E_batch)
            base_info = {"seq_stats": base_stats, "H_base_mean": float(H_base_mean)}
            self._base_cache[key] = base_info

        base_stats: _SeqStats = base_info["seq_stats"]
        H_base_mean: float = base_info["H_base_mean"]

        # --- New pass at θ' = θ + η v (params-only, fp32) ---
        new_stats = self._score_batch_new(E_batch, v_named, eta)

        # --- SNIS reducer over sequences ---
        H_new_snis, ess, lw_stats = self._snis_reduce(
            base=base_stats, new=new_stats, clip_c=clip_c
        )

        if self.logger:
            self.logger.info(
                f"[SNIS] η={eta:g}  H_new={H_new_snis:.6e}  H_base={H_base_mean:.6e}  "
                f"ΔH_true={H_new_snis - H_base_mean:.6e}  ESS={ess:.1f}/{len(base_stats.seq_logprob)}  "
                f"lw[min/med/max]={lw_stats}"
            )

        return float(H_new_snis - H_base_mean)

    # ---------- Internals ----------

    def _batch_key(self, E_batch: Dict[str, Any]) -> int:
        """
        Lightweight identity key for caching within a single run. We assume the same
        E_batch object is reused across η sweeps.
        """
        return id(E_batch)

    @torch.no_grad()
    def _score_batch_base(self, E_batch: Dict[str, Any]) -> Tuple[_SeqStats, float]:
        """
        One TF no-grad pass on θ to collect sequence logprobs and the integrand.
        Returns per-sequence stats and the mean baseline H across sequences.
        """
        # Reconstruct BatchedSequences from E_batch data
        from entropy_experiments.utils.sequence_processor import BatchedSequences
        seqs = BatchedSequences(
            sequences=E_batch["sequences"],
            prompt_lens=E_batch["prompt_lens"], 
            gen_lens=E_batch["gen_lens"],  # Use the original gen_lens
            attention_masks=E_batch["attention_masks"],
            responses_text=[]  # Not needed for teacher forcing
        )
        use_simple = bool(self.config.get("estimator", {}).get("use_simple_entropy_for_x", False))

        # SP path (no params_override for baseline θ)
        # We keep tf_batch_size=1 for reproducibility; SP will loop over (B,G).
        lp, diag = self.sp.teacher_force_logprobs_with_diagnostics(
            sequences=seqs,
            with_grad=False,
            tf_batch_size=1,
            compute_rb=not use_simple,
            return_baseline_features=False,
            params_override=None,
            buffers_override=None,
        )

        stats = self._extract_seq_stats(lp, diag, use_simple=use_simple)
        H_base_mean = float(np.mean(stats.integrand_seq)) if stats.integrand_seq.size else 0.0
        return stats, H_base_mean

    @torch.no_grad()
    def _score_batch_new(
        self,
        E_batch: Dict[str, Any],
        v_named: Dict[str, torch.Tensor],
        eta: float,
    ) -> _SeqStats:
        """
        One TF no-grad pass on θ' = θ + η v using a *params-only* functional mapping in fp32.
        """
        # Reconstruct BatchedSequences from E_batch data
        from entropy_experiments.utils.sequence_processor import BatchedSequences
        seqs = BatchedSequences(
            sequences=E_batch["sequences"],
            prompt_lens=E_batch["prompt_lens"],
            gen_lens=E_batch["gen_lens"],  # Use the original gen_lens
            attention_masks=E_batch["attention_masks"],
            responses_text=[]  # Not needed for teacher forcing
        )
        use_simple = bool(self.config.get("estimator", {}).get("use_simple_entropy_for_x", False))

        # Build params-only mapping (fp32) – single source of truth for overrides.
        params_override, _ = build_functional_params_named(
            self.sp._unwrap(self.model),  # keep PEFT/LoRA wrapper intact; SP unwraps DDP only
            v_named=v_named,
            eta=float(eta),
            detach_params=True,
            detach_buffers=True,
            force_param_dtype=torch.float32,
            force_buffer_dtype=None,
        )

        lp, diag = self.sp.teacher_force_logprobs_with_diagnostics(
            sequences=seqs,
            with_grad=False,
            tf_batch_size=1,
            compute_rb=not use_simple,
            return_baseline_features=False,
            params_override=params_override,
            buffers_override=None,
        )

        return self._extract_seq_stats(lp, diag, use_simple=use_simple)

    def _extract_seq_stats(
        self,
        lp_res: Any,
        diag_res: Any,
        *,
        use_simple: bool,
    ) -> _SeqStats:
        """
        Flatten SP results into per-sequence numpy arrays.

        Expected lp_res fields (from SP._teacher_force_no_grad):
          - sequence_logprobs: List[List[float]]
          - entropies:         List[List[np.ndarray]]  # per-seq arrays of naive surprisal (−log p_token)
          - rb_entropies:      List[List[np.ndarray]]  # per-seq arrays if compute_rb=True; else empty
        """
        # Flatten helpers
        def _flatten_nested(nested):
            out = []
            for row in nested:
                out.extend(row)
            return out

        seq_logprob_list = _flatten_nested(lp_res.sequence_logprobs)  # per-seq float
        entropies_list   = _flatten_nested(lp_res.entropies)          # per-seq np.ndarray
        rb_list          = _flatten_nested(lp_res.rb_entropies)       # per-seq np.ndarray (or [])

        # Sequence integrand: per-sequence SUM of chosen tokenwise entropy
        integrand_vals = []
        T_tokens = []
        for i in range(len(seq_logprob_list)):
            if use_simple:
                arr = np.asarray(entropies_list[i]) if i < len(entropies_list) else np.array([])
            else:
                arr = np.asarray(rb_list[i]) if i < len(rb_list) and len(rb_list[i]) else np.array([])
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            T = int(arr.size)
            T_tokens.append(T)
            integrand_vals.append(float(arr.sum()) if T > 0 else 0.0)

        seq_lp_np = np.asarray(seq_logprob_list, dtype=np.float64)
        integ_np  = np.asarray(integrand_vals, dtype=np.float64)
        T_np      = np.asarray(T_tokens, dtype=np.int32)
        return _SeqStats(seq_logprob=seq_lp_np, integrand_seq=integ_np, T_tokens=T_np)

    def _snis_reduce(
        self,
        *,
        base: _SeqStats,
        new: _SeqStats,
        clip_c: float,
    ) -> Tuple[float, float, Tuple[float, float, float]]:
        """
        Self-normalized IS at sequence level.

        Returns:
            (H_new_snis, ESS, (lw_min, lw_med, lw_max))
        """
        # log-weights in float64 on CPU
        lw = (new.seq_logprob - base.seq_logprob).astype(np.float64, copy=False)
        if not np.isfinite(lw).all():
            lw = np.nan_to_num(lw, neginf=-1e3, posinf=1e3)

        if clip_c and clip_c > 0:
            lw = np.clip(lw, -clip_c, +clip_c)

        # stability shift
        m = float(lw.max()) if lw.size else 0.0
        w = np.exp(lw - m, dtype=np.float64)

        Z = float(w.sum()) if w.size else 1.0
        if Z <= 0.0 or not np.isfinite(Z):
            # Degenerate weights => fall back to simple mean (rare with top_p=1.0)
            H_new = float(np.mean(new.integrand_seq)) if new.integrand_seq.size else 0.0
            ess = 0.0
        else:
            H_new = float(np.dot(new.integrand_seq, w) / Z)
            ess = float((Z ** 2) / float(np.dot(w, w))) if w.size else 0.0

        # lw diagnostics
        if lw.size:
            lw_sorted = np.sort(lw)
            lw_min = float(lw_sorted[0])
            lw_max = float(lw_sorted[-1])
            lw_med = float(lw_sorted[lw_sorted.size // 2])
        else:
            lw_min = lw_med = lw_max = 0.0

        return H_new, ess, (lw_min, lw_med, lw_max)
