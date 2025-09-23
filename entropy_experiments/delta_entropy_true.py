# delta_entropy_true.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

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
        *,
        return_details: bool = False,
        symmetric_eta: Optional[float] = None,
        clip_overrides: Optional[Sequence[float]] = None,
    ) -> Union[float, Dict[str, Any]]:
        """Compute ΔH_true(η) on the given E-batch using sequence-level SNIS."""
        cfg = cfg or {}
        clip_c = float(cfg.get("clip_c", 10.0))
        report_per_token = bool(cfg.get("report_per_token", False))
        use_simple = bool(self.config.get("estimator", {}).get("use_simple_entropy_for_x", False))

        key = (id(E_batch), int(report_per_token), int(use_simple))
        base_info = self._base_cache.get(key)
        if base_info is None:
            base_stats, H_base_mean = self._score_batch_base(
                E_batch, report_per_token=report_per_token
            )
            base_info = {"seq_stats": base_stats, "H_base_mean": float(H_base_mean)}
            self._base_cache[key] = base_info

        base_stats: _SeqStats = base_info["seq_stats"]
        H_base_mean: float = base_info["H_base_mean"]

        new_stats = self._score_batch_new(E_batch, v_named, eta)

        clip_overrides = tuple(float(x) for x in (clip_overrides or ()))
        H_new_snis, ess, lw_stats, detail = self._snis_reduce(
            base=base_stats,
            new=new_stats,
            clip_c=clip_c,
            report_per_token=report_per_token,
            return_details=True,
        )
        delta_val = H_new_snis - H_base_mean

        if not return_details and not clip_overrides and symmetric_eta is None:
            if self.logger:
                lab = "per-token" if report_per_token else "per-sequence"
                total_T_base = int(base_stats.T_tokens.sum()) if base_stats.T_tokens.size else 0
                total_T_new = int(new_stats.T_tokens.sum()) if new_stats.T_tokens.size else 0
                self.logger.info(
                    f"[SNIS:{lab}] η={eta:g}  H_new={H_new_snis:.6e}  H_base={H_base_mean:.6e}  "
                    f"ΔH_true={delta_val:.6e}  ESS={ess:.1f}/{len(base_stats.seq_logprob)}  "
                    f"lw[min/med/max]={lw_stats}  T_base={total_T_base} T_new={total_T_new}"
                )
            return float(delta_val)

        detail = detail or {}
        diag: Dict[str, Any] = {
            "delta_h_true": float(delta_val),
            "base_entropy": float(H_base_mean),
            "ess": float(ess),
            "logweight_stats": {
                "min": lw_stats[0],
                "median": lw_stats[1],
                "max": lw_stats[2],
            },
            "clip_fraction": float(detail.get("clip_fraction", 0.0)),
            "weights_sum": float(detail.get("weights_sum", 0.0)),
            "normalized_weights": list(detail.get("norm_weights", [])),
            "log_weights": list(detail.get("log_weights", [])),
            "h_base": base_stats.integrand_seq.tolist(),
            "h_new": new_stats.integrand_seq.tolist(),
            "delta_h_seq": (new_stats.integrand_seq - base_stats.integrand_seq).tolist(),
            "token_counts": new_stats.T_tokens.astype(int).tolist(),
        }

        if clip_overrides:
            clip_data: Dict[str, Any] = {}
            for clip_value in clip_overrides:
                H_clip, ess_clip, _, clip_detail = self._snis_reduce(
                    base=base_stats,
                    new=new_stats,
                    clip_c=clip_value,
                    report_per_token=report_per_token,
                    return_details=True,
                )
                logs = clip_detail.get("log_weights") or []
                logs_arr = np.asarray(logs, dtype=np.float64) if logs else None
                clip_data[f"{clip_value:.6g}"] = {
                    "delta_h_true": float(H_clip - H_base_mean),
                    "ess": float(ess_clip),
                    "logweight_stats": {
                        "min": float(logs_arr.min()) if logs_arr is not None and logs_arr.size else 0.0,
                        "median": float(np.median(logs_arr)) if logs_arr is not None and logs_arr.size else 0.0,
                        "max": float(logs_arr.max()) if logs_arr is not None and logs_arr.size else 0.0,
                    },
                    "clip_fraction": float(clip_detail.get("clip_fraction", 0.0)),
                    "weights_sum": float(clip_detail.get("weights_sum", 0.0)),
                }
            diag["clip_overrides"] = clip_data
            if self.logger and clip_data:
                worst_key, worst_val = max(
                    clip_data.items(), key=lambda kv: kv[1].get("clip_fraction", 0.0)
                )
                clip_summary = {
                    "clip_fraction": worst_val.get("clip_fraction", 0.0),
                    "ess": worst_val.get("ess"),
                }
                self.logger.info(
                    f"[SNIS:clip] worst_clip={worst_key} summary={clip_summary}"
                )

        if symmetric_eta is not None:
            diag["symmetric_fd"] = self._compute_symmetric_fd(
                base_stats=base_stats,
                H_base=H_base_mean,
                E_batch=E_batch,
                v_named=v_named,
                eta=float(symmetric_eta),
                clip_c=clip_c,
                report_per_token=report_per_token,
            )

        if self.logger:
            diag_summary = {
                "clip_fraction": diag.get("clip_fraction"),
                "weights_sum": diag.get("weights_sum"),
                "ess": diag.get("ess"),
            }
            self.logger.info(f"[SNIS:detail] summary={diag_summary}")

        return diag

    @torch.no_grad()
    def compute_delta_h_true_multi(
        self,
        E_batch: Dict[str, Any],
        v_named_list: List[Dict[str, torch.Tensor]],
        eta: float,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if not v_named_list:
            return []

        cfg = cfg or {}
        clip_c = float(cfg.get("clip_c", 10.0))
        report_per_token = bool(cfg.get("report_per_token", False))
        use_simple = bool(self.config.get("estimator", {}).get("use_simple_entropy_for_x", False))

        key = (id(E_batch), int(report_per_token), int(use_simple))
        base_info = self._base_cache.get(key)
        if base_info is None:
            base_stats, H_base_mean = self._score_batch_base(
                E_batch, report_per_token=report_per_token
            )
            base_info = {"seq_stats": base_stats, "H_base_mean": float(H_base_mean)}
            self._base_cache[key] = base_info

        base_stats: _SeqStats = base_info["seq_stats"]
        H_base_mean: float = base_info["H_base_mean"]

        new_stats_list = self._score_batch_new_multi(E_batch, v_named_list, eta)

        results: List[Dict[str, Any]] = []
        for new_stats in new_stats_list:
            H_new_snis, ess, lw_stats, detail = self._snis_reduce(
                base=base_stats,
                new=new_stats,
                clip_c=clip_c,
                report_per_token=report_per_token,
                return_details=True,
            )
            delta_val = H_new_snis - H_base_mean

            diag: Dict[str, Any] = {
                "delta_h_true": float(delta_val),
                "base_entropy": float(H_base_mean),
                "ess": float(ess),
                "logweight_stats": {
                    "min": lw_stats[0],
                    "median": lw_stats[1],
                    "max": lw_stats[2],
                },
                "clip_fraction": float(detail.get("clip_fraction", 0.0)),
                "weights_sum": float(detail.get("weights_sum", 0.0)),
                "normalized_weights": list(detail.get("norm_weights", [])),
                "log_weights": list(detail.get("log_weights", [])),
                "h_base": base_stats.integrand_seq.tolist(),
                "h_new": new_stats.integrand_seq.tolist(),
                "delta_h_seq": (new_stats.integrand_seq - base_stats.integrand_seq).tolist(),
                "token_counts": new_stats.T_tokens.astype(int).tolist(),
            }

            results.append(diag)

        return results

    def _batch_key(self, E_batch: Dict[str, Any]) -> int:
        """
        Lightweight identity key for caching within a single run. We assume the same
        E_batch object is reused across η sweeps.
        """
        return id(E_batch)

    @torch.no_grad()
    def _score_batch_base(self, E_batch: Dict[str, Any], *, report_per_token: bool = False) -> Tuple[_SeqStats, float]:
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
        if report_per_token:
            num = float(stats.integrand_seq.sum()) if stats.integrand_seq.size else 0.0
            den = float(stats.T_tokens.sum()) if stats.T_tokens.size else 1.0
            H_base_mean = (num / max(den, 1.0))
        else:
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

    @torch.no_grad()
    def _score_batch_new_multi(
        self,
        E_batch: Dict[str, Any],
        v_named_list: List[Dict[str, torch.Tensor]],
        eta: float,
    ) -> List[_SeqStats]:
        if not v_named_list:
            return []

        from entropy_experiments.utils.sequence_processor import BatchedSequences

        seqs = BatchedSequences(
            sequences=E_batch["sequences"],
            prompt_lens=E_batch["prompt_lens"],
            gen_lens=E_batch["gen_lens"],
            attention_masks=E_batch["attention_masks"],
            responses_text=[],
        )
        use_simple = bool(self.config.get("estimator", {}).get("use_simple_entropy_for_x", False))

        params_overrides: List[Dict[str, torch.Tensor]] = []
        for v_named in v_named_list:
            params_override, _ = build_functional_params_named(
                self.sp._unwrap(self.model),
                v_named=v_named,
                eta=float(eta),
                detach_params=True,
                detach_buffers=True,
                force_param_dtype=torch.float32,
                force_buffer_dtype=None,
            )
            params_overrides.append(params_override)

        lp_diag_list = self.sp.teacher_force_logprobs_with_diagnostics(
            sequences=seqs,
            with_grad=False,
            tf_batch_size=1,
            compute_rb=not use_simple,
            return_baseline_features=False,
            params_override=params_overrides,
            buffers_override=None,
        )

        stats_list: List[_SeqStats] = []
        for lp, diag in lp_diag_list:
            stats_list.append(self._extract_seq_stats(lp, diag, use_simple=use_simple))
        return stats_list

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
        report_per_token: bool = False,
        return_details: bool = False,
    ) -> Union[
        Tuple[float, float, Tuple[float, float, float]],
        Tuple[float, float, Tuple[float, float, float], Dict[str, Any]],
    ]:
        """Self-normalized IS at sequence level."""
        lw_raw = (new.seq_logprob - base.seq_logprob).astype(np.float64, copy=False)
        if not np.isfinite(lw_raw).all():
            lw_raw = np.nan_to_num(lw_raw, neginf=-1e3, posinf=1e3)

        lw = lw_raw.copy()
        clip_fraction = 0.0
        if clip_c and clip_c > 0:
            lw = np.clip(lw, -clip_c, +clip_c)
            if lw_raw.size:
                clip_fraction = float(np.mean(np.abs(lw_raw) > clip_c))

        m = float(lw.max()) if lw.size else 0.0
        w = np.exp(lw - m, dtype=np.float64)

        Z = float(w.sum()) if w.size else 1.0
        if Z <= 0.0 or not np.isfinite(Z):
            if report_per_token:
                num = float(new.integrand_seq.sum()) if new.integrand_seq.size else 0.0
                den = float(new.T_tokens.sum()) if new.T_tokens.size else 1.0
                H_new = num / max(den, 1.0)
            else:
                H_new = float(np.mean(new.integrand_seq)) if new.integrand_seq.size else 0.0
            ess = 0.0
            norm_weights = np.zeros_like(lw_raw)
        else:
            if report_per_token:
                num = float(np.dot(new.integrand_seq, w))
                den = float(np.dot(new.T_tokens.astype(np.float64, copy=False), w))
                den = den if np.isfinite(den) and den > 0.0 else 1.0
                H_new = num / den
            else:
                H_new = float(np.dot(new.integrand_seq, w) / Z)
            ess = float((Z ** 2) / float(np.dot(w, w))) if w.size else 0.0
            norm_weights = w / Z if Z else np.zeros_like(w)

        if lw.size:
            lw_sorted = np.sort(lw)
            lw_min = float(lw_sorted[0])
            lw_max = float(lw_sorted[-1])
            lw_med = float(lw_sorted[lw_sorted.size // 2])
        else:
            lw_min = lw_med = lw_max = 0.0

        if return_details:
            details = {
                "clip_fraction": clip_fraction,
                "norm_weights": norm_weights.tolist(),
                "log_weights": lw_raw.tolist(),
                "weights_sum": float(Z),
            }
            return H_new, ess, (lw_min, lw_med, lw_max), details
        return H_new, ess, (lw_min, lw_med, lw_max)


    def _compute_symmetric_fd(
        self,
        *,
        base_stats: _SeqStats,
        H_base: float,
        E_batch: Dict[str, Any],
        v_named: Dict[str, torch.Tensor],
        eta: float,
        clip_c: float,
        report_per_token: bool,
    ) -> Dict[str, Any]:
        eta_abs = abs(float(eta))
        if eta_abs == 0.0:
            return {"eta": 0.0, "finite_difference": 0.0}

        stats_plus = self._score_batch_new(E_batch, v_named, eta_abs)
        H_plus, ess_plus, _, detail_plus = self._snis_reduce(
            base=base_stats,
            new=stats_plus,
            clip_c=clip_c,
            report_per_token=report_per_token,
            return_details=True,
        )
        delta_plus = H_plus - H_base

        stats_minus = self._score_batch_new(E_batch, v_named, -eta_abs)
        H_minus, ess_minus, _, detail_minus = self._snis_reduce(
            base=base_stats,
            new=stats_minus,
            clip_c=clip_c,
            report_per_token=report_per_token,
            return_details=True,
        )
        delta_minus = H_minus - H_base

        finite_diff = (delta_plus - delta_minus) / (2.0 * eta_abs)

        def _log_stats(detail: Dict[str, Any]) -> Dict[str, float]:
            logs = detail.get("log_weights") or []
            if not logs:
                return {"min": 0.0, "median": 0.0, "max": 0.0}
            arr = np.asarray(logs, dtype=np.float64)
            return {
                "min": float(arr.min()),
                "median": float(np.median(arr)),
                "max": float(arr.max()),
            }

        return {
            "eta": eta_abs,
            "delta_plus": float(delta_plus),
            "delta_minus": float(delta_minus),
            "finite_difference": float(finite_diff),
            "ess_plus": float(ess_plus),
            "ess_minus": float(ess_minus),
            "logweight_stats_plus": _log_stats(detail_plus),
            "logweight_stats_minus": _log_stats(detail_minus),
            "clip_fraction_plus": float(detail_plus.get("clip_fraction", 0.0)),
            "clip_fraction_minus": float(detail_minus.get("clip_fraction", 0.0)),
            "weights_sum_plus": float(detail_plus.get("weights_sum", 0.0)),
            "weights_sum_minus": float(detail_minus.get("weights_sum", 0.0)),
        }
