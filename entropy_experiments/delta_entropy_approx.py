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
import numpy as np
import torch
from torch.func import jvp, functional_call
from entropy_experiments.utils.sequence_processor import (
    BatchedSequences,
    LogprobResults,
    DiagnosticsResults,
)


from entropy_experiments.utils.jvp_utils import (
    snapshot_base_functional_state, intersect_jvp_primals_tangents,
    make_seq_outputs_closure, make_mb_outputs_closure,
)
from entropy_experiments.baselines import get_strategy, EmaState, build_weights_base, get_timewise_strategy
from entropy_experiments.baselines.strategies import RidgeConfig
from entropy_experiments.utils.param_overrides import (
    build_functional_params_named,
    merge_params_and_buffers,
)


@dataclass
class _Intersections:
    matched: int
    v_named_total: int
    trainable_total: int
    missing_in_model: List[str]
    missing_in_v: List[str]


class DeltaEntropyApprox:
    DEFAULT_CV_FEATURES: Tuple[str, ...] = (
        "length_log",
        "sum_w",
        "sum_w2",
        "rb_entropy_sum",
        "var_logp",
    )

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
        # Regression baseline configuration (guarded; defaults keep behavior unchanged)
        self.baseline_reg_l2 = float(baseline_cfg.get("regression_l2", 0.0))
        self.baseline_reg_intercept = bool(baseline_cfg.get("include_intercept", True))
        self.baseline_reg_fit_dtype = str(baseline_cfg.get("fit_dtype", "float64")).lower()
        self.baseline_reg_normalize = bool(baseline_cfg.get("normalize", False))
        self.baseline_reg_clip_min = baseline_cfg.get("clip_min", None)
        self.baseline_reg_clip_max = baseline_cfg.get("clip_max", None)

        est_cfg = (self.cfg.get("estimator", {}) or {})
        self.use_rb = not bool(est_cfg.get("use_simple_entropy_for_x", False))

        self.debug = bool(self.cfg.get("debug", False)) or bool(approx_cfg.get("debug", False))

        simple_baseline_cfg = (approx_cfg.get("simple_baseline", {}) or {})
        self.simple_baseline_kind = str(simple_baseline_cfg.get("kind", "time_loo")).lower()

        self._ema_beta = float(baseline_cfg.get("ema_beta", 0.95))
        self._pos_bins = int(baseline_cfg.get("pos_bins", 32))
        self._ema_resid = torch.zeros(self._pos_bins, dtype=torch.float32)  # CPU ok
        self._ema_cnt   = torch.zeros(self._pos_bins, dtype=torch.int64)


        # Variance diagnostics
        var_cfg = (approx_cfg.get("variance", {}) or {})
        self.var_enabled = bool(var_cfg.get("enabled", False))
        self.var_jackknife = bool(var_cfg.get("jackknife", True))

        # Ridge baseline (JVP helper) knobs
        ridge_cfg = (approx_cfg.get("ridge", {}) or {})
        self.ridge_lambda = float(ridge_cfg.get("lambda", 1e-3))
        self.ridge_eps = float(ridge_cfg.get("eps", 1e-8))

        # Cache for baseline entropy estimates keyed by E-batch identity.
        self._h_base_cache: Dict[int, float] = {}



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

    def _compute_h_base_mean(self, E_batch: Dict[str, Any]) -> float:
        """Return the per-token baseline entropy for the provided E batch."""

        if self.normalize != "per_token":
            return 0.0

        key = id(E_batch)
        cached = self._h_base_cache.get(key)
        if cached is not None:
            return cached

        seqs = BatchedSequences(
            sequences=E_batch["sequences"],
            prompt_lens=E_batch["prompt_lens"],
            gen_lens=E_batch["gen_lens"],
            attention_masks=E_batch["attention_masks"],
            responses_text=[],
        )

        lp, _ = self.sp.teacher_force_logprobs_with_diagnostics(
            sequences=seqs,
            with_grad=False,
            tf_batch_size=1,
            compute_rb=True,
            return_baseline_features=False,
            params_override=None,
            buffers_override=None,
        )

        if lp.rb_entropies:
            ent_nested = lp.rb_entropies
        else:
            ent_nested = lp.entropies

        rb_sums: List[float] = []
        token_counts: List[int] = []
        for row in ent_nested:
            for arr in row:
                if arr is None:
                    continue
                arr_np = np.asarray(arr, dtype=np.float64)
                rb_sums.append(float(arr_np.sum()))
                token_counts.append(int(arr_np.size))

        total_tokens = int(np.sum(token_counts)) if token_counts else 0
        if total_tokens > 0:
            h_base = float(np.sum(rb_sums) / total_tokens)
        else:
            h_base = 0.0

        self._h_base_cache[key] = h_base
        return h_base

    def compute_delta_h_approx(
        self,
        *,
        E_batch: Dict[str, Any],
        v_named: Dict[str, torch.Tensor],
        include_quadratic: bool = False,
        return_per_sequence: bool = False,
        control_variate_features: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Compute the linearized entropy change and optional diagnostics."""
        features = tuple(control_variate_features or self.DEFAULT_CV_FEATURES)
        result = self.compute_delta_h_approx_jvp(
            E_batch=E_batch,
            v_named=v_named,
            return_per_sequence=return_per_sequence,
            features=features,
        )
        if include_quadratic:
            quad = self.compute_dir_linear_and_quadratic_jvp(
                E_batch=E_batch,
                v_named=v_named,
                return_per_sequence=return_per_sequence,
            )
            result["quadratic"] = quad
            result["eta_star"] = quad.get("eta_star")
            if return_per_sequence and "per_sequence" in result and "per_sequence_vhvv" in quad:
                seq_vals = quad.get("per_sequence_vhvv") or []
                records = result.get("per_sequence") or []
                if len(records) == len(seq_vals):
                    for rec, vhvv in zip(records, seq_vals):
                        rec["vhvv"] = vhvv
                else:
                    # Length mismatch implies diagnostics bug; preserve raw list for debugging
                    result.setdefault("per_sequence_quadratic", seq_vals)
        return result

    def compute_delta_h_approx_jvp(
        self,
        *,
        E_batch: Dict[str, Any],
        v_named: Dict[str, torch.Tensor],
        return_per_sequence: bool = False,
        features: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Forward-mode JVP estimator with optional per-sequence diagnostics."""
        device = next(self.model.parameters()).device
        mdl = self.model
        mdl.eval()

        base_map = snapshot_base_functional_state(self.model)
        names, primals, tangents = intersect_jvp_primals_tangents(self.model, base_map, v_named)
        feature_tuple = tuple(features or self.DEFAULT_CV_FEATURES)

        B_total = int(E_batch["sequences"].shape[0])
        T_total = self._count_total_gen_tokens(E_batch)
        apply_denom_correction = self.normalize == "per_token"
        h_base_mean = self._compute_h_base_mean(E_batch) if apply_denom_correction else 0.0

        contribs_mb: List[float] = []
        total_tokens_used = 0
        scale_sum = 0.0

        baseline_kind = str(self.baseline_kind).lower()

        for mb_E in self._iter_microbatches(E_batch, self.mb):
            B_mb = int(mb_E.sequences.shape[0])
            tf_bs = min(self.mb, B_mb)

            ema_state = None
            if baseline_kind == "hk_ema":
                ema_state = EmaState(
                    pos_bins=self._pos_bins,
                    ema_beta=self._ema_beta,
                    ema_resid=self._ema_resid,
                    ema_cnt=self._ema_cnt,
                )
            ridge_cfg = RidgeConfig(lambda_=self.ridge_lambda, eps=self.ridge_eps)
            w_list = build_weights_base(
                kind=baseline_kind,
                model=self.model,
                sp=self.sp,
                mb_E=mb_E,
                tf_bs=tf_bs,
                ema=ema_state,
                ridge=ridge_cfg,
            )

            input_ids_mb = mb_E.sequences[:, 0].to(device=device)
            attention_mask_mb = mb_E.attention_masks[:, 0].to(device=device)
            prompt_lens = [int(x) for x in mb_E.prompt_lens]
            T_list = [int(mb_E.gen_lens[b][0]) for b in range(B_mb)]
            T_mb = sum(t for t in T_list if t > 0)
            if T_mb == 0:
                continue

            w_cat = torch.cat([w_list[b] for b in range(B_mb) if T_list[b] > 0], dim=0)

            f_mb = make_mb_outputs_closure(
                model=self.model,
                base_map=base_map,
                names=list(names),
                input_ids_mb=input_ids_mb,
                attention_mask_mb=attention_mask_mb,
                prompt_lens=prompt_lens,
                gen_lens=T_list,
            )
            (logp_cat, H_sum), (j_logp_cat, j_H_sum) = jvp(f_mb, (primals,), (tangents,))

            offsets: List[Tuple[int, int, int]] = []
            acc = 0
            for b in range(B_mb):
                if T_list[b] > 0:
                    offsets.append((b, acc, acc + T_list[b]))
                    acc += T_list[b]

            denom_corr_mb = 0.0
            if apply_denom_correction and offsets:
                for b, s, e in offsets:
                    j_sum = float(j_logp_cat[s:e].sum().item())
                    denom_corr_mb += -h_base_mean * float(T_list[b]) * j_sum

            mb_contrib = float(((w_cat.to(j_logp_cat) * j_logp_cat).sum() + j_H_sum).item())
            mb_contrib += denom_corr_mb

            scale = self._scale_for_derivative(B_total, T_total)
            total_tokens_used += T_mb
            scale_sum += float(scale)
            contribs_mb.append(mb_contrib * float(scale))

        delta_h_per_lr = float(sum(contribs_mb))
        out: Dict[str, Any] = {
            "delta_h_per_lr": delta_h_per_lr,
            "num_sequences": B_total,
            "num_tokens": int(T_total),
            "estimator": "rb",
            "baseline": {"kind": self.baseline_kind},
            "method": "jvp",
            "audit": {"scale_sum": scale_sum, "total_tokens_used": total_tokens_used},
        }
        if self.var_enabled:
            from entropy_experiments.utils.variance import compute_variance_info

            out["variance"] = compute_variance_info(
                contribs_mb,
                debug=self.debug,
                use_jackknife=self.var_jackknife,
            )

        if return_per_sequence:
            out.update(
                self._compute_per_sequence_jvp(
                    E_batch=E_batch,
                    v_named=v_named,
                    base_map=base_map,
                    names=tuple(names),
                    primals=tuple(primals),
                    tangents=tuple(tangents),
                    B_total=B_total,
                    T_total=T_total,
                    H_base_mean=h_base_mean,
                    features=feature_tuple,
                )
            )

        if self.logger:
            self.logger.info(
                f"[delta-h approx JVP] gdotv={out['delta_h_per_lr']:.6e} | B={B_total} T={T_total} | baseline={self.baseline_kind}"
            )
            if self.var_enabled:
                vinfo = out.get("variance", {})
                self.logger.info(
                    f"[delta-h approx JVP][variance] shards={vinfo.get('num_shards', 0)} "
                    f"SE(shard)={vinfo.get('se_shard', 0.0):.3e} "
                    f"SE(jack)={vinfo.get('se_jackknife', 0.0):.3e}"
                )
            self.logger.info(
                f"[delta-h approx JVP][audit] deriv_scale={self._scale_for_derivative(B_total, T_total):.6e}, "
                f"sum_deriv_scales={scale_sum:.6e} total_tokens_used={total_tokens_used} pre_count={T_total}"
            )
        return out

    def _compute_per_sequence_jvp(
        self,
        *,
        E_batch: Dict[str, Any],
        v_named: Dict[str, torch.Tensor],
        base_map: Dict[str, torch.Tensor],
        names: Tuple[str, ...],
        primals: Tuple[torch.Tensor, ...],
        tangents: Tuple[torch.Tensor, ...],
        B_total: int,
        T_total: int,
        H_base_mean: float,
        features: Tuple[str, ...],
    ) -> Dict[str, Any]:
        scale = self._scale_for_derivative(B_total, T_total)
        baseline_kind = str(self.baseline_kind).lower()
        records: List[Dict[str, Any]] = []
        total_tokens_used = 0
        seq_index = 0
        apply_denom_correction = self.normalize == "per_token"

        ema_state = None
        if baseline_kind == "hk_ema":
            ema_state = EmaState(
                pos_bins=self._pos_bins,
                ema_beta=self._ema_beta,
                ema_resid=self._ema_resid,
                ema_cnt=self._ema_cnt,
            )
        ridge_cfg = RidgeConfig(lambda_=self.ridge_lambda, eps=self.ridge_eps)

        for mb_E in self._iter_microbatches(E_batch, self.mb):
            B_mb = int(mb_E.sequences.shape[0])
            tf_bs = min(self.mb, B_mb)
            w_list = build_weights_base(
                kind=baseline_kind,
                model=self.model,
                sp=self.sp,
                mb_E=mb_E,
                tf_bs=tf_bs,
                ema=ema_state,
                ridge=ridge_cfg,
            )

            for b in range(B_mb):
                T_b = int(mb_E.gen_lens[b][0])
                if T_b <= 0:
                    records.append({
                        "index": seq_index,
                        "gdotv": 0.0,
                        "length": 0,
                        "sum_logp": 0.0,
                        "mean_logp": 0.0,
                        "var_logp": 0.0,
                        "max_logp": -1e30,
                        "min_logp": 1e30,
                        "extras": {},
                        "features": {},
                    })
                    seq_index += 1
                    continue

                total_tokens_used += T_b
                w_b = w_list[b]
                sum_w = float(w_b.sum().item())
                sum_w2 = float((w_b ** 2).sum().item())
                sum_w_abs = float(w_b.abs().sum().item())

                seq_view = self._slice_single_sequence(mb_E, b)
                device = primals[0].device
                input_ids = seq_view.sequences[:, 0].to(device=device)
                att_mask = seq_view.attention_masks[:, 0].to(device=device)
                prompt_lens = [int(seq_view.prompt_lens[0])]
                gen_lens = [int(seq_view.gen_lens[0][0])]

                f_seq = make_mb_outputs_closure(
                    model=self.model,
                    base_map=base_map,
                    names=list(names),
                    input_ids_mb=input_ids,
                    attention_mask_mb=att_mask,
                    prompt_lens=prompt_lens,
                    gen_lens=gen_lens,
                )
                (logp_cat, H_sum), (j_logp_cat, j_H_sum) = jvp(
                    f_seq,
                    (primals,),
                    (tangents,),
                )

                wjlogp_sum = float((w_b.to(j_logp_cat) * j_logp_cat).sum().item())
                jH_sum = float(j_H_sum.item())
                j_logp_sum = float(j_logp_cat.sum().item())
                denom_corr = 0.0
                if apply_denom_correction:
                    denom_corr = -H_base_mean * float(T_b) * j_logp_sum
                gdotv_i = float(scale * (wjlogp_sum + jH_sum + denom_corr))

                lp = logp_cat.detach()
                length = int(T_b)
                sum_lp = float(lp.sum().item())
                mean_lp = sum_lp / max(length, 1)
                if length > 0:
                    diffs = lp - lp.mean()
                    var_lp = float((diffs.pow(2).sum() / float(length)).item())
                    max_lp = float(lp.max().item())
                    min_lp = float(lp.min().item())
                else:
                    var_lp, max_lp, min_lp = 0.0, -1e30, 1e30

                sum_abs_jlogp = float(j_logp_cat.abs().sum().item())
                mean_abs_jlogp = sum_abs_jlogp / max(length, 1)
                rb_sum = float(H_sum.detach().item())

                extras = {
                    "rb_entropy_sum": rb_sum,
                    "sum_w": sum_w,
                    "sum_w2": sum_w2,
                    "sum_w_abs": sum_w_abs,
                    "wjlogp_sum": wjlogp_sum,
                    "jH_sum": jH_sum,
                    "sum_abs_jlogp": sum_abs_jlogp,
                    "mean_abs_jlogp": mean_abs_jlogp,
                    "token_share": float(length / max(T_total, 1)),
                }

                if apply_denom_correction:
                    extras["denom_corr_scaled"] = float(scale * denom_corr)
                    extras["denom_corr_unscaled"] = float(denom_corr)

                feature_vals: Dict[str, float] = {}
                for name in dict.fromkeys(features):
                    if name == "length_log":
                        feature_vals[name] = float(math.log(max(length, 1)))
                    elif name == "sum_w":
                        feature_vals[name] = sum_w
                    elif name == "sum_w2":
                        feature_vals[name] = sum_w2
                    elif name == "rb_entropy_sum":
                        feature_vals[name] = rb_sum
                    elif name == "var_logp":
                        feature_vals[name] = var_lp
                    elif name == "length":
                        feature_vals[name] = float(length)
                    elif name == "sum_logp":
                        feature_vals[name] = sum_lp
                    elif name == "mean_logp":
                        feature_vals[name] = mean_lp
                    elif name == "mean_abs_jlogp":
                        feature_vals[name] = mean_abs_jlogp
                    else:
                        if name in extras:
                            feature_vals[name] = float(extras[name])

                records.append({
                    "index": seq_index,
                    "gdotv": gdotv_i,
                    "length": length,
                    "sum_logp": sum_lp,
                    "mean_logp": mean_lp,
                    "var_logp": var_lp,
                    "max_logp": max_lp,
                    "min_logp": min_lp,
                    "extras": extras,
                    "features": feature_vals,
                })
                seq_index += 1

        meta = {
            "B_total": B_total,
            "T_total": int(T_total),
            "scale": float(scale),
            "baseline_kind": baseline_kind,
            "total_tokens_used": int(total_tokens_used),
            "features": list(features),
            "H_base_mean": float(H_base_mean) if apply_denom_correction else None,
        }
        return {"per_sequence": records, "per_sequence_meta": meta}

    def _slice_single_sequence(self, mb_E: BatchedSequences, idx: int) -> BatchedSequences:
        seq = mb_E.sequences[idx:idx + 1]
        att = mb_E.attention_masks[idx:idx + 1]
        prom = [int(mb_E.prompt_lens[idx])]
        gen = [[int(mb_E.gen_lens[idx][0])]]
        resp = [[mb_E.responses_text[idx][0] if mb_E.responses_text else ""]]
        return BatchedSequences(
            sequences=seq,
            prompt_lens=prom,
            gen_lens=gen,
            attention_masks=att,
            responses_text=resp,
        )
    def compute_dir_linear_and_quadratic_jvp(
        self,
        *,
        E_batch: Dict[str, Any],
        v_named: Dict[str, torch.Tensor],
        etas: Optional[List[float]] = None,
        return_per_sequence: bool = False,
    ) -> Dict[str, Any]:
        """
        Compute the linear term g·v and the quadratic curvature term v^T H v for the RB surrogate on E-batch
        using nested forward-mode JVP:
            g·v     = jvp(f, θ; v)
            v^T H v = jvp( θ ↦ jvp(f, θ; v), θ; v )
        where f aggregates over the microbatch with detached weights w_t = (G_t - b_t).
        """
        device = next(self.model.parameters()).device
        mdl = self.model; mdl.eval()

        # --- IMPORTANT: Disable any model mutations that might call .requires_grad_() in forward ---
        restore_ckpt = False
        restore_inputreq = False
        try:
            if getattr(mdl, "is_gradient_checkpointing", False):
                try:
                    mdl.gradient_checkpointing_disable()
                    restore_ckpt = True
                except Exception:
                    pass
            if hasattr(mdl, "disable_input_require_grads"):
                try:
                    # Prevent forward hooks that do x.requires_grad_() on inputs
                    mdl.disable_input_require_grads()
                    restore_inputreq = True
                except Exception:
                    pass

            # Eta list only for reporting predicted ratio
            est_cfg = (self.cfg.get("estimator", {}) or {})
            if etas is None:
                if bool(est_cfg.get("eta_sweep", False)) and est_cfg.get("eta_list"):
                    etas = [float(x) for x in est_cfg.get("eta_list")]
                else:
                    etas = [float(est_cfg.get("single_eta", 2e-6))]

            # Base functional state (params+buffers), LoRA-only intersection
            p_dict, b_dict = build_functional_params_named(
                self.model,
                v_named=None,
                eta=0.0,
                strict=True,
                allow_frozen_updates=False,
                detach_params=True,
                detach_buffers=True,
                force_param_dtype=torch.float32,
                force_buffer_dtype=None,
            )
            base_map = merge_params_and_buffers(p_dict, b_dict)
            names: List[str] = []
            primals: List[torch.Tensor] = []
            tangents: List[torch.Tensor] = []
            for n, p in self.model.named_parameters():
                if (not p.requires_grad) or (n not in v_named):
                    continue
                names.append(n)
                primals.append(base_map[n].to(device=device))
                tangents.append(v_named[n].to(device=device, dtype=base_map[n].dtype))
            if len(names) == 0:
                raise ValueError("[Phase1 JVP] No intersecting trainables for JVP.")
            primals = tuple(primals); tangents = tuple(tangents)

            B_total = int(E_batch["sequences"].shape[0])
            T_total = self._count_total_gen_tokens(E_batch)
            apply_denom_correction = self.normalize == "per_token"
            h_base_mean = self._compute_h_base_mean(E_batch) if apply_denom_correction else 0.0

            gdotv_total = 0.0
            vHvv_total = 0.0
            scale_sum = 0.0
            total_tokens_used = 0
            g_contribs_mb: List[float] = []
            h_contribs_mb: List[float] = []
            per_seq_vhvv: List[float] = [] if return_per_sequence else []

            # ---- Microbatch loop: single forward + nested JVPs per microbatch ----
            for mb_E in self._iter_microbatches(E_batch, self.mb):
                B_mb = int(mb_E.sequences.shape[0])
                tf_bs = min(self.mb, B_mb)

                baseline_kind = str(self.baseline_kind).lower()
                ema_state = None
                if baseline_kind == "hk_ema":
                    ema_state = EmaState(
                        pos_bins=self._pos_bins,
                        ema_beta=self._ema_beta,
                        ema_resid=self._ema_resid,
                        ema_cnt=self._ema_cnt,
                    )
                    ridge_cfg = RidgeConfig(lambda_=getattr(self, "ridge_lambda", 1e-3),
                                            eps=getattr(self, "ridge_eps", 1e-8))
                    w_list = build_weights_base(
                        kind=baseline_kind,
                        model=self.model,
                        sp=self.sp,
                        mb_E=mb_E,
                        tf_bs=tf_bs,
                        ema=ema_state,
                        ridge=ridge_cfg,
                    )

                input_ids_mb = mb_E.sequences[:, 0].to(device=device)          # [B_mb, L]
                attention_mask_mb = mb_E.attention_masks[:, 0].to(device=device)
                prompt_lens = [int(x) for x in mb_E.prompt_lens]
                T_list = [int(mb_E.gen_lens[b][0]) for b in range(B_mb)]
                T_mb = sum(t for t in T_list if t > 0)
                if T_mb == 0:
                    if return_per_sequence:
                        per_seq_vhvv.extend([0.0] * B_mb)
                    continue
                total_tokens_used += T_mb
                w_cat = torch.cat([w_list[b] for b in range(B_mb) if T_list[b] > 0], dim=0)

                # Microbatch closures WITHOUT any toggling inside (safe for nested JVP):
                # F_mb: returns (logp_cat[T_mb], H_sum[scalar]) for the linear term
                def F_mb(params_tuple):
                    params_map = dict(base_map)
                    for n, t in zip(names, params_tuple):
                        params_map[n] = t
                    out = functional_call(
                        mdl, params_map, (input_ids_mb,),
                        {"attention_mask": attention_mask_mb, "use_cache": False}
                    )
                    logits = out.logits  # [B_mb, L, V]
                    logp_pieces = []
                    H_sum = torch.zeros((), dtype=torch.float32, device=logits.device)
                    for b in range(B_mb):
                        T_b = int(T_list[b])
                        if T_b <= 0:
                            continue
                        start = max(int(prompt_lens[b]) - 1, 0)
                        end = start + T_b
                        logits_slice = logits[b:b+1, start:end, :]                # [1,T_b,V]
                        logp_full = torch.log_softmax(logits_slice.to(torch.float32), dim=-1)
                        p = torch.exp(logp_full)
                        H_t = -(p * logp_full).sum(dim=-1).squeeze(0)             # [T_b]
                        H_sum = H_sum + H_t.sum()
                        targets = input_ids_mb[b, prompt_lens[b]: prompt_lens[b] + T_b]
                        logp_vec_b = logp_full.gather(-1, targets.view(1, T_b, 1)).squeeze(-1).squeeze(0)
                        logp_pieces.append(logp_vec_b)
                    logp_cat = torch.cat(logp_pieces, dim=0) if logp_pieces else torch.empty(0, device=logits.device)
                    return logp_cat, H_sum

                # F_pair: per-seq scalars f_vec (sum entropies) and L_vec (sum realized logprobs)
                eff_idx = [b for b in range(B_mb) if T_list[b] > 0]
                eff_prompt = [prompt_lens[b] for b in eff_idx]
                eff_T      = [T_list[b]      for b in eff_idx]

                def F_pair(params_tuple):
                    params_map = dict(base_map)
                    for n, t in zip(names, params_tuple):
                        params_map[n] = t
                    out = functional_call(
                        mdl, params_map, (input_ids_mb,),
                        {"attention_mask": attention_mask_mb, "use_cache": False}
                    )
                    logits = out.logits  # [B_mb, L, V]
                    f_list, L_list = [], []
                    for i, b in enumerate(eff_idx):
                        start = max(int(eff_prompt[i]) - 1, 0)
                        T_b   = int(eff_T[i]); end = start + T_b
                        logits_slice = logits[b:b+1, start:end, :]
                        logp_full = torch.log_softmax(logits_slice.to(torch.float32), dim=-1)
                        p = torch.exp(logp_full)
                        H_t = -(p * logp_full).sum(dim=-1).squeeze(0)      # [T_b]
                        f_b = H_t.sum()
                        targets = input_ids_mb[b, eff_prompt[i]: eff_prompt[i] + T_b]
                        logp_vec = logp_full.gather(-1, targets.view(1, T_b, 1)).squeeze(-1).squeeze(0)
                        L_b = logp_vec.sum()
                        f_list.append(f_b); L_list.append(L_b)
                    return torch.stack(f_list, 0), torch.stack(L_list, 0)   # [B_eff],[B_eff]

                (f_vec, L_vec), (j_f_vec, j_L_vec) = jvp(F_pair, (primals,), (tangents,))

                # ===== Correct quadratic term via (A)+(B)+(C)+(D) per sequence =====
                # Second (nested) JVPs: v^T(∇^2 f)v and D_v S_v, both per sequence
                def Jf(params_tuple):
                    return jvp(F_pair, (params_tuple,), (tangents,))[1][0]        # j_f_vec
                def JL(params_tuple):
                    return jvp(F_pair, (params_tuple,), (tangents,))[1][1]        # j_L_vec
                _, vHessH_vec = jvp(Jf, (primals,), (tangents,))                  # [B_eff]
                _, DvSv_vec   = jvp(JL, (primals,), (tangents,))                  # [B_eff]

                # ----- Linear term g·v (three normalization modes) -----
                # Also need per-token j_logp; reuse F_mb once:
                (logp_cat, H_sum), (j_logp_cat, j_H_sum) = jvp(F_mb, (primals,), (tangents,))
                # Offsets to split j_logp_cat back into per-seq segments
                offsets = []
                acc = 0
                for b in range(B_mb):
                    if T_list[b] > 0:
                        offsets.append((b, acc, acc + T_list[b]))
                        acc += T_list[b]
                denom_corr_mb = 0.0
                if apply_denom_correction and offsets:
                    for b, s, e in offsets:
                        j_sum = float(j_logp_cat[s:e].sum().item())
                        denom_corr_mb += -h_base_mean * float(T_list[b]) * j_sum
                # Per-seq dot(w, j_logp) + j_f (or j_H_sum/T_b) as required
                gdotv_mb = 0.0
                if self.normalize == "per_token":
                    # original global per-token objective: use unnormalized w_cat and scalar j_H_sum
                    gdotv_mb = (w_cat.to(j_logp_cat) * j_logp_cat).sum() + j_H_sum
                elif getattr(self, "normalize", "") == "per_seq_token_mean":
                    # mean per-token per sequence: divide both pieces by T_b, average by 1/B_total later
                    for i, (b, s, e) in enumerate(offsets):
                        T_b = float(T_list[b])
                        if T_b <= 0: 
                            continue
                        w_b = w_list[b] / T_b
                        gdotv_mb = gdotv_mb + (w_b.to(j_logp_cat[s:e]) * j_logp_cat[s:e]).sum() + (j_f_vec[i] / T_b)
                else:
                    # 'per_sequence' (sum per sequence): original surrogate linear piece
                    gdotv_mb = (w_cat.to(j_logp_cat) * j_logp_cat).sum() + j_H_sum

                if isinstance(gdotv_mb, torch.Tensor):
                    gdotv_mb_value = float(gdotv_mb.item())
                else:
                    gdotv_mb_value = float(gdotv_mb)
                gdotv_mb_value += denom_corr_mb

                # ----- Quadratic term v^T H v (three normalization modes) -----
                if getattr(self, "normalize", "") == "per_seq_token_mean":
                    # divide per-seq contributions by T_b
                    vHvv_seq = []
                    ii = 0
                    for b in eff_idx:
                        T_b = float(T_list[b])
                        vHvv_b = (vHessH_vec[ii] + 2.0 * j_f_vec[ii] * j_L_vec[ii] + f_vec[ii] * (DvSv_vec[ii] + j_L_vec[ii] * j_L_vec[ii])) / max(T_b, 1.0)
                        vHvv_seq.append(vHvv_b)
                        ii += 1
                    vHvv_mb = torch.stack(vHvv_seq, 0).sum()
                else:
                    # original (per_token or per_sequence): no per-seq 1/T_b division inside
                    vHvv_seq = vHessH_vec + 2.0 * j_f_vec * j_L_vec + f_vec * (DvSv_vec + j_L_vec * j_L_vec)
                    vHvv_mb = vHvv_seq.sum()

                # Normalize this microbatch contribution
                # derivative scaling constant depends on objective:
                if getattr(self, "normalize", "") == "per_seq_token_mean":
                    # average over sequences of per-seq mean ⇒ constant 1/B_total
                    scale = 1.0 / max(B_total, 1)
                else:
                    # 'per_token' ⇒ 1/T_total ; 'per_sequence' ⇒ 1/B_total (as before)
                    scale = self._scale_for_derivative(B_total, T_total)
                scale_sum += float(scale)
                g_val = gdotv_mb_value * float(scale)
                h_val = float(vHvv_mb.item()) * float(scale)
                g_contribs_mb.append(g_val); gdotv_total += g_val
                h_contribs_mb.append(h_val); vHvv_total  += h_val

                if return_per_sequence:
                    eff_lookup = {b: idx for idx, b in enumerate(eff_idx)}
                    for b in range(B_mb):
                        if T_list[b] <= 0:
                            per_seq_vhvv.append(0.0)
                        else:
                            idx_local = eff_lookup[b]
                            val = float(vHvv_seq[idx_local].item()) * float(scale)
                            per_seq_vhvv.append(val)

            eps = 1e-30
            eta_star = (2.0 * abs(gdotv_total) / max(abs(vHvv_total), eps)) if abs(vHvv_total) > 0 else float("inf")
            kappa = (vHvv_total / max(gdotv_total, eps)) if abs(gdotv_total) > 0 else float("nan")
            ratio_pred = {float(eta): 1.0 / (1.0 + 0.5 * float(eta) * kappa) for eta in etas}

            out = {
                "gdotv": gdotv_total,
                "vHvv": vHvv_total,
                "eta_star": eta_star,
                "ratio_pred": ratio_pred,
                "num_sequences": int(E_batch["sequences"].shape[0]),
                "num_tokens": int(T_total),
                "method": "jvp_nested",
                "baseline": {"kind": str(self.baseline_kind)},
                "audit": {"scale_sum": scale_sum, "total_tokens_used": total_tokens_used},
            }
            M = len(g_contribs_mb)
            if M > 1:
                mean_g = sum(g_contribs_mb) / M
                var_g = sum((x - mean_g) ** 2 for x in g_contribs_mb) / (M - 1)
                mean_h = sum(h_contribs_mb) / M
                var_h = sum((x - mean_h) ** 2 for x in h_contribs_mb) / (M - 1)
                out["variance"] = {
                    "num_shards": M,
                    "se_gdotv": (var_g ** 0.5) / (M ** 0.5),
                    "se_vHvv": (var_h ** 0.5) / (M ** 0.5),
                }
            if return_per_sequence:
                out["per_sequence_vhvv"] = per_seq_vhvv
            if self.logger:
                self.logger.info(
                    f"[dir JVP] g·v={gdotv_total:.6e}  vHv={vHvv_total:.6e}  eta*={eta_star:.3e}  "
                    f"B={out['num_sequences']} T={out['num_tokens']} baseline={self.baseline_kind}"
                )
                self.logger.info(
                    f"[dir JVP][audit] mode={self.normalize}, "
                    f"deriv_scale={(1.0/max(B_total,1) if self.normalize=='per_seq_token_mean' else self._scale_for_derivative(B_total,T_total)):.6e}, "
                    f"sum_deriv_scales={scale_sum:.6e}, tokens_used={total_tokens_used}, pre_count={T_total}"
                )
                assert T_total == int(sum(t[0] for t in E_batch["gen_lens"])), "Token count mismatch."

                try:
                    rs = ", ".join([f"η={eta:.1e}→R_pred={ratio_pred[eta]:.3f}" for eta in ratio_pred])
                    self.logger.info(f"[dir JVP] ratio_pred: {rs}")
                except Exception:
                    pass
            return out
        # --- Restore model states outside transformed regions ---
        finally:
            if restore_inputreq and hasattr(mdl, "enable_input_require_grads"):
                try:
                    mdl.enable_input_require_grads()
                except Exception:
                    pass
            if restore_ckpt:
                try:
                    mdl.gradient_checkpointing_enable()
                except Exception:
                    pass

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

    def _scale_for_derivative(self, B_total: int, T_total: int) -> float:
        """
        Scale to average **directional derivatives** over the E-batch.
        For per-token mean:  (1 / T_total) * Σ_mb D_v L_mb
        For per-sequence mean: (1 / B_total) * Σ_mb D_v L_mb
        """
        if self.normalize == "per_token":
            return 1.0 / max(T_total, 1)
        elif self.normalize == "per_sequence":
            return 1.0 / max(B_total, 1)
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








