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
from torch.func import jvp
from entropy_experiments.utils.param_registry import (
    dot_named,
    flatten_named,

)
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

        # For variance diagnostics (microbatch-level directional contributions)
        contribs: List[float] = []
        last_dot_val = 0.0
        # Prepare v on CPU/fp32 up-front (used inside the loop if variance enabled)
        v_named_cpu = {k: (v.detach().to("cpu", torch.float32) if isinstance(v, torch.Tensor) else v)
                       for k, v in v_named.items()}

        for mb_E in self._iter_microbatches(E_batch, self.mb):
            # Teacher-forced with-grad forward
            # Note: tf_batch_size can be ≤ microbatch size; cap by current B_mb
            B_mb = int(mb_E.sequences.shape[0])
            tf_bs = min(self.mb, B_mb)

            want_feats = self.use_rb and (self.baseline_kind in {"regression", "reg", "reg_ridge", "regression_ridge"})
            res, _diag = self.sp.teacher_force_logprobs_with_diagnostics(
                sequences=mb_E,
                tf_batch_size=tf_bs,
                compute_rb=True,
                with_grad=True,
                return_baseline_features=want_feats,
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
                    # Create strategy once per microbatch (EMA state is persistent)
                    baseline_kind = str(self.baseline_kind).lower()
                    if baseline_kind == "hk_ema":
                        ema_state = EmaState(
                            pos_bins=self._pos_bins,
                            ema_beta=self._ema_beta,
                            ema_resid=self._ema_resid,
                            ema_cnt=self._ema_cnt,
                        )
                        strat = get_strategy("hk_ema", ema=ema_state)
                    elif baseline_kind in {"regression", "reg", "reg_ridge", "regression_ridge"}:
                        strat = get_strategy(
                            baseline_kind,
                            l2=self.baseline_reg_l2,
                            include_intercept=self.baseline_reg_intercept,
                            fit_dtype=self.baseline_reg_fit_dtype,
                            normalize=self.baseline_reg_normalize,
                            clip_min=self.baseline_reg_clip_min,
                            clip_max=self.baseline_reg_clip_max,
                        )
                    elif baseline_kind == "hk":
                        strat = get_strategy("hk")
                    elif baseline_kind == "none":
                        strat = None
                    else:
                        strat = get_strategy("hk")

                    phi = None
                    if strat is not None and (hasattr(res, "baseline_feats_torch") and res.baseline_feats_torch is not None) and len(res.baseline_feats_torch[b]) > 0:
                        phi = res.baseline_feats_torch[b][0]
                        if phi.dim() == 2 and int(phi.shape[0]) != T:
                            T_eff = min(T, int(phi.shape[0]))
                            phi = phi[:T_eff]
                            lp = lp[:T_eff]
                            H_rb = H_rb[:T_eff]
                            G = G[:T_eff]

                    if strat is None:
                        b_k = torch.zeros_like(G)
                    else:
                        b_k = strat.compute_bk_rb(H_rb=H_rb, G=G, phi=phi, update_state=(baseline_kind == "hk_ema"))

                    # Accumulate statistics for diagnostics
                    b_vals.append(b_k.mean().item())
                    G_vals.append(G.mean().item())

                    # Score term + RB term
                    sur = sur + ((G - b_k) * lp).sum() + H_rb.sum()

                else:
                    # -------- Simple estimator for ∇ E[-log π]:
                    # grad L = - (logπ - b).detach() * ∇ logπ
                    # where b is action-independent (timewise batch baseline).
                    # Gather ragged list first to build baselines jointly.
                    lp_list: List[torch.Tensor] = []
                    lengths: List[int] = []
                    for b in range(len(res.logprobs)):
                        lp_b: torch.Tensor = res.logprobs[b][0]  # [T_b], graph-carrying
                        lengths.append(int(lp_b.numel()))
                        if int(lp_b.numel()) > 0:
                            lp_list.append(lp_b)

                    # Build baselines for all sequences/time steps at once (detached)
                    tw_strat = get_timewise_strategy(self.simple_baseline_kind)
                    baselines = tw_strat.compute_timewise_baselines(lp_list)

                    # Now accumulate the surrogate
                    list_idx = 0
                    for b in range(len(res.logprobs)):
                        lp_b: torch.Tensor = res.logprobs[b][0]
                        T_b = int(lp_b.numel())
                        if T_b == 0:
                            continue
                        b_b = baselines[list_idx]  # same shape as lp_b
                        list_idx += 1

                        # Surrogate: - Σ_t (logπ_t - b_t).detach() * logπ_t
                        sur = sur - ((lp_b - b_b).detach() * lp_b).sum()
                        # Diagnostics
                        b_vals.append(float(b_b.mean().item()))
                        # No meaningful "G" in this estimator; track mean logπ as a proxy
                        G_vals.append(float(lp_b.detach().mean().item()))


            # Normalize this microbatch contribution
            scale = self._scale_for_average(B_total, T_total, B_mb, T_mb)
            total_tokens_used += T_mb
            baseline_means.append(float(sum(b_vals) / max(len(b_vals), 1)) if b_vals else 0.0)
            G_means.append(float(sum(G_vals) / max(len(G_vals), 1)) if G_vals else 0.0)

            (sur * float(scale)).backward()

            # --- Variance: record this microbatch's contribution to g·v (optional)
            if self.var_enabled:
                from entropy_experiments.utils.variance import update_shard_contrib
                last_dot_val = update_shard_contrib(name_to_param, v_named_cpu, contribs, last_dot_val)


        # Collect grads on intersection (CPU/fp32) and contract with v
        from entropy_experiments.utils.variance import gather_named_grads, compute_variance_info
        grads_named = gather_named_grads(name_to_param)

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

        # Variance diagnostics
        if self.var_enabled:
            out["variance"] = compute_variance_info(contribs, debug=self.debug, use_jackknife=self.var_jackknife)




        if self.logger:
            self.logger.info(
                f"[delta-h approx] ⟨∇H, v⟩={out['delta_h_per_lr']:.6e} | "
                f"B={out['num_sequences']} T={out['num_tokens']} | "
                f"||g||={grad_l2:.3e} ||v||={v_l2:.3e} cos={cosine:.3f} | "
                f"est={out['estimator']} baseline={self.baseline_kind}"
            )

            if self.var_enabled:
                vinfo = out.get("variance", {})
                self.logger.info(
                    f"[delta-h approx][variance] shards={vinfo.get('num_shards', 0)} "
                    f"SE(shard)={vinfo.get('se_shard', 0.0):.3e} "
                    f"SE(jack)={vinfo.get('se_jackknife', 0.0):.3e}"
                )


        return out

    # ---------------------------------------------------------------------------------------------
    # Phase 1: linear + quadratic directional terms via nested JVP (LoRA-only, microbatched)
    # ---------------------------------------------------------------------------------------------
    def compute_dir_linear_and_quadratic_jvp(
        self,
        *,
        E_batch: Dict[str, Any],
        v_named: Dict[str, torch.Tensor],
        etas: Optional[List[float]] = None,
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

        gdotv_total = 0.0
        vHvv_total = 0.0
        scale_sum = 0.0
        total_tokens_used = 0
        g_contribs_mb: List[float] = []
        h_contribs_mb: List[float] = []

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

            input_ids_mb = mb_E.sequences[:, 0].to(device=device)
            attention_mask_mb = mb_E.attention_masks[:, 0].to(device=device)
            prompt_lens = [int(x) for x in mb_E.prompt_lens]
            T_list = [int(mb_E.gen_lens[b][0]) for b in range(B_mb)]
            T_mb = sum(t for t in T_list if t > 0)
            if T_mb == 0:
                continue
            total_tokens_used += T_mb
            w_cat = torch.cat([w_list[b] for b in range(B_mb) if T_list[b] > 0], dim=0)

            F_mb = make_mb_outputs_closure(
                model=self.model,
                base_map=base_map,
                names=names,
                input_ids_mb=input_ids_mb,
                attention_mask_mb=attention_mask_mb,
                prompt_lens=prompt_lens,
                gen_lens=T_list,
            )

            def f_scalar(params_tuple):
                logp_cat, H_sum = F_mb(params_tuple)
                return (w_cat.to(logp_cat) * logp_cat).sum() + H_sum

            # First JVP: g·v
            _, gdotv_mb = jvp(f_scalar, (primals,), (tangents,))

            # Nested: v^T H v
            def gdot_fn(params_tuple):
                return jvp(f_scalar, (params_tuple,), (tangents,))[1]
            _, vHvv_mb = jvp(gdot_fn, (primals,), (tangents,))

            # Derivative averaging: constant scale per microbatch
            scale = self._scale_for_derivative(B_total, T_total)
            scale_sum += float(scale)
            g_contribs_mb.append(float(gdotv_mb.item()) * float(scale))
            h_contribs_mb.append(float(vHvv_mb.item()) * float(scale))
            gdotv_total += g_contribs_mb[-1]
            vHvv_total += h_contribs_mb[-1]

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
        if self.logger:
            self.logger.info(
                f"[dir JVP] g·v={gdotv_total:.6e}  vHv={vHvv_total:.6e}  eta*={eta_star:.3e}  "
                f"B={out['num_sequences']} T={out['num_tokens']} baseline={self.baseline_kind}"
            )
            self.logger.info(
                f"[dir JVP][audit] deriv_scale={self._scale_for_derivative(B_total, T_total):.6e}, "
                f"sum_deriv_scales={scale_sum:.6e} "
                f"(per_token ⇒ ≈ #microbatches/T_total), tokens_used={total_tokens_used}, pre_count={T_total}"
            )
            assert T_total == int(sum(t[0] for t in E_batch["gen_lens"])), "Token count mismatch."


            try:
                rs = ", ".join([f"η={eta:.1e}→R_pred={ratio_pred[eta]:.3f}" for eta in ratio_pred])
                self.logger.info(f"[dir JVP] ratio_pred: {rs}")
            except Exception:
                pass
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


    # -----------------------------------------------------------
    # ---------- For JVP approach -----------------------------
    # -----------------------------------------------------------


    def compute_delta_h_approx_jvp(
        self,
        *,
        E_batch: Dict[str, Any],
        v_named: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        """
        Directional derivative using forward-mode JVP with a vectorized closure.
        Assumes top_p=1.0 (full-softmax for RB entropies).
        Baselines supported: 'hk' and 'none' (extendable: add EMA/ridge weights as detached terms).
        """
        device = next(self.model.parameters()).device
        mdl = self.model; mdl.eval()

        base_map = snapshot_base_functional_state(self.model)
        names, primals, tangents = intersect_jvp_primals_tangents(self.model, base_map, v_named)

        # Count totals for normalization
        B_total = int(E_batch["sequences"].shape[0])
        T_total = self._count_total_gen_tokens(E_batch)

        contribs_mb: List[float] = []
        total_tokens_used = 0
        scale_sum = 0.0


        for mb_E in self._iter_microbatches(E_batch, self.mb):
            B_mb = int(mb_E.sequences.shape[0])
            tf_bs = min(self.mb, B_mb)
            # Build detached per-seq weights w_t = (G - b_t) using baseline strategies
            baseline_kind = str(self.baseline_kind).lower()
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
            # Build one microbatch closure and one concatenated weight vector
            input_ids_mb = mb_E.sequences[:, 0].to(device=device)          # [B_mb, L]
            attention_mask_mb = mb_E.attention_masks[:, 0].to(device=device)
            prompt_lens = [int(x) for x in mb_E.prompt_lens]
            T_list = [int(mb_E.gen_lens[b][0]) for b in range(B_mb)]
            T_mb = sum(t for t in T_list if t > 0)
            if T_mb == 0:
                continue
            # Concatenate weights in the same order we will concatenate logp vectors
            w_cat = torch.cat([w_list[b] for b in range(B_mb) if T_list[b] > 0], dim=0)  # [T_mb]

            f_mb = make_mb_outputs_closure(
                model=self.model,
                base_map=base_map,
                names=names,
                input_ids_mb=input_ids_mb,
                attention_mask_mb=attention_mask_mb,
                prompt_lens=prompt_lens,
                gen_lens=T_list,
            )
            (logp_cat, H_sum), (j_logp_cat, j_H_sum) = jvp(f_mb, (primals,), (tangents,))
            mb_contrib = float(((w_cat.to(j_logp_cat) * j_logp_cat).sum() + j_H_sum).item())

            scale = self._scale_for_average(B_total, T_total, B_mb, T_mb)
            total_tokens_used += T_mb
            scale_sum += float(scale)
            contribs_mb.append(mb_contrib * float(scale))

        delta_h_per_lr = float(sum(contribs_mb))
        out = {
            "delta_h_per_lr": delta_h_per_lr,
            "num_sequences": int(E_batch["sequences"].shape[0]),
            "num_tokens": int(T_total),
            "estimator": "rb",
            "baseline": {"kind": self.baseline_kind},
            "method": "jvp",
            "audit": {"scale_sum": scale_sum, "total_tokens_used": total_tokens_used},
        }
        if self.var_enabled:
            from entropy_experiments.utils.variance import compute_variance_info
            out["variance"] = compute_variance_info(contribs_mb, debug=self.debug, use_jackknife=self.var_jackknife)
        if self.logger:
            self.logger.info(
                f"[delta-h approx JVP] ⟨∇H, v⟩={out['delta_h_per_lr']:.6e} | "
                f"B={out['num_sequences']} T={out['num_tokens']} | baseline={self.baseline_kind}"
            )
            if self.var_enabled:
                vinfo = out.get("variance", {})
                self.logger.info(
                    f"[delta-h approx JVP][variance] shards={vinfo.get('num_shards', 0)} "
                    f"SE(shard)={vinfo.get('se_shard', 0.0):.3e} "
                    f"SE(jack)={vinfo.get('se_jackknife', 0.0):.3e}"
                )
            self.logger.info(
                f"[dir JVP][audit] deriv_scale={self._scale_for_derivative(B_total, T_total):.6e}, "
                f"sum_deriv_scales={scale_sum:.6e} "
                f"(per_token ⇒ ≈ #microbatches/T_total), total_tokens_used={total_tokens_used}, pre_count={T_total}"
            )
        return out






