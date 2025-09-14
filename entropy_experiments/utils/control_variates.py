# entropy_experiments/control_variates.py
# -*- coding: utf-8 -*-
"""
Control-variates study for the first-order estimator ⟨∇H, v⟩ on E-batches.

What this module does
---------------------
1) Recomputes the first-order JVP estimator of ΔH (linear term) but returns a
   *per-sequence* decomposition y_i that sums to the batch estimator under the
   chosen normalization (e.g., "per_token").
2) Collects per-sequence candidate control-variates (CV) features (length,
   surprisal stats, RB entropy aggregates, baseline-weight stats, etc.).
3) Fits centered OLS (optionally cross-fittable later) to estimate variance
   reduction from CVs. Reports both *per-sequence variance ratios* and the
   *batch-level* standard error (SE) before/after CV.
4) Writes tidy artifacts: CSV (one row per sequence), JSON summary, and plots.

How to add a new feature (two small steps)
------------------------------------------
A. Emit (record) the feature during per-sequence computation:
   - In `compute_jvp_per_sequence(...)`, compute a scalar per-sequence value
     and place it in `extras` (e.g., `extras["my_feat"] = value`).
   - Prefer *primal*, detached features (no grads). Derivatives (like jH_sum)
     are also acceptable for analysis; we center them in OLS so unbiasedness is
     preserved for the batch-mean.

B. Expose the feature to the fitter:
   - In `fit_control_variates(...)`, extend `feature_map` with:
       `"my_feat": np.array([r.extras.get("my_feat", 0.0) for r in records])`
   - Then pass `--features my_feat ...` via your runner or put it in the config.

Where to change the runner defaults
-----------------------------------
In your Colab driver (e.g., `run_control_variate_analysis.py`), the features
used by default come from:
    cfg["control_variates"]["features"]
or, if missing, whatever default list the runner provides. To try the stronger
set by default, set:
    cfg["control_variates"]["features"] =
        ["rb_entropy_sum", "sum_w", "sum_w2", "surprisal_mean"]
You can also pass `--features ...` on the CLI.

Notes
-----
- We compute one JVP per sequence inside each microbatch. This is intentionally
  redundant to guarantee a clean per-sequence split without touching production
  estimator code paths.
- Normalization is handled by the same internal scaling used in production
  (`_scale_for_derivative`), so `sum_i y_i` equals the batch JVP estimate.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple, Optional
import os
import json
import math
from datetime import datetime

import numpy as np
import torch
from torch.func import jvp  # forward-mode
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# Project imports (same ones used by DeltaEntropyApprox)
from entropy_experiments.utils.sequence_processor import BatchedSequences
from entropy_experiments.utils.jvp_utils import (
    snapshot_base_functional_state,
    make_mb_outputs_closure,
)
from entropy_experiments.baselines import (
    EmaState,
    build_weights_base,
)
from entropy_experiments.baselines.strategies import RidgeConfig


# ----------------------------- Data structures ----------------------------- #

@dataclass
class CVBatchRecord:
    """One row per sequence in the E-batch."""
    gdotv_i: float
    length: int
    sum_logp: float
    mean_logp: float
    var_logp: float
    max_logp: float
    min_logp: float
    # Arbitrary extra scalars keyed by name
    extras: Dict[str, float]


@dataclass
class CVSummary:
    n_seqs: int
    normalization: str           # "per_token" or "per_sequence"
    mean_gdotv: float            # sample mean of y_i (equals batch estimator / B)
    var_gdotv: float             # per-sequence variance (for diagnostics)
    corr: Dict[str, float]       # Pearson correlations with features
    ols_beta: Dict[str, float]   # OLS coefficients (centered; tiny ridge)
    var_after_cv: Dict[str, float]      # per-seq variance ratio for each single CV
    var_after_cv_joint: float           # per-seq variance ratio for joint OLS
    diagnostics: Dict[str, Any]         # includes batch-level SEs and meta


# ----------------------------- Small utilities ----------------------------- #

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _center_columns(y: np.ndarray, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    y_mean = float(y.mean()) if y.size > 0 else 0.0
    y_c = y - y_mean
    Z_mean = Z.mean(axis=0, keepdims=True) if Z.size else np.zeros((1, Z.shape[1]), dtype=Z.dtype)
    Z_c = Z - Z_mean
    return y_c, Z_c, y_mean, Z_mean.ravel()


def _ols_centered(y_c: np.ndarray, Z_c: np.ndarray, ridge: float = 1e-8) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Solve argmin_beta ||y_c - Z_c beta||^2 + ridge ||beta||^2.
    Returns (beta, var_ratio, residuals), where var_ratio = Var(resid)/Var(y).
    """
    if Z_c.size == 0:
        return np.zeros((0,), dtype=np.float64), 1.0, y_c.copy()

    G = Z_c.T @ Z_c + ridge * np.eye(Z_c.shape[1], dtype=np.float64)
    beta = np.linalg.solve(G, Z_c.T @ y_c)
    resid = y_c - Z_c @ beta
    var_y = float(y_c.var(ddof=1)) if y_c.size > 1 else 0.0
    var_resid = float(resid.var(ddof=1)) if resid.size > 1 else 0.0
    var_ratio = (var_resid / var_y) if var_y > 0 else 1.0
    return beta, var_ratio, resid


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return 0.0
    xc = x - x.mean()
    yc = y - y.mean()
    vx = float(xc.var(ddof=1))
    vy = float(yc.var(ddof=1))
    if vx <= 0.0 or vy <= 0.0:
        return 0.0
    return float((xc @ yc) / math.sqrt(vx * vy) / (x.size - 1))


def _feature_vector(records: List[CVBatchRecord], name: str) -> np.ndarray:
    """
    Resolve a feature name to a numpy vector over records.
    Supports direct fields, derived features, and extras.
    """
    # Direct fields
    if name in {"length", "sum_logp", "mean_logp", "var_logp", "max_logp", "min_logp"}:
        return np.array([getattr(r, name) for r in records], dtype=np.float64)

    # Derived from direct fields
    if name == "surprisal_sum":
        return -np.array([r.sum_logp for r in records], dtype=np.float64)
    if name == "surprisal_mean":
        return -np.array([r.mean_logp for r in records], dtype=np.float64)
    if name == "length_inv":
        return np.array([1.0 / max(r.length, 1) for r in records], dtype=np.float64)
    if name == "length_log":
        return np.array([math.log(max(r.length, 1.0)) for r in records], dtype=np.float64)

    # Extras / analysis-time metrics
    def ex(key: str, default: float = 0.0) -> np.ndarray:
        return np.array([float(r.extras.get(key, default)) for r in records], dtype=np.float64)

    if name in {
        "rb_entropy_sum", "rb_entropy_mean",
        "sum_w", "sum_w2", "sum_w_abs",
        "wjlogp_sum", "jH_sum",
        "sum_abs_jlogp", "mean_abs_jlogp",
        "token_share",
        "length_x_surprisal_mean", "length_x_rb_entropy_mean",
    }:
        # compose from primitives where needed
        if name == "rb_entropy_sum":
            return ex("H_sum", 0.0)
        if name == "rb_entropy_mean":
            H = ex("H_sum", 0.0)
            L = np.array([max(r.length, 1) for r in records], dtype=np.float64)
            return H / L
        if name == "sum_w":
            return ex("sum_w", 0.0)
        if name == "sum_w2":
            return ex("sum_w2", 0.0)
        if name == "sum_w_abs":
            return ex("sum_w_abs", 0.0)
        if name == "wjlogp_sum":
            return ex("wjlogp_sum", 0.0)
        if name == "jH_sum":
            return ex("jH_sum", 0.0)
        if name == "sum_abs_jlogp":
            return ex("sum_abs_jlogp", 0.0)
        if name == "mean_abs_jlogp":
            return ex("mean_abs_jlogp", 0.0)
        if name == "token_share":
            return ex("token_share", 0.0)
        if name == "length_x_surprisal_mean":
            return np.array([r.length * (-(r.mean_logp)) for r in records], dtype=np.float64)
        if name == "length_x_rb_entropy_mean":
            H = ex("H_sum", 0.0)
            L = np.array([max(r.length, 1) for r in records], dtype=np.float64)
            return (H / L) * L  # equals H, but keeps naming consistent

    raise ValueError(f"Unknown feature '{name}'. See module docstring for how to add new features.")


# ----------------------------- Core computation ----------------------------- #

@torch.no_grad()
def _make_single_seq_view(mb_E: BatchedSequences, idx: int) -> BatchedSequences:
    """
    Slice a BatchedSequences microbatch to keep only sequence `idx`, with G=1.
    """
    seq = mb_E.sequences[idx:idx+1]             # [1, G, L]
    att = mb_E.attention_masks[idx:idx+1]       # [1, G, L]
    prom = [int(mb_E.prompt_lens[idx])]         # [1]
    gen  = [[int(mb_E.gen_lens[idx][0])]]       # [1][1]
    resp = [[mb_E.responses_text[idx][0]]]      # [1][1]
    return BatchedSequences(
        sequences=seq,
        prompt_lens=prom,
        gen_lens=gen,
        attention_masks=att,
        responses_text=resp,
    )


def compute_jvp_per_sequence(
    *,
    delta_approx,      # DeltaEntropyApprox instance
    E_batch: Dict[str, Any],
    v_named: Dict[str, torch.Tensor],
    normalization: str = "per_token",
    device: Optional[torch.device] = None,
) -> Tuple[List[CVBatchRecord], Dict[str, Any]]:
    """
    Compute per-sequence contributions y_i to ⟨∇H, v⟩ via the forward-mode JVP path.
    Uses the same derivative scaling as production (_scale_for_derivative), so
    sum_i y_i == batch estimator.

    Returns: (records, meta)
      - records: list[CVBatchRecord] of length B_total
      - meta: dict with audit information
    """
    assert normalization in {"per_token", "per_sequence"}, \
        f"Unknown normalization: {normalization}"

    mdl = delta_approx.model
    sp = delta_approx.sp
    mdl.eval()
    if device is None:
        device = next(mdl.parameters()).device

    base_map = snapshot_base_functional_state(mdl)

    # Intersect primals/tangents with v_named
    names: List[str] = []
    primals: List[torch.Tensor] = []
    tangents: List[torch.Tensor] = []
    for n, p in mdl.named_parameters():
        if (not p.requires_grad) or (n not in v_named):
            continue
        names.append(n)
        primals.append(base_map[n].to(device=device))
        tangents.append(v_named[n].to(device=device, dtype=base_map[n].dtype))
    if not names:
        raise ValueError("[control_variates] No intersecting trainables for JVP.")

    # Totals and derivative scaling (must mirror production)
    B_total = int(E_batch["sequences"].shape[0])
    T_total = delta_approx._count_total_gen_tokens(E_batch)
    scale = delta_approx._scale_for_derivative(B_total, T_total)

    records: List[CVBatchRecord] = []
    total_tokens_used = 0
    baseline_kind = str(getattr(delta_approx, "baseline_kind", "hk_ema")).lower()

    # EMA and ridge config mirrors your JVP estimator
    ema_state = None
    if baseline_kind == "hk_ema":
        ema_state = EmaState(
            pos_bins=getattr(delta_approx, "_pos_bins", 32),
            ema_beta=getattr(delta_approx, "_ema_beta", 0.95),
            ema_resid=getattr(delta_approx, "_ema_resid", 0.0),
            ema_cnt=getattr(delta_approx, "_ema_cnt", 0),
        )
    ridge_cfg = RidgeConfig(
        lambda_=getattr(delta_approx, "ridge_lambda", 1e-3),
        eps=getattr(delta_approx, "ridge_eps", 1e-8),
    )

    # Iterate over microbatches exactly as production does
    for mb_E in delta_approx._iter_microbatches(E_batch, delta_approx.mb):
        B_mb = int(mb_E.sequences.shape[0])
        tf_bs = min(delta_approx.mb, B_mb)

        # Build detached per-time weights w_t = (G - b_t) with the same baseline
        w_list = build_weights_base(
            kind=baseline_kind,
            model=mdl,
            sp=sp,
            mb_E=mb_E,
            tf_bs=tf_bs,
            ema=ema_state,
            ridge=ridge_cfg,
        )

        # Sequence-level JVP (one closure per sequence for a clean split)
        for b in range(B_mb):
            T_b = int(mb_E.gen_lens[b][0])
            if T_b <= 0:
                records.append(CVBatchRecord(
                    gdotv_i=0.0, length=0,
                    sum_logp=0.0, mean_logp=0.0, var_logp=0.0,
                    max_logp=-1e30, min_logp=+1e30, extras={}
                ))
                continue

            total_tokens_used += T_b
            w_b = w_list[b]  # [T_b], detached

            # Per-seq weight stats (cheap and often useful)
            sum_w   = float(w_b.sum().item())
            sum_w2  = float((w_b**2).sum().item())
            sum_w_abs = float(w_b.abs().sum().item())

            # Single-sequence view + functional closure
            seq_view = _make_single_seq_view(mb_E, b)
            input_ids = seq_view.sequences[:, 0].to(device=device)         # [1, L]
            att_mask  = seq_view.attention_masks[:, 0].to(device=device)   # [1, L]
            prompt_lens = [int(seq_view.prompt_lens[0])]
            gen_lens    = [int(seq_view.gen_lens[0][0])]

            f_seq = make_mb_outputs_closure(
                model=mdl,
                base_map=base_map,
                names=names,
                input_ids_mb=input_ids,
                attention_mask_mb=att_mask,
                prompt_lens=prompt_lens,
                gen_lens=gen_lens,
            )
            # jvp returns:
            #   primals_out = (logp_cat[T_b], H_sum[scalar]),
            #   tangents_out = (j_logp_cat[T_b], j_H_sum[scalar])
            (logp_cat, H_sum), (j_logp_cat, j_H_sum) = jvp(
                f_seq, (tuple(primals),), (tuple(tangents),)
            )

            # Component terms of the directional derivative
            wjlogp_sum = float((w_b.to(j_logp_cat) * j_logp_cat).sum().item())
            jH_sum_f   = float(j_H_sum.item())

            # Aligned per-seq contribution: scale * (w·j_logp + jH_sum)
            y_i = float(scale * (wjlogp_sum + jH_sum_f))

            # Per-seq features from primal log-probs
            lp = logp_cat.detach()
            length = int(T_b)
            sum_lp  = float(lp.sum().item())
            mean_lp = sum_lp / max(length, 1)
            if length > 0:
                diffs = lp - lp.mean()
                var_lp = float((diffs.pow(2).sum() / float(length)).item())
                max_lp = float(lp.max().item())
                min_lp = float(lp.min().item())
            else:
                var_lp, max_lp, min_lp = 0.0, -1e30, +1e30

            # JVP magnitude diagnostics (cheap)
            sum_abs_jlogp  = float(j_logp_cat.abs().sum().item())
            mean_abs_jlogp = sum_abs_jlogp / max(length, 1)

            # Extras (add any new features here)
            extras = {
                "H_sum": float(H_sum.detach().item()),   # RB entropy sum (primal)
                "sum_w": sum_w,
                "sum_w2": sum_w2,
                "sum_w_abs": sum_w_abs,
                "wjlogp_sum": wjlogp_sum,
                "jH_sum": jH_sum_f,
                "sum_abs_jlogp": sum_abs_jlogp,
                "mean_abs_jlogp": mean_abs_jlogp,
                "token_share": float(length / max(T_total, 1)),
            }

            records.append(CVBatchRecord(
                gdotv_i=y_i,
                length=length,
                sum_logp=sum_lp,
                mean_logp=mean_lp,
                var_logp=var_lp,
                max_logp=max_lp,
                min_logp=min_lp,
                extras=extras
            ))

    meta = {
        "B_total": B_total,
        "T_total": int(T_total),
        "scale": float(scale),
        "normalization": normalization,
        "baseline_kind": baseline_kind,
        "total_tokens_used": int(total_tokens_used),
    }
    return records, meta


# ----------------------------- CV fitting & reporting ----------------------------- #

def fit_control_variates(
    records: List[CVBatchRecord],
    features: List[str] | str = ("length", "mean_logp", "var_logp"),
    ridge: float = 1e-8,
    crossfit_folds: int = 0,
) -> CVSummary:
    """
    Build centered design Z from chosen features and response y = gdotv_i.
    Reports:
      - per-sequence variance (diagnostic)
      - batch-level SE before/after CV
      - for *every* available feature: correlation, single-feature var ratio, and SE_after
      - rankings of features by |corr| and by variance reduction

    Pass features="all" or "*" to include all available features.
    """
    if not records:
        raise ValueError("[control_variates] Empty records.")

    # Response and batch-level quantities
    y = np.array([r.gdotv_i for r in records], dtype=np.float64)
    B = y.shape[0]
    var_y = float(y.var(ddof=1)) if B > 1 else 0.0
    gdotv_batch = float(y.sum())                  # batch estimator (sum of per-seq contributions)
    var_batch = var_y / max(B, 1)                 # variance of the batch estimator
    se_batch = float(var_batch ** 0.5)

    # Build a *complete* feature map (direct + derived + extras + simple interactions)
    feature_map: Dict[str, np.ndarray] = {
        # direct
        "length":         _feature_vector(records, "length"),
        "sum_logp":       _feature_vector(records, "sum_logp"),
        "mean_logp":      _feature_vector(records, "mean_logp"),
        "var_logp":       _feature_vector(records, "var_logp"),
        "max_logp":       _feature_vector(records, "max_logp"),
        "min_logp":       _feature_vector(records, "min_logp"),
        # derived
        "surprisal_sum":  _feature_vector(records, "surprisal_sum"),
        "surprisal_mean": _feature_vector(records, "surprisal_mean"),
        "length_inv":     _feature_vector(records, "length_inv"),
        "length_log":     _feature_vector(records, "length_log"),
        # extras from JVP computation
        "rb_entropy_sum":  _feature_vector(records, "rb_entropy_sum"),
        "rb_entropy_mean": _feature_vector(records, "rb_entropy_mean"),
        "sum_w":           _feature_vector(records, "sum_w"),
        "sum_w2":          _feature_vector(records, "sum_w2"),
        "sum_w_abs":       _feature_vector(records, "sum_w_abs"),
        "wjlogp_sum":      _feature_vector(records, "wjlogp_sum"),
        "jH_sum":          _feature_vector(records, "jH_sum"),
        "sum_abs_jlogp":   _feature_vector(records, "sum_abs_jlogp"),
        "mean_abs_jlogp":  _feature_vector(records, "mean_abs_jlogp"),
        "token_share":     _feature_vector(records, "token_share"),
        # simple interactions (safe & cheap)
        "length_x_surprisal_mean": _feature_vector(records, "length_x_surprisal_mean"),
        "length_x_rb_entropy_mean": _feature_vector(records, "length_x_rb_entropy_mean"),
    }
    all_feature_names = list(feature_map.keys())

    # Allow "all" / "*" to select everything
    if isinstance(features, str) and features.lower() in {"all", "*"}:
        selected = all_feature_names
    else:
        selected = list(features) if isinstance(features, (list, tuple)) else [str(features)]

    # Correlations for *all* features (diagnostic)
    corr_all = {c: _pearson(feature_map[c], y) for c in all_feature_names}

    # Center y once
    y_c = y - (y.mean() if y.size else 0.0)

    # Single-feature variance ratios & SE_after for *all* features
    var_ratio_single: Dict[str, float] = {}
    se_after_single: Dict[str, float] = {}
    for c in all_feature_names:
        z = feature_map[c]
        zc = z - (z.mean() if z.size else 0.0)
        # solve OLS with 1 column (tiny ridge)
        if zc.ndim == 1:
            zc = zc[:, None]
        _, r1, _ = _ols_centered(y_c, zc, ridge=ridge)
        var_ratio_single[c] = float(r1)
        se_after_single[c] = float(((r1 * var_y) / max(B, 1)) ** 0.5)

    # Joint OLS on the *selected* subset
    Z = np.stack([feature_map[f] for f in selected], axis=1) if selected else np.zeros((B, 0), dtype=np.float64)
    y_c2, Z_c, y_mean, _ = _center_columns(y, Z)
    beta, var_ratio_joint, resid = _ols_centered(y_c2, Z_c, ridge=ridge)

    # Also report single-feature numbers for the selected subset (for backwards compat fields)
    var_after_subset = {c: var_ratio_single[c] for c in selected}
    se_after_subset  = {c: se_after_single[c]  for c in selected}

    # Batch-level SE after *joint* CV on the selected subset
    se_batch_after_joint = float(((var_ratio_joint * var_y) / max(B, 1)) ** 0.5)

    # Rankings
    top_by_abs_corr = sorted(all_feature_names, key=lambda k: abs(corr_all[k]), reverse=True)
    top_by_var_reduction = sorted(all_feature_names, key=lambda k: var_ratio_single[k])  # smaller is better

    summary = CVSummary(
        n_seqs=int(B),
        normalization="per_token",  # overwritten by caller with actual selection
        mean_gdotv=float(y.mean()),
        var_gdotv=var_y,            # per-sequence variance (kept for diagnostics)
        corr={k: float(corr_all[k]) for k in selected},
        ols_beta={c: float(b) for c, b in zip(selected, beta.tolist() if beta.size else [])},
        var_after_cv=var_after_subset,                 # per-seq variance ratios for selected subset
        var_after_cv_joint=float(var_ratio_joint),
        diagnostics={
            "features": selected,
            "features_all": all_feature_names,
            "ridge": float(ridge),
            "crossfit_folds": int(crossfit_folds),
            "y_mean": float(y_mean),
            # Batch-level quantities:
            "gdotv_batch": gdotv_batch,
            "var_batch": var_batch,
            "se_batch": se_batch,
            "se_batch_after_cv": se_after_subset,              # per-feature for selected
            "se_batch_after_cv_joint": se_batch_after_joint,   # joint OLS on selected
            # Full per-feature diagnostics (for *all* features)
            "all_features_stats": {
                "corr": {k: float(v) for k, v in corr_all.items()},
                "var_ratio_single": {k: float(v) for k, v in var_ratio_single.items()},
                "se_after_single": {k: float(v) for k, v in se_after_single.items()},
                "rank_top_abs_corr": top_by_abs_corr,
                "rank_top_var_reduction": top_by_var_reduction,
            },
        },
    )
    return summary


def save_cv_artifacts(
    out_dir: str,
    records: List[CVBatchRecord],
    summary: CVSummary,
    make_plots: bool = True,
) -> Dict[str, str]:
    """
    Save per-seq table (CSV), per-feature stats (CSV), summary (JSON), and plots (PNG).
    Returns dict of file paths.
    """
    _ensure_dir(out_dir)

    # ----- Per-sequence CSV (unchanged except for robustness) -----
    import csv
    csv_path = os.path.join(out_dir, "cv_per_seq.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = [
            "gdotv_i",
            "length",
            "sum_logp", "mean_logp", "var_logp", "max_logp", "min_logp",
            "rb_entropy_sum", "rb_entropy_mean",
            "sum_w", "sum_w2", "sum_w_abs",
            "wjlogp_sum", "jH_sum",
            "sum_abs_jlogp", "mean_abs_jlogp",
            "surprisal_sum", "surprisal_mean",
            "token_share",
            "length_inv", "length_log",
            "length_x_surprisal_mean", "length_x_rb_entropy_mean",
        ]
        writer.writerow(header)
        for r in records:
            rb_entropy_sum = float(r.extras.get("H_sum", 0.0))
            rb_entropy_mean = rb_entropy_sum / max(r.length, 1)
            sum_w   = float(r.extras.get("sum_w", 0.0))
            sum_w2  = float(r.extras.get("sum_w2", 0.0))
            sum_w_abs = float(r.extras.get("sum_w_abs", 0.0))
            wjlogp_sum = float(r.extras.get("wjlogp_sum", 0.0))
            jH_sum     = float(r.extras.get("jH_sum", 0.0))
            sum_abs_jlogp  = float(r.extras.get("sum_abs_jlogp", 0.0))
            mean_abs_jlogp = float(r.extras.get("mean_abs_jlogp", 0.0))
            surprisal_sum  = -float(r.sum_logp)
            surprisal_mean = -float(r.mean_logp)
            token_share = float(r.extras.get("token_share", 0.0))
            length_inv  = 1.0 / max(r.length, 1)
            length_log  = math.log(max(r.length, 1.0))
            length_x_surprisal_mean = r.length * surprisal_mean
            length_x_rb_entropy_mean = rb_entropy_mean * r.length  # equals rb_entropy_sum

            writer.writerow([
                r.gdotv_i,
                r.length,
                r.sum_logp, r.mean_logp, r.var_logp, r.max_logp, r.min_logp,
                rb_entropy_sum, rb_entropy_mean,
                sum_w, sum_w2, sum_w_abs,
                wjlogp_sum, jH_sum,
                sum_abs_jlogp, mean_abs_jlogp,
                surprisal_sum, surprisal_mean,
                token_share,
                length_inv, length_log,
                length_x_surprisal_mean, length_x_rb_entropy_mean,
            ])

    # ----- Per-feature stats CSV (new) -----
    feat_csv = os.path.join(out_dir, "cv_feature_stats.csv")
    with open(feat_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "feature",
            "selected_in_joint",       # whether this feature was in the joint model
            "corr",                    # Pearson corr(y, feature)
            "var_ratio_single",        # Var(resid)/Var(y) using this single CV
            "se_after_single",         # batch SE after single-feature CV
            "beta_joint",              # joint OLS coefficient (if selected, else "")
        ])
        diag = summary.diagnostics
        selected = set(diag.get("features", []))
        stats_all = diag.get("all_features_stats", {})
        corr_all = stats_all.get("corr", {})
        vr_all   = stats_all.get("var_ratio_single", {})
        se_all   = stats_all.get("se_after_single", {})
        beta_map = summary.ols_beta or {}
        # iterate over all features so you can sort in a spreadsheet
        names = diag.get("features_all", list(corr_all.keys()))
        for name in names:
            writer.writerow([
                name,
                int(name in selected),
                corr_all.get(name, float("nan")),
                vr_all.get(name, float("nan")),
                se_all.get(name, float("nan")),
                beta_map.get(name, ""),
            ])

    # ----- JSON summary -----
    json_path = os.path.join(out_dir, "cv_summary.json")
    with open(json_path, "w") as f:
        d = asdict(summary)
        d["timestamp"] = datetime.utcnow().isoformat() + "Z"
        json.dump(d, f, indent=2)

    paths = {"csv_path": csv_path, "feature_stats_path": feat_csv, "summary_path": json_path}

    # ----- Plots (top-3 by |corr| across *all* features) -----
    if make_plots:
        try:
            ys = np.array([r.gdotv_i for r in records], dtype=np.float64)
            stats_all = summary.diagnostics.get("all_features_stats", {})
            top_corr = stats_all.get("rank_top_abs_corr", [])[:3]
            for name in top_corr:
                try:
                    xs = _feature_vector(records, name)
                except Exception:
                    continue
                fig = plt.figure()
                plt.scatter(xs, ys, s=9)
                rho = _pearson(xs, ys)
                plt.xlabel(name)
                plt.ylabel("per-seq contribution (y_i)")
                plt.title(f"y_i vs {name}  (ρ≈{rho:.3f})")
                outp = os.path.join(out_dir, f"scatter_{name}.png")
                fig.savefig(outp, dpi=120, bbox_inches="tight")
                plt.close(fig)
                paths[f"plot_{name}"] = outp

            # Histogram of y_i
            fig = plt.figure()
            plt.hist(ys, bins=40)
            plt.xlabel("y_i")
            plt.ylabel("count")
            plt.title("Distribution of per-seq contributions y_i")
            outp = os.path.join(out_dir, f"hist_yi.png")
            fig.savefig(outp, dpi=120, bbox_inches="tight")
            plt.close(fig)
            paths["plot_hist_yi"] = outp

        except Exception:
            pass

    return paths


def run_control_variate_analysis(
    *,
    delta_approx,
    E_batch: Dict[str, Any],
    v_named: Dict[str, torch.Tensor],
    normalization: str = "per_token",
    out_dir: str = "entropy_experiments/cv_runs",
    features: List[str] = ("length", "mean_logp", "var_logp"),
    ridge: float = 1e-8,
    crossfit_folds: int = 0,
) -> Dict[str, Any]:
    """
    Orchestrates compute_jvp_per_sequence → fit_control_variates → save_cv_artifacts.

    Returns a dict with: summary (dict), csv_path, summary_path, and plot file paths if created.
    """
    # Timestamped subdir
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(out_dir, f"cv_{run_id}")
    _ensure_dir(out_dir)

    # 1) Per-seq contributions and features
    records, meta = compute_jvp_per_sequence(
        delta_approx=delta_approx,
        E_batch=E_batch,
        v_named=v_named,
        normalization=normalization,
    )

    # 2) CV fitting
    summary = fit_control_variates(
        records=records,
        features=list(features),
        ridge=float(ridge),
        crossfit_folds=int(crossfit_folds),
    )
    summary.normalization = normalization
    summary.diagnostics["meta"] = meta

    # 3) Save artifacts
    paths = save_cv_artifacts(out_dir, records, summary, make_plots=True)

    result = {"summary": asdict(summary)}
    result.update(paths)
    return result
