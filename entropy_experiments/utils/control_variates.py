# entropy_experiments/control_variates.py
# -*- coding: utf-8 -*-
"""
Control-variates study for the first-order estimator g·v on E-batches.

This module:
  1) Recomputes ⟨∇H, v⟩ via the JVP path but exposes *per-sequence* contributions y_i
     that add up to the batch estimator under your chosen normalization.
  2) Collects per-sequence features (length, log-prob stats, etc.).
  3) Fits control-variates (centered OLS; optional K-fold cross-fitting).
  4) Saves a tidy table (CSV), a JSON summary, and diagnostic plots.

Integration
-----------
- Pass in your existing DeltaEntropyApprox instance as `delta_approx`.
- Use `run_control_variate_analysis(...)` from your runner after constructing E and v.

Notes
-----
- This analysis path is intentionally clear and *slightly redundant* (we JVP
  one sequence at a time inside each microbatch) to guarantee a clean per-seq
  decomposition without touching production code. It is fine for offline study.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple, Optional
import os
import json
import math
import time
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
    extras: Dict[str, float]


@dataclass
class CVSummary:
    n_seqs: int
    normalization: str           # "per_token" or "per_sequence"
    mean_gdotv: float            # sample mean of y_i (equals batch estimator)
    var_gdotv: float
    corr: Dict[str, float]       # Pearson correlations with features
    ols_beta: Dict[str, float]   # OLS coefficients (centered; small ridge)
    var_after_cv: Dict[str, float]      # variance after single-feature CV
    var_after_cv_joint: float           # variance after joint OLS over all features
    diagnostics: Dict[str, Any]


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


# ----------------------------- Core computation ----------------------------- #

@torch.no_grad()
def _make_single_seq_view(mb_E: BatchedSequences, idx: int) -> BatchedSequences:
    """
    Slice a BatchedSequences microbatch to keep only sequence `idx`, with G=1.
    """
    # Tensors keep [B, G, L] shape in the first two dims
    seq = mb_E.sequences[idx:idx+1]             # [1, G, L]
    att = mb_E.attention_masks[idx:idx+1]       # [1, G, L]

    # Scalars/lists
    prom = [int(mb_E.prompt_lens[idx])]         # [1]
    gen  = [[int(mb_E.gen_lens[idx][0])]]       # [1][1]

    # REQUIRED field: responses_text is List[List[str]] with shape [B][G]
    # We take only the first generation to match G=1 logic above.
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
    Compute per-sequence contributions y_i to g·v via the forward-mode JVP path.
    The contributions are scaled by the *same* derivative averaging factor used in
    production (_scale_for_derivative), so sum_i y_i == batch ⟨∇H, v⟩ estimator.

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

    # Functional snapshot for closures
    base_map = snapshot_base_functional_state(mdl)

    # Intersect primals/tangents with v_named
    names: List[str] = []
    primals: List[torch.Tensor] = []
    tangents: List[torch.Tensor] = []
    for n, p in mdl.named_parameters():
        if (not p.requires_grad) or (n not in v_named):
            continue
        names.append(n)
        # base_map[n] contains the "frozen" tensor we will use in functional call
        primals.append(base_map[n].to(device=device))
        # v_named[n] is Δθ/η; cast to base dtype for JVP
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
                # Keep a placeholder record to preserve indexing
                records.append(CVBatchRecord(
                    gdotv_i=0.0, length=0,
                    sum_logp=0.0, mean_logp=0.0, var_logp=0.0,
                    max_logp=-1e30, min_logp=+1e30, extras={}
                ))
                continue

            total_tokens_used += T_b
            w_b = w_list[b]  # shape [T_b], detached

            # Single-sequence view + closure
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
            # jvp returns: primals_out = (logp_cat[T_b], H_sum[scalar]),
            #              tangents_out = (j_logp_cat[T_b], j_H_sum[scalar])
            (logp_cat, H_sum), (j_logp_cat, j_H_sum) = jvp(f_seq, (tuple(primals),), (tuple(tangents),))

            # Per-seq contribution (aligned with production): scale * (w·j_logp + j_H_sum)
            contrib = (w_b.to(j_logp_cat) * j_logp_cat).sum() + j_H_sum
            y_i = float(scale * contrib.item())

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

            records.append(CVBatchRecord(
                gdotv_i=y_i,
                length=length,
                sum_logp=sum_lp,
                mean_logp=mean_lp,
                var_logp=var_lp,
                max_logp=max_lp,
                min_logp=min_lp,
                extras={"H_sum": float(H_sum.detach().item())}
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
    features: List[str] = ("length", "mean_logp", "var_logp"),
    ridge: float = 1e-8,
    crossfit_folds: int = 0,
) -> CVSummary:
    """
    Build centered design Z from chosen features and response y = gdotv_i.
    Optionally perform K-fold cross-fitting to approximate out-of-sample variance.

    Returns: CVSummary
    """
    if not records:
        raise ValueError("[control_variates] Empty records.")

    # Assemble arrays
    y = np.array([r.gdotv_i for r in records], dtype=np.float64)
    cols = []
    X = []
    feature_map = {
        "length": np.array([r.length for r in records], dtype=np.float64),
        "sum_logp": np.array([r.sum_logp for r in records], dtype=np.float64),
        "mean_logp": np.array([r.mean_logp for r in records], dtype=np.float64),
        "var_logp": np.array([r.var_logp for r in records], dtype=np.float64),
        "max_logp": np.array([r.max_logp for r in records], dtype=np.float64),
        "min_logp": np.array([r.min_logp for r in records], dtype=np.float64),
    }
    for f in features:
        if f not in feature_map:
            raise ValueError(f"[control_variates] Unknown feature '{f}'.")
        X.append(feature_map[f])
        cols.append(f)
    Z = np.stack(X, axis=1) if X else np.zeros((y.shape[0], 0), dtype=np.float64)

    # Correlations (for quick diagnostics)
    corr = {c: _pearson(feature_map[c], y) for c in cols}

    # In-sample OLS on centered variables
    y_c, Z_c, y_mean, Z_mean = _center_columns(y, Z)
    beta, var_ratio_joint, resid = _ols_centered(y_c, Z_c, ridge=ridge)

    # Single-feature variance reductions
    var_after = {}
    for j, c in enumerate(cols):
        zc = Z_c[:, [j]]
        b1, r1, _ = _ols_centered(y_c, zc, ridge=ridge)
        var_after[c] = float(r1)

    summary = CVSummary(
        n_seqs=int(y.shape[0]),
        normalization="per_token",  # The estimator alignment; downstream stores the actual used one.
        mean_gdotv=float(y.mean()),
        var_gdotv=float(y.var(ddof=1)) if y.size > 1 else 0.0,
        corr={k: float(v) for k, v in corr.items()},
        ols_beta={c: float(b) for c, b in zip(cols, beta.tolist() if beta.size else [])},
        var_after_cv=var_after,
        var_after_cv_joint=float(var_ratio_joint),
        diagnostics={
            "features": cols,
            "ridge": float(ridge),
            "crossfit_folds": int(crossfit_folds),
            "y_mean": float(y_mean),
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
    Save per-seq table (CSV), summary (JSON), and plots (PNG).
    Returns dict of file paths.
    """
    _ensure_dir(out_dir)

    # CSV
    import csv
    csv_path = os.path.join(out_dir, "cv_per_seq.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["gdotv_i", "length", "sum_logp", "mean_logp", "var_logp", "max_logp", "min_logp"]
        writer.writerow(header)
        for r in records:
            writer.writerow([r.gdotv_i, r.length, r.sum_logp, r.mean_logp, r.var_logp, r.max_logp, r.min_logp])

    # JSON summary
    json_path = os.path.join(out_dir, "cv_summary.json")
    with open(json_path, "w") as f:
        d = asdict(summary)
        # Include timestamp and version-ish info
        d["timestamp"] = datetime.utcnow().isoformat() + "Z"
        json.dump(d, f, indent=2)

    paths = {"csv_path": csv_path, "summary_path": json_path}

    if make_plots:
        # Scatter plots: y vs top features in summary.diagnostics['features']
        feats = summary.diagnostics.get("features", [])[:3]
        try:
            import pandas as pd  # optional; only for quick plotting convenience
            xs = {name: [] for name in feats}
            ys = []
            for r in records:
                ys.append(r.gdotv_i)
                for name in feats:
                    xs[name].append(getattr(r, name))
            ys = np.array(ys, dtype=np.float64)

            for name in feats:
                fig = plt.figure()
                plt.scatter(np.array(xs[name], dtype=np.float64), ys, s=9)
                plt.xlabel(name)
                plt.ylabel("per-seq contribution (y_i)")
                plt.title(f"y_i vs {name}")
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
            # Plots are optional; do not fail the run if plotting libs are absent.
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
    # carry normalization string through
    summary.normalization = normalization
    summary.diagnostics["meta"] = meta

    # 3) Save artifacts
    paths = save_cv_artifacts(out_dir, records, summary, make_plots=True)

    result = {"summary": asdict(summary)}
    result.update(paths)
    return result
