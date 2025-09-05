#!/usr/bin/env python3
"""
Parity harness: compare manual AdamW direction (per-unit-lr) vs actual optimizer step Δθ/lr.

Usage:
  python entropy_experiments/debug/compare_update_directions.py \
    --config entropy_experiments/configs/A100_config.yaml \
    --eta 2e-6 --B_U 8 --G_U 4 --mb_size 1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch

from entropy_experiments.offline_entropy_probe import OfflineEntropyProbe
from entropy_experiments.update_vector import compute_update_vector_step, compute_update_vector_adamw_manual
from entropy_experiments.param_registry import flatten_named


def cosine_named(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> float:
    va = flatten_named(a)
    vb = flatten_named(b)
    if va.numel() == 0 or vb.numel() == 0:
        return float("nan")
    n = min(va.numel(), vb.numel())
    va = va[:n]
    vb = vb[:n]
    num = (va.double() * vb.double()).sum()
    den = va.double().norm() * vb.double().norm()
    if den.item() == 0.0:
        return float("nan")
    return float((num / den).item())


def l2_diff_named(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> float:
    acc = torch.zeros((), dtype=torch.float64)
    for k, va in a.items():
        vb = b.get(k)
        if vb is None:
            continue
        d = (va.double() - vb.double())
        acc += (d * d).sum()
    return float(acc.sqrt().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--eta", type=float, required=False, default=None, help="Trial LR; if set, overrides optimizer LRs for the step path only")
    ap.add_argument("--B_U", type=int, default=None)
    ap.add_argument("--G_U", type=int, default=None)
    ap.add_argument("--mb_size", type=int, default=None)
    ap.add_argument("--out", type=str, default="entropy_experiments/results/update_vector_parity.json")
    args = ap.parse_args()

    probe = OfflineEntropyProbe.from_config_file(args.config)
    ckpt = probe.config["checkpoint"]["checkpoint_path"]
    optp = probe.config["checkpoint"].get("optimizer_path")
    probe.load_checkpoint(ckpt, optp)

    if args.B_U is not None:
        probe.config["batch_config"]["B_U"] = args.B_U
    if args.G_U is not None:
        probe.config["batch_config"]["G"] = args.G_U
    if args.mb_size is not None:
        probe.config.setdefault("true_delta_h", {})["microbatch_size"] = args.mb_size

    probe._ensure_sequence_processor()
    B_U = probe.config["batch_config"]["B_U"]
    G_U = probe.config["batch_config"]["G"]
    U_batch = probe._get_or_sample_U(B_U, G_U)

    # Optional: override LRs for the step path only
    old_lrs = [g.get("lr", None) for g in probe.optimizer.param_groups]
    if args.eta is not None:
        for g in probe.optimizer.param_groups:
            g["lr"] = float(args.eta)

    vec_B, stats_B = compute_update_vector_step(
        model=probe.model,
        optimizer=probe.optimizer,
        U_batch=U_batch,
        config=probe.config,
        logger=probe.logger,
    )

    # Restore original LRs if overridden
    if args.eta is not None:
        for g, lr0 in zip(probe.optimizer.param_groups, old_lrs):
            if lr0 is not None:
                g["lr"] = lr0

    vec_C, stats_C = compute_update_vector_adamw_manual(
        model=probe.model,
        optimizer=probe.optimizer,
        U_batch=U_batch,
        config=probe.config,
        logger=probe.logger,
    )

    cos_BC = cosine_named(vec_B, vec_C)
    norm_B = float(torch.sqrt(sum((v.double() ** 2).sum() for v in vec_B.values())).item())
    norm_C = float(torch.sqrt(sum((v.double() ** 2).sum() for v in vec_C.values())).item())
    ratio = (norm_C / max(norm_B, 1e-38))
    diff = l2_diff_named(vec_B, vec_C)

    print("=== Parity Harness Results ===")
    print(f"cos(B,C)={cos_BC:.6f}  ||C||/||B||={ratio:.6f}  L2||B−C||={diff:.6e}")
    print(f"B: {stats_B}")
    print(f"C: {stats_C}")

    out = {
        "cos_BC": cos_BC,
        "norm_B": norm_B,
        "norm_C": norm_C,
        "ratio_C_over_B": ratio,
        "l2_diff_BC": diff,
        "stats_B": stats_B,
        "stats_C": stats_C,
        "B_U": B_U,
        "G_U": G_U,
        "mb_size": probe.config.get("true_delta_h", {}).get("microbatch_size", None),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()

