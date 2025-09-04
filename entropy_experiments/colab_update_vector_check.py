#!/usr/bin/env python3
"""
Colab-friendly script to compare update vectors (Option A vs B).

Usage in Colab:
    !python entropy_experiments/colab_update_vector_check.py \
        --config entropy_experiments/configs/A100_config.yaml \
        --B_U 8 --G_U 4 --mb_size 1

Assumes you've cloned the repo and the config points to a valid LoRA + optimizer checkpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch

from entropy_experiments.offline_entropy_probe import OfflineEntropyProbe
from entropy_experiments.update_vector import (
    compute_update_vector_adamw,
    compute_update_vector_step,
)
from entropy_experiments.param_registry import flatten_named


def cosine_named(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> float:
    va = flatten_named(a)
    vb = flatten_named(b)
    if va.numel() == 0 or vb.numel() == 0:
        return float("nan")
    # Align lengths (intersection of names already ensured by flatten order in practice)
    n = min(va.numel(), vb.numel())
    va = va[:n]
    vb = vb[:n]
    num = (va.double() * vb.double()).sum()
    den = va.double().norm() * vb.double().norm()
    if den.item() == 0.0:
        return float("nan")
    return float((num / den).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--B_U", type=int, default=None, help="Override B_U for sampling")
    ap.add_argument("--G_U", type=int, default=None, help="Override G for U batch")
    ap.add_argument("--mb_size", type=int, default=None, help="Override microbatch size for grad/step")
    ap.add_argument("--out", type=str, default="entropy_experiments/results/update_vector_check.json")
    args = ap.parse_args()

    # Load probe and checkpoint
    probe = OfflineEntropyProbe.from_config_file(args.config)
    ckpt = probe.config["checkpoint"]["checkpoint_path"]
    optp = probe.config["checkpoint"].get("optimizer_path")
    probe.load_checkpoint(ckpt, optp)

    # Override sizes if provided
    if args.B_U is not None:
        probe.config["batch_config"]["B_U"] = args.B_U
    if args.G_U is not None:
        probe.config["batch_config"]["G"] = args.G_U
    if args.mb_size is not None:
        probe.config.setdefault("true_delta_h", {})["microbatch_size"] = args.mb_size

    # Prepare U batch via SequenceProcessor
    probe._ensure_sequence_processor()
    B_U = probe.config["batch_config"]["B_U"]
    G_U = probe.config["batch_config"]["G"]
    U_batch = probe._get_or_sample_U(B_U, G_U)

    # Compute both update vectors
    vec_B, stats_B = compute_update_vector_step(
        model=probe.model,
        optimizer=probe.optimizer,
        U_batch=U_batch,
        config=probe.config,
        logger=probe.logger,
    )
    vec_A, stats_A = compute_update_vector_adamw(
        model=probe.model,
        optimizer=probe.optimizer,
        U_batch=U_batch,
        config=probe.config,
        logger=probe.logger,
    )

    # Compare
    cos = cosine_named(vec_A, vec_B)
    norm_A = float(torch.sqrt(sum((v.double() ** 2).sum() for v in vec_A.values())).item())
    norm_B = float(torch.sqrt(sum((v.double() ** 2).sum() for v in vec_B.values())).item())
    ratio = (norm_A / norm_B) if norm_B > 0 else float("inf")

    print("=== Update Vector Comparison ===")
    print(f"Option A (adamw_from_grads): norm={norm_A:.6e}")
    print(f"Option B (single_step):      norm={norm_B:.6e}; lr_used={stats_B.get('lr_used', float('nan')):.3e}")
    print(f"Cosine similarity: {cos:.6f}")
    print(f"Norm ratio A/B:    {ratio:.6f}")

    # Save summary JSON
    out = {
        "cosine": cos,
        "norm_A": norm_A,
        "norm_B": norm_B,
        "norm_ratio_A_over_B": ratio,
        "stats_A": stats_A,
        "stats_B": stats_B,
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

