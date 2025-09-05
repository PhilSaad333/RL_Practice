#!/usr/bin/env python3
"""
Colab-friendly script to compare update vectors (Options A, B, C).

Usage in Colab:
  !python entropy_experiments/colab_update_vector_check.py \
      --config entropy_experiments/configs/A100_config.yaml \
      --B_U 8 --G_U 4 --mb_size 1

Assumes the repo is cloned and the config points to a valid LoRA + optimizer checkpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch

from entropy_experiments.offline_entropy_probe import OfflineEntropyProbe
from entropy_experiments.update_vector import (
    compute_update_vector_adamw,
    compute_update_vector_step,
    compute_update_vector_adamw_manual,
)
from entropy_experiments.param_registry import flatten_named, get_trainable_named


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


def topk_by_norm(vec: Dict[str, torch.Tensor], k: int = 10):
    items = [(name, float(v.double().norm().item())) for name, v in vec.items()]
    items.sort(key=lambda x: x[1], reverse=True)
    return items[:k]


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
    ap.add_argument("--B_U", type=int, default=None)
    ap.add_argument("--G_U", type=int, default=None)
    ap.add_argument("--mb_size", type=int, default=None)
    ap.add_argument("--out", type=str, default="entropy_experiments/results/update_vector_check.json")
    args = ap.parse_args()

    # Load probe and checkpoint
    probe = OfflineEntropyProbe.from_config_file(args.config)
    ckpt = probe.config["checkpoint"]["checkpoint_path"]
    optp = probe.config["checkpoint"].get("optimizer_path")
    probe.load_checkpoint(ckpt, optp)

    # Overrides
    if args.B_U is not None:
        probe.config["batch_config"]["B_U"] = args.B_U
    if args.G_U is not None:
        probe.config["batch_config"]["G"] = args.G_U
    if args.mb_size is not None:
        probe.config.setdefault("true_delta_h", {})["microbatch_size"] = args.mb_size

    # Prepare U batch
    probe._ensure_sequence_processor()
    B_U = probe.config["batch_config"]["B_U"]
    G_U = probe.config["batch_config"]["G"]
    U_batch = probe._get_or_sample_U(B_U, G_U)

    # Compute update vectors with memory tracking
    print(f"\n=== Starting update vector computations ===")
    print(f"U_batch shapes: sequences={U_batch['sequences'].shape}, device={U_batch['sequences'].device}")
    
    # Get GPU memory usage helper
    def get_gpu_memory():
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            return f"Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
        return "No GPU"
    
    print(f"Initial GPU memory: {get_gpu_memory()}")
    
    print("\n>>> Computing Option B (single_step)...")
    vec_B, stats_B = compute_update_vector_step(
        model=probe.model,
        optimizer=probe.optimizer,
        U_batch=U_batch,
        config=probe.config,
        logger=probe.logger,
    )
    print(f"After Option B: {get_gpu_memory()}")
    print(f"Option B stats: {stats_B}")
    
    print("\n>>> Computing Option A (adamw_from_grads)...")
    vec_A, stats_A = compute_update_vector_adamw(
        model=probe.model,
        optimizer=probe.optimizer,
        U_batch=U_batch,
        config=probe.config,
        logger=probe.logger,
    )
    print(f"After Option A: {get_gpu_memory()}")
    print(f"Option A stats: {stats_A}")
    
    print("\n>>> Computing Option C (adamw_manual)...")
    vec_C, stats_C = compute_update_vector_adamw_manual(
        model=probe.model,
        optimizer=probe.optimizer,
        U_batch=U_batch,
        config=probe.config,
        logger=probe.logger,
    )
    print(f"After Option C: {get_gpu_memory()}")
    print(f"Option C stats: {stats_C}")

    # Metrics
    cos_AB = cosine_named(vec_A, vec_B)
    cos_AC = cosine_named(vec_A, vec_C)
    cos_BC = cosine_named(vec_B, vec_C)
    norm_A = float(torch.sqrt(sum((v.double() ** 2).sum() for v in vec_A.values())).item())
    norm_B = float(torch.sqrt(sum((v.double() ** 2).sum() for v in vec_B.values())).item())
    norm_C = float(torch.sqrt(sum((v.double() ** 2).sum() for v in vec_C.values())).item())
    ratio_AB = (norm_A / norm_B) if norm_B > 0 else float("inf")

    print("=== Update Vector Comparison ===")
    print(f"Option A (adamw_from_grads): norm={norm_A:.6e}")
    lr_groups = stats_B.get('lr_groups', [])
    print(f"Option B (single_step):      norm={norm_B:.6e}; lr_groups={lr_groups}")
    print(f"Option C (adamw_manual):     norm={norm_C:.6e}")
    print(f"Cosine A↔B: {cos_AB:.6f}  A↔C: {cos_AC:.6f}  B↔C: {cos_BC:.6f}")
    print(f"Norm ratio A/B: {ratio_AB:.6f}")

    diff_BC = l2_diff_named(vec_B, vec_C)
    diff_AC = l2_diff_named(vec_A, vec_C)
    print(f"L2 diff ||B−C||: {diff_BC:.6e}   ||A−C||: {diff_AC:.6e}")

    print("\nTop-10 params by update norm (A):")
    for n, v in topk_by_norm(vec_A, 10):
        print(f"  {n}: {v:.3e}")
    print("Top-10 params by update norm (B):")
    for n, v in topk_by_norm(vec_B, 10):
        print(f"  {n}: {v:.3e}")
    print("Top-10 params by update norm (C):")
    for n, v in topk_by_norm(vec_C, 10):
        print(f"  {n}: {v:.3e}")

    # Optimizer diagnostics
    print("\nOptimizer Param Groups:")
    for i, g in enumerate(probe.optimizer.param_groups):
        betas = g.get("betas", (0.9, 0.999))
        eps = g.get("eps", 1e-8)
        wd = g.get("weight_decay", 0.0)
        lr = g.get("lr", float('nan'))
        n = len(g.get("params", []))
        print(f"  Group {i}: lr={lr:.3e} betas={betas} eps={eps:.1e} weight_decay={wd:.3e} n_params={n}")
    # Warn if multiple distinct LRs
    lr_set = sorted({float(g.get('lr', float('nan'))) for g in probe.optimizer.param_groups})
    if len(lr_set) > 1:
        print(f"WARNING: Multiple distinct learning rates across param groups: {lr_set}")

    total = have_v = have_step = 0
    for group in probe.optimizer.param_groups:
        for p in group["params"]:
            total += 1
            st = probe.optimizer.state.get(p, {})
            if isinstance(st.get("exp_avg_sq", None), torch.Tensor):
                have_v += 1
            s = st.get("step", None)
            if isinstance(s, int) or isinstance(s, torch.Tensor):
                have_step += 1
    cov_v = have_v / max(total, 1)
    cov_step = have_step / max(total, 1)
    print(f"\nOptimizer state coverage: exp_avg_sq {have_v}/{total} = {cov_v:.1%}, step {have_step}/{total} = {cov_step:.1%}")

    name2param = get_trainable_named(probe.model)
    union_top = list({n for n, _ in (topk_by_norm(vec_A, 5) + topk_by_norm(vec_B, 5) + topk_by_norm(vec_C, 5))})
    print("\nSampled parameter states:")
    for n in union_top:
        p = name2param.get(n)
        if p is None:
            print(f"  {n}: not found among trainables")
            continue
        st = probe.optimizer.state.get(p, {})
        step = st.get("step", None)
        has_v = isinstance(st.get("exp_avg_sq", None), torch.Tensor)
        v_shape = tuple(st.get("exp_avg_sq", torch.tensor(())).shape) if has_v else None
        has_m = isinstance(st.get("exp_avg", None), torch.Tensor)
        print(f"  {n}: step={int(step) if isinstance(step, int) else step} exp_avg={'Y' if has_m else 'N'} exp_avg_sq={'Y' if has_v else 'N'} v_shape={v_shape}")

    # Save summary JSON
    out = {
        "cos_AB": cos_AB,
        "cos_AC": cos_AC,
        "cos_BC": cos_BC,
        "norm_A": norm_A,
        "norm_B": norm_B,
        "norm_C": norm_C,
        "norm_ratio_A_over_B": ratio_AB,
        "l2_diff_BC": diff_BC,
        "l2_diff_AC": diff_AC,
        "stats_A": stats_A,
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
