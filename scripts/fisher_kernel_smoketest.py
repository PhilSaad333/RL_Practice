#!/usr/bin/env python3
"""Colab-ready Fisher kernel smoke test with saved plots and diagnostics."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from entropy_experiments.fisher_kernel import (
    BatchRequest,
    BatchRequestKind,
    FisherKernelPlan,
    FisherKernelRunner,
    WorkspaceSpec,
)


@dataclass
class TestConfig:
    config_path: Path
    update_prompts: int = 8
    update_completions: int = 8
    eval_prompts: int = 64
    eval_completions: int = 1
    microbatch_size: int = 2
    seed: int = 123
    topk: int = 5
    output_dir: Path = Path("fisher_kernel_outputs")
    heatmap_percentile: float = 99.0
    outlier_top_k: int = 20


def parse_args() -> TestConfig:
    parser = argparse.ArgumentParser(description="Fisher kernel smoke test")
    parser.add_argument("--config", type=Path, default=Path("entropy_experiments/configs/config_template.yaml"))
    parser.add_argument("--update-prompts", type=int, default=8)
    parser.add_argument("--update-completions", type=int, default=8)
    parser.add_argument("--eval-prompts", type=int, default=64)
    parser.add_argument("--eval-completions", type=int, default=1)
    parser.add_argument("--microbatch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("fisher_kernel_outputs"))
    parser.add_argument("--heatmap-percentile", type=float, default=99.0)
    parser.add_argument("--outlier-top-k", type=int, default=20)
    args = parser.parse_args()
    return TestConfig(
        config_path=args.config,
        update_prompts=args.update_prompts,
        update_completions=args.update_completions,
        eval_prompts=args.eval_prompts,
        eval_completions=args.eval_completions,
        microbatch_size=args.microbatch_size,
        seed=args.seed,
        topk=args.topk,
        output_dir=args.output_dir,
        heatmap_percentile=args.heatmap_percentile,
        outlier_top_k=args.outlier_top_k,
    )


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def build_plan(cfg: TestConfig, base_config: Dict[str, Any]) -> FisherKernelPlan:
    batch_cfg = base_config.get("batch_config", {}) or {}
    u_split = batch_cfg.get("U_split", batch_cfg.get("split", "train"))
    e_split = batch_cfg.get("E_split", batch_cfg.get("split", "train"))

    workspace_spec = WorkspaceSpec(
        kind=BatchRequestKind.UPDATE,
        params={
            "batch_size_prompts": cfg.update_prompts,
            "completions_per_prompt": cfg.update_completions,
            "dataset_split": u_split,
            "seed": cfg.seed,
        },
        capture_self_kernel=True,
    )

    evaluation_request = BatchRequest(
        kind=BatchRequestKind.EVALUATION,
        params={
            "batch_size_prompts": cfg.eval_prompts,
            "completions_per_prompt": cfg.eval_completions,
            "with_replacement": True,
            "dataset_split": e_split,
            "seed": cfg.seed + 1,
        },
        capture_full_kernel=True,
        topk_contributors=cfg.topk,
    )

    plan = FisherKernelPlan(
        workspace=workspace_spec,
        evaluation_requests=[evaluation_request],
        microbatch_size=cfg.microbatch_size,
        capture_full_kernel=True,
        topk_contributors=cfg.topk,
    )
    return plan


def save_histogram(values: np.ndarray, title: str, path: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=40, color="steelblue", alpha=0.8)
    plt.title(title)
    plt.xlabel("value")
    plt.ylabel("count")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_heatmap(matrix: np.ndarray, title: str, path: Path, percentile: float) -> None:
    clipped = matrix.copy()
    if clipped.size > 0:
        clip_val = np.percentile(np.abs(clipped), percentile)
        if clip_val > 0:
            clipped = np.clip(clipped, -clip_val, clip_val)
    plt.figure(figsize=(6, 5))
    plt.imshow(clipped, aspect="auto", cmap="RdBu_r")
    plt.colorbar()
    plt.title(f"{title} (clipped at ±{percentile}th pct)")
    plt.xlabel("U index")
    plt.ylabel("E index")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_outliers(matrix: np.ndarray, row_ids: List[str], col_ids: List[str], top_k: int, path: Path) -> None:
    if matrix.size == 0:
        path.write_text(json.dumps([], indent=2))
        return
    flat = matrix.ravel()
    idx = np.argpartition(np.abs(flat), -top_k)[-top_k:]
    entries = []
    num_cols = matrix.shape[1]
    for index in idx:
        r, c = divmod(index, num_cols)
        entries.append(
            {
                "row_index": int(r),
                "col_index": int(c),
                "row_sequence_id": row_ids[r] if r < len(row_ids) else None,
                "col_sequence_id": col_ids[c] if c < len(col_ids) else None,
                "value": float(matrix[r, c]),
            }
        )
    entries.sort(key=lambda item: abs(item["value"]), reverse=True)
    path.write_text(json.dumps(entries, indent=2))


def save_sequences(batch: GeneratedBatch, path: Path) -> None:
    serializable = [asdict(record) for record in batch.sequences]
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def save_numpy(matrix: np.ndarray, path: Path) -> None:
    np.save(path, matrix)


def main() -> None:
    cfg = parse_args()
    base_config = load_config(cfg.config_path)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    plan = build_plan(cfg, base_config)
    runner = FisherKernelRunner(base_config)

    print("[INFO] Running Fisher kernel plan ...")
    results = runner.run(plan)
    print("[INFO] Completed.")

    workspace = results.workspace
    self_kernel = workspace.self_kernel
    self_influence = workspace.self_influence

    outputs = []

    save_sequences(workspace.batch, cfg.output_dir / "workspace_sequences.json")
    outputs.append(cfg.output_dir / "workspace_sequences.json")

    if self_influence is not None:
        influence_np = self_influence.detach().cpu().numpy()
        path = cfg.output_dir / "influence_update_self_hist.png"
        save_histogram(influence_np, "Influence on update batch (self kernel)", path)
        outputs.append(path)
        save_numpy(influence_np, cfg.output_dir / "influence_update_self.npy")

    if self_kernel is not None:
        kernel_np = self_kernel.detach().cpu().numpy()
        save_numpy(kernel_np, cfg.output_dir / "kernel_update.npy")
        path = cfg.output_dir / "kernel_update_heatmap.png"
        save_heatmap(kernel_np, "Update batch Fisher kernel (self)", path, cfg.heatmap_percentile)
        outputs.append(path)
        outlier_path = cfg.output_dir / "kernel_update_top_outliers.json"
        row_ids = [cache.sequence_id for cache in workspace.gradient_caches]
        save_outliers(kernel_np, row_ids, row_ids, cfg.outlier_top_k, outlier_path)
        outputs.append(outlier_path)

    if results.evaluations:
        primary_eval = results.evaluations[0]
        save_sequences(primary_eval.batch, cfg.output_dir / "evaluation_sequences.json")
        outputs.append(cfg.output_dir / "evaluation_sequences.json")
        influence = primary_eval.influence
        kernel_block = primary_eval.kernel_block
        if influence is not None:
            influence_np = influence.delta_logprobs.detach().cpu().numpy()
            path = cfg.output_dir / "influence_eval_hist.png"
            save_histogram(influence_np, "Influence on evaluation batch", path)
            outputs.append(path)
            save_numpy(influence_np, cfg.output_dir / "influence_eval.npy")
        if kernel_block is not None and kernel_block.matrix is not None:
            matrix_np = kernel_block.matrix.detach().cpu().numpy()
            save_numpy(matrix_np, cfg.output_dir / "kernel_eval.npy")
            path = cfg.output_dir / "kernel_eval_heatmap.png"
            save_heatmap(matrix_np, "Evaluation vs update Fisher kernel", path, cfg.heatmap_percentile)
            outputs.append(path)
            outlier_path = cfg.output_dir / "kernel_eval_top_outliers.json"
            save_outliers(
                matrix_np,
                kernel_block.row_sequence_ids,
                kernel_block.col_sequence_ids,
                cfg.outlier_top_k,
                outlier_path,
            )
            outputs.append(outlier_path)

    if outputs:
        print("Saved outputs:")
        for path in outputs:
            print(f"  - {path}")
    else:
        print("No outputs generated (check evaluation results).")


if __name__ == "__main__":
    torch.set_grad_enabled(True)
    main()
