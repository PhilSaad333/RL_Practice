#!/usr/bin/env python3
"""Run Fisher-kernel with a custom workspace dominated by Shiloh prompt variants."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import List

import json
import matplotlib.pyplot as plt
import numpy as np
import yaml

from entropy_experiments.fisher_kernel import (
    BatchRequest,
    BatchRequestKind,
    FisherKernelPlan,
    FisherKernelRunner,
    WorkspaceSpec,
)

CONFIG_PATH = Path("entropy_experiments/configs/config_lambda.yaml")
OUTPUT_DIR = Path("entropy_experiments/results/fisher_kernel/shiloh_variants")
MICROBATCH_SIZE = 2
TOPK = 5
EVAL_PROMPTS = 64
EVAL_COMPLETIONS = 1
COMPLETIONS_PER_PROMPT = 8
SEED = 321
HEATMAP_PERCENTILE = 99.0

BASE_PROMPT = (
    "Shiloh is 44 years old today. In 7 years, he will be three times as old as his nephew. "
    "How old is his nephew today?"
)

UNRELATED_PROMPT = (
    "Rachel bought 23 cookies and Janet gave her 42 more. Her brother later ate 44 of them. "
    "How many cookies does Rachel have now?"
)

VARIANTS: List[str] = [
    BASE_PROMPT,
    "Today Shiloh is 44. In seven years his age will be triple his nephew's. Determine the nephew's age now.",
    "Shiloh currently is 44; after 7 years he will be three times older than his nephew. How old is the nephew today?",
    "In seven years Shiloh will be 51, exactly three times his nephew. How old is the nephew right now?",
    "If Shiloh (age 44) expects to be triple his nephew's age in seven years, what is the nephew's current age?",
    "Shiloh plans for seven years from now when his age equals 3 times his nephew's. Given he is 44 now, find the nephew's age.",
    "An uncle aged 44 will be three times his nephew in 7 years. What is the nephew's present age?",
    UNRELATED_PROMPT,
]


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


def save_heatmap(matrix: np.ndarray, title: str, path: Path) -> None:
    clipped = matrix.copy()
    if clipped.size > 0:
        clip_val = np.percentile(np.abs(clipped), HEATMAP_PERCENTILE)
        if clip_val > 0:
            clipped = np.clip(clipped, -clip_val, clip_val)
    plt.figure(figsize=(6, 5))
    plt.imshow(clipped, aspect="auto", cmap="RdBu_r")
    plt.colorbar()
    plt.title(f"{title} (clipped at +/-{HEATMAP_PERCENTILE}th pct)")
    plt.xlabel("U index")
    plt.ylabel("E index")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    cfg_dict = yaml.safe_load(CONFIG_PATH.read_text())
    runner = FisherKernelRunner(cfg_dict)

    workspace_spec = WorkspaceSpec(
        kind=BatchRequestKind.CUSTOM_PROMPTS,
        params={
            "prompts": VARIANTS,
            "completions_per_prompt": COMPLETIONS_PER_PROMPT,
            "seed": SEED,
            "apply_template": True,
        },
        capture_self_kernel=True,
    )

    eval_request = BatchRequest(
        kind=BatchRequestKind.EVALUATION,
        params={
            "batch_size_prompts": EVAL_PROMPTS,
            "completions_per_prompt": EVAL_COMPLETIONS,
            "with_replacement": True,
            "dataset_split": cfg_dict.get("batch_config", {}).get("E_split", cfg_dict.get("batch_config", {}).get("split", "train")),
            "seed": SEED + 1,
        },
        capture_full_kernel=True,
        topk_contributors=TOPK,
    )

    plan = FisherKernelPlan(
        workspace=workspace_spec,
        evaluation_requests=[eval_request],
        microbatch_size=MICROBATCH_SIZE,
        capture_full_kernel=True,
        topk_contributors=TOPK,
    )

    results = runner.run(plan)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    workspace_meta = [
        {
            **asdict(record),
            "gradient_norm": cache.gradient_norm,
            "preconditioned_norm": cache.preconditioned_norm,
        }
        for record, cache in zip(results.workspace.batch.sequences, results.workspace.gradient_caches)
    ]
    (OUTPUT_DIR / "workspace_sequences.json").write_text(
        json.dumps(workspace_meta, indent=2), encoding="utf-8"
    )

    if results.workspace.self_kernel is not None:
        kernel_np = results.workspace.self_kernel.detach().cpu().numpy()
        np.save(OUTPUT_DIR / "kernel_update.npy", kernel_np)
        save_heatmap(kernel_np, "Workspace Fisher kernel", OUTPUT_DIR / "kernel_update_heatmap.png")
    if results.workspace.self_influence is not None:
        influence_np = results.workspace.self_influence.detach().cpu().numpy()
        np.save(OUTPUT_DIR / "influence_update_self.npy", influence_np)
        save_histogram(influence_np, "Workspace influence", OUTPUT_DIR / "influence_update_self_hist.png")

    for idx, eval_res in enumerate(results.evaluations):
        prefix = f"eval_{idx:02d}"
        eval_meta = [asdict(record) for record in eval_res.batch.sequences]
        (OUTPUT_DIR / f"{prefix}_sequences.json").write_text(
            json.dumps(eval_meta, indent=2), encoding="utf-8"
        )
        if idx == 0:
            (OUTPUT_DIR / "evaluation_sequences.json").write_text(
                json.dumps(eval_meta, indent=2), encoding="utf-8"
            )
        if eval_res.kernel_block and eval_res.kernel_block.matrix is not None:
            mat = eval_res.kernel_block.matrix.detach().cpu().numpy()
            np.save(OUTPUT_DIR / f"{prefix}_kernel.npy", mat)
            save_heatmap(mat, f"Evaluation kernel {idx}", OUTPUT_DIR / f"{prefix}_kernel_heatmap.png")
            if idx == 0:
                np.save(OUTPUT_DIR / "evaluation_kernel.npy", mat)
                save_heatmap(mat, "Evaluation kernel", OUTPUT_DIR / "evaluation_kernel_heatmap.png")
        if eval_res.influence is not None:
            influence_np = eval_res.influence.delta_logprobs.detach().cpu().numpy()
            np.save(OUTPUT_DIR / f"{prefix}_influence.npy", influence_np)
            save_histogram(influence_np, f"Evaluation influence {idx}", OUTPUT_DIR / f"{prefix}_influence_hist.png")
            if idx == 0:
                np.save(OUTPUT_DIR / "evaluation_influence.npy", influence_np)
                save_histogram(influence_np, "Evaluation influence", OUTPUT_DIR / "evaluation_influence_hist.png")

    print("Workspace sequences:")
    for cache in results.workspace.gradient_caches:
        print(f"  {cache.sequence_id}")
    print(f"Results written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
