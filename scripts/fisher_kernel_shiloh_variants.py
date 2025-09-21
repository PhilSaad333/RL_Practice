#!/usr/bin/env python3
"""Run Fisher-kernel with a custom workspace dominated by Shiloh prompt variants."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Sequence

import json
import random

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
MICROBATCH_SIZE = 4
TOPK = 5
EVAL_COMPLETIONS = 1
COMPLETIONS_PER_PROMPT = 8
SEED = 321
HEATMAP_PERCENTILE = 99.0

DATASET_PATHS = {
    "train": Path("rlp_datasets/processed/gsm8k_latex_train.jsonl"),
    "test": Path("rlp_datasets/processed/gsm8k_latex_test.jsonl"),
}

BASE_PROMPT = (
    "Shiloh is 44 years old today. In 7 years, he will be three times as old as his nephew. "
    "How old is his nephew today?"
)

VARIANT_PROMPTS: List[str] = [
    "Paul is 44 years old today. In 7 years, he will be three times as old as his cousin. How old is his cousin today?",
    "Shiloh is 44 years old today. In 7 years, he will be three times as old as his niece. How old is his niece today?",
    "Shiloh is 40 years old today. In 5 years, he will be three times as old as his nephew. How old is his nephew today?",
    "Mira is 53 years old today. In 7 years, she will be three times as old as her nephew. How old is her nephew today?",
    "Shiloh is 50 years old today. In 8 years, he will be three times as old as his younger cousin. How old is his younger cousin today?",
    "Caleb is 38 years old today. In 4 years, he will be three times as old as his nephew. How old is his nephew today?",
    "At present, Shiloh is 44. Seven years from now his age will triple that of his nephew. Determine the nephew's current age.",
]


def _load_questions(split: str) -> List[str]:
    path = DATASET_PATHS.get(split)
    if path is None:
        raise ValueError(f"Unsupported split '{split}'. Expected one of: {list(DATASET_PATHS)}")
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    questions: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            question = record["text"].split("<think>")[0].strip()
            questions.append(question)
    return questions


def sample_questions(
    *,
    split: str,
    count: int,
    rng: random.Random,
    exclude: Sequence[str] | None = None,
) -> List[str]:
    if count <= 0:
        return []
    questions = _load_questions(split)
    exclude_set = set(exclude or ())
    candidates = [q for q in questions if q not in exclude_set]
    if len(candidates) < count:
        raise ValueError(
            f"Requested {count} prompts from split '{split}' but only {len(candidates)} available after exclusions."
        )
    return rng.sample(candidates, count)


def record_prompts(path: Path, *, name: str, prompts: Iterable[str]) -> None:
    payload = {
        "label": name,
        "prompts": list(prompts),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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

    rng = random.Random(SEED)

    workspace_random = sample_questions(
        split="test",
        count=7,
        rng=rng,
        exclude=[BASE_PROMPT],
    )
    workspace_prompts = [BASE_PROMPT, *workspace_random]

    eval_random = sample_questions(
        split="train",
        count=1,
        rng=rng,
        exclude=[BASE_PROMPT, *VARIANT_PROMPTS, *workspace_random],
    )
    eval_prompts = [*VARIANT_PROMPTS, *eval_random]

    workspace_spec = WorkspaceSpec(
        kind=BatchRequestKind.CUSTOM_PROMPTS,
        params={
            "prompts": workspace_prompts,
            "completions_per_prompt": COMPLETIONS_PER_PROMPT,
            "seed": SEED,
            "apply_template": True,
        },
        capture_self_kernel=True,
    )

    eval_request = BatchRequest(
        kind=BatchRequestKind.CUSTOM_PROMPTS,
        params={
            "prompts": eval_prompts,
            "completions_per_prompt": EVAL_COMPLETIONS,
            "seed": SEED + 1,
            "apply_template": True,
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
    record_prompts(OUTPUT_DIR / "workspace_prompts.json", name="workspace", prompts=workspace_prompts)
    record_prompts(OUTPUT_DIR / "evaluation_prompts.json", name="evaluation", prompts=eval_prompts)

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
                np.save(OUTPUT_DIR / "kernel_eval.npy", mat)
                save_heatmap(mat, "Evaluation kernel", OUTPUT_DIR / "evaluation_kernel_heatmap.png")
                save_heatmap(mat, "Evaluation kernel", OUTPUT_DIR / "kernel_eval_heatmap.png")
        if eval_res.influence is not None:
            influence_np = eval_res.influence.delta_logprobs.detach().cpu().numpy()
            np.save(OUTPUT_DIR / f"{prefix}_influence.npy", influence_np)
            save_histogram(influence_np, f"Evaluation influence {idx}", OUTPUT_DIR / f"{prefix}_influence_hist.png")
            if idx == 0:
                np.save(OUTPUT_DIR / "evaluation_influence.npy", influence_np)
                np.save(OUTPUT_DIR / "influence_eval.npy", influence_np)
                save_histogram(influence_np, "Evaluation influence", OUTPUT_DIR / "evaluation_influence_hist.png")
                save_histogram(influence_np, "Evaluation influence", OUTPUT_DIR / "influence_eval_hist.png")

    print("Workspace sequences:")
    for cache in results.workspace.gradient_caches:
        print(f"  {cache.sequence_id}")
    print(f"Results written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
