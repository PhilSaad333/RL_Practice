#!/usr/bin/env python3
"""Custom Fisher-kernel run anchored on a specific prompt (sequence ID or global ID)."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

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

DEFAULT_WORKSPACE_JSON = Path(
    "entropy_experiments/results/fisher_kernel/outlier_diagnostics/workspace_sequences.json"
)


@dataclass
class Config:
    config_path: Path
    target_prompt_id: int
    prompt_sequences_path: Path
    total_prompts: int = 8
    completions_per_prompt: int = 8
    microbatch_size: int = 2
    seed: int = 123
    topk: int = 5
    eval_prompts: int = 64
    eval_completions: int = 1
    output_dir: Path = Path("fisher_kernel_anchor")
    heatmap_percentile: float = 99.0


def parse_key(text: str, prompt_sequences_path: Path) -> int:
    """Accept either a numeric global prompt id or a sequence id such as U-004-07."""

    try:
        return int(text)
    except ValueError:
        if not prompt_sequences_path.exists():
            raise ValueError(
                "Prompt sequences JSON not found and non-numeric target supplied"
            )
        data = json.loads(prompt_sequences_path.read_text(encoding="utf-8"))
        for entry in data:
            if entry.get("sequence_id") == text:
                gid = entry.get("global_prompt_id")
                if gid is None:
                    raise ValueError(f"Sequence {text} lacks global_prompt_id")
                return int(gid)
        raise ValueError(f"Sequence ID {text} not present in {prompt_sequences_path}")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Fisher kernel with anchored prompt IDs")
    parser.add_argument("--config", type=Path, default=Path("entropy_experiments/configs/config_template.yaml"))
    parser.add_argument("--target", required=True, help="Global prompt id or sequence id (e.g. U-004-07)")
    parser.add_argument("--prompt-json", type=Path, default=DEFAULT_WORKSPACE_JSON)
    parser.add_argument("--total-prompts", type=int, default=8)
    parser.add_argument("--completions", type=int, default=8)
    parser.add_argument("--microbatch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--eval-prompts", type=int, default=64)
    parser.add_argument("--eval-completions", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("fisher_kernel_anchor"))
    parser.add_argument("--heatmap-percentile", type=float, default=99.0)
    args = parser.parse_args()
    target_prompt_id = parse_key(args.target, args.prompt_json)
    return Config(
        config_path=args.config,
        target_prompt_id=target_prompt_id,
        prompt_sequences_path=args.prompt_json,
        total_prompts=args.total_prompts,
        completions_per_prompt=args.completions,
        microbatch_size=args.microbatch_size,
        seed=args.seed,
        topk=args.topk,
        eval_prompts=args.eval_prompts,
        eval_completions=args.eval_completions,
        output_dir=args.output_dir,
        heatmap_percentile=args.heatmap_percentile,
    )


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def sample_prompt_ids(cfg: Config) -> List[int]:
    """Return the target prompt id plus randomly sampled additional ids."""

    candidate_ids: List[int] = []
    if cfg.prompt_sequences_path.exists():
        data = json.loads(cfg.prompt_sequences_path.read_text(encoding="utf-8"))
        for entry in data:
            gid = entry.get("global_prompt_id")
            if gid is not None and int(gid) != cfg.target_prompt_id:
                candidate_ids.append(int(gid))
    unique_ids: List[int] = []
    for gid in candidate_ids:
        if gid not in unique_ids:
            unique_ids.append(gid)
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(unique_ids)
    needed = max(0, cfg.total_prompts - 1)
    sampled = unique_ids[:needed]
    while len(sampled) < needed:
        sampled.append(sampled[-1] if sampled else cfg.target_prompt_id)
    return [cfg.target_prompt_id] + sampled


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
    plt.title(f"{title} (clipped at +/-{percentile}th pct)")
    plt.xlabel("U index")
    plt.ylabel("E index")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def serialize_results(cfg: Config, results) -> None:
    output = cfg.output_dir
    output.mkdir(parents=True, exist_ok=True)

    workspace = results.workspace
    workspace_meta = [
        {
            **cache.metadata,
            "sequence_id": cache.sequence_id,
            "gradient_norm": cache.gradient_norm,
            "preconditioned_norm": cache.preconditioned_norm,
        }
        for cache in workspace.gradient_caches
    ]
    (output / "workspace_sequences.json").write_text(
        json.dumps(workspace_meta, indent=2), encoding="utf-8"
    )

    if workspace.self_kernel is not None:
        kernel_np = workspace.self_kernel.detach().cpu().numpy()
        np.save(output / "kernel_update.npy", kernel_np)
        save_heatmap(kernel_np, "Workspace Fisher kernel", output / "kernel_update_heatmap.png", cfg.heatmap_percentile)
    if workspace.self_influence is not None:
        influence_np = workspace.self_influence.detach().cpu().numpy()
        np.save(output / "influence_update_self.npy", influence_np)
        save_histogram(influence_np, "Workspace influence", output / "influence_update_self_hist.png")

    for idx, eval_res in enumerate(results.evaluations):
        prefix = f"eval_{idx:02d}"
        if eval_res.kernel_block and eval_res.kernel_block.matrix is not None:
            mat = eval_res.kernel_block.matrix.detach().cpu().numpy()
            np.save(output / f"{prefix}_kernel.npy", mat)
            save_heatmap(mat, f"Evaluation kernel {idx}", output / f"{prefix}_kernel_heatmap.png", cfg.heatmap_percentile)
        if eval_res.influence is not None:
            influence_np = eval_res.influence.delta_logprobs.detach().cpu().numpy()
            np.save(output / f"{prefix}_influence.npy", influence_np)
            save_histogram(influence_np, f"Evaluation influence {idx}", output / f"{prefix}_influence_hist.png")


def main() -> None:
    cfg = parse_args()
    base_config = load_config(cfg.config_path)
    prompt_ids = sample_prompt_ids(cfg)
    runner = FisherKernelRunner(base_config)
    plan = FisherKernelPlan(
        workspace=WorkspaceSpec(
            kind=BatchRequestKind.PROMPT_IDS,
            params={
                "prompt_ids": prompt_ids,
                "completions_per_prompt": cfg.completions_per_prompt,
                "dataset_split": base_config.get("batch_config", {}).get("U_split", base_config.get("batch_config", {}).get("split", "train")),
                "seed": cfg.seed,
            },
            capture_self_kernel=True,
        ),
        evaluation_requests=[
            BatchRequest(
                kind=BatchRequestKind.EVALUATION,
                params={
                    "batch_size_prompts": cfg.eval_prompts,
                    "completions_per_prompt": cfg.eval_completions,
                    "with_replacement": True,
                    "dataset_split": base_config.get("batch_config", {}).get("E_split", base_config.get("batch_config", {}).get("split", "train")),
                    "seed": cfg.seed + 1,
                },
                capture_full_kernel=True,
                topk_contributors=cfg.topk,
            )
        ],
        microbatch_size=cfg.microbatch_size,
        capture_full_kernel=True,
        topk_contributors=cfg.topk,
    )
    results = runner.run(plan)
    serialize_results(cfg, results)
    print("Workspace sequences:")
    for cache in results.workspace.gradient_caches:
        print(f"  {cache.sequence_id}")
    print(f"Run complete. Results saved to {cfg.output_dir}")


if __name__ == "__main__":
    main()
