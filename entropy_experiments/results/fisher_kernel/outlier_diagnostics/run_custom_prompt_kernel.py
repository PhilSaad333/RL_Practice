#!/usr/bin/env python3
"""Custom Fisher-kernel run with a workspace anchored on specific prompt IDs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

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
from entropy_experiments.utils.sample_generator import SampleGenerator


@dataclass
class Config:
    config_path: Path
    target_prompt_id: int
    prompt_ids_path: Path
    total_prompts: int = 8
    completions_per_prompt: int = 8
    microbatch_size: int = 2
    seed: int = 123
    topk: int = 5
    eval_prompts: int = 64
    eval_completions: int = 1
    output_dir: Path = Path("fisher_kernel_anchor")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Fisher kernel with anchored prompt IDs")
    parser.add_argument("--config", type=Path, default=Path("entropy_experiments/configs/config_template.yaml"))
    parser.add_argument("--target-prompt-id", type=int, required=True)
    parser.add_argument("--prompt-ids-path", type=Path, default=Path("entropy_experiments/results/fisher_kernel/outlier_diagnostics/workspace_sequences.json"))
    parser.add_argument("--total-prompts", type=int, default=8)
    parser.add_argument("--completions", type=int, default=8)
    parser.add_argument("--microbatch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--eval-prompts", type=int, default=64)
    parser.add_argument("--eval-completions", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("fisher_kernel_anchor"))
    args = parser.parse_args()
    return Config(
        config_path=args.config,
        target_prompt_id=args.target_prompt_id,
        prompt_ids_path=args.prompt_ids_path,
        total_prompts=args.total_prompts,
        completions_per_prompt=args.completions,
        microbatch_size=args.microbatch_size,
        seed=args.seed,
        topk=args.topk,
        eval_prompts=args.eval_prompts,
        eval_completions=args.eval_completions,
        output_dir=args.output_dir,
    )


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def sample_prompt_ids(cfg: Config, base_config: Dict[str, Any]) -> List[int]:
    workspace_path = cfg.prompt_ids_path
    data = json.loads(workspace_path.read_text()) if workspace_path.exists() else None
    candidate_ids: List[int] = []
    if data:
        for entry in data:
            gid = entry.get("global_prompt_id")
            if gid is not None and int(gid) != cfg.target_prompt_id:
                candidate_ids.append(int(gid))
    candidate_ids = list(dict.fromkeys(candidate_ids))
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(candidate_ids)
    num_needed = max(0, cfg.total_prompts - 1)
    sampled = candidate_ids[:num_needed]
    sampled = [gid for gid in sampled if gid != cfg.target_prompt_id]
    while len(sampled) < num_needed:
        sampled.append(sampled[-1] if sampled else cfg.target_prompt_id)
    prompt_ids = [cfg.target_prompt_id] + sampled
    return prompt_ids


def build_plan(cfg: Config, base_config: Dict[str, Any], prompt_ids: List[int]) -> FisherKernelPlan:
    batch_cfg = base_config.get("batch_config", {}) or {}
    u_split = batch_cfg.get("U_split", batch_cfg.get("split", "train"))
    e_split = batch_cfg.get("E_split", batch_cfg.get("split", "train"))
    workspace_spec = WorkspaceSpec(
        kind=BatchRequestKind.PROMPT_IDS,
        params={
            "prompt_ids": prompt_ids,
            "completions_per_prompt": cfg.completions_per_prompt,
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


def main() -> None:
    cfg = parse_args()
    base_config = load_config(cfg.config_path)
    prompt_ids = sample_prompt_ids(cfg, base_config)
    runner = FisherKernelRunner(base_config)
    plan = build_plan(cfg, base_config, prompt_ids)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    results = runner.run(plan)
    print("Workspace sequence IDs:")
    for cache in results.workspace.gradient_caches:
        print(cache.sequence_id)
    print(f"Done. Save outputs in {cfg.output_dir}")


if __name__ == "__main__":
    main()
