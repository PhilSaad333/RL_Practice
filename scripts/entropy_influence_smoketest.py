#!/usr/bin/env python3
"""Minimal smoke-test runner for entropy influence experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml

from entropy_experiments.entropy_influence import (
    AggregateEntropyResult,
    EntropyInfluencePlan,
    EntropyInfluenceRunner,
)
from entropy_experiments.fisher_kernel import BatchRequest, BatchRequestKind, WorkspaceSpec


CONFIG_PATH = Path("entropy_experiments/configs/config_template.yaml")


def load_config() -> Dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_smoke_plan() -> EntropyInfluencePlan:
    workspace = WorkspaceSpec(
        kind=BatchRequestKind.UPDATE,
        params={
            "batch_size_prompts": 1,
            "completions_per_prompt": 2,
            "dataset_split": "test",
            "seed": 1234,
        },
    )

    evaluation_request = BatchRequest(
        kind=BatchRequestKind.EVALUATION,
        params={
            "batch_size_prompts": 2,
            "completions_per_prompt": 1,
            "dataset_split": "test",
            "seed": 4321,
        },
    )

    return EntropyInfluencePlan(
        workspace=workspace,
        evaluation_requests=[evaluation_request],
        etas=[1e-4],
        microbatch_size=2,
        auto_scale=True,
        auto_scale_target=1e-6,
    )


def main() -> None:
    config = load_config()
    runner = EntropyInfluenceRunner(config)
    plan = build_smoke_plan()
    results = runner.run(plan)

    workspace_meta = {
        "update_stats": results.workspace.update_stats,
        "num_sequences": len(results.workspace.batch.sequences),
    }

    evaluations_summary = []
    for eval_result in results.evaluations:
        aggregate: AggregateEntropyResult = eval_result.aggregate[0]
        evaluations_summary.append(
            {
                "eta": aggregate.eta,
                "delta_h": aggregate.delta_h,
                "num_eval_sequences": len(eval_result.batch.sequences),
            }
        )

    payload = {
        "workspace": workspace_meta,
        "evaluations": evaluations_summary,
    }

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

