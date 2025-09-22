#!/usr/bin/env python3
"""Run a larger entropy-influence experiment and dump summaries."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

from entropy_experiments.entropy_influence import (
    EntropyInfluencePlan,
    EntropyInfluenceRunner,
    WorkspaceSpec,
    BatchRequest,
    BatchRequestKind,
)

CONFIG_PATH = Path("entropy_experiments/configs/config_template.yaml")
OUTPUT_ROOT = Path("entropy_experiments/results/entropy_influence")

ETAS = [2e-6, 4e-6, 6e-6, 8e-6, 1e-5]


def to_serializable(obj: Any) -> Any:
    if isinstance(obj, (int, float, str)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(x) for x in obj]
    return str(obj)


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    runner = EntropyInfluenceRunner(cfg)

    plan = EntropyInfluencePlan(
        workspace=WorkspaceSpec(
            kind=BatchRequestKind.UPDATE,
            params={
                "batch_size_prompts": 32,
                "completions_per_prompt": 8,
                "dataset_split": "test",
                "seed": 20250203,
            },
        ),
        evaluation_requests=[
            BatchRequest(
                kind=BatchRequestKind.EVALUATION,
                params={
                    "batch_size_prompts": 256,
                    "completions_per_prompt": 1,
                    "dataset_split": "train",
                    "seed": 13579,
                },
            )
        ],
        etas=ETAS,
        microbatch_size=1,
        auto_scale=False,
        record_per_sequence_eta=True,
    )

    results = runner.run(plan)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = OUTPUT_ROOT / f"large_run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "plan": {
            "workspace": to_serializable(plan.workspace.params),
            "evaluation": [to_serializable(req.params) for req in plan.evaluation_requests],
            "etas": plan.etas,
            "microbatch_size": plan.microbatch_size,
        },
        "workspace": {
            "num_sequences": len(results.workspace.batch.sequences),
            "update_stats": to_serializable(results.workspace.update_stats),
        },
        "evaluations": [],
    }

    for eval_idx, eval_res in enumerate(results.evaluations):
        eval_entry: Dict[str, Any] = {
            "index": eval_idx,
            "num_sequences": len(eval_res.batch.sequences),
            "aggregate": [],
        }

        for agg in eval_res.aggregate:
            eval_entry["aggregate"].append(
                {
                    "eta": agg.eta,
                    "delta_h": agg.delta_h,
                    "per_sequence_delta_sum": float(sum(agg.per_sequence_delta or [])),
                    "diagnostics": to_serializable(agg.diagnostics),
                }
            )

        per_seq = eval_res.per_sequence
        if per_seq:
            matrix = np.array(per_seq.delta_matrix, dtype=np.float64)
            np.save(run_dir / f"eval_{eval_idx:02d}_delta_matrix.npy", matrix)
            eval_entry.update(
                {
                    "eta_reference": per_seq.eta_reference,
                    "row_sums": matrix.sum(axis=1).tolist(),
                    "column_sums": matrix.sum(axis=0).tolist(),
                    "max_abs": float(np.abs(matrix).max()),
                    "sequence_ids": per_seq.sequence_ids,
                    "eta_per_sequence": per_seq.eta_per_sequence,
                }
            )

            grad_dir = run_dir / f"eval_{eval_idx:02d}_grad_eta"
            grad_dir.mkdir(parents=True, exist_ok=True)
            grad_summary: Dict[str, Any] = {}
            for eta_key, values in per_seq.grad_eta_deltas.items():
                arr = np.array(values, dtype=np.float64)
                safe_key = eta_key.replace('.', 'p').replace('-', 'm')
                np.save(grad_dir / f"delta_{safe_key}.npy", arr)
                grad_summary[eta_key] = {
                    "sum": float(arr.sum()),
                    "mean": float(arr.mean()),
                    "max_abs": float(np.abs(arr).max()),
                }
            eval_entry["grad_eta_summaries"] = grad_summary
            eval_entry["baseline_eta_delta"] = per_seq.baseline_eta_delta

            diag_path = run_dir / f"eval_{eval_idx:02d}_diagnostics.json"
            diag_path.write_text(
                json.dumps(to_serializable(per_seq.diagnostics), indent=2),
                encoding="utf-8",
            )

        summary["evaluations"].append(eval_entry)

    (run_dir / "summary.json").write_text(
        json.dumps(to_serializable(summary), indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
