#!/usr/bin/env python3
"""Single-run entropy experiment CLI with diagnostics and repeat support."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np
import yaml

from entropy_experiments.entropy_experiment_runner import (
    EntropyMeasurements,
    ExperimentPlan,
)

DEFAULT_CONFIG = Path("entropy_experiments/configs/config_template.yaml")
_COLAB_DRIVE_ROOT = Path("/content/drive/MyDrive/RL_Practice_Files/experiment_runs")
if _COLAB_DRIVE_ROOT.anchor and Path("/content/drive").exists():
    DEFAULT_OUTPUT_ROOT = _COLAB_DRIVE_ROOT
else:
    DEFAULT_OUTPUT_ROOT = Path.home() / "RL_Practice_Files" / "experiment_runs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run entropy diagnostics and save results")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML config path")
    p.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory to store run outputs",
    )
    p.add_argument("--label", type=str, default="single_run", help="Label for this run series")
    p.add_argument("--n-runs", type=int, default=1, help="Number of runs to execute")

    toggles = p.add_argument_group("measurement toggles")
    toggles.add_argument("--no-true", action="store_true", help="Skip ?H_true measurements")
    toggles.add_argument("--no-linear", action="store_true", help="Skip linear approximation")
    toggles.add_argument("--linquad", action="store_true", help="Enable curvature / lin+quad output")
    toggles.add_argument(
        "--control-variates",
        action="store_true",
        help="Run control-variates analysis",
    )
    toggles.add_argument(
        "--capture-per-sequence",
        dest="capture_per_sequence",
        action="store_true",
        help="Return per-sequence diagnostics",
    )
    toggles.add_argument(
        "--no-capture-per-sequence",
        dest="capture_per_sequence",
        action="store_false",
        help="Disable per-sequence diagnostics (enabled by default)",
    )
    p.set_defaults(capture_per_sequence=True)

    diag = p.add_argument_group("diagnostic options")
    diag.add_argument(
        "--eta",
        type=float,
        action="append",
        dest="etas",
        help="Learning rate(s) ? to evaluate (repeat flag for multiple values)",
    )
    diag.add_argument(
        "--clip-override",
        type=float,
        action="append",
        dest="clip_overrides",
        help="Alternative clip thresholds for SNIS diagnostics (repeatable)",
    )
    diag.add_argument(
        "--eta-sweep",
        action="store_true",
        help="Force config estimator.eta_sweep = True",
    )

    return p.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r") as fh:
        return yaml.safe_load(fh)


def flatten_option(values: Iterable[float] | None) -> list[float] | None:
    if not values:
        return None
    return [float(v) for v in values]


def to_serializable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_serializable(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    try:
        import torch

        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
    except Exception:
        pass
    if hasattr(obj, "__dict__"):
        return to_serializable(vars(obj))
    return str(obj)


def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    if args.eta_sweep:
        config.setdefault("estimator", {})["eta_sweep"] = True

    runner = EntropyMeasurements(config)

    eta_list = flatten_option(args.etas)
    clip_overrides = flatten_option(args.clip_overrides)

    plan = ExperimentPlan(
        compute_true=not args.no_true,
        compute_linear=not args.no_linear,
        compute_linquad=args.linquad,
        run_control_variates=args.control_variates,
        capture_per_sequence=args.capture_per_sequence,
        eta_list=eta_list,
        clip_overrides=clip_overrides,
    )

    timestamp = datetime.now()
    base_dir = (
        args.out_root.expanduser()
        / timestamp.strftime("%Y-%m-%d")
        / f"{args.label}_{timestamp.strftime('%H-%M-%S')}"
    )
    base_dir.mkdir(parents=True, exist_ok=True)

    for run_idx in range(1, args.n_runs + 1):
        run_dir = base_dir / f"run_{run_idx:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        result = runner.run_experiments(plan=plan)

        results_path = run_dir / "results.json"
        plan_path = run_dir / "plan.json"
        config_copy_path = run_dir / "config.yaml"

        with results_path.open("w") as fh:
            json.dump(to_serializable(result), fh, indent=2)

        with plan_path.open("w") as fh:
            json.dump(to_serializable(asdict(plan)), fh, indent=2)

        if run_idx == 1:
            shutil.copy(args.config, config_copy_path)

        print(f"Saved run {run_idx} -> {results_path}")

    print(f"All runs saved under {base_dir}")


if __name__ == "__main__":
    main()
