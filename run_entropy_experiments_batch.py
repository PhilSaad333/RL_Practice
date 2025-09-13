#!/usr/bin/env python3
"""
Run EntropyMeasurements multiple times with small config overrides and save results to Drive.

Defaults are chosen for a small batch:
  - approx_delta_h.baseline.kind = hk_ema
  - approx_delta_h.baseline.ema_beta = 0.9
  - estimator.single_eta = 4e-6 (eta_sweep = false)
  - batch_config.B_E = 512, B_U = 64
  - n_runs = 8

Usage examples:
  python run_entropy_experiments_batch.py
  python run_entropy_experiments_batch.py --n_runs 8 --config entropy_experiments/configs/config_template.yaml
  python run_entropy_experiments_batch.py --output_root \
      /content/drive/MyDrive/RL_Practice_Files/ --label hkema_overnight

Enable eta sweeps (runner handles sweep per run):
  python run_entropy_experiments_batch.py --eta_sweep --etas 1e-6 2e-6 4e-6
  # If --etas omitted with --eta_sweep, defaults to [1e-6, 2e-6, 4e-6]

You can also override knobs from CLI, e.g.:
  --baseline_kind hk_ema --ema_beta 0.9 --eta 4e-6 --BE 512 --BU 64
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml

from entropy_experiments.entropy_experiment_runner import EntropyMeasurements


DEFAULT_CONFIG = Path("entropy_experiments/configs/config_template.yaml")
DEFAULT_OUTPUT_ROOT = "/content/drive/MyDrive/RL_Practice_Files/"


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg = copy.deepcopy(cfg)

    # Batch sizes
    bc = cfg.setdefault("batch_config", {})
    if args.BE is not None:
        bc["B_E"] = int(args.BE)
    if args.BU is not None:
        bc["B_U"] = int(args.BU)

    # Estimator / eta sweep controls
    est = cfg.setdefault("estimator", {})
    if args.eta_sweep:
        est["eta_sweep"] = True
        # Use provided list or default sweep
        if args.etas and len(args.etas) > 0:
            est["eta_list"] = [float(x) for x in args.etas]
        else:
            est["eta_list"] = [1e-6, 2e-6, 4e-6]
    else:
        est["eta_sweep"] = False
        if args.eta is not None:
            est["single_eta"] = float(args.eta)

    # Approx settings
    approx = cfg.setdefault("approx_delta_h", {})
    if args.method:
        approx["method"] = str(args.method)
    base = approx.setdefault("baseline", {})
    if args.baseline_kind:
        base["kind"] = str(args.baseline_kind)
    # hk_ema beta if provided
    if args.ema_beta is not None:
        base["ema_beta"] = float(args.ema_beta)

    # Ensure variance enabled for diagnostics
    var = approx.setdefault("variance", {})
    if args.enable_variance:
        var["enabled"] = True
        var["jackknife"] = bool(args.jackknife)

    return cfg


def run_once(cfg: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    probe = EntropyMeasurements(cfg)
    results = probe.run_experiments()

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run entropy experiments multiple times and save to Drive.")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Base YAML config path")
    p.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT, help="Output root directory")
    p.add_argument("--label", type=str, default="overnight_runs", help="Label name for the run folder")
    p.add_argument("--n_runs", type=int, default=8, help="Number of runs")

    # Overrides
    p.add_argument("--baseline_kind", type=str, default="hk_ema", help="Baseline kind (hk|hk_ema|hk_ridge|regression|none)")
    p.add_argument("--ema_beta", type=float, default=0.9, help="hk_ema beta")
    # Eta controls: single or sweep
    p.add_argument("--eta", type=float, default=4e-6, help="Single eta (used when --eta_sweep is not set)")
    p.add_argument("--eta_sweep", action="store_true", help="Enable eta sweep; runner will iterate over eta_list per run")
    p.add_argument(
        "--etas",
        type=float,
        nargs="+",
        default=None,
        help="Eta list for sweep (space separated). If omitted with --eta_sweep, defaults to [1e-6, 2e-6, 4e-6]",
    )
    p.add_argument("--BE", type=int, default=512, help="B_E")
    p.add_argument("--BU", type=int, default=64, help="B_U")
    p.add_argument("--method", type=str, default=None, help="Approx method (jvp|grad_dot)")
    p.add_argument("--enable_variance", action="store_true", default=True, help="Ensure variance is enabled")
    p.add_argument("--jackknife", action="store_true", default=True, help="Enable jackknife in variance")
    return p.parse_args()


def main():
    args = parse_args()
    base_cfg = load_config(args.config)
    cfg = apply_overrides(base_cfg, args)

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    root = Path(os.path.join(args.output_root, f"{args.label}_{ts}"))
    root.mkdir(parents=True, exist_ok=True)

    index = []
    for i in range(1, int(args.n_runs) + 1):
        run_dir = root / f"run_{i:02d}"
        print(f"\n[RUN {i}/{args.n_runs}] -> {run_dir}")
        try:
            results = run_once(cfg, run_dir)
            index.append({
                "run": i,
                "path": str(run_dir),
                "timing": results.get("timing", {}),
                "variance": results.get("variance", {}),
                "approx": results.get("approx", {}),
            })
        except Exception as e:
            err_path = run_dir / "error.txt"
            err_path.parent.mkdir(parents=True, exist_ok=True)
            with open(err_path, "w") as f:
                f.write(str(e))
            index.append({"run": i, "path": str(run_dir), "error": str(e)})

    with open(root / "index.json", "w") as f:
        json.dump(index, f, indent=2, default=str)
    print("\nAll runs complete. Index saved to:", root / "index.json")


if __name__ == "__main__":
    main()
