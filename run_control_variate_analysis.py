#!/usr/bin/env python3
# run_control_variates.py
"""
Run control-variates analysis for the first-order JVP estimator and save results to Drive.

- Loads the same checkpoint specified in your YAML config (as in your batch script).
- Samples E and U via EntropyMeasurements, computes v on U, and runs CV analysis on E.
- Saves:
    * the per-sequence table (CSV),
    * a JSON summary (β's, correlations, variance reduction),
    * diagnostic plots (PNGs),
  under /content/drive/MyDrive/RL_Practice_Files/<label>_<ts>/run_01/cv_YYYYMMDD_HHMMSS/.

Usage examples (in Colab, after mounting Drive):
  !python run_control_variates.py
  !python run_control_variates.py --config entropy_experiments/configs/config_template.yaml \\
        --output_root /content/drive/MyDrive/RL_Practice_Files/ --label cv_runs
  !python run_control_variates.py --features length mean_logp var_logp --normalization per_token

You can also override a few knobs similar to your batch script:
  --baseline_kind hk_ema --ema_beta 0.9 --BE 512 --BU 64
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml

from entropy_experiments.entropy_experiment_runner import EntropyMeasurements
from entropy_experiments.delta_entropy_approx import DeltaEntropyApprox  # ensures module available
# We call the runner method `run_control_variate_analysis`, which should be present
# per our earlier patch.

DEFAULT_CONFIG = Path("entropy_experiments/configs/config_template.yaml")
DEFAULT_OUTPUT_ROOT = "/content/drive/MyDrive/RL_Practice_Files/"

def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)

def apply_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Minimal overrides consistent with your batch script."""
    cfg = dict(cfg)  # shallow copy is fine; we only set a few leaves

    # Batch sizes
    bc = cfg.setdefault("batch_config", {})
    if args.BE is not None:
        bc["B_E"] = int(args.BE)
    if args.BU is not None:
        bc["B_U"] = int(args.BU)
    if args.G is not None:
        bc["G"] = int(args.G)

    # Approx settings (baseline kind / EMA beta)
    approx = cfg.setdefault("approx_delta_h", {})
    base = approx.setdefault("baseline", {})
    if args.baseline_kind:
        base["kind"] = str(args.baseline_kind)
    if args.ema_beta is not None:
        base["ema_beta"] = float(args.ema_beta)

    # Normalization default (used by CV; still stored here for consistency)
    if args.normalization:
        approx["normalize"] = str(args.normalization)

    # Control-variates config (for reproducibility)
    cv_cfg = cfg.setdefault("control_variates", {})
    if args.features:
        cv_cfg["features"] = list(args.features)
    if args.ridge is not None:
        cv_cfg["ridge"] = float(args.ridge)
    if args.crossfit_folds is not None:
        cv_cfg["crossfit_folds"] = int(args.crossfit_folds)

    return cfg

def run_once(cfg: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save the *effective* config used
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    probe = EntropyMeasurements(cfg)

    # Delegate to the new runner method (adds its own timestamped cv_* subfolder)
    # The method returns paths + JSON-serializable summary.
    out_root = str(run_dir)  # pass run_dir as base; method creates cv_<ts>/ under it
    normalization = (cfg.get("approx_delta_h", {}) or {}).get("normalize", "per_token")
    cv_cfg = (cfg.get("control_variates", {}) or {})
    #features = cv_cfg.get("features", ["length", "mean_logp", "var_logp"])
    features = cv_cfg.get("features", ["length", "mean_logp", "var_logp", "rb_entropy_sum", "sum_w", "sum_w2", "surprisal_mean"])

    ridge = float(cv_cfg.get("ridge", 1e-8))
    crossfit = int(cv_cfg.get("crossfit_folds", 0))

    results = probe.run_control_variate_analysis(
        out_dir=out_root,
        normalization=normalization,
        features=features,
        ridge=ridge,
        crossfit_folds=crossfit,
    )

    # Also save a small index at the run level
    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run control-variates analysis and save to Drive.")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Base YAML config path")
    p.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT, help="Output root directory")
    p.add_argument("--label", type=str, default="cv_runs", help="Label name for the run folder")
    p.add_argument("--n_runs", type=int, default=1, help="Optional number of independent runs")

    # Light overrides (keep parity with your batch script where helpful)
    p.add_argument("--baseline_kind", type=str, default='hk_ema', help="Baseline kind (hk|hk_ema|hk_ridge|regression|none)")
    p.add_argument("--ema_beta", type=float, default=0.9, help="hk_ema beta")
    p.add_argument("--BE", type=int, default=512, help="B_E for E-batch")
    p.add_argument("--BU", type=int, default=64, help="B_U for U-batch (for v computation)")
    p.add_argument("--G", type=int, default=8, help="G for U-batch generations (for v computation)")

    # Control-variates knobs
    p.add_argument("--features", type=str, nargs="+", default=None,
                   help="Feature names for CV (e.g., length mean_logp var_logp)")
    p.add_argument("--normalization", type=str, default=None, choices=["per_token", "per_sequence"],
                   help="Target normalization for ⟨∇H, v⟩")
    p.add_argument("--ridge", type=float, default=None, help="Tiny ridge added to OLS normal equations")
    p.add_argument("--crossfit_folds", type=int, default=None, help="K-fold cross-fitting (0 = in-sample OLS)")

    return p.parse_args()

def main():
    args = parse_args()
    base_cfg = load_config(args.config)
    cfg = apply_overrides(base_cfg, args)

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    root = Path(os.path.join(args.output_root, f"{args.label}_{ts}"))
    root.mkdir(parents=True, exist_ok=True)

    index: List[Dict[str, Any]] = []
    for i in range(1, int(args.n_runs) + 1):
        run_dir = root / f"run_{i:02d}"
        print(f"\n[RUN {i}/{args.n_runs}] -> {run_dir}")
        try:
            results = run_once(cfg, run_dir)
            # Pull out the cv subfolder path if available
            cv_subdir = results.get("paths", {}).get("csv_path")
            index.append({
                "run": i,
                "run_dir": str(run_dir),
                "cv_csv": results.get("paths", {}).get("csv_path"),
                "cv_summary": results.get("paths", {}).get("summary_path"),
                "plots": {k: v for k, v in results.get("paths", {}).items() if k.startswith("plot_")},
                "summary": results.get("summary", {}),
            })
        except Exception as e:
            err_path = run_dir / "error.txt"
            err_path.parent.mkdir(parents=True, exist_ok=True)
            with open(err_path, "w") as f:
                f.write(repr(e))
            index.append({"run": i, "run_dir": str(run_dir), "error": repr(e)})

    with open(root / "index.json", "w") as f:
        json.dump(index, f, indent=2, default=str)
    print("\nAll runs complete. Index saved to:", root / "index.json")

if __name__ == "__main__":
    main()
