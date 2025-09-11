#!/usr/bin/env python3
"""
Batch driver to study variance across baseline settings.

Reads the base config (entropy_experiments/configs/config_template.yaml),
constructs a small grid of (method, baseline.kind, baseline knobs, jackknife)
settings, and runs EntropyMeasurements for each setting with a few replicates.

Results are saved under:
  /content/drive/MyDrive/RL_Practice_Files/entropy_variance_tests_YYYYmmdd_HHMM/
with one subfolder per setting and replicate, containing:
  - results.json   (runner output)
  - config.yaml    (effective config used for the run)

Default grid (about ~24 runs at ~20 min each ≈ 8 hours):
  Methods: [jvp, grad_dot]
  Controls: hk (both)
  hk_ema: ema_beta in [0.90, 0.95] (both), pos_bins fixed from base config
  hk_ridge (JVP): lambda in [1e-4, 1e-3, 1e-2]
  regression (grad_dot): regression_l2 in [0.0, 1e-3, 1e-2]
  Variance: jackknife in [false, true] (enabled=true)
  Replicates per setting: 2

You can tweak the grid or replicates below if you need to shorten runtime.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml

from entropy_experiments.entropy_experiment_runner import EntropyMeasurements


BASE_OUTPUT_ROOT = "/content/drive/MyDrive/RL_Practice_Files/"
DEFAULT_CONFIG_PATH = Path("entropy_experiments/configs/config_template.yaml")


@dataclass
class Setting:
    method: str
    baseline_kind: str
    jackknife: bool
    knobs: Dict[str, Any]
    label: str  # concise folder label


def load_base_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_grid(base_cfg: Dict[str, Any]) -> List[Setting]:
    # Use the current batch sizes and a single eta
    # We override estimator settings here
    # Define grids
    jack_list = [False, True]

    # hk controls (both methods)
    grid: List[Setting] = []

    def add(method: str, baseline_kind: str, jackknife: bool, knobs: Dict[str, Any], label_parts: List[str]):
        label = "__".join(label_parts)
        grid.append(Setting(method=method, baseline_kind=baseline_kind, jackknife=jackknife, knobs=knobs, label=label))

    # Controls: hk
    for jk in jack_list:
        add("jvp", "hk", jk, {}, ["jvp", "hk", f"jk={int(jk)}"]) 
        add("grad_dot", "hk", jk, {}, ["gd", "hk", f"jk={int(jk)}"]) 

    # hk_ema: ema_beta in {0.90, 0.95}
    for jk in jack_list:
        for beta in [0.90, 0.95]:
            add("jvp", "hk_ema", jk, {"baseline.ema_beta": beta}, ["jvp", "hk_ema", f"beta={beta}", f"jk={int(jk)}"]) 
            add("grad_dot", "hk_ema", jk, {"baseline.ema_beta": beta}, ["gd", "hk_ema", f"beta={beta}", f"jk={int(jk)}"]) 

    # hk_ridge for JVP only: lambda in {1e-4,1e-3,1e-2}
    for jk in jack_list:
        for lam in [1e-4, 1e-3, 1e-2]:
            add("jvp", "hk_ridge", jk, {"ridge.lambda": lam}, ["jvp", "hk_ridge", f"lam={lam}", f"jk={int(jk)}"]) 

    # regression for grad_dot only: regression_l2 in {0.0, 1e-3, 1e-2}
    for jk in jack_list:
        for l2 in [0.0, 1e-3, 1e-2]:
            add("grad_dot", "regression", jk, {"baseline.regression_l2": l2}, ["gd", "reg", f"l2={l2}", f"jk={int(jk)}"]) 

    return grid


def set_nested(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    """Set nested config value from a dot path like 'baseline.ema_beta'."""
    parts = dotted.split(".")
    d = cfg
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def apply_setting(base_cfg: Dict[str, Any], s: Setting) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    # Force only single eta
    est = cfg.setdefault("estimator", {})
    est["eta_sweep"] = False
    est["single_eta"] = 4e-6

    approx = cfg.setdefault("approx_delta_h", {})
    approx["method"] = s.method
    base = approx.setdefault("baseline", {})
    base["kind"] = s.baseline_kind
    var = approx.setdefault("variance", {})
    var["enabled"] = True
    var["jackknife"] = bool(s.jackknife)

    # Apply extra knobs for this setting
    for k, v in s.knobs.items():
        set_nested(approx, k, v)  # knobs are relative to approx_delta_h

    return cfg


def run_setting(cfg: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save effective config
    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    probe = EntropyMeasurements(cfg)
    results = probe.run_experiments()
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


def main():
    base_cfg = load_base_config(DEFAULT_CONFIG_PATH)

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    root = Path(os.path.join(BASE_OUTPUT_ROOT, f"entropy_variance_tests_{ts}"))
    root.mkdir(parents=True, exist_ok=True)

    grid = build_grid(base_cfg)
    replicates = 2  # keep total runtime ~ 8 hours.

    index: List[Dict[str, Any]] = []
    for s in grid:
        for rep in range(1, replicates + 1):
            cfg = apply_setting(base_cfg, s)
            setting_dir = root / s.label / f"rep_{rep}"
            print(f"\n[RUN] {s.label}  rep={rep}  -> {setting_dir}")
            try:
                results = run_setting(cfg, setting_dir)
                index.append({
                    "label": s.label,
                    "method": s.method,
                    "baseline_kind": s.baseline_kind,
                    "jackknife": s.jackknife,
                    "knobs": s.knobs,
                    "rep": rep,
                    "path": str(setting_dir),
                    "timing": results.get("timing", {}),
                    "variance": results.get("variance", {}),
                })
            except Exception as e:
                err_path = setting_dir / "error.txt"
                err_path.parent.mkdir(parents=True, exist_ok=True)
                with open(err_path, "w") as f:
                    f.write(str(e))
                index.append({
                    "label": s.label,
                    "method": s.method,
                    "baseline_kind": s.baseline_kind,
                    "jackknife": s.jackknife,
                    "knobs": s.knobs,
                    "rep": rep,
                    "path": str(setting_dir),
                    "error": str(e),
                })

    # Save index summary
    with open(root / "index.json", "w") as f:
        json.dump(index, f, indent=2, default=str)

    print("\nAll runs complete. Index saved to:", root / "index.json")


if __name__ == "__main__":
    main()

