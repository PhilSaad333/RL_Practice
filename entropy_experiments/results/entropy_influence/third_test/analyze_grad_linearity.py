#!/usr/bin/env python3
"""Assess linearity of per-sequence grad-only entropy deltas across learning rates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def load_grad_deltas(grad_dir: Path) -> Dict[float, np.ndarray]:
    mapping: Dict[float, np.ndarray] = {}
    for npy_path in grad_dir.glob("delta_*.npy"):
        key = npy_path.stem.replace("delta_", "").replace("m", "-").replace("p", ".")
        eta = float(key)
        mapping[eta] = np.load(npy_path)
    if not mapping:
        raise FileNotFoundError(f"No grad delta files found in {grad_dir}")
    return dict(sorted(mapping.items()))


def fit_line(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float, float]:
    coeffs = np.polyfit(x, y, deg=1)
    slope, intercept = coeffs
    pred = np.polyval(coeffs, x)
    residuals = y - pred
    sse = np.sum(residuals ** 2)
    mean = np.mean(y)
    sst = np.sum((y - mean) ** 2)
    r2 = 1.0 - sse / sst if sst > 0 else 1.0
    rmse = np.sqrt(sse / x.size)
    max_abs = float(np.max(np.abs(residuals)))
    return float(slope), float(intercept), float(r2), float(rmse), max_abs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory with summary.json and eval_* data")
    parser.add_argument("--evaluation-index", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5, help="How many worst-fitting sequences to display")
    parser.add_argument("--json-out", type=Path, help="Optional path to dump per-sequence metrics as JSON")
    args = parser.parse_args()

    grad_dir = args.data_dir / f"eval_{args.evaluation_index:02d}_grad_eta"
    grad_deltas = load_grad_deltas(grad_dir)

    etas = np.array(list(grad_deltas.keys()), dtype=np.float64)
    matrix = np.stack([grad_deltas[eta] for eta in etas], axis=0)  # [num_eta, num_sequences]

    results = []
    for idx in range(matrix.shape[1]):
        values = matrix[:, idx]
        slope, intercept, r2, rmse, max_abs = fit_line(etas, values)
        results.append(
            {
                "index": idx,
                "slope": slope,
                "intercept": intercept,
                "r_squared": r2,
                "rmse": rmse,
                "max_abs_residual": max_abs,
            }
        )

    results.sort(key=lambda r: r["r_squared"])  # ascending => worst first
    worst = results[: args.top_k]

    print("Worst sequences by R^2 (lower is worse):")
    for item in worst:
        print(
            f"idx={item['index']:03d}  R^2={item['r_squared']:.6f}  rmse={item['rmse']:.2e}  max|resid|={item['max_abs_residual']:.2e}"
        )

    r2_values = np.array([r["r_squared"] for r in results])
    print(
        f"R^2 stats -> min={r2_values.min():.6f}, mean={r2_values.mean():.6f}, median={np.median(r2_values):.6f}, max={r2_values.max():.6f}"
    )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=2))
        print(f"Wrote metrics to {args.json_out}")


if __name__ == "__main__":
    main()
