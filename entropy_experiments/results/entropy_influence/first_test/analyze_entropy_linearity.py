#!/usr/bin/env python3
"""Check whether aggregate entropy deltas scale linearly with the learning rate.

The script expects the directory layout produced by scripts/run_entropy_influence_large.py:
<run>/
  data/
    summary.json
    eval_00_delta_matrix.npy
    ...

Usage
-----
python entropy_experiments/results/entropy_influence/first_test/analyze_entropy_linearity.py \
  --data-dir entropy_experiments/results/entropy_influence/first_test/data
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class AggregatePoint:
    eta: float
    delta_sum: float
    delta_h: float


@dataclass
class FitResult:
    slope: float
    intercept: float
    r_squared: float
    rmse: float
    max_abs_residual: float
    residuals: np.ndarray
    predicted: np.ndarray


def load_summary(summary_path: Path, evaluation_index: int) -> List[AggregatePoint]:
    raw = json.loads(summary_path.read_text(encoding="utf-8"))
    try:
        evaluation = raw["evaluations"][evaluation_index]
    except (KeyError, IndexError) as exc:
        raise ValueError(
            f"No evaluation index {evaluation_index} in {summary_path}; "
            f"found {len(raw.get('evaluations', []))} entries"
        ) from exc

    points: List[AggregatePoint] = []
    for entry in evaluation.get("aggregate", []):
        eta = float(entry["eta"])
        delta_h = float(entry.get("delta_h", 0.0))
        delta_sum = float(entry.get("per_sequence_delta_sum", delta_h))
        points.append(AggregatePoint(eta=eta, delta_sum=delta_sum, delta_h=delta_h))

    if not points:
        raise ValueError(
            f"No aggregate entries found at evaluation index {evaluation_index} in {summary_path}"
        )

    points.sort(key=lambda p: p.eta)
    return points


def fit_line(points: Iterable[AggregatePoint]) -> FitResult:
    etas = np.array([p.eta for p in points], dtype=np.float64)
    delta = np.array([p.delta_sum for p in points], dtype=np.float64)

    if etas.ndim != 1 or etas.size < 2:
        raise ValueError("Need at least two points to fit a line")

    # Least squares fit: y = slope * x + intercept.
    a = np.vstack([etas, np.ones_like(etas)]).T
    solution, residuals, _, _ = np.linalg.lstsq(a, delta, rcond=None)
    slope, intercept = solution

    predicted = slope * etas + intercept
    residual_vec = delta - predicted
    sse = float(np.sum(residual_vec**2))
    mean = float(np.mean(delta))
    sst = float(np.sum((delta - mean) ** 2))
    r_squared = 1.0 - sse / sst if sst > 0 else 1.0
    rmse = math.sqrt(sse / etas.size)
    max_abs_residual = float(np.max(np.abs(residual_vec)))

    return FitResult(
        slope=float(slope),
        intercept=float(intercept),
        r_squared=float(r_squared),
        rmse=float(rmse),
        max_abs_residual=max_abs_residual,
        residuals=residual_vec,
        predicted=predicted,
    )


def format_table(points: Iterable[AggregatePoint], fit: FitResult) -> str:
    header = "eta        delta_sum    predicted    residual"
    rows = []
    for point, pred, resid in zip(points, fit.predicted, fit.residuals):
        rows.append(
            f"{point.eta:>8.1e}  {point.delta_sum:>+.6e}  {pred:>+.6e}  {resid:>+.2e}"
        )
    metrics = (
        f"slope={fit.slope:+.6e}, intercept={fit.intercept:+.6e}, "
        f"R^2={fit.r_squared:.6f}, rmse={fit.rmse:.2e}, "
        f"max|residual|={fit.max_abs_residual:.2e}"
    )
    return "\n".join([header, *rows, "", metrics])


def build_report(points: Iterable[AggregatePoint], fit: FitResult) -> dict:
    return {
        "points": [
            {
                "eta": point.eta,
                "delta_sum": point.delta_sum,
                "delta_h": point.delta_h,
                "predicted": float(pred),
                "residual": float(resid),
            }
            for point, pred, resid in zip(points, fit.predicted, fit.residuals)
        ],
        "fit": {
            "slope": fit.slope,
            "intercept": fit.intercept,
            "r_squared": fit.r_squared,
            "rmse": fit.rmse,
            "max_abs_residual": fit.max_abs_residual,
        },
    }


def plot_linearity(points: Iterable[AggregatePoint], fit: FitResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    etas = np.array([p.eta for p in points], dtype=np.float64)
    delta = np.array([p.delta_sum for p in points], dtype=np.float64)

    x_grid = np.linspace(etas.min(), etas.max(), 200)
    y_grid = fit.slope * x_grid + fit.intercept

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(etas, delta, color="#1f77b4", label="aggregate ΔH")
    ax.plot(x_grid, y_grid, color="#ff7f0e", label="linear fit")
    ax.set_xlabel("learning rate (eta)")
    ax.set_ylabel("aggregate entropy change")
    ax.set_title("Entropy influence vs learning rate")
    ax.legend()
    annotation = (
        f"slope={fit.slope:+.2e}\nintercept={fit.intercept:+.2e}\n"
        f"R²={fit.r_squared:.5f}\nRMSE={fit.rmse:.1e}"
    )
    ax.text(
        0.05,
        0.95,
        annotation,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "aggregate_delta_vs_eta.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
    ax.scatter(etas, fit.residuals, color="#d62728", label="residual")
    ax.set_xlabel("learning rate (eta)")
    ax.set_ylabel("residual (observed - predicted)")
    ax.set_title("Fit residuals vs learning rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "residuals_vs_eta.png", dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_data = script_dir / "data"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data,
        help="Directory that contains summary.json (default: %(default)s)",
    )
    parser.add_argument(
        "--summary-name",
        default="summary.json",
        help="Filename of the summary JSON inside the data directory.",
    )
    parser.add_argument(
        "--evaluation-index",
        type=int,
        default=0,
        help="Which evaluation entry to analyse (default: %(default)s)",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        help="Optional path to dump a JSON report with fit statistics.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        help="Directory to store plot images (default: <script dir>/plots)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    plots_dir = args.plots_dir or (script_dir / "plots")

    summary_path = args.data_dir / args.summary_name
    if not summary_path.exists():
        raise FileNotFoundError(f"Could not find {summary_path}")

    points = load_summary(summary_path, args.evaluation_index)
    fit = fit_line(points)

    table = format_table(points, fit)
    print(table)

    if plots_dir:
        plot_linearity(points, fit, plots_dir)

    if args.report_path:
        report = build_report(points, fit)
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
