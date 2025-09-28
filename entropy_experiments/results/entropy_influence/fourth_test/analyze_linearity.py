#!/usr/bin/env python3
"""Analyze entropy-influence linearity for the fourth_test dataset."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"


@dataclass
class LineFit:
    slope: float
    intercept: float
    r_squared: float
    rmse: float
    max_abs_residual: float


@dataclass
class ComponentMetrics:
    eta: float
    full_delta: float
    baseline_delta: float
    sum_grad_components: float
    sum_baseline_components: float
    residual_full_vs_grad: float
    residual_full_vs_baseline: float


def _decode_eta(name: str) -> float:
    token = name.replace("delta_", "").replace("eta_", "").replace(".npy", "")
    token = token.replace("m", "-").replace("p", ".")
    if not token:
        raise ValueError(f"Could not parse eta from '{name}'")
    return float(token)


def _fit_line(etas: np.ndarray, values: np.ndarray) -> LineFit:
    if values.size == 0:
        return LineFit(0.0, 0.0, 1.0, 0.0, 0.0)
    coeffs = np.polyfit(etas, values, deg=1)
    pred = np.polyval(coeffs, etas)
    residuals = values - pred
    sse = float(np.sum(residuals ** 2))
    mean = float(np.mean(values))
    sst = float(np.sum((values - mean) ** 2))
    r_squared = 1.0 if sst == 0.0 else 1.0 - (sse / sst)
    rmse = float(np.sqrt(sse / values.size))
    max_abs = float(np.max(np.abs(residuals))) if residuals.size else 0.0
    slope, intercept = float(coeffs[0]), float(coeffs[1])
    return LineFit(slope, intercept, r_squared, rmse, max_abs)


def _summarise_fits(fits: Sequence[LineFit]) -> Dict[str, float]:
    if not fits:
        return {}
    r2_values = np.array([f.r_squared for f in fits], dtype=np.float64)
    rmse_values = np.array([f.rmse for f in fits], dtype=np.float64)
    max_abs_values = np.array([f.max_abs_residual for f in fits], dtype=np.float64)
    return {
        "r2_min": float(r2_values.min()),
        "r2_mean": float(r2_values.mean()),
        "r2_median": float(np.median(r2_values)),
        "r2_max": float(r2_values.max()),
        "rmse_mean": float(rmse_values.mean()),
        "rmse_median": float(np.median(rmse_values)),
        "rmse_max": float(rmse_values.max()),
        "max_abs_mean": float(max_abs_values.mean()),
        "max_abs_median": float(np.median(max_abs_values)),
        "max_abs_max": float(max_abs_values.max()),
    }


def _load_grad_components(grad_dir: Path) -> Dict[float, np.ndarray]:
    mapping: Dict[float, np.ndarray] = {}
    for npy_path in sorted(grad_dir.glob("delta_*.npy")):
        eta = _decode_eta(npy_path.name)
        mapping[eta] = np.load(npy_path)
    if not mapping:
        raise FileNotFoundError(f"No grad component files found in {grad_dir}")
    return mapping


def _load_per_sequence_matrix(base_dir: Path, subdir: str) -> Dict[float, np.ndarray]:
    target = base_dir / subdir
    mapping: Dict[float, np.ndarray] = {}
    for npy_path in sorted(target.glob("eta_*.npy")):
        eta = _decode_eta(npy_path.name)
        mapping[eta] = np.load(npy_path)
    if not mapping:
        raise FileNotFoundError(f"No entries found in {target}")
    return mapping


def _build_component_metrics(
    etas: Sequence[float],
    full_summary: Dict[float, float],
    baseline_summary: Dict[float, float],
    grad_components: Dict[float, np.ndarray],
    grad_baseline_components: Dict[float, np.ndarray],
) -> List[ComponentMetrics]:
    records: List[ComponentMetrics] = []
    for eta in etas:
        grad_vec = grad_components.get(eta)
        grad_baseline_mat = grad_baseline_components.get(eta)
        if grad_vec is None or grad_baseline_mat is None:
            raise KeyError(f"Missing component data for eta={eta:g}")
        baseline_delta = baseline_summary.get(eta, 0.0)
        sum_grad = float(np.sum(grad_vec))
        sum_baseline = float(np.sum(grad_baseline_mat))
        full_val = full_summary.get(eta, 0.0)
        residual_grad = full_val - (baseline_delta + sum_grad)
        residual_baseline = full_val - (baseline_delta + sum_baseline)
        records.append(
            ComponentMetrics(
                eta=eta,
                full_delta=full_val,
                baseline_delta=baseline_delta,
                sum_grad_components=sum_grad,
                sum_baseline_components=sum_baseline,
                residual_full_vs_grad=residual_grad,
                residual_full_vs_baseline=residual_baseline,
            )
        )
    return records


def _stack_direction_values(mapping: Dict[float, np.ndarray], etas: Sequence[float]) -> np.ndarray:
    vectors = [mapping[eta] for eta in etas]
    return np.stack(vectors, axis=0)


def _compute_direction_fits(etas: np.ndarray, matrix: np.ndarray) -> List[LineFit]:
    fits: List[LineFit] = []
    for col in range(matrix.shape[1]):
        values = matrix[:, col]
        fits.append(_fit_line(etas, values))
    return fits


def _encode_eta(eta: float) -> str:
    return f"{eta:.6g}"


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Path to entropy influence data folder (default: fourth_test/data)",
    )
    parser.add_argument(
        "--evaluation-index",
        type=int,
        default=0,
        help="Evaluation batch index to analyze",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Optional path to dump analysis summary as JSON",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of worst-fitting directions to report for each proxy",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    data_dir = args.data_dir
    summary_path = data_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    eval_entry = summary["evaluations"][args.evaluation_index]

    aggregate_entries = eval_entry["aggregate"]
    eta_values = [float(item["eta"]) for item in aggregate_entries]
    full_summary = {float(item["eta"]): float(item.get("delta_h", 0.0)) for item in aggregate_entries}

    baseline_eta_delta = {
        float(_decode_eta(f"eta_{key}.npy")): float(val)
        for key, val in (eval_entry.get("baseline_eta_delta") or {}).items()
    }

    grad_dir = data_dir / f"eval_{args.evaluation_index:02d}_grad_eta"
    grad_components = _load_grad_components(grad_dir)

    per_seq_dir = data_dir / f"eval_{args.evaluation_index:02d}_per_sequence"
    grad_baseline_components = _load_per_sequence_matrix(per_seq_dir, "baseline_plus_grad")

    component_metrics = _build_component_metrics(
        eta_values,
        full_summary,
        baseline_eta_delta,
        grad_components,
        grad_baseline_components,
    )

    etas_array = np.array(eta_values, dtype=np.float64)
    grad_matrix = _stack_direction_values(grad_components, eta_values)
    grad_baseline_matrix = np.stack(
        [grad_baseline_components[eta].sum(axis=1) for eta in eta_values],
        axis=0,
    )

    grad_fits = _compute_direction_fits(etas_array, grad_matrix)
    baseline_fits = _compute_direction_fits(etas_array, grad_baseline_matrix)

    grad_stats = _summarise_fits(grad_fits)
    baseline_stats = _summarise_fits(baseline_fits)

    print("Aggregate delta-H comparisons:")
    for record in component_metrics:
        eta_str = _encode_eta(record.eta)
        print(
            f"  eta={eta_str}: full={record.full_delta:.6e}, baseline={record.baseline_delta:.6e}, "
            f"sum_grad={record.sum_grad_components:.6e}, sum_baseline_grad={record.sum_baseline_components:.6e}, "
            f"res(full - baseline - sum_grad)={record.residual_full_vs_grad:.3e}, "
            f"res(full - baseline - sum_baseline_grad)={record.residual_full_vs_baseline:.3e}"
        )

    def _report_worst(label: str, fits: Sequence[LineFit]) -> List[Dict[str, float]]:
        paired = list(enumerate(fits))
        paired.sort(key=lambda item: item[1].r_squared)
        worst = paired[: args.top_k]
        print(f"\nWorst {label} directions by R^2:")
        results: List[Dict[str, float]] = []
        for idx, fit in worst:
            print(
                f"  idx={idx:03d}  R^2={fit.r_squared:.6f}  slope={fit.slope:.3e}  "
                f"rmse={fit.rmse:.3e}  max|resid|={fit.max_abs_residual:.3e}"
            )
            results.append(
                {
                    "index": idx,
                    "r_squared": fit.r_squared,
                    "slope": fit.slope,
                    "rmse": fit.rmse,
                    "max_abs_residual": fit.max_abs_residual,
                }
            )
        return results

    worst_grad = _report_worst("raw grad-only", grad_fits)
    worst_baseline = _report_worst("baseline-adjusted", baseline_fits)

    print("\nGradient-only fit stats:")
    for key, value in grad_stats.items():
        print(f"  {key}: {value:.6e}")

    print("\nBaseline-adjusted fit stats:")
    for key, value in baseline_stats.items():
        print(f"  {key}: {value:.6e}")

    if args.json_out:
        payload = {
            "etas": eta_values,
            "aggregate_metrics": [record.__dict__ for record in component_metrics],
            "gradient_fit_stats": grad_stats,
            "baseline_fit_stats": baseline_stats,
            "worst_gradient_directions": worst_grad,
            "worst_baseline_directions": worst_baseline,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote summary to {args.json_out}")


if __name__ == "__main__":
    main()
