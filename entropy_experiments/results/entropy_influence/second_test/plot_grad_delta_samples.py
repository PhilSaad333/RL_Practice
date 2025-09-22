#!/usr/bin/env python3
"""Plot per-sequence grad-only entropy deltas across learning rates.

Usage
-----
python plot_grad_delta_samples.py \
  --data-dir entropy_experiments/results/entropy_influence/second_test/data \
  --indices 0 1 2 3 \
  --out-dir entropy_experiments/results/entropy_influence/second_test/plots

If --indices is omitted, a random sample of sequences is drawn.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def load_grad_deltas(grad_dir: Path) -> Dict[float, np.ndarray]:
    mapping: Dict[float, np.ndarray] = {}
    for npy_path in grad_dir.glob("delta_*.npy"):
        key = npy_path.stem.replace("delta_", "").replace("m", "-").replace("p", ".")
        eta = float(key)
        mapping[eta] = np.load(npy_path)
    if not mapping:
        raise FileNotFoundError(f"No grad delta files found in {grad_dir}")
    return dict(sorted(mapping.items()))


def choose_indices(total: int, requested: Sequence[int] | None, sample_size: int, valid: Sequence[int] | None = None) -> List[int]:
    if requested:
        for idx in requested:
            if idx < 0 or idx >= total:
                raise ValueError(f"Index {idx} is out of range for total {total}")
        return list(dict.fromkeys(requested))
    candidates = list(valid) if valid else list(range(total))
    if not candidates:
        return []
    if len(candidates) <= sample_size:
        return candidates
    return random.sample(candidates, sample_size)


def plot_samples(
    grad_deltas: Dict[float, np.ndarray],
    indices: Iterable[int],
    out_dir: Path,
    baseline: Dict[str, float] | None = None,
    title_prefix: str = "sequence",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    etas = np.array(sorted(grad_deltas.keys()), dtype=np.float64)
    baseline_vals = None
    if baseline:
        baseline_float = {float(k): float(v) for k, v in baseline.items()}
        baseline_vals = np.array([baseline_float.get(float(eta), np.nan) for eta in etas], dtype=np.float64)
        if np.isnan(baseline_vals).all():
            baseline_vals = None

    stacked = np.stack([grad_deltas[eta] for eta in etas], axis=0)
    for idx in indices:
        values = stacked[:, idx]
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        ax.scatter(etas, values, color=DEFAULT_COLORS[0], label="grad-only ΔH")
        ax.plot(etas, values, color=DEFAULT_COLORS[0])
        if baseline_vals is not None:
            ax.plot(etas, baseline_vals, color=DEFAULT_COLORS[1], linestyle="--", label="baseline ΔH")
        coeffs = np.polyfit(etas, values, deg=1)
        fit_vals = np.polyval(coeffs, etas)
        ax.plot(etas, fit_vals, color=DEFAULT_COLORS[2], linestyle=":", label="linear fit")
        ax.set_title(f"{title_prefix} {idx}")
        ax.set_xlabel("learning rate η")
        ax.set_ylabel("ΔH_true (grad only)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"grad_delta_seq_{idx:03d}.png", dpi=300)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing summary.json and grad data")
    parser.add_argument("--evaluation-index", type=int, default=0, help="Evaluation batch index to process (default: 0)")
    parser.add_argument("--indices", type=int, nargs="*", help="Specific update sequence indices to plot")
    parser.add_argument("--sample-size", type=int, default=6, help="Number of sequences to sample if indices not specified")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for output plots")
    args = parser.parse_args()

    summary = json.loads((args.data_dir / "summary.json").read_text())
    eval_entry = summary["evaluations"][args.evaluation_index]
    baseline = eval_entry.get("baseline_eta_delta")

    grad_dir = args.data_dir / f"eval_{args.evaluation_index:02d}_grad_eta"
    grad_deltas = load_grad_deltas(grad_dir)
    stacked = np.stack([grad_deltas[eta] for eta in grad_deltas], axis=0)
    total_sequences = stacked.shape[1]
    nonzero_mask = np.any(stacked != 0.0, axis=0)
    valid = [i for i, flag in enumerate(nonzero_mask) if flag]
    indices = choose_indices(total_sequences, args.indices, args.sample_size, valid=valid)
    if not indices:
        print("No nonzero sequences found; nothing to plot.")
        return

    plot_samples(grad_deltas, indices, args.out_dir, baseline=baseline)
    print(f"Wrote plots for indices {indices} to {args.out_dir}")


if __name__ == "__main__":
    main()
