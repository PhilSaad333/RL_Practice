# evals/plot_metrics.py
#
# Usage example (Colab):
#   %run evals/plot_metrics.py --model_name phi2 \
#                              --train_dataset math \
#                              --eval_dataset math
#
# Optional --run_name lets you override the auto-detected tag if you keep more
# than one (e.g. r4 vs r8) under a step folder.

import os
import re
from pathlib import Path
from itertools import product
from typing import List, Tuple, Set, Optional

import pandas as pd
import matplotlib.pyplot as plt
import tyro


def _discover(
    base_root: Path,
    eval_dataset: str,
) -> Tuple[List[int], List[float], List[float], Set[str]]:
    """
    Walk only those step_*_{eval_dataset} folders under `base_root` and
    return:
      • steps   – sorted list of ints
      • temps   – sorted list of floats
      • ps      – sorted list of floats
      • runs    – set of distinct run tags (strings)
    """
    # Only match step directories for the eval_dataset we care about:
    step_pattern = re.compile(rf"^step_(\d+)_{re.escape(eval_dataset)}$")
    sub_re       = re.compile(r"^temp(?P<temp>[0-9.]+)_p(?P<p>[0-9.]+)_(?P<run>.+)$")

    steps:   Set[int]   = set()
    temps:   Set[float] = set()
    ps:      Set[float] = set()
    run_set: Set[str]   = set()

    # Glob only step_*_{eval_dataset}
    for step_dir in base_root.glob(f"step_*_{eval_dataset}"):
        if not step_dir.is_dir():
            continue

        m_step = step_pattern.match(step_dir.name)
        if not m_step:
            continue

        step_num = int(m_step.group(1))
        steps.add(step_num)

        # inside each step folder, look for temp…_p…_{run}
        for sub in step_dir.iterdir():
            if not sub.is_dir():
                continue
            m = sub_re.match(sub.name)
            if not m:
                continue
            temps.add(float(m.group("temp")))
            ps.add(float(m.group("p")))
            run_set.add(m.group("run"))

    return sorted(steps), sorted(temps), sorted(ps), run_set


def main(
    model_name: str,
    train_dataset: str,
    eval_dataset: str,
    run_name: Optional[str] = None,
    show: bool = True,
    save_dir: Optional[str] = None,
):
    base_root = Path(
        f"/content/drive/MyDrive/RL_Practice_Files/eval_runs/"
        f"{model_name}_{train_dataset}_finetuned"
    )

    # 1) Discover only the step_*_{eval_dataset} folders:
    steps, temps, ps, run_tags = _discover(base_root, eval_dataset)

    if not steps:
        raise RuntimeError(f"No `step_*_{eval_dataset}` folders found in {base_root}")

    if run_name is None:
        if len(run_tags) != 1:
            raise ValueError(
                f"Multiple run tags found ({', '.join(sorted(run_tags))}); "
                "pass --run_name to select one."
            )
        run_name = next(iter(run_tags))

    print(
        f"Discovered for eval_dataset={eval_dataset!r}:\n"
        f"  steps = {steps}\n"
        f"  temps = {temps}\n"
        f"  top_p = {ps}\n"
        f"  run   = {run_name!r}"
    )

    # 2) Pick an example to read metric columns
    example_csv = (
        base_root
        / f"step_{steps[0]}_{eval_dataset}"
        / f"temp{temps[0]}_p{ps[0]}_{run_name}"
        / "metrics.csv"
    )
    if not example_csv.exists():
        raise FileNotFoundError(f"Expected metrics.csv at {example_csv}")

    metric_cols = [
        c for c in pd.read_csv(example_csv, nrows=1).columns
        if c != "q_idx"
    ]

    # 3) Prepare an empty DataFrame per metric
    idx  = temps
    cols = pd.MultiIndex.from_product([steps, ps], names=["step", "top_p"])
    agg  = {m: pd.DataFrame(index=idx, columns=cols, dtype=float) for m in metric_cols}

    # 4) Fill in means
    for step, temp, p in product(steps, temps, ps):
        metrics_path = (
            base_root
            / f"step_{step}_{eval_dataset}"
            / f"temp{temp}_p{p}_{run_name}"
            / "metrics.csv"
        )
        if not metrics_path.exists():
            print(f"[WARNING] missing {metrics_path}")
            continue

        mean_vals = pd.read_csv(metrics_path)[metric_cols].mean()
        for m in metric_cols:
            agg[m].loc[temp, (step, p)] = mean_vals[m]

    # 5) Plot (and optionally save)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for m, df in agg.items():
        plt.figure(figsize=(6, 4))
        for step in steps:
            for p in ps:
                plt.plot(
                    df.index,
                    df[(step, p)],
                    marker="o",
                    label=f"step {step}, p={p}"
                )
        plt.xlabel("Temperature")
        plt.ylabel(m)
        plt.title(f"{m} vs Temperature ({eval_dataset})")
        plt.grid(True)
        plt.legend(fontsize="small")
        plt.tight_layout()

        if save_dir:
            out = Path(save_dir) / f"{eval_dataset}_{m}.png"
            plt.savefig(out, dpi=140)
            print(f"Saved plot to {out}")

        if show:
            plt.show()
        else:
            plt.close()

    print("All done!")


if __name__ == "__main__":
    tyro.cli(main)
