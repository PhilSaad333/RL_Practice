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


def _discover(base_root: Path) -> Tuple[List[int], List[float], List[float], Set[str]]:
    """
    Walk `base_root` and return:
      • steps   – sorted list of ints
      • temps   – sorted list of floats
      • ps      – sorted list of floats
      • runs    – set of distinct run tags (strings) that appear
    Folder structure assumed:
        base_root/
          step_{STEP}_{eval_dataset}/
            temp{TEMP}_p{P}_{RUN}/metrics.csv
    """
    step_re = re.compile(r"step_(\d+)")
    sub_re  = re.compile(r"temp(?P<temp>[0-9.]+)_p(?P<p>[0-9.]+)_(?P<run>.+)")

    steps:   Set[int]   = set()
    temps:   Set[float] = set()
    ps:      Set[float] = set()
    run_set: Set[str]   = set()

    for step_dir in base_root.glob("step_*"):
        if not step_dir.is_dir():
            continue
        step_match = step_re.match(step_dir.name)
        if step_match:
            steps.add(int(step_match.group(1)))

        for sub in step_dir.iterdir():
            if not sub.is_dir():
                continue
            m = sub_re.match(sub.name)
            if m:
                temps.add(float(m.group("temp")))
                ps.add(float(m.group("p")))
                run_set.add(m.group("run"))

    return sorted(steps), sorted(temps), sorted(ps), run_set


def main(
    model_name: str,
    train_dataset: str,
    eval_dataset: str,
    run_name: Optional[str] = None,              # override if you wish
    show: bool = True,                           # set False inside Colab jobs
    save_dir: Optional[str] = None,              # if you want PNGs saved
):
    """
    Plot the mean of each metric over temperatures, one figure per metric.

    Parameters
    ----------
    model_name, train_dataset, eval_dataset : str
        Identify the folder `.../eval_runs/{model}_{train}_finetuned/`.
    run_name : str, optional
        If None we auto-detect it.  Override when more than one exists.
    show : bool
        If True call plt.show(); set False for headless batch jobs.
    save_dir : str, optional
        Directory to save each figure as a PNG (created if missing).
    """

    base_root = Path(
        f"/content/drive/MyDrive/RL_Practice_Files/eval_runs/"
        f"{model_name}_{train_dataset}_finetuned"
    )

    # ---------------------------------------------------------------------
    # 1) Discover available (steps, temps, ps, runs) automatically
    # ---------------------------------------------------------------------
    steps, temps, ps, run_tags = _discover(base_root)

    if run_name is None:
        if len(run_tags) != 1:
            raise ValueError(
                f"Multiple run folders found ({', '.join(sorted(run_tags))}); "
                "pass --run_name to disambiguate."
            )
        run_name = next(iter(run_tags))

    print(
        f"Discovered:\n  steps  = {steps}\n  temps  = {temps}"
        f"\n  top_p  = {ps}\n  run    = '{run_name}'"
    )

    # ---------------------------------------------------------------------
    # 2) Determine metric columns from one example CSV
    # ---------------------------------------------------------------------
    example_path = (
        base_root
        / f"step_{steps[0]}_{eval_dataset}"
        / f"temp{temps[0]}_p{ps[0]}_{run_name}"
        / "metrics.csv"
    )

    if not example_path.exists():
        raise FileNotFoundError(f"Expected metrics file not found: {example_path}")

    metric_cols = [
        c for c in pd.read_csv(example_path, nrows=1).columns if c != "q_idx"
    ]

    # ---------------------------------------------------------------------
    # 3) Build <metric> → DataFrame( index=temp , columns = MultiIndex(step,p) )
    # ---------------------------------------------------------------------
    idx = temps
    cols = pd.MultiIndex.from_product([steps, ps], names=["step", "top_p"])
    agg = {m: pd.DataFrame(index=idx, columns=cols, dtype=float) for m in metric_cols}

    # ---------------------------------------------------------------------
    # 4) Populate with means
    # ---------------------------------------------------------------------
    for step, temp, p in product(steps, temps, ps):
        csv_path = (
            base_root
            / f"step_{step}_{eval_dataset}"
            / f"temp{temp}_p{p}_{run_name}"
            / "metrics.csv"
        )
        if not csv_path.exists():
            print(f"[WARN] missing {csv_path}")
            continue

        means = pd.read_csv(csv_path)[metric_cols].mean()
        for m in metric_cols:
            agg[m].loc[temp, (step, p)] = means[m]

    # ---------------------------------------------------------------------
    # 5) Plot
    # ---------------------------------------------------------------------
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for m, df in agg.items():
        plt.figure(figsize=(6, 4))
        for step in steps:
            for p in ps:
                label = f"step {step}, p={p}"
                plt.plot(df.index, df[(step, p)], marker="o", label=label)

        plt.xlabel("Temperature")
        plt.ylabel(m)
        plt.title(f"{m} vs. Temperature")
        plt.grid(True)
        plt.legend(fontsize="small")
        plt.tight_layout()

        if save_dir:
            out = Path(save_dir) / f"{m}.png"
            plt.savefig(out, dpi=140)
            print(f"Saved {out}")

        if show:
            plt.show()
        else:
            plt.close()

    print("Done!")


if __name__ == "__main__":
    tyro.cli(main)
