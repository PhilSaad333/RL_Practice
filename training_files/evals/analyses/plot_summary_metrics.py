# evals/analyses/plot_summary_metrics.py
#
# Usage example (Colab):
# !pip --quiet install tyro
# !python -m evals.cli \
#     --analysis     plot_summary_metrics \
#     --base-root    /content/drive/MyDrive/RL_Practice_Files/eval_runs/phi2_gsm8k_latex_finetuned \
#     --show \
#     --save-dir /content/plots
# from IPython.display import Image, display
# import glob, os
# 
# for png in glob.glob("/content/plots/*.png"):
#     display(Image(png))


import os
import re
from pathlib import Path
from itertools import product
from typing import List, Tuple, Set, Optional

import pandas as pd
import matplotlib.pyplot as plt
import tyro


def _discover_metrics(
    base_root: Path,
) -> Tuple[List[int], List[float], List[float], Set[str]]:
    """
    Walk all step_* folders under `base_root` and
    # Match any step directory:
    """
    # Only match step directories for the eval_dataset we care about:
    step_pattern = re.compile(r"^step_(\d+)_")    
    sub_re       = re.compile(r"^temp(?P<temp>[0-9.]+)_p(?P<p>[0-9.]+)_(?P<run>.+)$")

    steps:   Set[int]   = set()
    temps:   Set[float] = set()
    ps:      Set[float] = set()
    run_set: Set[str]   = set()

    # Glob every step_* folder
    for step_dir in base_root.glob("step_*"):
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
    base_root: Path,
    gens_per_prompt: int = 8,
    show: bool = True,
    save_dir: Optional[Path] = None,
):
    """
    Plot summary metrics for every step/temp/top_p under base_root.

    base_root should be something like
      /…/eval_runs/phi2_math_finetuned
    """


    # 1) Discover every step_* folder under base_root:
    steps, temps, ps, run_tags = _discover_metrics(base_root)

    if not steps:
        raise RuntimeError(f"No `step_*` folders found under {base_root}")

    # pick the actual on-disk folder for the first step, e.g. "step_394_gsm8k_latex"
    step_dirs = list(base_root.glob(f"step_{steps[0]}_*"))
    if not step_dirs:
        raise FileNotFoundError(f"No step_{steps[0]}_* folder under {base_root}")
    step_name = step_dirs[0].name

    # extract everything after "step_<num>_"
    m = re.match(r"^step_\d+_(.+)$", step_name)
    if not m:
        raise ValueError(f"Couldn't parse eval_dataset from {step_name}")
    eval_dataset = m.group(1)      # e.g. "gsm8k_latex"


    print(f"Found steps={steps}, temps={temps}, top_p={ps}, gens_per_prompt={gens_per_prompt}")


    # 2) Pick an example to read metric columns
    # find the actual step directory on disk (e.g. "step_394_gsm8k_latex")
    step_glob = list(base_root.glob(f"step_{steps[0]}_*"))
    if not step_glob:
        raise FileNotFoundError(f"No step directory matching step_{steps[0]}_* under {base_root}")
    step_dir = step_glob[0]

    example_csv = (
        step_dir
        / f"temp{temps[0]}_p{ps[0]}_r{gens_per_prompt}"
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
        # locate the correct step‐directory again
        step_glob = list(base_root.glob(f"step_{step}_*"))
        if not step_glob:
            print(f"[WARNING] no dir step_{step}_*")
            continue
        step_dir = step_glob[0]

        metrics_path = step_dir / f"temp{temp}_p{p}_r{gens_per_prompt}" / "metrics.csv"
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
            out = Path(save_dir) / f"{m}.png"
            plt.savefig(out, dpi=140)
            print(f"Saved plot to {out}")

        if show:
            plt.show()
        else:
            plt.close()

    print("All done!")


if __name__ == "__main__":
    tyro.cli(main)
