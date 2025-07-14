"""
Composite loader for the Hendrycks MATH dataset.
Loads *all* seven subject configs, keeps their step-by-step LaTeX rationales,
and wraps them in <think>/<answer> tags.
"""

import re
from pathlib import Path
from datasets import load_dataset, concatenate_datasets
from . import BaseDataset, Example, register

RAW_DIR  = Path("datasets/raw")
PROC_DIR = Path("datasets/processed/math_tagged")
SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]  # cf. dataset card  :contentReference[oaicite:2]{index=2}

BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")

@register("math")
class HendrycksMath(BaseDataset):
    def __iter__(self):
        # 1--load every subject split and concat
        splits = []
        for subj in SUBJECTS:
            ds = load_dataset(
                "EleutherAI/hendrycks_math",    # public mirror, no token needed :contentReference[oaicite:3]{index=3}
                name=subj,
                split=self.split,
                cache_dir=RAW_DIR,
            )
            ds = ds.add_column("subject", [subj] * len(ds))
            splits.append(ds)
        full = concatenate_datasets(splits)     # single unified dataset

        # 2--yield in the tag format your pipeline expects
        for row in full:
            prob = row["problem"].strip()
            sol  = row["solution"].strip()
            final = BOXED_RE.search(sol).group(1).strip() if BOXED_RE.search(sol) else ""
            yield Example(
                text=f"{prob}\n<think>\n{sol}\n</think>\n"
                     f"<answer>\n{final}\n</answer>",
                question=question,
                answer=final
                meta={"dataset": "math", "subject": row["subject"]},
            )

if __name__ == "__main__":
    HendrycksMath("train").dump(PROC_DIR)
