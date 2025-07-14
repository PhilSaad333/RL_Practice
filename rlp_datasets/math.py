import re
from pathlib import Path
from datasets import load_dataset
from . import BaseDataset, Example, register

RAW_DIR  = Path("datasets/raw")
PROC_DIR = Path("datasets/processed/math_tagged")

BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")

@register("math")
class HendrycksMath(BaseDataset):
    def __iter__(self):
        # subject splits are already shuffled; we keep as-is
        ds = load_dataset("EleutherAI/hendrycks_math",        # public mirror
                        split=self.split, cache_dir=RAW_DIR)        # :contentReference[oaicite:1]{index=1}
        for row in ds:
            prob = row["problem"].strip()
            sol  = row["solution"].strip()
            m = BOXED_RE.search(sol)
            final = m.group(1).strip() if m else ""
            yield Example(
                text=f"{prob}\n<think>\n{sol}\n</think>\n"
                     f"<answer>\n{final}\n</answer>",
                meta={"dataset": "math", "subject": row.get("level", "")},
            )

if __name__ == "__main__":
    HendrycksMath("train").dump(PROC_DIR)
