import re
from pathlib import Path
from datasets import load_dataset
from . import BaseDataset, Example, register

RAW_DIR  = Path("datasets/raw")
PROC_DIR = Path("datasets/processed/mathqa_tagged")

FINAL_RE = re.compile(r"(-?\d+(?:\.\d+)?)$")  # grab last number

@register("mathqa")
class MathQA(BaseDataset):
    """
    MathQA has 'question' and 'answer' (often multi-sentence explanation).
    We use the whole answer as <think> and extract the trailing number
    as the final <answer>.  If extraction fails, we keep full text.
    """
    def __iter__(self):
        ds = load_dataset("miike-ai/mathqa", split=self.split,
                          cache_dir=RAW_DIR)           # :contentReference[oaicite:10]{index=10}
        for row in ds:
            q = row["question"].strip()
            ans_full = row["answer"].strip()
            m = FINAL_RE.search(ans_full)
            final = m.group(1) if m else ans_full
            yield Example(
                text=f"{q}\n<think>\n{ans_full}\n</think>\n"
                     f"<answer>\n{final}\n</answer>",
                meta={"dataset": "mathqa"}
            )

if __name__ == "__main__":
    MathQA("train").dump(PROC_DIR)
