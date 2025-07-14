from pathlib import Path
from datasets import load_dataset
from . import BaseDataset, Example, register

RAW_DIR  = Path("datasets/raw")
PROC_DIR = Path("datasets/processed/openwebmath_tagged")

@register("openwebmath")
class OpenWebMath(BaseDataset):
    """
    OpenWebMath is raw LaTeX/HTML math text, not Q-A pairs.
    We treat each document as a 'problem' and leave think blank.
    """
    def __iter__(self):
        ds = load_dataset("open-web-math/open-web-math",
                          split=self.split, cache_dir=RAW_DIR)  # :contentReference[oaicite:8]{index=8}
        for row in ds:
            txt = row["text"].strip()
            yield Example(
                text=f"{txt}\n<think>\n\n</think>\n<answer>\n\n</answer>",
                meta={"dataset": "openwebmath"}
            )

if __name__ == "__main__":
    OpenWebMath("train").dump(PROC_DIR)
