from pathlib import Path
from datasets import load_dataset
from . import BaseDataset, Example, register

RAW_DIR  = Path("datasets/raw")
PROC_DIR = Path("datasets/processed/deepmind_math_tagged")

@register("deepmind_math")
class DeepMindMath(BaseDataset):
    """
    Each example has 'question' and 'answer' only.
    We leave <think> empty.
    """
    def __iter__(self):
        ds = load_dataset("deepmind/math_dataset", name="algebra",
                          split=self.split, cache_dir=RAW_DIR)    # :contentReference[oaicite:9]{index=9}
        for row in ds:
            q = row["question"].strip()
            ans = row["answer"].strip()
            yield Example(
                text=f"{q}\n<think>\n\n</think>\n<answer>\n{ans}\n</answer>",
                meta={"dataset": "deepmind_math"}
            )

if __name__ == "__main__":
    DeepMindMath("train").dump(PROC_DIR)
