from pathlib import Path
from datasets import load_dataset
from . import BaseDataset, Example, register

RAW_DIR  = Path("datasets/raw")
PROC_DIR = Path("datasets/processed/gsm8k_tagged")

@register("gsm8k")
class GSM8K(BaseDataset):
    """Grade-School Math 8K, with think/answer tags."""
    def __iter__(self):
        ds = load_dataset(
            "openai/gsm8k",
            name="main",
            split=self.split,
            cache_dir=RAW_DIR,
            )
       
        for row in ds:
            question = row["question"].strip()
            rationale, final = row["answer"].split("####")
            yield Example(
                text=f"{question}\n<think>\n{rationale.strip()}\n</think>\n"
                     f"<answer>\n{final.strip()}\n</answer>",
                question=question,
                answer=final.strip(),
                meta={"dataset": "gsm8k"}
            )

if __name__ == "__main__":
    GSM8K("train").dump(PROC_DIR)
    GSM8K("test").dump(PROC_DIR)
