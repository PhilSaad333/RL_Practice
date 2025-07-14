from pathlib import Path, PurePath
import json
from . import BaseDataset, Example, register

RAW_DIR  = Path("datasets/raw/svamp")          # manual download once
PROC_DIR = Path("datasets/processed/svamp_tagged")
SVAMP_JSON = RAW_DIR / "SVAMP.json"            # file from repo :contentReference[oaicite:11]{index=11}

@register("svamp")
class SVAMP(BaseDataset):
    SPLIT_MAP = {"train": "train", "test": "test", "dev": "dev"}  # not used

    def __iter__(self):
        with SVAMP_JSON.open() as f:
            data = json.load(f)
        for ex in data:                     # each ex has 'Body' + 'Answer'
            q = ex["Body"].strip()
            ans = str(ex["Answer"]).strip()
            yield Example(
                text=f"{q}\n<think>\n\n</think>\n<answer>\n{ans}\n</answer>",
                meta={"dataset": "svamp"}
            )

if __name__ == "__main__":
    SVAMP("train").dump(PROC_DIR)
