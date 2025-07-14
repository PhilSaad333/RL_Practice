# datasets/__init__.py
from typing import Dict, Iterator
from pathlib import Path
from dataclasses import dataclass

import datasets as hf_ds



DATASET_REGISTRY: Dict[str, "BaseDataset"] = {}

def register(name: str):
    def wrap(cls):
        DATASET_REGISTRY[name] = cls
        return cls
    return wrap

@dataclass
class Example:
    text: str
    question: str
    answer: str
    meta: dict


class BaseDataset:
    """
    Sub-class this.  Only __iter__() is strictly required.

    Each yielded Example must have the *final* string ready for tokenisation:
    Question\n<think>\nrationale\n</think>\n<answer>\nfinal\n</answer>
    """

    SPLIT_MAP = {"train": "train", "test": "test"}

    def __init__(self, split: str = "train"):
        self.split = self.SPLIT_MAP.get(split, split)

    # subclasses implement
    def __iter__(self) -> Iterator[Example]: ...

    # helper – save to arrow / jsonl like your GSM8K builder
    def dump(self, root: Path):
        import json, pyarrow as pa, pyarrow.dataset as ds

        root.mkdir(parents=True, exist_ok=True)
        rows = [e.text for e in self]
        # 1) JSONL
        with (root / f"{self.split}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps({"text": r}, ensure_ascii=False) + "\n")
        # 2) Arrow
        arrow_table = pa.Table.from_pylist([{"text": r} for r in rows])
        ds.write_dataset(arrow_table, root / self.split, format="arrow")



# --- auto-import all sibling modules so their @register(...) executes ----------
from importlib import import_module
from pathlib import Path
import pkgutil

_pkg_dir = Path(__file__).resolve().parent     # <— directory, not “__init__.py”
for mod in pkgutil.iter_modules([str(_pkg_dir)]):
    if mod.name.startswith("_"):              # skip private helpers like _utils.py
        continue
    import_module(f"{__name__}.{mod.name}")

