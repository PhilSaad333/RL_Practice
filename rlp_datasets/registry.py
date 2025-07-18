# rlp_datasets/registry.py
from dataclasses import dataclass, asdict
from typing import Dict, Any, List

# ↓ simple global used everywhere else
DATASET_REGISTRY: Dict[str, callable] = {}

@dataclass
class Example:
    text: str        # full prompt incl. tags
    question: str    # natural-language problem statement
    answer: str      # final numeric / symbolic answer
    meta: Dict[str, Any]

    # helpful for quick → dict conversions in SFT trainer
    def asdict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["text"] = self.text
        out.update(self.meta)
        return out
