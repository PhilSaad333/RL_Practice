# rlp_datasets/gsm8k.py
import re, json, os
from datasets import load_from_disk
from rlp_datasets.registry import DATASET_REGISTRY, Example
from rlp_datasets.local_paths import BASE   # "/content/drive/.../arrow"

ANS_RE = re.compile(r"####\s*([-\d,\.]+)")   # GSM8K answer spec  :contentReference[oaicite:2]{index=2}

def _parse_one(rec: dict, split: str) -> Example:
    q   = rec["question"].strip()
    sol = rec["answer"].strip()
    m   = ANS_RE.search(sol)
    if m:
        # split off the final answer from the chain-of-thought
        think = sol[: m.start()].strip()            # content before "#### 42" :contentReference[oaicite:0]{index=0}
        ans   = m.group(1)
    else:
        think = sol
        ans   = sol.split()[-1]
    text = (
        f"{q}\n"
        f"<think>\n{think}\n</think>\n"
        f"<answer>\n{ans}\n</answer>"
    )
    meta = dict(dataset="gsm8k", split=split)
    return Example(text=text, question=q, answer=ans, meta=meta)

def build_gsm8k(split: str = "train") -> list[Example]:
    ds = load_from_disk(os.path.join(BASE, f"gsm8k_{split}"))
    return [_parse_one(rec, split) for rec in ds]

DATASET_REGISTRY["gsm8k"] = build_gsm8k
