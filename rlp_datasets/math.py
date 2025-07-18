# rlp_datasets/math.py
import re, os, itertools
from datasets import load_from_disk, concatenate_datasets
from rlp_datasets.registry import DATASET_REGISTRY, Example
from rlp_datasets.local_paths import BASE, BAD_WORDS

BOX_RE = re.compile(r"\\boxed\{([^}]+)\}")      # official format  :contentReference[oaicite:4]{index=4}
SUBJECTS = {
    "alg":       "alg_math",
    "prealg":    "prealg_math",
    "cnt_prob":  "counting_prob_math",
}

def _visual(rec):
    txt = (rec["problem"] or "") + " " + (rec.get("solution") or "")
    return any(bad in txt.lower() for bad in BAD_WORDS)

def _parse_one(rec: dict, split: str) -> Example:
    q   = rec["problem"].strip()
    sol = rec["solution"].strip()
    m   = BOX_RE.search(sol)
    ans = m.group(1) if m else sol.split()[-1]
    subj = rec.get("type", "unknown")
    lvl  = rec.get("level", "NA")
    text = f"{q}\n<think>\n{sol}\n</think>\n<answer>\n{ans}\n</answer>"
    meta = dict(dataset="hendrycks_math", subject=subj, level=lvl, split=split)
    return Example(text=text, question=q, answer=ans, meta=meta)

def build_math(split: str = "train") -> list[Example]:
    parts = [
        load_from_disk(os.path.join(BASE, f"{name}_{split}"))
        for name in SUBJECTS.values()
    ]
    ds = concatenate_datasets(parts).filter(lambda r: not _visual(r))
    return [_parse_one(rec, split) for rec in ds]

DATASET_REGISTRY["math"] = build_math
