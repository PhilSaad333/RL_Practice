# rlp_datasets/gsm8k_latex.py
import re, json, os
from rlp_datasets.registry import DATASET_REGISTRY, Example

BASE = '/content/drive/MyDrive/RL_Practice_Files/datasets'


def _parse_one(rec: dict, split: str) -> Example:
    text = rec['text']
    q = text.split('<think>')[0].strip()
    sol = text.split('<think>')[-1].split('</think>')[0].strip()
    ans = text.split('<answer>')[-1].split('</answer>')[0].strip()

    meta = dict(dataset="gsm8k_latex", split=split)
    return Example(text=text, question=q, answer=ans, meta=meta)

def build_gsm8k(split: str = "train") -> list[Example]:
    ds = []
    with open(os.path.join(BASE, f"gsm8k_latex_{split}.jsonl"), 'r') as f:
        for line in f:
            ds.append(json.loads(line))
    return [_parse_one(rec, split) for rec in ds]

DATASET_REGISTRY["gsm8k_latex"] = build_gsm8k
