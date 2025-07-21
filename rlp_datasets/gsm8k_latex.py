# rlp_datasets/gsm8k_latex.py
import re, json, os
from datasets import load_from_disk
from rlp_datasets.registry import DATASET_REGISTRY, Example

DIR = '/content/drive/MyDrive/RL_Practice_Files/datasets/gsm8k_latex


def _parse_one(rec: dict, split: str) -> Example:
    text = rec['text']
    q = text.split('<think>')[0].strip()
    sol = text.split('<think>')[-1].split('</think>')[0].strip()
    ans = text.split('<answer>')[-1].split('</answer>')[0].strip()

    meta = dict(dataset="gsm8k_latex", split=split)
    return Example(text=text, question=q, answer=ans, meta=meta)

def build_gsm8k(split: str = "train") -> list[Example]:
    ds = load_from_disk(os.path.join(BASE, f"_{split}"))
    return [_parse_one(rec, split) for rec in ds]

DATASET_REGISTRY["gsm8k_latex"] = build_gsm8k
