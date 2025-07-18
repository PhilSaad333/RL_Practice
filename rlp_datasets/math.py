# rlp_datasets/hendrycks_math.py
from datasets import load_from_disk, concatenate_datasets, Dataset
from rlp_datasets.registry import DATASET_REGISTRY
from rlp_datasets.local_paths import BASE, BAD_WORDS

SUBJECTS = {
    "algebra":           "alg_math",
    "prealgebra":        "prealg_math",
    "counting_prob":     "counting_prob_math",
}

def _load_subject(sub, split):
    return load_from_disk(f"{BASE}/{SUBJECTS[sub]}_{split}")

def _filter_bad_words(example):
    text = (example.get("problem") or "") + " " + (example.get("question") or "")
    return not any(bw in text.lower() for bw in BAD_WORDS)

def build_math(split: str = "train"):
    parts = [_load_subject(s, split) for s in SUBJECTS]
    ds = concatenate_datasets(parts)
    return ds.filter(_filter_bad_words, batched=False)

DATASET_REGISTRY["math"] = build_math
