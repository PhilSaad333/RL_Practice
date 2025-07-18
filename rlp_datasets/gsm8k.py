# rlp_datasets/gsm8k.py
from datasets import load_from_disk
from rlp_datasets.registry import DATASET_REGISTRY
from rlp_datasets.local_paths import BASE



def build_gsm8k(split: str = "train"):
    return load_from_disk(f"{BASE}/gsm8k_{split}")

DATASET_REGISTRY["gsm8k"] = build_gsm8k
