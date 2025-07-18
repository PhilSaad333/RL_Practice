# rlp_datasets/math_mix.py
from datasets import interleave_datasets
from rlp_datasets.registry import DATASET_REGISTRY

def build_mix(split="train"):
    gsm  = DATASET_REGISTRY["gsm8k"](split)
    math = DATASET_REGISTRY["hendrycks_math"](split)
    return interleave_datasets([gsm, math], probabilities=[0.5, 0.5], seed=42)

DATASET_REGISTRY["math_mix"] = build_mix
