# rlp_datasets/math_mix.py
from datasets import interleave_datasets
from rlp_datasets.registry import DATASET_REGISTRY

def build_mix(split="train"):
    gsm  = DATASET_REGISTRY["gsm8k"](split)
    math = DATASET_REGISTRY["math"](split)
    return interleave_datasets([gsm, math], probabilities=[0.5, 0.5], seed=42)

DATASET_REGISTRY["mathmix"] = build_mix
