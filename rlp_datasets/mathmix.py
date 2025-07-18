# rlp_datasets/mathmix.py
import random
from rlp_datasets.registry import DATASET_REGISTRY, Example

def build_mix(split="train", p_gsm=0.5, seed=42) -> list[Example]:
    gsm  = DATASET_REGISTRY["gsm8k"](split)
    math = DATASET_REGISTRY["math"](split)
    rng = random.Random(seed)
    mixed = []
    for g, m in zip(itertools.cycle(gsm), itertools.cycle(math)):
        if len(mixed) >= max(len(gsm), len(math))*2: break
        mixed.append(g if rng.random() < p_gsm else m)
    return mixed

DATASET_REGISTRY["mathmix"] = build_mix
