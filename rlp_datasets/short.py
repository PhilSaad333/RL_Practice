# rlp_datasets/short.py
import random
import itertools
from rlp_datasets.registry import DATASET_REGISTRY, Example
from transformers import AutoTokenizer


# Explicitly using phi2 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/phi-2",
    trust_remote_code=True,
    add_prefix_space=False
)

def text_ntokens(text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])

def filter_lens(unfiltered, tolerance: int = 200):
    filtered = []
    for ex in unfiltered:
        if text_ntokens(ex.text) <= tolerance:
            filtered.append(ex)
    return filtered

def build_mix(split="train", p_gsm=0.5, seed=42) -> list[Example]:
    gsm  = DATASET_REGISTRY["gsm8k"](split)
    math = DATASET_REGISTRY["math"](split)
    rng = random.Random(seed)
    mixed = []
    for g, m in zip(itertools.cycle(gsm), itertools.cycle(math)):
        if len(mixed) >= max(len(gsm), len(math))*2: break
        mixed.append(g if rng.random() < p_gsm else m)
    return filter_lens(mixed)



    

DATASET_REGISTRY["short"] = build_mix