# rlp_datasets/gsm8k_r1_template.py
import re, json, os
from rlp_datasets.registry import DATASET_REGISTRY, Example
from transformers import AutoTokenizer


#BASE = '/content/drive/MyDrive/RL_Practice_Files/datasets'
BASE =  '/home/ubuntu/dataset'

# Explicitly using qwen2 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2-0.5B",
    trust_remote_code=True,
    add_prefix_space=False
)


def prompt_template(question: str) -> str:
    template = (
        "You are solving math problems. Respond by reasoning through the problem "
        "then providing a final answer. Enclose the reasoning process within <think> </think> "
        "and the answer within <answer> </answer> tags, i.e., <think> reasoning process here </think> "
        "<answer> answer here </answer>.\n"
        f"Question: {question}\nResponse: <think>"
    )
    return template


def text_ntokens(text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])

def filter_lens(unfiltered, tolerance: int = 200):
    filtered = []
    for ex in unfiltered:
        if text_ntokens(ex.text) <= tolerance:
            filtered.append(ex)
    return filtered

def _parse_one(rec: dict, split: str) -> Example:
    text = rec['text']
    q = text.split('<think>')[0].strip()
    sol = text.split('<think>')[-1].split('</think>')[0].strip()
    ans = text.split('<answer>')[-1].split('</answer>')[0].strip()

    meta = dict(dataset="gsm8k_latex", split=split)
    return Example(text=text, question=prompt_template(q), answer=ans, meta=meta)

def build_gsm8k(split: str = "train") -> list[Example]:
    ds = []
    with open(os.path.join(BASE, f"gsm8k_latex_{split}.jsonl"), 'r') as f:
        for line in f:
            ds.append(json.loads(line))
    return filter_lens([_parse_one(rec, split) for rec in ds])

DATASET_REGISTRY["gsm8k_r1_template"] = build_gsm8k







    

