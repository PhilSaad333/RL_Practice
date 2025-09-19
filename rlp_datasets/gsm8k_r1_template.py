# rlp_datasets/gsm8k_r1_template.py
import re, json, os
from rlp_datasets.registry import DATASET_REGISTRY, Example
from transformers import AutoTokenizer


#BASE = '/content/drive/MyDrive/RL_Practice_Files/datasets'
#BASE =  '/home/ubuntu/dataset'  # old path
import pathlib
BASE = str(pathlib.Path(__file__).parent / "processed")

# Explicitly using qwen2 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-1.5B",
    trust_remote_code=True,
    add_prefix_space=False
)

GLOBAL_ID_OFFSETS = {
    "train": 0,
    "test": 1_000_000,
    "validation": 2_000_000,
    "val": 2_000_000,
}




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

def _parse_one(rec: dict, split: str, global_prompt_id: int) -> Example:
    text = rec['text']
    q = text.split('<think>')[0].strip()
    sol = text.split('<think>')[-1].split('</think>')[0].strip()
    ans = text.split('<answer>')[-1].split('</answer>')[0].strip()

    meta = dict(dataset="gsm8k_latex", split=split, global_prompt_id=int(global_prompt_id))
    return Example(text=text, question=prompt_template(q), answer=ans, meta=meta)

def build_gsm8k(split: str = "train", root: str = None) -> list[Example]:
    ds: list[dict] = []
    dataset_path = os.path.join(BASE, f"gsm8k_latex_{split}.jsonl")
    with open(dataset_path, "r") as f:
        for line in f:
            ds.append(json.loads(line))

    offset = GLOBAL_ID_OFFSETS.get(split, GLOBAL_ID_OFFSETS.get(split.lower(), 3000000))
    examples = [_parse_one(rec, split, offset + idx) for idx, rec in enumerate(ds)]
    return filter_lens(examples)

DATASET_REGISTRY["gsm8k_r1_template"] = build_gsm8k
