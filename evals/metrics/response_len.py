# evals/metrics/response_len.py
import numpy as np
from typing import List, Dict
from evals.records import EvalRecord


from transformers import AutoTokenizer

# just use qwen tokenizer hard coded for now
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B", trust_remote_code=True)
pad_id = tok.pad_token_id

def _count_think_tokens(text: str) -> int:
    if "</think>" not in text:
        return 0
    inner = text.split("</think>", 1)[0].strip()
    return len(tok(inner, add_special_tokens=False).input_ids)

def response_len_metrics(records: List[EvalRecord]) -> List[Dict]:
    rows: List[Dict] = []
    for r in records:
        lens = []
        for g in r.generations:
            cot = g.split("</think>", 1)[0].strip() if "</think>" in g else g # fallback to whole gen
            lens.append(len(tok(cot, add_special_tokens=False).input_ids))
        arr = np.asarray(lens)
        rows.append(
            dict(
                q_idx=r.q_idx,
                len_min=int(arr.min()),
                len_max=int(arr.max()),
                len_mean=float(arr.mean()),
                len_std=float(arr.std(ddof=0)),
            )
        )
    return rows
