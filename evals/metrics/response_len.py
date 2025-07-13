# evals/metrics/response_len.py
import re, numpy as np
from typing import List, Dict
from evals.records import EvalRecord

_THINK_RGX = re.compile(r"<think>(.*?)</think>", re.S)

def response_len_metrics(records: List[EvalRecord]) -> List[Dict]:
    rows: List[Dict] = []
    for r in records:
        lens = []
        for g in r.generations:
            m = _THINK_RGX.search(g)
            span = m.group(1) if m else g            # fallback = whole gen
            lens.append(len(span.strip().split()))
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
