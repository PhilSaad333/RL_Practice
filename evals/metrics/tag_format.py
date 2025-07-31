# evals/metrics/tag_format.py
import re
from typing import List, Dict
from evals.records import EvalRecord

TAG_RGX = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.S)

def has_good_tags(txt: str) -> bool:
    """Return True if a single generated string `txt` has the <think>…</think><answer>…</answer> pattern."""
    txt = '<think>\n' + txt.split('<think>')[-1].strip()
    return bool(TAG_RGX.search(txt))

def tag_format_metrics(records: List[EvalRecord]) -> List[Dict]:
    """
    Top-level metric: for each record, count whether
    1) the first sample is well-formatted
    2) any of the N samples is well-formatted
    and emit a dict with q_idx and those flags.
    """
    out = []
    for r in records:
        flags = [has_good_tags(g) for g in r.generations]
        N = len(flags)
        out.append({
            "q_idx":            r.q_idx,
            "tag_ok_ave":     sum(flags)/N,
            "tag_ok_any":       int(any(flags)),
        })
    return out
