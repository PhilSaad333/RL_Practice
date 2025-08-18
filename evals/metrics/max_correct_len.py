# evals/metrics/max_correct_len.py
from typing import List, Dict
import numpy as np
from evals.metrics.tag_format import has_good_tags
from math_verify import parse, verify
from evals.records import EvalRecord

from transformers import AutoTokenizer

# just use qwen tokenizer hard coded for now
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)
pad_id = tok.pad_token_id

def max_correct_len_metric(records: List[EvalRecord]) -> List[Dict]:
    out = []
    for r in records:
        gold_expr = parse(f"${r.gold}$")
        max_l_cot = 0
        for g in r.generations:
            if not has_good_tags(g):
                continue
            pred = g.split("</answer>")[0].split("<answer>")[-1].strip()
            correct = verify(gold_expr, parse(f"${pred}$"))
            if correct:
                # compute len
                # first extract c.o.t.
                cot = g.split('</think>', 1)[0].strip()
                max_l_cot = max(max_l_cot, len(tok(cot, add_special_tokens=False).input_ids))
        out.append({'max_l_cot': max_l_cot})
    return out


