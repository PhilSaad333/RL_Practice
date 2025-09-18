# evals/metrics/passk.py
from typing import List, Dict
import numpy as np
from evals.metrics.tag_format import has_good_tags
from math_verify import parse, verify
from evals.records import EvalRecord

def passk_metrics(records: List[EvalRecord], k_vals=(1, 2, 4, 8)) -> List[Dict]:
    """
    For each EvalRecord compute:
      • pass_rate  = (# correct among max(k_vals)) / max(k_vals)
      • pass@k = 1 if any of first k samples correct, else 0
    """
    rows = []
    for r in records:
        gold_expr = parse(f"${r.gold}$")          # <-- reliable gold
        flags = []
        for g in r.generations:
            if not has_good_tags(g):
                flags.append(False)
                continue
            # pull predicted answer text
            pred = g.split("</answer>")[0].split("<answer>")[-1].strip()
            ok = verify(gold_expr, parse(f"${pred}$"))
            flags.append(ok)

        row = {
            "q_idx": r.q_idx,
            "pass_rate": sum(flags)/(max(k_vals))
            }
        for k in k_vals:
            row[f"pass@{k}"] = int(any(flags[:k]))
        rows.append(row)
    return rows
