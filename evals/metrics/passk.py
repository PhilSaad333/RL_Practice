# evals/metrics/passk.py
from typing import List, Dict
import numpy as np
from evals.metrics.tag_format import has_good_tags
from math_verify import parse, verify
from evals.records import EvalRecord

def passk(records: List[EvalRecord], k_vals=(1, 2, 4, 8)) -> List[Dict]:
    """
    For each EvalRecord compute:
      • pass@k  = (# correct among first k samples) / k
      • pass_any@k = 1 if any of first k samples correct, else 0
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

        row = {"q_idx": r.q_idx}
        for k in k_vals:
            row[f"pass@{k}"]     = sum(flags[:k]) / k        # fraction correct
            row[f"pass_any@{k}"] = int(any(flags[:k]))       # Codex metric
        rows.append(row)
    return rows
