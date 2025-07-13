# evals/metrics/entropy.py
import numpy as np
from typing import List, Dict
from evals.records import EvalRecord


def _stats(arr: np.ndarray, prefix: str) -> Dict[str, float]:
    """Helper: return {prefix_mean, prefix_std, prefix_max, prefix_p95}."""
    return {
        f"{prefix}_mean": float(arr.mean()),
        f"{prefix}_std":  float(arr.std(ddof=0)),
        f"{prefix}_max":  float(arr.max()),
        f"{prefix}_p95":  float(np.percentile(arr, 95)),
    }


def entropy_metrics(records: List[EvalRecord]) -> List[Dict]:
    rows: List[Dict] = []
    for r in records:
        # --- token-level arrays for this prompt (all completions concatenated)
        surp = -np.concatenate(r.logprobs)     # surprisal  = −log p₍chosen₎
        ent  =  np.concatenate(r.entropies)    # true entropy for each step

        row = {"q_idx": r.q_idx}
        row.update(_stats(surp, "surp"))
        row.update(_stats(ent,  "ent"))
        rows.append(row)

    return rows
