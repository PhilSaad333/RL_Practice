# evals/metrics/entropy.py
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
from typing import List, Dict
from evals.records import EvalRecord


def _stats(arr: np.ndarray, prefix: str) -> Dict[str, float]:
    """
    Return {prefix_mean, prefix_std, prefix_max, prefix_p95}.
    Any non-finite entries are dropped before computing stats; if nothing
    finite is left, all four stats are set to NaN.
    """
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std":  np.nan,
            f"{prefix}_max":  np.nan,
            f"{prefix}_p95":  np.nan,
        }

    return {
        f"{prefix}_mean": float(finite.mean()),
        f"{prefix}_std":  float(finite.std(ddof=0)),
        f"{prefix}_max":  float(finite.max()),
        f"{prefix}_p95":  float(np.percentile(finite, 95)),
    }


def entropy_metrics(records: List[EvalRecord]) -> List[Dict]:
    """
    For each prompt (q_idx) emit statistics over all completions' tokens:
        • entropy = −log p(chosen) = surprisal (correct entropy estimator)
        
    NOTE: Previously computed expensive Shannon entropy −∑ₖ pₖ log pₖ which was
    incorrect as an entropy estimator. The surprisal is the proper estimator.
    """
    rows: List[Dict] = []

    for r in records:
        # r.logprobs and r.entropies are List[N] of np.ndarray[T]
        if not r.logprobs:                  # empty list → emit all-NaN row
            row = {"q_idx": r.q_idx}
            row.update(
                {f"entropy_{s}": np.nan for s in ("mean", "std", "max", "p95")}
            )
            rows.append(row)
            continue

        # The entropies are now correctly computed as surprisal = -log p(chosen)
        entropy = np.concatenate(r.entropies)   # (tokens, ) - this is now surprisal

        row = {"q_idx": r.q_idx}
        row.update(_stats(entropy, "entropy"))
        rows.append(row)

    return rows
