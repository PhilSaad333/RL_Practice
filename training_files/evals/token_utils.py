"""
Helpers for analysing the JSONL-GZ dumps produced by
`evals/analyses/token_dump.py`.
"""
from __future__ import annotations
import gzip, json, io, itertools, pathlib
from typing import Iterator, List, Callable, Dict, Any, Sequence

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase


# ---------------------------------------------------------------------------

def load_dump(path: str | pathlib.Path) -> Iterator[Dict[str, Any]]:
    """Yield dicts from a .jsonl[.gz] dump."""
    path = pathlib.Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        for line in f:
            yield json.loads(line)


# --- token ↔ text helpers --------------------------------------------------

#def seq_to_text(tok_ids: Sequence[int], tokenizer: PreTrainedTokenizerBase) -> str:
#    return tokenizer.decode(tok_ids, skip_special_tokens=True)

def highlight_token(
    text: str,
    pos: int,
    tokenizer: PreTrainedTokenizerBase,
    window: int = 5,
    marker: str = ">>{}<<",
) -> str:
    toks = tokenizer.encode(text)
    left  = tokenizer.decode(toks[max(0, pos - window): pos])
    mid   = tokenizer.decode([toks[pos]])
    right = tokenizer.decode(toks[pos + 1: pos + 1 + window])
    return f"{left}{marker.format(mid)}{right}"


# --- statistics ------------------------------------------------------------

def autocorr(series: np.ndarray, k: int) -> float:
    """
    Sample autocorrelation r_k with mean subtraction.
    """
    if k == 0:
        return 1.0
    x  = series - series.mean()
    n  = len(x) - k
    if n <= 0:
        return np.nan
    num   = np.dot(x[:-k], x[k:]) / n
    denom = np.dot(x, x) / len(x)
    return num / denom


def get_outlier_tokens(
    dump_iter: Iterator[dict],
    stat_fn: Callable[[dict], float],
    tokenizer: PreTrainedTokenizerBase,
    top_n: int = 10,
) -> List[dict]:
    """
    Generic helper: compute `stat_fn(row)` for each generation row, keep
    the top-N rows with extreme values.  Returns a list of enriched dicts.
    """
    scored = []
    for row in dump_iter:
        try:
            score = stat_fn(row)
        except Exception:
            continue
        scored.append((score, row))
    scored.sort(key=lambda x: x[0], reverse=True)
    keep = scored[: top_n]
    # attach human-readable text:
    for score, row in keep:
        row["stat"] = score
    return [row for _, row in keep]
