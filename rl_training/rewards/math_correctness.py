# rl_training/rewards/math_correctness.py

from __future__ import annotations
import re
from typing import Dict, List

import torch
from math_verify import parse, verify  # pip install math-verify

# ──────────────────────────────────────────────────────────────────────────────
# Dataset registry (prompt → gold LaTeX string)
# ──────────────────────────────────────────────────────────────────────────────
_DATASET: Dict[str, str] = {}


def register_dataset(mapping: Dict[str, str]) -> None:
    """
    Add or overwrite reference answers.

    Call this once at start-up, e.g. the scheduler can do:
        from rl_training.rewards.math_correctness import register_dataset
        register_dataset({p["prompt"]: p["gold"] for p in dataset})
    """
    _DATASET.update(mapping)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def _extract_answer_block(text: str) -> str | None:
    """Return inner string if exactly one <answer>...</answer>; else None."""
    m = _TAG_RE.findall(text)
    if len(m) != 1:
        return None
    return m[0].strip()


# ──────────────────────────────────────────────────────────────────────────────
# Public reward function
# ──────────────────────────────────────────────────────────────────────────────
def reward_fn(prompt: str, answers: List[str]) -> torch.FloatTensor:
    """
    Args
    ----
    prompt   : the prompt string (must exist in the registered dataset)
    answers  : list of model outputs (length == G)

    Returns
    -------
    torch.FloatTensor shape (G,)   values in {0., 1.}
    """
    gold_latex = _DATASET.get(prompt)
    if gold_latex is None:
        # Unknown prompt → zero reward
        return torch.zeros(len(answers))

    try:
        gold_parsed = parse(gold_latex)
    except Exception:
        # Parsing gold failed → safest to give zero reward to all
        return torch.zeros(len(answers))

    out = []
    for ans in answers:
        # 1) Tag adherence
        inner = _extract_answer_block(ans)
        if inner is None:
            out.append(0.0)
            continue

        # 2) Correctness
        try:
            ans_parsed = parse(inner)
            ok = verify(gold_parsed, ans_parsed)
        except Exception:
            ok = False

        out.append(1.0 if ok else 0.0)

    return torch.tensor(out, dtype=torch.float32)
