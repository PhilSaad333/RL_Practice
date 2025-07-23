# rl_training/rewards/tag_pref.py

from __future__ import annotations
import re
from typing import Dict, List

import torch
from math_verify import parse, verify

# ── Hyperparameters ────────────────────────────────────────────────────────────
#hardcoded here rather than in config for now
W_TAG: float = 0.5   # weight for formatting adherence
W_ANS: float = 0.5   # weight for correctness when formatted properly

# ── Global prompt→gold map (populated once by your scheduler) ───────────────────
PROMPT2GOLD: Dict[int, str] = {}

# ── Regex to extract exactly one <answer>…</answer> block ────────────────────────
_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def set_prompt2gold(mapping: Dict[int, str]) -> None:
    """Initialize the PROMPT2GOLD dictionary once."""
    PROMPT2GOLD.update(mapping)


def _extract_answer_block(text: str) -> str | None:
    """
    Return the inner text of a single <answer>…</answer> block,
    or None if there are zero or multiple hits.
    """
    hits = _TAG_RE.findall(text)
    return hits[0].strip() if len(hits) == 1 else None


def reward_fn(prompt_id: int, answers: List[str]) -> torch.FloatTensor:
    """
    For each generated answer:
      • tag_score = 1.0 if exactly one <answer>…</answer> block, else 0.0
      • ans_score = 1.0 if verify(gold, inner) succeeds (only when tag_score == 1.0), else 0.0
      • reward = W_TAG * tag_score + W_ANS * ans_score
    """
    gold = PROMPT2GOLD.get(prompt_id)
    print(f'gold = {gold}')
    if gold is None:
        return torch.zeros(len(answers), dtype=torch.float32)

    try:
        gold_parsed = parse(gold)
    except Exception:
        # If the gold itself fails to parse, give zero reward
        return torch.zeros(len(answers), dtype=torch.float32)

    rewards: List[float] = []
    for ans in answers:
        inner = _extract_answer_block(ans)
        tag_score = 1.0 if inner is not None else 0.0

        if tag_score == 1.0:
            # Only check correctness when the tag is correct
            try:
                ok = verify(gold_parsed, parse(inner))  # ans_score ∈ {True,False}
            except Exception:
                ok = False
            ans_score = 1.0 if ok else 0.0
        else:
            ans_score = 0.0

        # Weighted sum of formatting and correctness
        rewards.append(W_TAG * tag_score + W_ANS * ans_score)

    return torch.tensor(rewards, dtype=torch.float32)
