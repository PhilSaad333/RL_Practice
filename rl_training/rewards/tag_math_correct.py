# rl_training/rewards/tag_math_correct.py
"""
reward(prompt_id, answers:list[str])  →  torch.FloatTensor[G]

* tag_score : 1 if exactly one <answer>…</answer> block, else 0
* math_score: 1 if math_verify says model answer ≡ gold answer, else 0
* reward    : tag_score × math_score   (∈ {0,1})

The module expects a global dict PROMPT2GOLD that maps *prompt_id* → gold_latex.
Prompt IDs are supplied by the scheduler.
"""
from __future__ import annotations
import re
from typing import Dict, List

import torch
from math_verify import parse, verify


PROMPT2GOLD: Dict[int, str] = {}    # filled once by the scheduler

_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def set_prompt2gold(mapping: Dict[int, str]) -> None:
    """Call exactly once at start-up."""
    PROMPT2GOLD.update(mapping)


def _extract_answer_block(text: str) -> str | None:
    hits = _TAG_RE.findall(text)
    return hits[0].strip() if len(hits) == 1 else None


def reward_fn(prompt_id: int, answers: List[str]) -> torch.FloatTensor:
    gold = PROMPT2GOLD.get(prompt_id)
    print(f'gold = {gold}')
    if gold is None:
        return torch.zeros(len(answers))

    try:
        gold_parsed = parse(gold)
    except Exception:
        return torch.zeros(len(answers))

    out = []
    for ans in answers:
        inner = _extract_answer_block(ans)
        if inner is None:
            out.append(0.0)
            continue
        try:
            ok = verify(gold_parsed, parse(inner))
        except Exception:
            ok = False
        out.append(1.0 if ok else 0.0)

    return torch.tensor(out, dtype=torch.float32)
