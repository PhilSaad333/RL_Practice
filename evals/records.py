from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass(frozen=True)
class EvalRecord:
    step: int
    q_idx: int
    prompt: str
    generations: List[str]               # raw text
    logprobs:   List[np.ndarray]         # (T,) per gen
    think_tokens: List[int]              # pre-computed
    cfg: dict                            # temperature, top_p, etc.
