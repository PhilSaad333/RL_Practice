# evals/records.py
from dataclasses import dataclass
from typing import List, Dict
import numpy as np
import json, gzip, pathlib

@dataclass(frozen=True, slots=True)
class EvalRecord:
    step: int
    q_idx: int
    prompt: str
    generations: List[str]          # raw decoded strings
    logprobs:   List[np.ndarray]    # per-token log-probs
    cfg: Dict                        # temp, top_p, etc.

def save_records(recs: List['EvalRecord'], path: str):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(p, "wt", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r.__dict__, default=lambda x: x.tolist()) + "\n")

def load_records(path: str):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            data["logprobs"] = [np.array(lp) for lp in data["logprobs"]]
            yield EvalRecord(**data)
