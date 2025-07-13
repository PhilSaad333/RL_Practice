# evals/records.py
import json, gzip, pathlib
from dataclasses import asdict, dataclass
from typing import List, Dict
import numpy as np


@dataclass(frozen=True, slots=True)
class EvalRecord:
    step: int
    q_idx: int
    prompt: str
    generations: List[str]          # raw decoded strings
    gold: str
    logprobs:   List[np.ndarray]    # per-token log-probs
    entropies:  List[np.ndarray]
    cfg: Dict                        # temp, top_p, etc.

def save_records(recs: List[EvalRecord], path: str):
    """
    Write one JSONL line per record, gzipped.
    Converts numpy arrays to lists so JSON is happy.
    """
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(p, "wt", encoding="utf-8") as f:
        for r in recs:
            # asdict returns a dict; convert np.ndarray → list
            data = asdict(r)
            data["logprobs"] = [lp.tolist() for lp in data["logprobs"]]
            data["entropies"] = [lp.tolist() for lp in data["entropies"]]
            f.write(json.dumps(data) + "\n")

def load_records(path: str):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            data["logprobs"] = [np.array(lp) for lp in data["logprobs"]]
            data["entropies"] = [np.array(lp) for lp in data["entropies"]]
            yield EvalRecord(**data)
