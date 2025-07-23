# evals/analyses/high_entropy_tokens.py
from __future__ import annotations
from pathlib import Path
import gzip, json, heapq, re
from typing import List, Tuple, Optional

import torch
from transformers import AutoTokenizer
import tyro

# ------------------------------------------------------------------ utils
def _load_records(records_path: Path) -> List[dict]:
    out = []
    with gzip.open(records_path, "rt", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out

def _model_from_base(base_root: Path) -> str:
    # e.g. ".../eval_runs/phi2_math_finetuned"  ->  "phi2"
    m = re.search(r"/([^/_]+)(?:_[^/]+)?_finetuned$", str(base_root))
    if not m:
        raise ValueError(f"Cannot infer model name from {base_root}")
    return m.group(1)

def _discover_one_run(base_root: Path) -> Path:
    """Return path to exactly ONE records.jsonl.gz or raise."""
    rec_paths = list(base_root.glob("step_*/*/records.jsonl.gz"))
    if not rec_paths:
        raise FileNotFoundError(f"No records.jsonl.gz under {base_root}")
    if len(rec_paths) > 1:
        raise RuntimeError(
            "Multiple runs found; pass --records_path to disambiguate:\n" +
            "\n".join(str(p) for p in rec_paths)
        )
    return rec_paths[0]

def _top_ent(records: List[dict], k: int) -> List[Tuple[float,int,int,int]]:
    heap: List[Tuple[float,int,int,int]] = []
    for q, rec in enumerate(records):
        for g, ent_arr in enumerate(rec["entropies"]):
            vals, idxs = torch.tensor(ent_arr).topk(min(k, len(ent_arr)))
            for v, s in zip(vals.tolist(), idxs.tolist()):
                if len(heap) < k:
                    heapq.heappush(heap, (v, q, g, s))
                elif v > heap[0][0]:
                    heapq.heapreplace(heap, (v, q, g, s))
                else:
                    break
    return sorted(heap, reverse=True)

def _highlight(gen: str, pos: int, tok, window=5) -> str:
    ids   = tok(gen, add_special_tokens=False)["input_ids"]
    toks  = tok.convert_ids_to_tokens(ids)
    toks[pos] = f"«{toks[pos]}»"
    s = max(0, pos-window); e = min(len(toks), pos+window+1)
    return tok.convert_tokens_to_string(toks[s:e])

# ------------------------------------------------------------------ main
def main(
    base_root: Path,
    records_path: Optional[Path] = None,
    k: int = 15,
    window: int = 5,
):
    """
    Print the top-k highest-entropy tokens for one evaluation run.

    If --records_path is omitted we search for a single records.jsonl.gz
    under `base_root` (raises if more than one).
    """
    if records_path is None:
        records_path = _discover_one_run(base_root)

    records = _load_records(records_path)

    model_name = _model_from_base(base_root)
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    top = _top_ent(records, k)
    if not top:
        print("[INFO] No entropy values found – is this the right records file?")
        return

    for H, q, g, s in top:
        rec  = records[q]
        gen  = rec["generations"][g]
        # entropy arrays already trimmed: index aligns 1-to-1
        snippet = _highlight(gen, s, tok, window)
        print("="*80)
        print(f"Prompt  : {rec['prompt']}")
        print(f"H = {H:.3f} at token #{s}  →  {snippet}")
        print(f"Full gen:\n{gen}\n")

if __name__ == "__main__":
    tyro.cli(main)
