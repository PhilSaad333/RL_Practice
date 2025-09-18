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
    m = re.search(r"/([^/_]+)(?:_[^/]+)?_finetuned$", str(base_root))
    if not m:
        raise ValueError(f"Cannot infer model name from {base_root}")
    name = m.group(1).lower()

    # --- special-case short aliases -----------------------------------
    alias = {"phi2": "microsoft/phi-2", "phi-2": "microsoft/phi-2"}
    return alias.get(name, name)


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

def _top_ent(records: list[dict], k: int) -> list[tuple[float,int,int,int]]:
    heap: list[tuple[float,int,int,int]] = []
    seen = 0                                                # NEW ✔
    for q_idx, rec in enumerate(records):
        for g_idx, ent_arr in enumerate(rec["entropies"]):
            tens = torch.tensor(ent_arr, dtype=torch.float32)
            seen += tens.numel()                            # count total scalars
            if tens.numel() == 0 or torch.isnan(tens).all():
                continue                                    # skip empty / all-nan

            vals, idxs = tens.topk(min(k, tens.numel()))    # local top-k
            for v, s in zip(vals.tolist(), idxs.tolist()):
                if len(heap) < k:
                    heapq.heappush(heap, (v, q_idx, g_idx, s))
                elif v > heap[0][0]:
                    heapq.heapreplace(heap, (v, q_idx, g_idx, s))
                else:                                       # vals are descending
                    break

    # always print a short summary so you know what happened
    print(f"[DEBUG] scalar entropies seen: {seen}")
    print(f"[DEBUG] global-top heap size : {len(heap)}")
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
