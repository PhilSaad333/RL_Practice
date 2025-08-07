# evals/analyses/token_dump.py
"""
Dump per-token statistics (sampled log-prob, top-k / top-p mass) for a model
responding to GSM8K-style prompts.

Usage (Colab example)
---------------------
!python -m evals.analyses.token_dump \
        --model_name_or_path qwen/Qwen2.5-1.5B \
        --ckpt_path        /content/drive/.../checkpoint-32 \
        --eval_dataset     openai/gsm8k \
        --split            test \
        --batch_size       8 \
        --top_k            50 \
        --out_dir          /content/token_dumps/ckpt32_top50
"""
from __future__ import annotations

import gzip
import json
import math
import pathlib
import random
import os
from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch
import tyro
from datasets import load_dataset
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
from peft import PeftModel
from rlp_datasets import DATASET_REGISTRY


# ---------------------------------------------------------------------------

@dataclass
class Args:
    # --- model / data -------------------------------------------------------
    model_name_or_path: str
    ckpt_path: str = ""
    eval_dataset: str = "openai/gsm8k"
    split: str = "test"
    data_root = os.environ.get("DATA_ROOT", "./datasets/processed")
    subset_frac: float = 1.0
    max_prompts: Optional[int] = None
    # --- generation ---------------------------------------------------------
    batch_size: int = 8
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_k: int = 50                              # mutually exclusive with top_p
    top_p: float = 0.0
    # --- io -----------------------------------------------------------------
    out_dir: pathlib.Path = pathlib.Path("./token_dump")
    compresslevel: int = 6
    # misc
    seed: int = 11
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        if self.top_p and self.top_k:
            raise ValueError("Choose *either* top_k or top_p, not both.")
        if not self.top_p and not self.top_k:
            self.top_k = 50  # sensible default


# ---------------------------------------------------------------------------


def load_model_and_tok(args: Args):
    """Load HF model & tokenizer; merge LoRA if `ckpt_path` is given."""
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tok.padding_side = "left"
    tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if args.ckpt_path:
        model = PeftModel.from_pretrained(model, args.ckpt_path)
        model = model.merge_and_unload()  # fully-materialise weights

    model.eval()
    return model, tok


def prepare_prompts(args: Args) -> Sequence[str]:
    """Load prompts and optionally subsample."""

    ds = DATASET_REGISTRY[args.eval_dataset](split=args.split, root=DATA_ROOT)

    total = len(ds)
    ids   = list(range(total))
    random.Random(args.seed).shuffle(ids)

    if args.subset_frac < 1.0:
        ids = ids[: math.ceil(args.subset_frac * total)]
    if args.max_prompts is not None:
        ids = ids[: args.max_prompts]

    prompts = [ds[int(i)].question for i in ids]
    return prompts


def write_jsonl_gz(path: pathlib.Path, rows, compresslevel: int = 6):
    """Append iterable of JSON-serialisable rows to a .jsonl.gz file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "ab" if path.exists() else "wb"
    with gzip.open(path, mode, compresslevel=compresslevel) as f:  # :contentReference[oaicite:3]{index=3}
        for r in rows:
            f.write(json.dumps(r).encode("utf-8") + b"\n")


def topk_or_topp(log_probs: torch.Tensor, *, k: int, p: float):
    """
    Return indices & log-probs of either the top-k or the minimal set whose mass ≥ p.
    Shape in  _in_:  (..., V).  Shapes out: (..., K), (..., K)
    """
    if p > 0.0:                         # nucleus  dynamic K
        sorted_lp, sorted_idx = log_probs.sort(dim=-1, descending=True)
        cdf = sorted_lp.exp().cumsum(dim=-1)
        mask = cdf < p
        # -- always include the first token over the cutoff
        mask[..., 0] = True
        K = mask.sum(dim=-1).max().item()
        k_slice = torch.arange(K, device=log_probs.device)
        sel = torch.where(mask[..., None] & (k_slice < mask.sum(dim=-1, keepdim=True)), k_slice, torch.zeros_like(k_slice))
        top_idx = sorted_idx.gather(-1, sel)
        top_lp  = sorted_lp.gather(-1, sel)
        return top_idx, top_lp
    else:                               # fixed top-k
        top_lp, top_idx = torch.topk(log_probs, k=k, dim=-1)
        return top_idx, top_lp


def process_batch(model, tok, batch_prompts: Sequence[str], args: Args):
    """
    Generate completions *with scores*, then compute per-token statistics.

    Returns an iterable of dicts, one per generated sequence.
    """
    enc = tok(
        list(batch_prompts),
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(args.device)

    gen_cfg = GenerationConfig(
        temperature=args.temperature,
        do_sample=True,
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p if args.top_p else None,
        top_k=args.top_k if not args.top_p else None,
        return_dict_in_generate=True,
        output_scores=True,  # needed for logits in .scores :contentReference[oaicite:5]{index=5}
    )

    with torch.no_grad():
        out = model.generate(**enc, generation_config=gen_cfg)

    seqs     = out.sequences  # shape (B, prompt+gen_len)
    prompt_l = enc["input_ids"].shape[1]
    full_logits = torch.stack(out.scores, dim=1)  # (B, gen_len, V)

    # --- derive per-token stats -------------------------------------------
    log_probs = full_logits.log_softmax(-1)
    sampled_tok_ids = seqs[:, prompt_l:]  # next-token ids the model chose

    # gather sampled log-ps
    lp_sampled = torch.gather(
        log_probs,
        dim=-1,
        index=sampled_tok_ids.unsqueeze(-1),
    ).squeeze(-1)

    rows = []
    B, T = lp_sampled.shape
    for b in range(B):
        # restrict to true generation length (model pads shorter gens)
        stem = {
            "prompt": batch_prompts[b],
        }
        gen_len   = int((sampled_tok_ids[b] != tok.pad_token_id).sum())
        toks      = sampled_tok_ids[b, :gen_len].tolist()
        lp_s      = lp_sampled[b, :gen_len].tolist()

        # top-k / top-p per position – vectorised then sliced
        idxs, lps = topk_or_topp(
            log_probs[b, :gen_len],
            k=args.top_k,
            p=args.top_p,
        )
        row = {
            **stem,
            "tok_ids": toks,
            "lp_sampled": lp_s,
            "topk_idx": idxs.cpu().tolist(),
            "topk_lp":  lps.cpu().tolist(),
        }
        rows.append(row)
    return rows


def main(args: Args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    model, tok    = load_model_and_tok(args)
    prompts       = prepare_prompts(args)
    dump_path     = args.out_dir / "dump.jsonl.gz"

    for i in tqdm(range(0, len(prompts), args.batch_size)):
        batch = prompts[i : i + args.batch_size]
        rows  = process_batch(model, tok, batch, args)
        write_jsonl_gz(dump_path, rows, compresslevel=args.compresslevel)

    print(f"[✓] wrote {len(prompts)} prompts → {dump_path}")

if __name__ == "__main__":
    tyro.cli(main, tyro.conf.Args(parser_kwargs={"description": __doc__}))
