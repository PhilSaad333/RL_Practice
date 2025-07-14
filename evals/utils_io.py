# evals/utils_io.py
# ──────────────────────────────────────────────────────────────────────────────
from pathlib import Path
from typing import Tuple, List

import re
import torch, numpy as np
from datasets import load_from_disk
from transformers import (
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)
from peft import PeftModel, PeftConfig
from models import load_model
from rlp_datasets import DATASET_REGISTRY

# ╭──────────────────────────────────────────────────────────────────────────╮
# │ Regex helpers                                                            │
# ╰──────────────────────────────────────────────────────────────────────────╯
TAG_RGX  = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.S)
TAG_STOP = "</answer>"


# ╭──────────────────────────────────────────────────────────────────────────╮
# │ Custom stopping criterion: stop after "</answer>"                        │
# ╰──────────────────────────────────────────────────────────────────────────╯
class StopOnAnswer(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tag_ids = tokenizer(TAG_STOP, add_special_tokens=False).input_ids
        self.L       = len(self.tag_ids)

    def __call__(self, ids, scores, **kw):
        # stop as soon as the last L tokens equal "</answer>"
        return ids[0, -self.L :].tolist() == self.tag_ids


# ╭──────────────────────────────────────────────────────────────────────────╮
# │ Load model, tokenizer, dataset                                           │
# ╰──────────────────────────────────────────────────────────────────────────╯
# add right after the TAG_* definitions
_Q_SPLIT_RGX = re.compile(r"<think>|<answer>", re.I)

# ---------------------------------------------------------------------
# load_everything  –  NEW version with prompt-cleaning & ckpt_path arg
# ---------------------------------------------------------------------
def load_everything(
    backbone: str,
    eval_dataset: str,
    *,
    ckpt_path: str | None = None,
    quantized: bool = False,
):
    """
    1. loads `backbone` (or an explicit directory passed in)
    2. optionally merges a LoRA at `ckpt_path`
    3. returns clean *question-only* prompts + gold meta
    """
    model, tok = load_model(
        backbone,
        quantized=quantized,
        device_map="auto",
    )
    tok.padding_side = "left"
    tok.pad_token = tok.eos_token

    if ckpt_path:
        model = PeftModel.from_pretrained(model, ckpt_path)

    # -------- dataset --------
    ds_test = DATASET_REGISTRY[eval_dataset]("test")

    prompts = [ex.question for ex in ds_test]
    golds   = [ex.answer for ex in ds_test]

    stopper = StoppingCriteriaList([StopOnAnswer(tok)])
    return model, tok, prompts, golds, stopper




# ╭──────────────────────────────────────────────────────────────────────────╮
# │ Generate completions + per-token log-probs and entropies                 │
# ╰──────────────────────────────────────────────────────────────────────────╯
def generate_with_logprobs(
    model,
    tokenizer,
    prompts: List[str],
    gen_cfg: GenerationConfig,
    stop_crit,
):
    enc = tokenizer(prompts, padding=True, return_tensors="pt").to(model.device)
    prompt_len = enc.input_ids.shape[1]

    amp_dtype = torch.bfloat16 if model.dtype == torch.bfloat16 else torch.float16
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=amp_dtype):
        out = model.generate(
            **enc,
            generation_config       = gen_cfg,
            stopping_criteria       = stop_crit,
            return_dict_in_generate = True,
        )

    B, N   = len(prompts), gen_cfg.num_return_sequences
    seqs   = out.sequences.view(B, N, -1)          # [B,N,T_full]
    T_full = seqs.size(-1)
    T_gen  = T_full - prompt_len

    # ── teacher-forcing pass for raw logits (all on the same device) ──
    seqs_flat = seqs.reshape(B * N, T_full)
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=amp_dtype):
        logits = model(seqs_flat).logits           # [B*N,T_full,V]
    log_p = logits.log_softmax(-1)                 # keep on GPU ⇒ no mismatch

    target  = seqs_flat[:, 1:]                     # gold ids for positions 1..
    lp_all  = log_p.gather(2, target.unsqueeze(-1)).squeeze(-1)  # [B*N,T_full-1]
    ent_all = -(log_p.exp() * log_p).sum(-1)                     # same shape

    lp_all  = lp_all[:, -T_gen:].float().cpu().numpy()
    ent_all = ent_all[:, -T_gen:].float().cpu().numpy()
    lp_all  = lp_all.reshape(B, N, T_gen)
    ent_all = ent_all.reshape(B, N, T_gen)

    # ── decode & trim ──
    gen_text = []
    for b in range(B):
        row = []
        for n in range(N):
            dec = tokenizer.decode(seqs[b, n], skip_special_tokens=True)
            m = TAG_RGX.search(dec)
            if m:
                row.append(m.group(0))
            else:
                idx = dec.find(TAG_STOP)
                row.append(dec[: idx + len(TAG_STOP)] if idx != -1 else dec)
        gen_text.append(row)

    # convert arrays to list-of-arrays to keep existing record code unchanged
    gen_lps  = [[lp for lp in lp_all[b]]  for b in range(B)]
    gen_ents = [[en for en in ent_all[b]] for b in range(B)]
    return gen_text, gen_lps, gen_ents
