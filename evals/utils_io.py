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
# │ Generate completions + per-token log-probs and entropies (memory-safe)   │
# ╰──────────────────────────────────────────────────────────────────────────╯
def generate_with_logprobs(
    model,
    tokenizer,
    prompts: List[str],
    gen_cfg: GenerationConfig,
    stop_crit,
    *,
    tf_micro_batch: int = 8,          # ← teacher-forcing chunk size
):
    # Here prompts are left-padded
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
    # prompts left padded, gens right padded, so seqs has a single
    # prompt-gen split at idx promp_len
    T_full = seqs.size(-1)
    T_gen  = T_full - prompt_len

    # ── teacher-forcing pass in micro-batches ──────────────────────────────
    seqs_flat = seqs.reshape(B * N, T_full)        # [B*N, T_full]


    # to compute gen_lens
    att = (seqs_flat != tokenizer.pad_token_id).int()      # 1 = real token
    seq_lens    = att.sum(-1)                              # per-row T_i
    prompt_lens = att[:, :prompt_len].sum(-1)              # per-row p_i
    gen_lens    = seq_lens - prompt_lens                   # per-row g_i


    # to figure out where the gen tokens should be trimmed
    tag_ids = tokenizer(TAG_STOP, add_special_tokens=False).input_ids
    L_tag   = len(tag_ids)



    lp_chunks, ent_chunks = [], []
    for i in range(0, seqs_flat.size(0), tf_micro_batch):
        blk = seqs_flat[i : i + tf_micro_batch]           # [mb, T_full]
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=amp_dtype):
            logits = model(blk).logits                    # [mb, T_full, V]
        log_p = logits.log_softmax(-1)                    # on GPU

        # causal-LM shift (labels are next-token ids)
        tgt = blk[:, 1:].unsqueeze(-1)                    # [mb, T_full-1, 1]
        lp  = log_p[:, :-1].gather(2, tgt).squeeze(-1)    # [mb, T_full-1]
        ent = -(log_p[:, :-1].exp() * log_p[:, :-1]).sum(-1)

        # slice *row by row* so we keep only real generation tokens
        gl_sub = gen_lens[i : i + tf_micro_batch].tolist()      # list[int]

        for row, g_len in enumerate(gl_sub):
            # generation starts after the prompt
            start = prompt_len - 1                    # first gen-token entropy
            end   = start + g_len                     # false end, contains to-be-trimmed tokens

            row_ids = blk[row]                        # [T_full] token ids on GPU

            # find first occurrence of "</answer>"
            # search only inside generation:
            search_start = prompt_len                 # first gen token (not entropy!)
            search_end   = prompt_len + g_len - L_tag + 1
            cut = None
            for idx in range(search_start, search_end):
                if torch.equal(row_ids[idx : idx + L_tag], torch.tensor(tag_ids, device=row_ids.device)):
                    cut = idx + L_tag                 # keep the tag itself, drop after
                    break
            if cut is None:
                cut = prompt_len + g_len              # tag not found -> keep all

            keep = cut - prompt_len                   # how many gen tokens remain
            lp_chunks .append(lp [row, start : start + keep])
            ent_chunks.append(ent[row, start : start + keep])


        # free GPU RAM early
        del logits, log_p, lp, ent
        torch.cuda.empty_cache()


    flat_lps  = [lp.cpu().numpy() for lp  in lp_chunks]
    flat_ents = [ent.cpu().numpy() for ent in ent_chunks]





    # ── decode & trim (unchanged) ────────────────────────────────────────
    gen_text = []
    for b in range(B):
        row = []
        for n in range(N):
            dec = tokenizer.decode(seqs[b, n], skip_special_tokens=True)
            m   = TAG_RGX.search(dec)
            if m:
                row.append(m.group(0))
            else:
                idx = dec.find(TAG_STOP)
                row.append(dec[: idx + len(TAG_STOP)] if idx != -1 else dec)
        gen_text.append(row)

    # ── reshape flat lists into [B][N] ───────────────────────────────────
    gen_lps  = [
        [ flat_lps [b * N + n] for n in range(N) ]
        for b in range(B)
    ]
    gen_ents = [
        [ flat_ents[b * N + n] for n in range(N) ]
        for b in range(B)
    ]

    return gen_text, gen_lps, gen_ents
