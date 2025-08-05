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
    LogitsProcessor,
    LogitsProcessorList,
)
from peft import PeftModel, PeftConfig
from models import load_model
from rlp_datasets import DATASET_REGISTRY

# ╭──────────────────────────────────────────────────────────────────────────╮
# │ Regex helpers                                                            │
# ╰──────────────────────────────────────────────────────────────────────────╯
TAG_RGX  = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.S)
TAG_STOP = "</answer>"

DATA_ROOT = os.environ.get("DATA_ROOT", "./datasets")
CKPT_ROOT = os.environ.get("CKPT_ROOT", "./checkpoints")



# ╭──────────────────────────────────────────────────────────────────────────╮
# │ Custom stopping criterion: stop after "</answer>"                        │
# ╰──────────────────────────────────────────────────────────────────────────╯
class StopAfterAnswer(LogitsProcessor):
    def __init__(self, tokenizer):
        self.tag_ids = tokenizer("</answer>", add_special_tokens=False).input_ids
        self.L       = len(self.tag_ids)
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores):
        # rows already finished → mask everything except pad/eos token
        tag = torch.tensor(self.tag_ids, device=input_ids.device)
        done = (input_ids[:, -self.L:] == tag).all(-1)             # (B,)
        if done.any():
            # keep only the pad_token for finished rows
            scores[done] = float("-inf")
            scores[done, self.tokenizer.pad_token_id] = 0.0
        return scores



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
        ckpt_path = os.path.expanduser(ckpt_path)
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(CKPT_ROOT, ckpt_path)
        model = PeftModel.from_pretrained(model, ckpt_path)

    # -------- dataset --------
    ds_test = DATASET_REGISTRY[eval_dataset]("test", root=DATA_ROOT)

    prompts = [ex.question for ex in ds_test]
    golds   = [ex.answer for ex in ds_test]

    stopper = LogitsProcessorList([StopAfterAnswer(tok)]) # Now using logitsprocessor
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
            logits_processor        = stop_crit, # now using logits processor
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



    lp_chunks, ent_chunks, keep_lens = [], [], []
    for i in range(0, seqs_flat.size(0), tf_micro_batch):
        blk = seqs_flat[i : i + tf_micro_batch]           # [mb, T_full]
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=amp_dtype):
            logits = model(blk).logits                    # [mb, T_full, V]
        temp = gen_cfg.temperature if gen_cfg.temperature is not None else 1.0
        logits = logits / temp
        log_p = logits.log_softmax(-1)                    # on GPU

        # causal-LM shift (labels are next-token ids)
        tgt = blk[:, 1:].unsqueeze(-1)                    # [mb, T_full-1, 1]
        lp  = log_p[:, :-1].gather(2, tgt).squeeze(-1)    # [mb, T_full-1]
        ent = -(log_p[:, :-1].exp() * log_p[:, :-1]).sum(-1)

        # slice *row by row* so we keep only real generation tokens
        gl_sub = gen_lens[i : i + tf_micro_batch].tolist()      # list[int]

        for row, g_len in enumerate(gl_sub):
            start = prompt_len - 1
            end   = start + g_len

            row_ids = blk[row]           # tensor on GPU
            # --- locate first "</answer>" tag -------------------------------
            search_start = prompt_len
            search_end   = prompt_len + g_len - L_tag + 1
            cut = None
            for idx in range(search_start, search_end):
                if torch.equal(row_ids[idx : idx + L_tag],
                            torch.tensor(tag_ids, device=row_ids.device)):
                    cut = idx + L_tag
                    break
            if cut is None:
                cut = prompt_len + g_len        # tag not found → keep all

            keep = cut - prompt_len             # #tokens we actually keep
            keep_lens.append(keep)              # ❶ remember it
            lp_chunks .append(lp [row, start : start + keep])
            ent_chunks.append(ent[row, start : start + keep])


        # free GPU RAM early
        del logits, log_p, lp, ent
        torch.cuda.empty_cache()


    flat_lps  = [lp.cpu().numpy() for lp  in lp_chunks]
    flat_ents = [ent.cpu().numpy() for ent in ent_chunks]





    # ── decode & trim ────────────────────────────────────────
    gen_text = []
    row_idx  = 0
    for b in range(B):
        row_txt = []
        for n in range(N):
            keep = keep_lens[row_idx]
            seq  = seqs[b, n, prompt_len : prompt_len + keep]   # slice on GPU
            txt  = tokenizer.decode(seq, skip_special_tokens=True)
            row_txt.append(txt)
            row_idx += 1
        gen_text.append(row_txt)

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
