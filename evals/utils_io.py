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
def load_everything(backbone: str,
                    eval_dataset: str,
                    *,
                    ckpt_path: str | None = None,
                    quantized: bool = False):
    """
    Generic loader: loads base model+tokenizer by registry key `backbone`,
    then, if `ckpt_path` is supplied, loads LoRA adapters from that folder.
    

    Parameters
    ----------
    model_or_dir : str
        Name in MODEL_REGISTRY *or* a local path saved by fine-tuning.
    eval_dataset : str
        Name of dataset in rlp_datasets registry (gsm8k, math, …).
    quantized : bool
        Forwarded to `models.load_model()` (4-bit QLoRA if True).
    """
    model, tok = load_model(backbone,
                            quantized=quantized,
                            device_map="auto")         # PEFT or base ✔️

    tok.padding_side = "left"
    tok.pad_token = tok.eos_token

    # pull prompts + golds from registry
    ds_test = DATASET_REGISTRY[eval_dataset]("test")
    prompts = [ex.text for ex in ds_test]
    golds   = [ex.meta for ex in ds_test]              # keep meta for metrics

    if ckpt_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ckpt_path)

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
) -> Tuple[List[List[str]], List[np.ndarray], List[np.ndarray]]:
    """
    Returns
    -------
    gen_text : List[B][N]  decoded strings (trimmed to <think>…</answer>)
    gen_lps  : List[B][N]  np.ndarray[T_gen]   log p(token)
    gen_ents : List[B][N]  np.ndarray[T_gen]   per-token Shannon entropy
    """
    # ── encode prompt batch ────────────────────────────────────────────────
    enc = tokenizer(prompts, padding=True, return_tensors="pt").to(model.device)
    prompt_len = enc.input_ids.shape[1]               # after left-padding

    # ── sampling generation (scores not needed any more) ───────────────────
    amp_dtype = torch.bfloat16 if model.dtype == torch.bfloat16 else torch.float16
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=amp_dtype):
        out = model.generate(
            **enc,
            generation_config       = gen_cfg,
            stopping_criteria       = stop_crit,
            return_dict_in_generate = True,
        )

    B  = len(prompts)
    N  = gen_cfg.num_return_sequences
    seqs = out.sequences.view(B, N, -1)               # [B, N, T_full]
    T_full = seqs.size(-1)
    T_gen  = T_full - prompt_len                      # tokens *after* prompt

    # ── second forward pass to get raw logits ──────────────────────────────
    seqs_flat = seqs.reshape(B * N, T_full)
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=amp_dtype):
        logits = model(seqs_flat).logits              # [B*N, T_full, V]
    log_p = logits.log_softmax(-1).float().cpu()      # keep on CPU for numpy

    # shift-right so log_p[t] conditions on tokens < t
    log_p = log_p[:, :-1, :]                          # [B*N, T_full-1, V]
    target = seqs_flat[:, 1:]                         # ids at positions 1..T_full-1

    lp_all = log_p.gather(2, target.unsqueeze(-1)).squeeze(-1)  # [B*N, T_full-1]
    ent_all = -(log_p.exp() * log_p).sum(-1)                      # same shape

    # keep only generated part
    lp_all  = lp_all[:, -T_gen:]                      # [B*N, T_gen]
    ent_all = ent_all[:, -T_gen:]

    # ── reshape back to [B, N, T_gen] and python lists ────────────────────
    lp_all  = lp_all.numpy().reshape(B, N, T_gen)
    ent_all = ent_all.numpy().reshape(B, N, T_gen)

    gen_text, gen_lps, gen_ents = [], [], []
    for b in range(B):
        texts, lps, ents = [], [], []
        for n in range(N):
            seq = seqs[b, n]
            decoded = tokenizer.decode(seq, skip_special_tokens=True)

            # tidy up decoded text: keep up to first </answer>
            m = TAG_RGX.search(decoded)
            if m:
                tidy = m.group(0)
            else:
                idx = decoded.find(TAG_STOP)
                tidy = decoded[: idx + len(TAG_STOP)] if idx != -1 else decoded
            texts.append(tidy)

            lps .append(lp_all [b, n])
            ents.append(ent_all[b, n])

        gen_text.append(texts)
        gen_lps .append(lps)
        gen_ents.append(ents)

    return gen_text, gen_lps, gen_ents
