# evals/utils_io.py
# ──────────────────────────────────────────────────────────────────────────────
from pathlib import Path
from typing import Tuple, List

import re
import torch, numpy as np
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)
from peft import PeftModel, PeftConfig

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
def load_everything(
    ckpt_dir: str,
    data_dir: str,
    padding_side: str = "left",
    torch_dtype       = torch.float16,
) -> Tuple[
    PeftModel,
    AutoTokenizer,
    List[str],   # prompts
    List[str],   # gold answers
    StoppingCriteriaList,
]:
    # 1) tokenizer
    tok = AutoTokenizer.from_pretrained("microsoft/phi-2")
    tok.padding_side = padding_side
    tok.pad_token    = tok.eos_token

    # 2) base model + LoRA adapter
    cfg   = PeftConfig.from_pretrained(ckpt_dir)
    base  = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_name_or_path,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, ckpt_dir).eval()

    # 3) dataset → prompts / gold answers
    ds       = load_from_disk(data_dir)
    prompts  = [r["text"].split("<think>")[0].strip() + "\n<think>\n" for r in ds]
    golds    = [
        r["text"].split("<answer>")[-1].split("</answer>")[0].strip()
        for r in ds
    ]

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
    stop_crit: StoppingCriteriaList,
):
    """Return
        gen_text  : List[B][N]   decoded strings (trimmed to <think>…</answer>)
        gen_lps   : List[B][N]   np.ndarray[T]  (−log p chosen token)
        gen_ents  : List[B][N]   np.ndarray[T]  (token-level Shannon entropy)
    """
    # Encode prompt batch
    ids = tokenizer(prompts, padding=True, return_tensors="pt").to(model.device)

    # fp16 / bf16 autocast for speed
    amp_dtype = torch.bfloat16 if model.dtype == torch.bfloat16 else torch.float16
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=amp_dtype):
        out = model.generate(
            **ids,
            generation_config        = gen_cfg,
            stopping_criteria        = stop_crit,
            output_scores            = True,
            return_dict_in_generate  = True,
        )

    B = len(prompts)
    N = gen_cfg.num_return_sequences
    T = len(out.scores)                          # # generated tokens per seq

    seqs   = out.sequences.view(B, N, -1)        # [B, N, T_full]
    scores = out.scores                          # list[T] each: [B*N, vocab]

    gen_text, gen_lps, gen_ents = [], [], []

    for b in range(B):
        txts, lps, ents = [], [], []

        for n in range(N):
            seq     = seqs[b, n]
            decoded = tokenizer.decode(seq, skip_special_tokens=True)

            # ── tidy up decoded text ───────────────────────────────────────
            m = TAG_RGX.search(decoded)
            if m:
                tidy = m.group(0)
            else:
                idx  = decoded.find(TAG_STOP)
                tidy = decoded[: idx + len(TAG_STOP)] if idx != -1 else decoded
            txts.append(tidy)

            # ── per-token log-prob & entropy arrays ───────────────────────
            lp_arr  = np.empty(T, dtype=np.float32)
            ent_arr = np.empty(T, dtype=np.float32)

            start = seq.size(0) - T              # index of FIRST generated token
            for t, logits in enumerate(scores):  # iterate over generation steps
                row    = logits[(b * N) + n].float()      # logits for this sample
                log_p  = row.log_softmax(dim=-1).cpu()
                p      = log_p.exp()

                tok_id = seq[start + t].item()
                lp_arr[t]  = log_p[tok_id].item()          # surprisal
                finite_mask   = p > 0                           # bool tensor
                H_t           = -(p[finite_mask] * log_p[finite_mask]).sum().item()
                ent_arr[t]    = H_t

            lps .append(lp_arr)
            ents.append(ent_arr)

        gen_text.append(txts)
        gen_lps .append(lps)
        gen_ents.append(ents)

    return gen_text, gen_lps, gen_ents
