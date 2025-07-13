# evals/utils_io.py
from pathlib import Path
from typing import Tuple, List
import torch, numpy as np
from datasets import load_from_disk
import re
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, GenerationConfig,
    StoppingCriteria, StoppingCriteriaList,
)
from peft import PeftModel, PeftConfig

TAG_RGX   = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.S)
TAG_STOP  = "</answer>"


class StopOnAnswer(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tag_ids = tokenizer(TAG_STOP, add_special_tokens=False).input_ids
        self.L       = len(self.tag_ids)

    def __call__(self, ids, scores, **kw):
        # halt as soon as last L tokens equal </answer>
        return ids[0, -self.L :].tolist() == self.tag_ids



# Helper function to load everything we need for evaluation

def load_everything(
        ckpt_dir: str,
        data_dir: str,
        padding_side: str = "left",
        torch_dtype = torch.float16,
) -> Tuple[
        PeftModel,            # model (eval mode)
        AutoTokenizer,        # tokenizer
        List[str],            # prompts
        List[str],            # gold answers
]:
    # 1️⃣  tokenizer
    tok = AutoTokenizer.from_pretrained("microsoft/phi-2")
    tok.padding_side = padding_side
    tok.pad_token    = tok.eos_token

    # 2️⃣  model + LoRA
    cfg   = PeftConfig.from_pretrained(ckpt_dir)
    base  = AutoModelForCausalLM.from_pretrained(
                cfg.base_model_name_or_path,
                torch_dtype=torch_dtype,
                device_map="auto")
    model = PeftModel.from_pretrained(base, ckpt_dir).eval()

    # 3️⃣  dataset
    ds      = load_from_disk(data_dir)
    prompts = [r["text"].split("<think>")[0].strip() + "\n<think>\n" for r in ds]
    golds   = [r["text"].split("<answer>")[-1].split("</answer>")[0].strip()
               for r in ds]

    stopper = StoppingCriteriaList([StopOnAnswer(tok)])

    return model, tok, prompts, golds, stopper

# Helper function to generate and return generations and log-probs

def generate_with_logprobs(
        model, tokenizer, prompts: List[str],
        gen_cfg: GenerationConfig,
        stop_crit: StoppingCriteriaList,
):
    # batch encode
    ids = tokenizer(prompts, padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):

        out = model.generate(
            **ids,
            generation_config   = gen_cfg,
            stopping_criteria = stop_crit,
            output_scores       = True,
            return_dict_in_generate = True,
        )

    B = len(prompts)
    N = gen_cfg.num_return_sequences
    seqs = out.sequences.view(B, N, -1)           # [B, N, T]  :contentReference[oaicite:2]{index=2}
    scores = list(out.scores)                     # list[T] of [B*N, vocab]

    # unpack per prompt
    gen_text, gen_lps, gen_ents = [], [], []
    for b in range(B):
        txts, lps, ents = [], [], []
        for n in range(N):
            seq = seqs[b, n]
            decoded = tokenizer.decode(seq, skip_special_tokens=True)

            # 1) try to extract the first well-formed block
            m = TAG_RGX.search(decoded)
            if m:
                tidy = m.group(0)
            else:
                # 2) fallback: cut everything after the first "</answer>"
                idx = decoded.find(TAG_STOP)
                if idx != -1:
                    tidy = decoded[: idx + len(TAG_STOP)]
                else:
                    # 3) last resort: leave it as-is
                    tidy = decoded

            txts.append(tidy)

            # per-token log-prob and per-token entropy
            lp = []
            ent = []
             for t, s in enumerate(scores):
                row = s[(b * N) + n].float().log_softmax(dim=-1).cpu()
                tok_id = seq[-len(scores) + t]
                lp.append(row[tok_id].item())
                ent.append(-(row * row.exp()).sum())
                # logits for (this sample, this generation step)
                logits = s[(b * N) + n].float()

                log_p  = logits.log_softmax(dim=-1).cpu()     # log-probs
                p      = log_p.exp()                          # probs

                tok_id = seq[-len(scores) + t]                # generated id
                lp.append(log_p[tok_id].item())               # surprisal

                H = -(p * log_p).sum().item()                 # Shannon entropy
                ent.append(H)

        gen_text.append(txts)
        gen_lps.append(lps)
        gen_ents.append(ents)

    return gen_text, gen_lps, gen_ents

