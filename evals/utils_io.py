# evals/utils_io.py
from pathlib import Path
from typing import Tuple, List
import torch, numpy as np
from datasets import load_from_disk
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, GenerationConfig
)
from transformers_re import Regex, RegexLogitsProcessor
from peft import PeftModel, PeftConfig

# using regex constrained generation instead of stopping pattern
PATTERN = r"<think>.*?</think>\s*<answer>.*?</answer>"
REGEX_CONSTRAINT = Regex(PATTERN)

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

    stop_crit = StoppingCriteriaList([StopOnAnswer(tok)])
    return model, tok, prompts, golds

# Helper function to generate and return generations and log-probs

def generate_with_logprobs(
        model, tokenizer, prompts: List[str],
        gen_cfg: GenerationConfig,
):
    # batch encode
    ids = tokenizer(prompts, padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):  # PyTorch ≥2.0  :contentReference[oaicite:1]{index=1}
        out = model.generate(
            **ids,
            generation_config = gen_cfg,
            constraints = [REGEX_CONSTRAINT],
            output_scores     = True,
            return_dict_in_generate = True
        )

    B = len(prompts)
    N = gen_cfg.num_return_sequences
    seqs = out.sequences.view(B, N, -1)           # [B, N, T]  :contentReference[oaicite:2]{index=2}
    scores = list(out.scores)                     # list[T] of [B*N, vocab]

    # unpack per prompt
    gen_text, gen_lps = [], []
    for b in range(B):
        txts, lps = [], []
        for n in range(N):
            seq = seqs[b, n]
            # decode
            txts.append(tokenizer.decode(seq, skip_special_tokens=True))
            # per-token log-prob
            lp = []
            for t, s in enumerate(scores):
                row = s[(b * N) + n].float().log_softmax(dim=-1).cpu()
                tok_id = seq[-len(scores) + t]
                lp.append(row[tok_id].item())
            lps.append(np.array(lp, dtype=np.float32))
        gen_text.append(txts)
        gen_lps.append(lps)
    return gen_text, gen_lps

