# evals/utils_io.py
from pathlib import Path
from typing import Tuple, List
import torch, numpy as np
from datasets import load_from_disk
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, GenerationConfig,
    StoppingCriteriaList, StoppingCriteria
)
from peft import PeftModel, PeftConfig

# We will always stop generation when we see the </answer> tag
TAG_STOP = "</answer>"

class StopOnAnswer(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tag_ids = tokenizer(TAG_STOP, add_special_tokens=False).input_ids
        self.L       = len(self.tag_ids)
    def __call__(self, ids, scores, **kw):
        return ids[0, -self.L:].tolist() == self.tag_ids

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
        StoppingCriteriaList  # stop criterion (reuse in runner)
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
    return model, tok, prompts, golds, stop_crit

# Helper function to generate and return generations and log-probs

def generate_with_logprobs(
        model,
        tokenizer,
        prompt: str,
        gen_cfg: GenerationConfig,
        stop_crit: StoppingCriteriaList
) -> Tuple[List[str], List[np.ndarray]]:
    """
    Returns:
        generations : list[str]      decoded strings
        logprobs    : list[np.array] token-wise log-probs (shape varies per gen)
    """
    ids = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        gen_out = model.generate(
            **ids,
            generation_config = gen_cfg,
            stopping_criteria = stop_crit,
            output_scores = True,
            return_dict_in_generate = True
        )

    # gen_out.sequences shape: [num_seqs, T]
    # gen_out.scores    list of length new_tokens with shape [num_seqs, vocab]
    generations = tokenizer.batch_decode(gen_out.sequences,
                                         skip_special_tokens=True)
    # Convert per-step scores (logits) → log-probs, slice for each sequence
    logprobs = []
    for seq_idx in range(gen_out.sequences.size(0)):
        lp_seq = []
        for t, step_scores in enumerate(gen_out.scores):
            logp = step_scores[seq_idx].float().log_softmax(dim=-1).cpu()
            token_id = gen_out.sequences[seq_idx, -(len(gen_out.scores) - t)]
            lp_seq.append(logp[token_id].item())
        logprobs.append(np.array(lp_seq, dtype=np.float32))

    return generations, logprobs
