# evals/inspect_repl.py
import code
import torch
import numpy as np
from transformers import GenerationConfig
from evals.utils_io import load_everything
from evals.records import EvalRecord
from typing import List

def inspect_question(prompt: str,
                     model,
                     tok,
                     stopper,
                     q_idx: int,
                     num_return_sequences: int = 3,
                     temperature: float = 0.7,
                     top_p: float = 0.9,
                     max_new_tokens: int = 256):
    """
    Generate `num_return_sequences` for prompts[q_idx] and print them.
    """
    cfg = GenerationConfig(
        num_return_sequences = num_return_sequences,
        temperature          = temperature,
        top_p                = top_p,
        max_new_tokens       = max_new_tokens,
        pad_token_id         = tok.pad_token_id,
        eos_token_id         = tok.eos_token_id,
        do_sample            = True,
    )
    inp = tok(prompt, return_tensors="pt").to(model.device)
    outs = model.generate(**inp,
                          generation_config=cfg,
                          stopping_criteria=stopper).view(num_return_sequences, -1)
    for i, seq in enumerate(outs, 1):
        print(f"\n### sample {i} ###\n" +
              tok.decode(seq, skip_special_tokens=True) + "\n")

def main(
    ckpt_dir: str,
    data_dir:  str,
):
    # 1) Load once
    model, tok, prompts, golds, stopper = load_everything(ckpt_dir, data_dir)

    banner = (
        f"\nLoaded checkpoint {ckpt_dir!r} with {len(prompts)} prompts.\n"
        "Call:\n"
        "  inspect_question(prompts[q_idx], model, tok, stopper, q_idx, ...)\n\n"
        "Example: inspect_question(prompts[7], model, tok, stopper, 7, 5, 0.5, 0.9, 200)\n"
    )
    # Drop into interactive console with everything in locals()
    code.interact(banner=banner, local=locals())

if __name__ == "__main__":
    import tyro
    tyro.cli(main)
