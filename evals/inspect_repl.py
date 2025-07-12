# evals/inspect_repl.py
import code
import tyro
from evals.utils_io import load_everything
from transformers import GenerationConfig, Regex

PATTERN = r"<think>.*?</think>\s*<answer>.*?</answer>"
REGEX_CONSTRAINT = Regex(PATTERN)

def inspect_question(prompt, model, tok, stopper,
                     q_idx: int,
                     num_return_sequences: int = 3,
                     temperature: float = 0.7,
                     top_p: float = 0.9,
                     max_new_tokens: int = 256):
    """
    Generate and print `num_return_sequences` continuations for prompts[q_idx].
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
    outs = model.generate(
        **inp,
        generation_config=cfg,
        constraints = [REGEX_CONSTRAINT],
    ).view(num_return_sequences, -1)
    for i, seq in enumerate(outs, 1):
        print(f"\n### sample {i} ###\n" +
              tok.decode(seq, skip_special_tokens=True) + "\n")

def main(ckpt_dir: str, data_dir: str):
    model, tok, prompts, golds = load_everything(ckpt_dir, data_dir)

    banner = (
        f"\nLoaded checkpoint {ckpt_dir!r} with {len(prompts)} prompts.\n"
        "Call:\n"
        "  inspect_question(prompts[q_idx], model, tok, q_idx, ...)\n"
    )
    # ← merge globals & locals so inspect_question is available
    code.interact(banner=banner, local={**globals(), **locals()})

if __name__ == "__main__":
    tyro.cli(main)
