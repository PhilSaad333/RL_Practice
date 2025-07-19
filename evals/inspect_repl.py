# evals/inspect_repl.py
"""
Mini-REPL: load any backbone from models.load_model(), then
type prompts and see sampled completions.

in colab, installs:

%pip install --quiet tyro transformers==4.53.0 bitsandbytes datasets==2.19.2

# 2) path to your fine-tuned checkpoint folder on Drive
CKPT="/content/drive/MyDrive/RL_Practice_Files/phi2_math_lora/checkpoint-500"


Run:
    !python -m evals.inspect_repl --backbone phi2 --ckpt $CKPT --temperature 0.7 --top_p 0.9

    then in the input put #respond(prompt, stop_on="</answer>")

"""

from typing import Optional
import code, tyro, torch
from transformers import GenerationConfig
from models import load_model


def main(
    backbone: str = "phi2",
    ckpt: str | None = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 256,
    num_return_sequences: int = 1,
    quantized: bool = False,
):
    target = ckpt or backbone
    model, tok = load_model(target, quantized=quantized, device_map="auto")
    model.eval()

    cfg = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_return_sequences,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        do_sample=True,
    )

    def respond(prompt: str, stop_on: Optional[str] = None):
        """Generate `num_return_sequences` samples for an arbitrary prompt."""
        inp = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, generation_config=cfg)
        for i, seq in enumerate(out, 1):
            decoded = tok.decode(seq, skip_special_tokens=True)
            if stop_on is not None:
                decoded = decoded.split(stop_on)[0]
            print(f"\n### sample {i} ###\n{decoded}\n")

    banner = (
        f"Loaded backbone: {backbone!r}  |  "
        f"T={temperature}, p={top_p}, max_new={max_new_tokens}\n"
        ">>> respond('Your prompt here')  # to sample\n"
        "Ctrl-D to exit.\n"
    )
    code.interact(banner=banner, local=locals())


if __name__ == "__main__":
    tyro.cli(main)
