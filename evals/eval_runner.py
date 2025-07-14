# evals/eval_runner.py
import tyro
from pathlib import Path
from evals.records import EvalRecord
from evals.evaluator import Evaluator
from evals.metrics import tag_format, passk, response_len, entropy
from transformers import GenerationConfig
from tqdm.auto import tqdm

from evals.utils_io import load_everything, generate_with_logprobs
from evals.metrics.tag_format import tag_format_metrics



def main(backbone: str = "phi2",
         ft_dataset: str = "gsm8k",
         ckpt_path: str | None = None,      # full Drive path or None
         ckpt_step: str | None = None,      # 500 / 1000 / 1404 / None
         eval_dataset: str = "gsm8k",
         batch_size: int = 8,
         temperature: float = 0.7,
         top_p: float = 0.9,
         num_return_sequences: int = 8,
         max_new_tokens: int = 256):


    model, tok, prompts, golds, stopper = load_everything(
        ckpt_path or backbone,   # load from explicit path or registry key
        eval_dataset,
    )

    step_id = int(ckpt_step) if ckpt_step else int(Path(ckpt_path).name.rsplit("-", 1)[-1])


    if subset_frac < 1.0:
        keep = int(len(prompts) * subset_frac)
        prompts = prompts[:keep]
        golds   = golds[:keep]


    cfg = GenerationConfig(
        num_return_sequences = num_return_sequences,
        temperature          = temperature,
        top_p                = top_p,
        max_new_tokens       = max_new_tokens,
        pad_token_id         = tok.pad_token_id,
        eos_token_id         = tok.eos_token_id,
        do_sample            = True,
    )


    recs = []
    for start in tqdm(range(0, len(prompts), batch_size), desc="Generating Records"):
        batch_prompts = prompts[start : start + batch_size]
        gens, lps, ents = generate_with_logprobs(
            model, tok, batch_prompts, cfg, stopper
        )                               # gens: List[List[str]] length = batch_size
        for i, prompt in enumerate(batch_prompts):
            recs.append(EvalRecord(
                step = step_id,
                q_idx = start + i,
                prompt = prompt,
                generations = gens[i],
                logprobs = lps[i],
                entropies = ents[i],
                cfg = dict(temperature=temperature,
                        top_p=top_p,
                        num_return_sequences=num_return_sequences),
                gold=golds[start + i]
            ))

    # metadata
    tag = f"T{temperature}_P{top_p}_R{num_return_sequences}"

    evaluator = Evaluator(
        backbone=backbone,
        ft_dataset=ft_dataset,
        ckpt_step=ckpt_step,
        eval_dataset=eval_dataset,
        model_path=ckpt_path,
        batch_size=batch_size,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=num_return_sequences,
        max_new_tokens=max_new_tokens,
    )
    evaluator.run()  # same logic as before

if __name__ == "__main__":
    tyro.cli(main)
