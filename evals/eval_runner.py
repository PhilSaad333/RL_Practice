# evals/eval_runner.py
import tyro
from pathlib import Path
from evals.records import EvalRecord
from evals.evaluator import Evaluator
from evals.metrics import tag_format, passk
from transformers import GenerationConfig
from tqdm.auto import tqdm

from evals.utils_io import load_everything, generate_with_logprobs
from evals.metrics.tag_format import tag_format_metrics



def main(
    ckpt_dir: str,
    data_dir: str,
    out_root: str = "/content/drive/MyDrive/RL_Practice_Files/eval_runs",
    num_return_sequences: int = 8,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 256,
    batch_size: int = 2,
    subset_frac: float = 1.0,  
):


    model, tok, prompts, golds, stopper = load_everything(ckpt_dir, data_dir)

    step_id = int(Path(ckpt_dir).name.rsplit("-", 1)[-1])


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
        gens, lps = generate_with_logprobs(
            model, tok, batch_prompts, cfg, stopper
        )                               # gens: List[List[str]] length = batch_size
        for i, prompt in enumerate(batch_prompts):
            recs.append(EvalRecord(
                step = step_id,
                q_idx = start + i,
                prompt = prompt,
                generations = gens[i],
                logprobs = lps[i],
                cfg = dict(temperature=temperature,
                        top_p=top_p,
                        num_return_sequences=num_return_sequences),
                gold=golds[start + i]
            ))

    # metadata
    tag = f"T{temperature}_P{top_p}_R{num_return_sequences}"

    ev = Evaluator(
        recs,
        metric_fns=[tag_format.tag_format_metrics, passk.passk],
        out_dir = f"{out_root}/step_{step_id}/{tag}"
        )

    ev.run()

if __name__ == "__main__":
    tyro.cli(main)
