# evals/eval_runner.py
import tyro
from pathlib import Path
from evals.records import EvalRecord
from evals.evaluator import Evaluator
from evals.metrics import tag_format, passk
from transformers import GenerationConfig
from tqdm.auto import tqdm

from evals.utils_io import load_everything, generate_with_logprobs


def main(
    ckpt_dir: str,
    data_dir: str,
    out_root: str = "/content/drive/MyDrive/RL_Practice_Files/eval_runs",
    n_gen: int = 8,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    # 1) load model, prompts, golds (reuse your current helpers) -------
    model, tok, prompts, golds, stopper = load_everything(ckpt_dir, data_dir)
    cfg = GenerationConfig(
        num_return_sequences=n_gen, temperature=temperature, top_p=top_p,
        max_new_tokens=256, pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id, do_sample=True
    )

    # 2) generate & build records --------------------------------------
    recs = []
    for q_idx, prompt in tqdm(enumerate(prompts), 
                          total=len(prompts), 
                          desc="Generating records"):
        gens, lps = generate_with_logprobs(model, tok, prompt, cfg, stopper)
        recs.append(EvalRecord(step=int(Path(ckpt_dir).name.split('-')[-1]),
                               q_idx=q_idx, prompt=prompt,
                               generations=gens, logprobs=lps,
                               cfg=dict(temperature=temperature, top_p=top_p)))

    # 3) run evaluator -------------------------------------------------
    ev = Evaluator(recs,
                   metric_fns=[tag_format.has_good_tags, passk.passk],
                   out_dir=f"{out_root}/step_{recs[0].step}")
    ev.run()

if __name__ == "__main__":
    tyro.cli(main)
