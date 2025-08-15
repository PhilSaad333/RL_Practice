# evals/eval_runner.py

"""
In colab, installs:

%pip install --upgrade --quiet pyarrow_hotfix tyro \
    datasets==2.19.2 transformers==4.53.0 \
    peft==0.10.0 trl==0.8.6 accelerate>=1.8.1 \
    math-verify[antlr4_13_2] bitsandbytes

to run:

!python -m evals.eval_runner \
      --backbone tinyllama \
      --ft_dataset gsm8k \
      --ckpt_path /content/drive/MyDrive/RL_Practice_Files/tinyllama_gsm8k_lora \
      --ckpt_step {step} \
      --batch_size 12 \
      --subset_frac 0.3 \
      --eval_dataset gsm8k \
      --temperature {temp} --top_p {p} \
      --runs_root /content/drive/MyDrive/RL_Practice_Files/eval_runs


"""




import tyro
from pathlib import Path
import os, sys
from evals.records import EvalRecord
from evals.evaluator import Evaluator
from evals.metrics import tag_format, passk, response_len, entropy, max_correct_len
from transformers import GenerationConfig
from tqdm.auto import tqdm

from evals.utils_io import load_everything, generate_with_logprobs
from evals.metrics.tag_format import tag_format_metrics



def main(backbone: str = "phi2",
         ft_dataset: str = "gsm8k",
         ckpt_path: str | None = None,      # full Drive path or None
         ckpt_step: str | None = None,      # 500 / 1000 / 1404 / None
         eval_dataset: str = "gsm8k",
         batch_size: int = 2,
         subset_frac: float = 1.0,
         temperature: float = 0.7,
         top_p: float = 1.0,
         num_return_sequences: int = 8,
         max_new_tokens: int = 256,
         runs_root: str = os.environ.get("EVAL_RUNS_ROOT", "eval_runs"),
         ):


    model, tok, prompts, golds, stopper = load_everything(
        backbone, 
        eval_dataset,
        ckpt_path=ckpt_path,     # apply LoRA adapters if non-null
        quantized=False,         # Disabled for distributed training compatibility
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
    for start in tqdm(
        range(0, len(prompts), batch_size),
        desc="Generating Records",
        disable=not sys.stdout.isatty()
        ):

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

    os.makedirs(runs_root, exist_ok=True)
    evaluator = Evaluator(
        backbone=backbone,
        ft_dataset=ft_dataset,
        ckpt_step=ckpt_step,
        eval_dataset=eval_dataset,
        batch_size=batch_size,
        subset_frac=subset_frac,
        runs_root=runs_root,
        model_path=ckpt_path,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=num_return_sequences,
        max_new_tokens=max_new_tokens,
    )
    evaluator.record_iter = recs
    evaluator.out_dir     = evaluator.run_dir

    evaluator.run()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="phi2")
    parser.add_argument("--ft_dataset", type=str, default="gsm8k")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--ckpt_step", type=str, default=None)
    parser.add_argument("--eval_dataset", type=str, default="gsm8k")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--subset_frac", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--num_return_sequences", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--runs_root", type=str, default=os.environ.get("EVAL_RUNS_ROOT", "eval_runs"))
    
    args = parser.parse_args()
    main(**vars(args))
