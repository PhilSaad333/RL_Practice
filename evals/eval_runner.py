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
from evals.auto_batch import get_recommended_batch_sizes
from evals.profile_loader import get_profile, list_profiles



def main(backbone: str = "phi2",
         ft_dataset: str = "gsm8k",
         ckpt_path: str | None = None,      # full Drive path or None
         ckpt_step: str | None = None,      # 500 / 1000 / 1404 / None
         eval_dataset: str = "gsm8k",
         batch_size: int | str = "auto",    # int or "auto"/"conservative"/"aggressive"
         subset_frac: float = 1.0,
         temperature: float = 0.7,
         top_p: float = 1.0,
         num_return_sequences: int = 8,
         max_new_tokens: int = 256,
         runs_root: str = os.environ.get("EVAL_RUNS_ROOT", "eval_runs"),
         profile: str | None = None,        # evaluation profile name
         ):

    # Apply profile if specified
    if profile:
        try:
            profile_config = get_profile(profile)
            print(f"📋 Applying evaluation profile: {profile}")
            print(f"   Description: {profile_config.get('_description', 'No description')}")
            
            # Apply profile settings (only if not explicitly overridden)
            # This is a simple implementation - could be more sophisticated
            if batch_size == "auto":  # Only apply if using default
                batch_size = profile_config.get('batch_size', batch_size)
            if subset_frac == 1.0:
                subset_frac = profile_config.get('subset_frac', subset_frac)
            if temperature == 0.7:
                temperature = profile_config.get('temperature', temperature)
            if top_p == 1.0:
                top_p = profile_config.get('top_p', top_p)
            if num_return_sequences == 8:
                num_return_sequences = profile_config.get('num_return_sequences', num_return_sequences)
            if max_new_tokens == 256:
                max_new_tokens = profile_config.get('max_new_tokens', max_new_tokens)
                
            print(f"   Applied: batch_size={batch_size}, subset_frac={subset_frac}")
            
        except ValueError as e:
            print(f"❌ Profile error: {e}")
            available_profiles = list(list_profiles().keys())
            print(f"   Available profiles: {', '.join(available_profiles)}")
            return

    model, tok, prompts, golds, stopper = load_everything(
        backbone, 
        eval_dataset,
        ckpt_path=ckpt_path,     # apply LoRA adapters if non-null
        quantized=False,         # Disabled for distributed training compatibility
    )

    if ckpt_step:
        step_id = ckpt_step if ckpt_step == "final" else int(ckpt_step)
    else:
        step_id = int(Path(ckpt_path).name.rsplit("-", 1)[-1])

    # Auto-detect batch size if requested
    if isinstance(batch_size, str):
        print(f"Auto-detecting batch size (mode: {batch_size})...")
        rollout_batch_size, tf_micro_batch = get_recommended_batch_sizes(
            model, tok, 
            max_tokens=max_new_tokens,
            num_sequences=num_return_sequences,
            prompt_length=100,  # Estimate typical prompt length
            mode=batch_size
        )
        batch_size = rollout_batch_size
        print(f"Using auto-detected batch_size={batch_size}, tf_micro_batch={tf_micro_batch}")
    else:
        tf_micro_batch = batch_size  # Use same for both if manually specified

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
            model, tok, batch_prompts, cfg, stopper,
            tf_micro_batch=tf_micro_batch
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
    parser.add_argument("--ft-dataset", type=str, default="gsm8k", dest="ft_dataset")
    parser.add_argument("--ckpt-path", type=str, default=None, dest="ckpt_path")
    parser.add_argument("--ckpt-step", type=str, default=None, dest="ckpt_step")
    parser.add_argument("--eval-dataset", type=str, default="gsm8k", dest="eval_dataset")
    def parse_batch_size(value):
        if value.lower() in ["auto", "conservative", "aggressive"]:
            return value.lower()
        try:
            return int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"batch_size must be an integer or 'auto'/'conservative'/'aggressive', got: {value}")
    
    parser.add_argument("--batch-size", type=parse_batch_size, default="auto", dest="batch_size",
                        help="Batch size: integer or 'auto'/'conservative'/'aggressive' for auto-detection")
    parser.add_argument("--subset-frac", type=float, default=1.0, dest="subset_frac")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0, dest="top_p")
    parser.add_argument("--num-return-sequences", type=int, default=8, dest="num_return_sequences")
    parser.add_argument("--max-new-tokens", type=int, default=256, dest="max_new_tokens")
    parser.add_argument("--runs-root", type=str, default=os.environ.get("EVAL_RUNS_ROOT", "eval_runs"), dest="runs_root")
    parser.add_argument("--profile", type=str, help="Evaluation profile name (e.g., 'quick_test', 'full_evaluation')")
    parser.add_argument("--list-profiles", action="store_true", help="List available evaluation profiles and exit")
    
    args = parser.parse_args()
    
    if args.list_profiles:
        from evals.profile_loader import print_profile_info
        print_profile_info()
        sys.exit(0)
    
    main(**vars(args))
