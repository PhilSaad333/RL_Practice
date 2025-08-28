#!/usr/bin/env python
"""
Entropy Variance Estimator

Goal:
- Load a LoRA checkpoint
- Sample B prompts from a dataset
- Generate G responses per prompt
- Compute the expected value of per-sequence entropy (surprisal) over the batch
- Estimate variance either via:
  • Repeated estimates across N runs (optionally fixing prompts)
  • Jackknife over prompts from a single batch
- Log results to the same folder as this script

Notes:
- Reuses existing code paths as much as possible:
  • LoRA loader: entropy_experiments.offline_entropy_probe.load_peft_for_probe
  • Generation + per-token entropies: evals.utils_io.generate_with_logprobs
  • Dataset sampling: rlp_datasets.DATASET_REGISTRY

Definitions:
- Per-sequence entropy is defined here as the mean surprisal of generated tokens
  in a single completion (i.e., average of −log p(chosen_t) across the generated
  tokens kept by the StopAfterAnswer logic). The batch estimate is the mean of
  these per-sequence means across all B×G completions in the batch.

CLI Examples:
  python -m entropy_experiments.other_scripts.entropy_variance.entropy_variance \
      --adapter /path/to/lora/adapter_dir \
      --backbone Qwen/Qwen2.5-1.5B \
      --dataset gsm8k_r1_template --split test \
      --B 64 --G 8 --N 20 --method repeats --same-prompts \
      --max-new-tokens 64 --temperature 1.0 --top-p 1.0

  # Single-batch with jackknife variance over prompts
  python -m entropy_experiments.other_scripts.entropy_variance.entropy_variance \
      --adapter /path/to/lora/adapter_dir \
      --dataset gsm8k_r1_template --split test \
      --B 128 --G 4 --method jackknife

Outputs:
- JSON file: entropy_variance_YYYYmmdd_HHMMSS.json in this folder
  Contains estimate, variance, config, and prompt indices used.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import GenerationConfig, LogitsProcessorList

# Project imports (local package paths)
from entropy_experiments.offline_entropy_probe import load_peft_for_probe
from rlp_datasets import DATASET_REGISTRY
from evals.utils_io import StopAfterAnswer, generate_with_logprobs


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GenConfig:
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 1.0
    do_sample: bool = True
    pad_token_id: Optional[int] = None
    tf_micro_batch: int = 32


def set_global_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(backbone: str, adapter_path: str):
    """Load base + LoRA adapter using the probe's loader for consistency."""
    model = load_peft_for_probe(
        base_id=backbone,
        adapter_path=adapter_path,
        mode="lora_simple",
        dtype="bf16",  # Use bf16 like offline_entropy_probe
        device_map="cuda",
        use_checkpointing=False,
    )
    # Ensure adapter is active
    if hasattr(model, "set_adapter"):
        model.set_adapter("default")

    # Tokenizer: use the same base as the model's config id if available
    from transformers import AutoTokenizer

    # Try to infer tokenizer id from backbone
    tok = AutoTokenizer.from_pretrained(backbone, trust_remote_code=True)
    tok.padding_side = "left"
    tok.pad_token = tok.eos_token
    return model, tok


def sample_prompts(dataset_name: str, split: str, B: int, *, seed: Optional[int] = None,
                   fixed_indices: Optional[Sequence[int]] = None) -> Tuple[List[str], List[int]]:
    """Sample B prompts from the registry; optionally with fixed indices."""
    ds_builder = DATASET_REGISTRY[dataset_name]
    ds = ds_builder(split)

    if fixed_indices is not None:
        idx = list(fixed_indices)[:B]
        if len(idx) < B:
            idx = (idx * ((B // len(idx)) + 1))[:B]
    else:
        rng = random.Random(seed)
        pool = list(range(len(ds)))
        if B <= len(pool):
            idx = rng.sample(pool, B)
        else:
            # Repeat with wrap-around if dataset smaller than B
            idx = [pool[i % len(pool)] for i in range(B)]
            rng.shuffle(idx)

    prompts = [ds[i].question for i in idx]
    return prompts, idx


def compute_per_seq_entropy_stats(
    gen_entropies: List[List[np.ndarray]],
    *,
    agg: str = "mean",
) -> Tuple[float, List[float], List[float]]:
    """
    Compute per-sequence entropies and aggregate:
    - gen_entropies: [B][G] list of np arrays per token
    - agg: 'mean' (default) or 'sum' across tokens per sequence
    Returns (batch_mean, per_seq_vals, per_prompt_means)
    """
    B = len(gen_entropies)
    if B == 0:
        return float("nan"), [], []
    G = len(gen_entropies[0]) if gen_entropies[0] is not None else 0

    per_seq_vals: List[float] = []
    per_prompt_means: List[float] = []

    for b in range(B):
        row_vals: List[float] = []
        for g in range(G):
            ent = gen_entropies[b][g]
            if ent.size == 0:
                v = float("nan")
            else:
                if agg == "sum":
                    v = float(ent.sum())
                else:
                    v = float(ent.mean())
            row_vals.append(v)
            per_seq_vals.append(v)
        # per-prompt mean across its G sequences
        per_prompt_means.append(float(np.nanmean(row_vals)))

    batch_mean = float(np.nanmean(per_seq_vals))
    return batch_mean, per_seq_vals, per_prompt_means


def jackknife_variance(per_prompt_means: List[float]) -> float:
    """
    Jackknife variance over prompts for the batch estimator defined as the
    average of per-prompt means. If B prompts, θ̂ = (1/B)∑ μ_b, with μ_b being
    the mean over the G sequences for prompt b. Then jackknife estimate is:
      Var_jk(θ̂) = ((B - 1) / B) * ∑ (θ̂_(i) - θ̄_)^2
    where θ̂_(i) is the estimator leaving out prompt i, θ̄_ is their mean.
    """
    x = np.array(per_prompt_means, dtype=float)
    B = x.size
    if B <= 1 or not np.isfinite(x).all():
        return float("nan")

    total = x.sum()
    leave_one_out = (total - x) / (B - 1)  # vector of θ̂_(i)
    theta_bar = leave_one_out.mean()
    var = ((B - 1) / B) * np.square(leave_one_out - theta_bar).sum()
    return float(var)


def batched_generation(
    model,
    tokenizer,
    prompts: List[str],
    G: int,
    gen_cfg: GenConfig,
) -> Tuple[List[str], List[List[np.ndarray]]]:
    """Generate G responses per prompt with batched processing, return texts and entropies."""
    from transformers import GenerationConfig, LogitsProcessorList
    from evals.utils_io import StopAfterAnswer
    import torch.nn.functional as F
    
    # Generation config
    gc = GenerationConfig(
        max_new_tokens=gen_cfg.max_new_tokens,
        temperature=gen_cfg.temperature,
        top_p=gen_cfg.top_p,
        do_sample=gen_cfg.do_sample,
        num_return_sequences=G,
        pad_token_id=tokenizer.pad_token_id if gen_cfg.pad_token_id is None else gen_cfg.pad_token_id,
        return_dict_in_generate=True,
    )
    stop = LogitsProcessorList([StopAfterAnswer(tokenizer)])
    
    # Setup for batched generation (similar to probe_components)
    B = len(prompts)
    amp_dtype = torch.bfloat16 if model.dtype == torch.bfloat16 else torch.float16
    
    all_gen_texts = []
    all_entropies = []
    
    # Process in batches (simple approach: all prompts at once for now)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype):
        # Tokenize prompts with left padding
        tokenizer.padding_side = "left" 
        enc = tokenizer(prompts, padding=True, return_tensors="pt").to(model.device)
        prompt_len = enc.input_ids.shape[1]
        
        # Generate responses
        gen_out = model.generate(
            **enc,
            generation_config=gc,
            logits_processor=stop,
            return_dict_in_generate=True
        )
        
        # Reshape to [B, G, total_len]
        all_gen_sequences = gen_out.sequences
        sequences_reshaped = all_gen_sequences.view(B, G, -1)
        
        # Process each prompt
        for b in range(B):
            sequences = sequences_reshaped[b]  # [G, total_len]
            prompt_responses = []
            prompt_entropies = []
            
            for g in range(G):
                # Extract generated portion (skip prompt)
                gen_ids = sequences[g, prompt_len:]
                
                # Decode response text
                response_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                prompt_responses.append(response_text)
                
                # Compute per-token entropies via teacher forcing
                full_seq = sequences[g:g+1]  # [1, total_len] 
                with torch.no_grad():
                    logits = model(full_seq).logits  # [1, total_len, vocab_size]
                    
                # Apply temperature and compute log probs
                temp_logits = logits[0, prompt_len-1:-1] / gen_cfg.temperature  # [gen_len, vocab_size]
                log_probs = F.log_softmax(temp_logits, dim=-1)
                
                # Get actual token log probs
                actual_tokens = gen_ids[:len(temp_logits)]  # Handle potential length mismatch
                if len(actual_tokens) > 0:
                    token_log_probs = log_probs.gather(1, actual_tokens.unsqueeze(1)).squeeze(1)
                    # Convert to entropy (negative log prob)
                    entropies = -token_log_probs.float().cpu().numpy()
                else:
                    entropies = np.array([])
                
                prompt_entropies.append(entropies)
            
            all_gen_texts.extend(prompt_responses)
            all_entropies.append(prompt_entropies)
    
    return all_gen_texts, all_entropies


def one_batch_estimate(
    model,
    tokenizer,
    prompts: List[str],
    G: int,
    gen_cfg: GenConfig,
) -> Tuple[float, Dict[str, Any]]:
    """Generate G responses per prompt; return batch mean per-seq entropy and details."""
    
    # Use our custom batched generation
    gen_text, gen_entropies = batched_generation(model, tokenizer, prompts, G, gen_cfg)
    
    batch_mean, per_seq_vals, per_prompt_means = compute_per_seq_entropy_stats(gen_entropies, agg="mean")

    details = {
        "per_sequence_entropy_mean": batch_mean,
        "per_sequence_values": [float(v) for v in per_seq_vals],
        "per_prompt_means": [float(v) for v in per_prompt_means],
        "generated_text": gen_text,
    }
    return batch_mean, details


def repeats_variance(
    model,
    tokenizer,
    *,
    dataset: str,
    split: str,
    B: int,
    G: int,
    N: int,
    gen_cfg: GenConfig,
    same_prompts: bool,
    seed: Optional[int],
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Compute sample mean/variance across N independent estimates.
    - If same_prompts: sample a fixed set of prompts once; re-generate each run.
    - Else: resample prompts each run.
    Returns (mean_of_estimates, sample_variance, metadata)
    """
    set_global_seed(seed)

    fixed_prompts: Optional[List[str]] = None
    fixed_indices: Optional[List[int]] = None

    if same_prompts:
        fixed_prompts, fixed_indices = sample_prompts(dataset, split, B, seed=seed)

    estimates: List[float] = []
    runs_info: List[Dict[str, Any]] = []

    for k in range(N):
        run_seed = None if seed is None else seed + k + 1
        set_global_seed(run_seed)

        if same_prompts:
            prompts = list(fixed_prompts)  # copy
            idx = list(fixed_indices)
        else:
            prompts, idx = sample_prompts(dataset, split, B, seed=run_seed)

        mean_k, details = one_batch_estimate(model, tokenizer, prompts, G, gen_cfg)
        estimates.append(mean_k)
        runs_info.append({
            "seed": run_seed,
            "estimate": float(mean_k),
            "indices": idx,
        })

    arr = np.array(estimates, dtype=float)
    mean_est = float(arr.mean()) if arr.size > 0 else float("nan")
    var_est = float(arr.var(ddof=1)) if arr.size > 1 else float("nan")

    meta = {
        "runs": runs_info,
        "estimates": [float(x) for x in arr.tolist()],
        "fixed_indices": fixed_indices if same_prompts else None,
    }
    return mean_est, var_est, meta


def single_batch_jackknife(
    model,
    tokenizer,
    *,
    dataset: str,
    split: str,
    B: int,
    G: int,
    gen_cfg: GenConfig,
    seed: Optional[int],
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Single batch estimate with jackknife variance over prompts.
    Returns (batch_mean, jackknife_variance, metadata)
    """
    set_global_seed(seed)
    prompts, idx = sample_prompts(dataset, split, B, seed=seed)
    batch_mean, details = one_batch_estimate(model, tokenizer, prompts, G, gen_cfg)
    per_prompt_means = details["per_prompt_means"]
    var_jk = jackknife_variance(per_prompt_means)
    meta = {
        "indices": idx,
        "per_prompt_means": per_prompt_means,
    }
    return batch_mean, var_jk, meta


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-sequence entropy variance estimator")
    p.add_argument("--adapter", required=True, type=str, help="Path to LoRA adapter directory")
    p.add_argument("--backbone", default="Qwen/Qwen2.5-1.5B", type=str, help="Base model id or local path")

    p.add_argument("--dataset", default="gsm8k_r1_template", type=str, help="Dataset registry name")
    p.add_argument("--split", default="test", type=str, help="Dataset split")

    p.add_argument("--B", type=int, required=True, help="# of prompts")
    p.add_argument("--G", type=int, required=True, help="# of generations per prompt")

    p.add_argument("--method", choices=["repeats", "jackknife"], default="repeats",
                   help="Variance estimation method")
    p.add_argument("--N", type=int, default=10, help="# of repeated runs (for method=repeats)")
    p.add_argument("--same-prompts", action="store_true", help="Fix prompts across repeats (method=repeats)")

    # Generation and TF config
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--no-sampling", action="store_true", help="Disable sampling (do_sample=False)")
    p.add_argument("--tf-micro-batch", type=int, default=32, help="Teacher-forcing micro-batch size")

    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out-prefix", type=str, default="entropy_variance", help="Output filename prefix")
    return p.parse_args()


def main():
    args = parse_args()

    out_dir = Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{args.out_prefix}_{ts}.json"

    # Model + tokenizer
    model, tok = load_model_and_tokenizer(args.backbone, args.adapter)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    gen_cfg = GenConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=not args.no_sampling,
        pad_token_id=tok.pad_token_id,
        tf_micro_batch=args.tf_micro_batch,
    )

    start = time.time()

    if args.method == "repeats":
        mean_est, var_est, meta = repeats_variance(
            model,
            tok,
            dataset=args.dataset,
            split=args.split,
            B=args.B,
            G=args.G,
            N=args.N,
            gen_cfg=gen_cfg,
            same_prompts=args.same_prompts,
            seed=args.seed,
        )
        result = {
            "method": "repeats",
            "B": args.B,
            "G": args.G,
            "N": args.N,
            "same_prompts": bool(args.same_prompts),
            "estimate_mean_per_seq_entropy": float(mean_est),
            "variance_estimate": float(var_est),
            "dataset": args.dataset,
            "split": args.split,
            "backbone": args.backbone,
            "adapter": args.adapter,
            "generation": asdict(gen_cfg),
            "seed": args.seed,
            "meta": meta,
            "runtime_sec": time.time() - start,
        }
    else:
        batch_mean, var_jk, meta = single_batch_jackknife(
            model,
            tok,
            dataset=args.dataset,
            split=args.split,
            B=args.B,
            G=args.G,
            gen_cfg=gen_cfg,
            seed=args.seed,
        )
        result = {
            "method": "jackknife",
            "B": args.B,
            "G": args.G,
            "estimate_mean_per_seq_entropy": float(batch_mean),
            "variance_estimate": float(var_jk),
            "dataset": args.dataset,
            "split": args.split,
            "backbone": args.backbone,
            "adapter": args.adapter,
            "generation": asdict(gen_cfg),
            "seed": args.seed,
            "meta": meta,
            "runtime_sec": time.time() - start,
        }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Saved results to: {out_path}")
    print(json.dumps({k: v for k, v in result.items() if k not in ("meta",)}, indent=2))


if __name__ == "__main__":
    main()

