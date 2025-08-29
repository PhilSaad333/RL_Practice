#!/usr/bin/env python3
"""
Run comprehensive jackknife variance experiments for B=16, 32, 64.
Save all results to a single JSONL file.
"""

import json
import torch
import numpy as np
import random
from datetime import datetime
from pathlib import Path
import sys
sys.path.append('/home/ubuntu/RL_Practice')

def set_global_seed(seed):
    """Set global seed if provided."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer():
    """Load model and tokenizer."""
    from entropy_experiments.offline_entropy_probe import load_peft_for_probe
    from transformers import AutoTokenizer
    
    checkpoint_path = "/home/ubuntu/localfs/training_runs/run_2025-08-24_22-13-22/training_state/step_40/model"
    base_model_id = "Qwen/Qwen2.5-1.5B"
    
    model = load_peft_for_probe(
        base_id=base_model_id,
        adapter_path=checkpoint_path,
        mode="lora_simple",
        dtype="bf16",
        device_map="cuda",
    )
    
    if hasattr(model, "set_adapter"):
        model.set_adapter("default")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def sample_prompts(B, seed=None):
    """Sample B prompts from gsm8k_r1_template."""
    from rlp_datasets import DATASET_REGISTRY
    
    ds_builder = DATASET_REGISTRY["gsm8k_r1_template"]
    ds = ds_builder("test")
    
    rng = random.Random(seed)
    pool = list(range(len(ds)))
    if B <= len(pool):
        idx = rng.sample(pool, B)
    else:
        idx = [pool[i % len(pool)] for i in range(B)]
        rng.shuffle(idx)
    
    prompts = [ds[i].question for i in idx]
    return prompts, idx

def generate_with_sequence_processor(model, tokenizer, prompts, G):
    """Generate using SequenceProcessor."""
    from sequence_processing import SequenceProcessor, GenerationConfig
    
    # Create config - NO SEED anywhere
    config = GenerationConfig(
        temperature=1.0,
        top_p=1.0,
        max_new_tokens=200,
        do_sample=True,
        gen_batch_size=32,
        tf_batch_size=64
    )
    
    # Initialize processor
    processor = SequenceProcessor(model, tokenizer, config)
    
    # Generate sequences and compute logprobs
    sequences, logprob_results = processor.generate_with_logprobs(
        prompts=prompts,
        G=G,
        with_grad=False
    )
    
    # Extract entropy data properly
    all_entropies = []
    sample_info = None
    
    B = len(sequences.responses_text)
    for b in range(B):
        prompt_entropies = []
        
        for g in range(G):
            # Get logprobs for this sequence
            if (b < len(logprob_results.logprobs) and 
                g < len(logprob_results.logprobs[b])):
                
                token_logprobs = logprob_results.logprobs[b][g]
                if len(token_logprobs) > 0:
                    # Convert logprobs to entropies (negative logprobs)
                    entropies = -token_logprobs.detach().cpu().numpy()
                else:
                    entropies = np.array([])
            else:
                entropies = np.array([])
            
            prompt_entropies.append(entropies)
        
        all_entropies.append(prompt_entropies)
        
        # Capture sample for first batch size only
        if b == 0 and sample_info is None:
            sample_info = {
                "prompt": prompts[0],
                "response": sequences.responses_text[0][0] if len(sequences.responses_text[0]) > 0 else "",
                "prompt_length": sequences.prompt_lens[0] if len(sequences.prompt_lens) > 0 else 0,
                "response_length": sequences.gen_lens[0][0] if (len(sequences.gen_lens) > 0 and len(sequences.gen_lens[0]) > 0) else 0
            }
            if len(prompt_entropies) > 0 and len(prompt_entropies[0]) > 0:
                sample_info.update({
                    "entropy_mean": float(prompt_entropies[0].mean()),
                    "entropy_std": float(prompt_entropies[0].std()),
                    "entropy_length": len(prompt_entropies[0])
                })
    
    return all_entropies, sample_info

def compute_per_seq_entropy_stats(gen_entropies):
    """Compute per-sequence entropy statistics."""
    B = len(gen_entropies)
    if B == 0:
        return float("nan"), [], []
    
    per_seq_vals = []
    per_prompt_means = []
    
    for b in range(B):
        row_vals = []
        for g in range(len(gen_entropies[b])):
            ent = gen_entropies[b][g]
            if ent.size == 0:
                v = float("nan")
            else:
                v = float(ent.mean())  # Mean entropy per sequence
            row_vals.append(v)
            per_seq_vals.append(v)
        
        # Per-prompt mean across its G sequences
        per_prompt_means.append(float(np.nanmean(row_vals)))
    
    batch_mean = float(np.nanmean(per_seq_vals))
    return batch_mean, per_seq_vals, per_prompt_means

def jackknife_variance(per_prompt_means):
    """Jackknife variance estimator."""
    x = np.array(per_prompt_means, dtype=float)
    B = x.size
    if B <= 1 or not np.isfinite(x).all():
        return float("nan")
    
    total = x.sum()
    leave_one_out = (total - x) / (B - 1)  # θ̂_(i)
    theta_bar = leave_one_out.mean()
    var = ((B - 1) / B) * np.square(leave_one_out - theta_bar).sum()
    return float(var)

def run_single_experiment(model, tokenizer, B, G=8):
    """Run a single jackknife experiment for given B."""
    print(f"\n=== Running B={B} experiment ===")
    
    # IMPORTANT: No seed anywhere
    seed = None
    
    # Sample prompts (completely unseeded)
    set_global_seed(seed)  # This does nothing since seed=None
    prompts, indices = sample_prompts(B, seed=seed)
    print(f"Sampled {len(prompts)} prompts")
    
    # Generate responses
    print("Generating responses with SequenceProcessor...")
    gen_entropies, sample_info = generate_with_sequence_processor(model, tokenizer, prompts, G)
    print(f"Generated {len(gen_entropies)} prompt batches")
    
    # Compute statistics
    batch_mean, per_seq_vals, per_prompt_means = compute_per_seq_entropy_stats(gen_entropies)
    
    # Jackknife variance
    var_jk = jackknife_variance(per_prompt_means)
    std_jk = float(np.sqrt(var_jk)) if not np.isnan(var_jk) else float("nan")
    
    print(f"Mean entropy: {batch_mean:.6f}")
    print(f"Jackknife variance: {var_jk:.8f}")
    print(f"Jackknife std dev: {std_jk:.6f}")
    
    return {
        "B": B,
        "G": G, 
        "seed": seed,  # Always None
        "estimate_mean_per_seq_entropy": float(batch_mean),
        "variance_estimate": float(var_jk),
        "std_dev_estimate": std_jk,
        "per_prompt_means": [float(v) for v in per_prompt_means],
        "indices": indices,
        "sample_prompt_and_response": sample_info if B == 16 else None,  # Only save sample for first experiment
        "num_sequences_total": B * G,
        "num_finite_per_prompt_means": int(np.isfinite(per_prompt_means).sum())
    }

def main():
    """Run comprehensive jackknife experiments for B=16,32,64."""
    print("=== Comprehensive Jackknife Variance Experiments ===")
    print("Running B=16, 32, 64 with NO random seeds")
    print("max_new_tokens=200, G=8")
    
    # Load model once
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer()
    model.eval()
    
    # Run experiments for each batch size
    batch_sizes = [16, 32, 64]
    results = []
    
    for B in batch_sizes:
        experiment_result = run_single_experiment(model, tokenizer, B)
        results.append(experiment_result)
    
    # Create comprehensive results
    comprehensive_result = {
        "experiment_type": "jackknife_variance_comprehensive",
        "dataset": "gsm8k_r1_template",
        "split": "test",
        "checkpoint": "/home/ubuntu/localfs/training_runs/run_2025-08-24_22-13-22/training_state/step_40/model",
        "base_model": "Qwen/Qwen2.5-1.5B",
        "generation_config": {
            "max_new_tokens": 200,
            "temperature": 1.0,
            "top_p": 1.0,
            "do_sample": True,
            "G": 8
        },
        "seed_policy": "NO_SEED_USED",  # Explicitly document this
        "timestamp": datetime.now().isoformat(),
        "experiments": results,
        "summary": {
            "batch_sizes": batch_sizes,
            "mean_entropies": [r["estimate_mean_per_seq_entropy"] for r in results],
            "std_devs": [r["std_dev_estimate"] for r in results],
            "variances": [r["variance_estimate"] for r in results]
        }
    }
    
    # Save results
    output_file = "/home/ubuntu/RL_Practice/jackknife_comprehensive_results.jsonl"
    with open(output_file, "w") as f:
        f.write(json.dumps(comprehensive_result, indent=2))
    
    print(f"\n✅ Comprehensive results saved to: {output_file}")
    
    # Summary table
    print(f"\n=== FINAL SUMMARY ===")
    print(f"{'B':>3} {'Mean Entropy':>12} {'Std Dev':>10} {'Variance':>12}")
    print("-" * 40)
    for r in results:
        print(f"{r['B']:>3} {r['estimate_mean_per_seq_entropy']:>12.6f} {r['std_dev_estimate']:>10.6f} {r['variance_estimate']:>12.8f}")

if __name__ == "__main__":
    main()