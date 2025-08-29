#!/usr/bin/env python3
"""
Extensive jackknife variance experiments with optimized parallelization.

Runs B = 32, 64, 128, 256, 512, 1024 with 4x larger batch sizes to utilize GPU efficiently.
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

def generate_with_optimized_sequence_processor(model, tokenizer, prompts, G):
    """Generate using SequenceProcessor with optimized parallelization."""
    from sequence_processing import SequenceProcessor, GenerationConfig
    
    # OPTIMIZED CONFIG - 4x larger batch sizes
    config = GenerationConfig(
        temperature=1.0,
        top_p=1.0,
        max_new_tokens=200,
        do_sample=True,
        gen_batch_size=128,    # 4x increase: 32 -> 128
        tf_batch_size=256      # 4x increase: 64 -> 256
    )
    
    print(f"  Using optimized config: gen_batch_size={config.gen_batch_size}, tf_batch_size={config.tf_batch_size}")
    
    # Initialize processor
    processor = SequenceProcessor(model, tokenizer, config)
    
    # Generate sequences and compute logprobs
    sequences, logprob_results = processor.generate_with_logprobs(
        prompts=prompts,
        G=G,
        with_grad=False
    )
    
    # Extract entropy data
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
        
        # Capture sample for first experiment only (B=32)
        if b == 0 and sample_info is None:
            sample_info = {
                "prompt": prompts[0][:200] + "..." if len(prompts[0]) > 200 else prompts[0],
                "response": (sequences.responses_text[0][0][:200] + "...") if len(sequences.responses_text[0]) > 0 and len(sequences.responses_text[0][0]) > 200 else (sequences.responses_text[0][0] if len(sequences.responses_text[0]) > 0 else ""),
                "prompt_length": sequences.prompt_lens[0] if len(sequences.prompt_lens) > 0 else 0,
                "response_length": sequences.gen_lens[0][0] if (len(sequences.gen_lens) > 0 and len(sequences.gen_lens[0]) > 0) else 0
            }
    
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
    print(f"\n{'='*50}")
    print(f"EXPERIMENT: B={B}")
    print(f"{'='*50}")
    
    # Check memory before starting
    if torch.cuda.is_available():
        gpu_memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        print(f"GPU Memory before: {gpu_memory_used:.1f}GB / {gpu_memory_total:.1f}GB ({100*gpu_memory_used/gpu_memory_total:.1f}%)")
    
    # IMPORTANT: No seed anywhere
    seed = None
    
    # Sample prompts (completely unseeded)
    print(f"Sampling {B} prompts...")
    set_global_seed(seed)  # This does nothing since seed=None
    prompts, indices = sample_prompts(B, seed=seed)
    
    # Generate responses with optimized parallelization
    print(f"Generating {B}×{G}={B*G} sequences...")
    gen_entropies, sample_info = generate_with_optimized_sequence_processor(model, tokenizer, prompts, G)
    
    # Check memory after generation
    if torch.cuda.is_available():
        gpu_memory_used_after = torch.cuda.memory_allocated() / 1024**3  # GB
        print(f"GPU Memory after: {gpu_memory_used_after:.1f}GB / {gpu_memory_total:.1f}GB ({100*gpu_memory_used_after/gpu_memory_total:.1f}%)")
    
    print(f"Computing statistics...")
    
    # Compute statistics
    batch_mean, per_seq_vals, per_prompt_means = compute_per_seq_entropy_stats(gen_entropies)
    
    # Jackknife variance
    var_jk = jackknife_variance(per_prompt_means)
    std_jk = float(np.sqrt(var_jk)) if not np.isnan(var_jk) else float("nan")
    
    print(f"Results:")
    print(f"  Mean entropy: {batch_mean:.6f}")
    print(f"  Jackknife variance: {var_jk:.8f}")
    print(f"  Jackknife std dev: {std_jk:.6f}")
    print(f"  Total sequences: {len(per_seq_vals)}")
    print(f"  Finite per-prompt means: {np.isfinite(per_prompt_means).sum()}/{len(per_prompt_means)}")
    
    return {
        "B": B,
        "G": G, 
        "seed": seed,  # Always None
        "estimate_mean_per_seq_entropy": float(batch_mean),
        "variance_estimate": float(var_jk),
        "std_dev_estimate": std_jk,
        "per_prompt_means": [float(v) for v in per_prompt_means],
        "indices": indices[:20],  # Save only first 20 indices to reduce file size
        "sample_prompt_and_response": sample_info if B == 32 else None,  # Only for first experiment
        "num_sequences_total": B * G,
        "num_finite_per_prompt_means": int(np.isfinite(per_prompt_means).sum()),
        "parallelization_config": {
            "gen_batch_size": 128,
            "tf_batch_size": 256
        }
    }

def main():
    """Run extensive jackknife experiments with optimized parallelization."""
    print("=" * 80)
    print("EXTENSIVE JACKKNIFE VARIANCE EXPERIMENTS")
    print("=" * 80)
    print("Batch sizes: B = 32, 64, 128, 256, 512, 1024")
    print("Optimized parallelization: gen_batch_size=128, tf_batch_size=256")
    print("NO random seeds used")
    print("max_new_tokens=200, G=8")
    print("=" * 80)
    
    # Load model once
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer()
    model.eval()
    
    # Run experiments for each batch size
    batch_sizes = [32, 64, 128, 256, 512, 1024]
    results = []
    
    start_time = datetime.now()
    
    for i, B in enumerate(batch_sizes):
        exp_start = datetime.now()
        experiment_result = run_single_experiment(model, tokenizer, B)
        exp_duration = (datetime.now() - exp_start).total_seconds()
        
        experiment_result["experiment_duration_seconds"] = exp_duration
        results.append(experiment_result)
        
        print(f"Experiment {i+1}/{len(batch_sizes)} completed in {exp_duration:.1f}s")
        
        # Clear cache to prevent memory accumulation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    total_duration = (datetime.now() - start_time).total_seconds()
    
    # Create comprehensive results
    comprehensive_result = {
        "experiment_type": "extensive_jackknife_variance_optimized",
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
        "parallelization_optimization": {
            "gen_batch_size": 128,
            "tf_batch_size": 256,
            "optimization_factor": "4x increase from conservative defaults"
        },
        "seed_policy": "NO_SEED_USED",
        "timestamp": datetime.now().isoformat(),
        "total_duration_seconds": total_duration,
        "experiments": results,
        "summary": {
            "batch_sizes": batch_sizes,
            "mean_entropies": [r["estimate_mean_per_seq_entropy"] for r in results],
            "std_devs": [r["std_dev_estimate"] for r in results],
            "variances": [r["variance_estimate"] for r in results],
            "total_sequences_processed": sum(r["num_sequences_total"] for r in results)
        }
    }
    
    # Save results
    output_file = "/home/ubuntu/RL_Practice/extensive_jackknife_results.jsonl"
    with open(output_file, "w") as f:
        f.write(json.dumps(comprehensive_result, indent=2))
    
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Total duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")
    print(f"Results saved to: {output_file}")
    
    # Summary table
    print(f"\n{'B':>5} {'Mean Entropy':>12} {'Std Dev':>10} {'Variance':>12} {'Sequences':>9} {'Time(s)':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r['B']:>5} {r['estimate_mean_per_seq_entropy']:>12.6f} {r['std_dev_estimate']:>10.6f} {r['variance_estimate']:>12.8f} {r['num_sequences_total']:>9} {r['experiment_duration_seconds']:>8.1f}")
    
    print(f"\nTotal sequences processed: {comprehensive_result['summary']['total_sequences_processed']}")
    print("✅ All experiments completed successfully!")

if __name__ == "__main__":
    main()