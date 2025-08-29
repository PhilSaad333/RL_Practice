#!/usr/bin/env python3
"""
Conservative B=4096 jackknife experiment with memory-safe parallelization.
"""

import json
import torch
import numpy as np
import random
from datetime import datetime
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

def sample_prompts_train(B, seed=None):
    """Sample B prompts from gsm8k_r1_template TRAIN split."""
    from rlp_datasets import DATASET_REGISTRY
    
    ds_builder = DATASET_REGISTRY["gsm8k_r1_template"]
    ds = ds_builder("train")
    
    print(f"Train dataset size: {len(ds)} samples")
    
    rng = random.Random(seed)
    pool = list(range(len(ds)))
    
    if B <= len(pool):
        idx = rng.sample(pool, B)
        print(f"Sampled {B} unique prompts from train set")
    else:
        idx = [pool[i % len(pool)] for i in range(B)]
        rng.shuffle(idx)
        print(f"WARNING: B={B} > dataset size, using repeated samples")
    
    prompts = [ds[i].question for i in idx]
    return prompts, idx

def generate_with_conservative_sequence_processor(model, tokenizer, prompts, G):
    """Generate using SequenceProcessor with CONSERVATIVE memory-safe settings."""
    from sequence_processing import SequenceProcessor, GenerationConfig
    
    # CONSERVATIVE SETTINGS - much smaller batches for B=4096
    config = GenerationConfig(
        temperature=1.0,
        top_p=1.0,
        max_new_tokens=200,
        do_sample=True,
        gen_batch_size=128,    # Same as successful extensive experiment  
        tf_batch_size=256      # Same as successful extensive experiment
    )
    
    print(f"Using CONSERVATIVE memory-safe config:")
    print(f"  gen_batch_size={config.gen_batch_size}")
    print(f"  tf_batch_size={config.tf_batch_size}")
    print(f"  Total sequences to generate: {len(prompts)} × {G} = {len(prompts)*G}")
    
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
    
    B = len(sequences.responses_text)
    for b in range(B):
        prompt_entropies = []
        
        for g in range(G):
            if (b < len(logprob_results.logprobs) and 
                g < len(logprob_results.logprobs[b])):
                
                token_logprobs = logprob_results.logprobs[b][g]
                if len(token_logprobs) > 0:
                    entropies = -token_logprobs.detach().cpu().numpy()
                else:
                    entropies = np.array([])
            else:
                entropies = np.array([])
            
            prompt_entropies.append(entropies)
        
        all_entropies.append(prompt_entropies)
    
    # Minimal sample info to save memory
    sample_info = {
        "first_prompt_length": len(prompts[0]) if prompts else 0,
        "generated_sequences": B * G,
        "memory_management": "conservative_batching_used"
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
                v = float(ent.mean())
            row_vals.append(v)
            per_seq_vals.append(v)
        
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
    leave_one_out = (total - x) / (B - 1)
    theta_bar = leave_one_out.mean()
    var = ((B - 1) / B) * np.square(leave_one_out - theta_bar).sum()
    return float(var)

def analyze_distribution(per_prompt_means):
    """Analyze distribution with focus on heavy tail."""
    x = np.array(per_prompt_means, dtype=float)
    finite_x = x[np.isfinite(x)]
    
    if len(finite_x) == 0:
        return {}
    
    return {
        "count": len(finite_x),
        "mean": float(np.mean(finite_x)),
        "std": float(np.std(finite_x)),
        "min": float(np.min(finite_x)),
        "q01": float(np.percentile(finite_x, 1)),
        "q05": float(np.percentile(finite_x, 5)),
        "q25": float(np.percentile(finite_x, 25)),
        "median": float(np.median(finite_x)),
        "q75": float(np.percentile(finite_x, 75)),
        "q90": float(np.percentile(finite_x, 90)),
        "q95": float(np.percentile(finite_x, 95)),
        "q99": float(np.percentile(finite_x, 99)),
        "max": float(np.max(finite_x))
    }

def main():
    """Run memory-safe B=4096 jackknife experiment."""
    print("=" * 80)
    print("CONSERVATIVE B=4096 JACKKNIFE EXPERIMENT (Memory-Safe)")
    print("=" * 80)
    print("Using proven working batch sizes from extensive experiment:")
    print("  gen_batch_size=128, tf_batch_size=256")
    print("=" * 80)
    
    B = 4096
    G = 8
    seed = None
    
    start_time = datetime.now()
    
    # Load model
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer()
    model.eval()
    
    # Check memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear cache first
        gpu_memory_used = torch.cuda.memory_allocated() / 1024**3
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory: {gpu_memory_used:.1f}GB / {gpu_memory_total:.1f}GB")
    
    # Sample prompts
    print(f"Sampling {B} prompts from train dataset...")
    set_global_seed(seed)
    prompts, indices = sample_prompts_train(B, seed=seed)
    
    # Generate with conservative settings
    print(f"Generating sequences with conservative batching...")
    gen_start = datetime.now()
    gen_entropies, sample_info = generate_with_conservative_sequence_processor(model, tokenizer, prompts, G)
    generation_duration = (datetime.now() - gen_start).total_seconds()
    
    print(f"Generation completed in {generation_duration:.1f}s ({generation_duration/60:.1f} min)")
    
    # Compute statistics
    print("Computing statistics...")
    batch_mean, per_seq_vals, per_prompt_means = compute_per_seq_entropy_stats(gen_entropies)
    var_jk = jackknife_variance(per_prompt_means)
    std_jk = float(np.sqrt(var_jk)) if not np.isnan(var_jk) else float("nan")
    distribution_stats = analyze_distribution(per_prompt_means)
    
    total_duration = (datetime.now() - start_time).total_seconds()
    
    # Results
    result = {
        "experiment_type": "conservative_jackknife_B4096_train",
        "dataset": "gsm8k_r1_template", 
        "split": "train",
        "B": B,
        "G": G,
        "seed": seed,
        "estimate_mean_per_seq_entropy": float(batch_mean),
        "variance_estimate": float(var_jk),
        "std_dev_estimate": std_jk,
        "distribution_analysis": distribution_stats,
        "performance": {
            "total_duration_seconds": total_duration,
            "generation_duration_seconds": generation_duration,
            "sequences_per_second": (B*G)/generation_duration if generation_duration > 0 else 0
        },
        "memory_management": {
            "approach": "conservative_batching",
            "gen_batch_size": 128,
            "tf_batch_size": 256,
            "reason": "avoid_oom_with_large_B"
        },
        "timestamp": datetime.now().isoformat()
    }
    
    # Save results
    output_file = "/home/ubuntu/RL_Practice/jackknife_B4096_conservative_results.jsonl"
    with open(output_file, "w") as f:
        f.write(json.dumps(result, indent=2))
    
    print(f"\nRESULTS:")
    print(f"Mean entropy: {batch_mean:.6f}")
    print(f"Jackknife std: {std_jk:.6f}")
    print(f"Duration: {total_duration/60:.1f} min")
    print(f"Distribution range: {distribution_stats['min']:.3f} - {distribution_stats['max']:.3f}")
    print(f"Heavy tail Q99: {distribution_stats['q99']:.3f}")
    print(f"Results saved: {output_file}")

if __name__ == "__main__":
    main()