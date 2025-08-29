#!/usr/bin/env python3
"""
Large-scale jackknife variance experiment with B=4096 on TRAIN dataset.

Tests the heavy tail hypothesis: at B=4096, we should sample extensively from 
the right tail of the per-prompt entropy distribution, giving us a high mean
but stable jackknife variance estimate.
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
    ds = ds_builder("train")  # Use TRAIN split for larger dataset
    
    print(f"Train dataset size: {len(ds)} samples")
    
    rng = random.Random(seed)
    pool = list(range(len(ds)))
    
    if B <= len(pool):
        idx = rng.sample(pool, B)
        print(f"Sampled {B} unique prompts from train set")
    else:
        # This shouldn't happen with B=4096 and train set, but handle it
        idx = [pool[i % len(pool)] for i in range(B)]
        rng.shuffle(idx)
        print(f"WARNING: B={B} > dataset size, using repeated samples")
    
    prompts = [ds[i].question for i in idx]
    return prompts, idx

def generate_with_optimized_sequence_processor(model, tokenizer, prompts, G):
    """Generate using SequenceProcessor with maximum optimization for B=4096."""
    from sequence_processing import SequenceProcessor, GenerationConfig
    
    # MAXIMUM OPTIMIZATION - even larger batches for B=4096
    config = GenerationConfig(
        temperature=1.0,
        top_p=1.0,
        max_new_tokens=200,
        do_sample=True,
        gen_batch_size=256,    # 8x increase for massive scale
        tf_batch_size=512      # 8x increase for massive scale  
    )
    
    print(f"Using maximum optimization: gen_batch_size={config.gen_batch_size}, tf_batch_size={config.tf_batch_size}")
    
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
    
    # Create sample info for first prompt only (to save memory)
    sample_info = {
        "first_prompt": prompts[0][:200] + "..." if len(prompts[0]) > 200 else prompts[0],
        "first_response": (sequences.responses_text[0][0][:200] + "...") if len(sequences.responses_text[0]) > 0 and len(sequences.responses_text[0][0]) > 200 else (sequences.responses_text[0][0] if len(sequences.responses_text[0]) > 0 else ""),
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

def analyze_distribution(per_prompt_means):
    """Analyze the distribution of per-prompt means for tail characterization."""
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
        "q10": float(np.percentile(finite_x, 10)),
        "q25": float(np.percentile(finite_x, 25)),
        "median": float(np.median(finite_x)),
        "q75": float(np.percentile(finite_x, 75)),
        "q90": float(np.percentile(finite_x, 90)),
        "q95": float(np.percentile(finite_x, 95)),
        "q99": float(np.percentile(finite_x, 99)),
        "max": float(np.max(finite_x)),
        "range": float(np.max(finite_x) - np.min(finite_x)),
        "iqr": float(np.percentile(finite_x, 75) - np.percentile(finite_x, 25)),
        "skewness": float(float(np.mean((finite_x - np.mean(finite_x))**3)) / (np.std(finite_x)**3)),
        "kurtosis": float(float(np.mean((finite_x - np.mean(finite_x))**4)) / (np.std(finite_x)**4) - 3),
    }

def main():
    """Run large-scale jackknife experiment with B=4096 on train dataset."""
    print("=" * 80)
    print("LARGE-SCALE JACKKNIFE EXPERIMENT: B=4096 on TRAIN DATASET")
    print("=" * 80)
    print("Testing heavy tail hypothesis with maximum batch size")
    print("Using TRAIN split for sufficient dataset size")
    print("Maximum parallelization: gen_batch_size=256, tf_batch_size=512")
    print("NO random seeds used")
    print("=" * 80)
    
    B = 4096
    G = 8
    seed = None  # NO SEED
    
    print(f"Parameters: B={B}, G={G}, seed={seed}")
    print(f"Expected total sequences: {B*G}")
    
    # Load model
    print("\n1. Loading model...")
    model, tokenizer = load_model_and_tokenizer()
    model.eval()
    
    # Check initial memory
    if torch.cuda.is_available():
        gpu_memory_used = torch.cuda.memory_allocated() / 1024**3
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Initial GPU Memory: {gpu_memory_used:.1f}GB / {gpu_memory_total:.1f}GB ({100*gpu_memory_used/gpu_memory_total:.1f}%)")
    
    # Sample prompts from TRAIN dataset
    print(f"\n2. Sampling {B} prompts from train dataset...")
    start_time = datetime.now()
    set_global_seed(seed)
    prompts, indices = sample_prompts_train(B, seed=seed)
    
    # Generate responses
    print(f"\n3. Generating {B}×{G}={B*G} sequences with maximum optimization...")
    gen_start = datetime.now()
    gen_entropies, sample_info = generate_with_optimized_sequence_processor(model, tokenizer, prompts, G)
    generation_duration = (datetime.now() - gen_start).total_seconds()
    
    # Check memory after generation
    if torch.cuda.is_available():
        gpu_memory_after = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU Memory after generation: {gpu_memory_after:.1f}GB ({100*gpu_memory_after/gpu_memory_total:.1f}%)")
    
    print(f"Generation completed in {generation_duration:.1f}s ({generation_duration/60:.1f} min)")
    print(f"Throughput: {(B*G)/generation_duration:.1f} sequences/second")
    
    # Compute statistics
    print(f"\n4. Computing entropy statistics and jackknife variance...")
    stats_start = datetime.now()
    batch_mean, per_seq_vals, per_prompt_means = compute_per_seq_entropy_stats(gen_entropies)
    
    # Jackknife variance
    var_jk = jackknife_variance(per_prompt_means)
    std_jk = float(np.sqrt(var_jk)) if not np.isnan(var_jk) else float("nan")
    
    # Distribution analysis
    distribution_stats = analyze_distribution(per_prompt_means)
    
    stats_duration = (datetime.now() - stats_start).total_seconds()
    total_duration = (datetime.now() - start_time).total_seconds()
    
    print(f"Statistics computed in {stats_duration:.1f}s")
    
    # Results
    result = {
        "experiment_type": "large_scale_jackknife_B4096_train",
        "dataset": "gsm8k_r1_template",
        "split": "train",  # TRAIN split used
        "B": B,
        "G": G,
        "seed": seed,
        "checkpoint": "/home/ubuntu/localfs/training_runs/run_2025-08-24_22-13-22/training_state/step_40/model",
        "base_model": "Qwen/Qwen2.5-1.5B",
        "generation_config": {
            "max_new_tokens": 200,
            "temperature": 1.0,
            "top_p": 1.0,
            "do_sample": True,
            "G": G
        },
        "parallelization_config": {
            "gen_batch_size": 256,
            "tf_batch_size": 512,
            "optimization_level": "maximum_8x"
        },
        "estimate_mean_per_seq_entropy": float(batch_mean),
        "variance_estimate": float(var_jk),
        "std_dev_estimate": std_jk,
        "num_sequences_total": B * G,
        "num_finite_per_prompt_means": int(np.isfinite(per_prompt_means).sum()),
        "distribution_analysis": distribution_stats,
        "sample_prompt_and_response": sample_info,
        "performance_metrics": {
            "total_duration_seconds": total_duration,
            "generation_duration_seconds": generation_duration,
            "statistics_duration_seconds": stats_duration,
            "sequences_per_second": (B*G)/generation_duration if generation_duration > 0 else 0,
            "prompts_per_second": B/generation_duration if generation_duration > 0 else 0
        },
        "timestamp": datetime.now().isoformat(),
        "heavy_tail_hypothesis": "Testing if B=4096 samples extensively from right tail of entropy distribution"
    }
    
    # Save results (without per_prompt_means to keep file size reasonable)
    output_file = "/home/ubuntu/RL_Practice/jackknife_B4096_train_results.jsonl"
    with open(output_file, "w") as f:
        f.write(json.dumps(result, indent=2))
    
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"Dataset: GSM8K R1 Template (TRAIN split)")
    print(f"Batch size: {B} prompts")
    print(f"Total sequences: {B*G}")
    print(f"Duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")
    print(f"Throughput: {(B*G)/generation_duration:.1f} sequences/sec")
    print()
    print(f"Mean per-sequence entropy: {batch_mean:.6f}")
    print(f"Jackknife variance: {var_jk:.8f}")
    print(f"Jackknife std dev: {std_jk:.6f}")
    print()
    print("Distribution characteristics:")
    print(f"  Min: {distribution_stats['min']:.4f}")
    print(f"  Q01: {distribution_stats['q01']:.4f}")
    print(f"  Q25: {distribution_stats['q25']:.4f}")
    print(f"  Median: {distribution_stats['median']:.4f}")
    print(f"  Q75: {distribution_stats['q75']:.4f}")
    print(f"  Q99: {distribution_stats['q99']:.4f}")
    print(f"  Max: {distribution_stats['max']:.4f}")
    print(f"  Skewness: {distribution_stats['skewness']:.4f}")
    print(f"  Kurtosis: {distribution_stats['kurtosis']:.4f}")
    print()
    print(f"Results saved to: {output_file}")
    print("✅ Large-scale jackknife experiment completed successfully!")

if __name__ == "__main__":
    main()