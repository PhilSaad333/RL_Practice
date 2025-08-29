#!/usr/bin/env python3
"""
Run naive variance estimator for B=64, N=16 as cross-check against jackknife.
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

def generate_and_compute_entropy(model, tokenizer, prompts, G):
    """Generate and compute mean per-sequence entropy for a batch."""
    from sequence_processing import SequenceProcessor, GenerationConfig
    
    config = GenerationConfig(
        temperature=1.0,
        top_p=1.0,
        max_new_tokens=200,
        do_sample=True,
        gen_batch_size=32,
        tf_batch_size=64
    )
    
    processor = SequenceProcessor(model, tokenizer, config)
    sequences, logprob_results = processor.generate_with_logprobs(
        prompts=prompts,
        G=G,
        with_grad=False
    )
    
    # Compute per-sequence entropies
    per_seq_vals = []
    
    B = len(sequences.responses_text)
    for b in range(B):
        for g in range(G):
            if (b < len(logprob_results.logprobs) and 
                g < len(logprob_results.logprobs[b])):
                
                token_logprobs = logprob_results.logprobs[b][g]
                if len(token_logprobs) > 0:
                    entropies = -token_logprobs.detach().cpu().numpy()
                    per_seq_entropy = float(entropies.mean())
                    per_seq_vals.append(per_seq_entropy)
    
    # Return batch mean
    return float(np.nanmean(per_seq_vals)) if per_seq_vals else float("nan")

def main():
    """Run naive variance estimator with B=64, N=16."""
    print("=== Naive Variance Estimator Cross-Check ===")
    print("B=64, N=16, NO random seeds")
    
    B = 64
    G = 8 
    N = 16
    seed = None  # NO SEED
    
    print(f"B={B}, G={G}, N={N}, seed={seed}")
    
    # Load model
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer()
    model.eval()
    
    # Run N independent experiments
    estimates = []
    
    for run in range(N):
        print(f"\nRun {run+1}/{N}")
        
        # NO SEED - completely independent sampling
        set_global_seed(seed)  # Does nothing since seed=None
        prompts, indices = sample_prompts(B, seed=seed)
        
        # Generate and compute batch mean entropy
        batch_mean = generate_and_compute_entropy(model, tokenizer, prompts, G)
        estimates.append(batch_mean)
        
        print(f"  Batch mean entropy: {batch_mean:.6f}")
    
    # Compute naive variance estimate
    estimates_array = np.array(estimates)
    mean_estimate = float(estimates_array.mean())
    naive_variance = float(estimates_array.var(ddof=1))  # Sample variance
    naive_std = float(np.sqrt(naive_variance))
    
    result = {
        "method": "naive_variance",
        "dataset": "gsm8k_r1_template", 
        "split": "test",
        "B": B,
        "G": G,
        "N": N,
        "seed": seed,
        "checkpoint": "/home/ubuntu/localfs/training_runs/run_2025-08-24_22-13-22/training_state/step_40/model",
        "generation_config": {
            "max_new_tokens": 200,
            "temperature": 1.0,
            "top_p": 1.0,
            "do_sample": True
        },
        "estimate_mean_per_seq_entropy": mean_estimate,
        "variance_estimate": naive_variance,
        "std_dev_estimate": naive_std,
        "individual_estimates": [float(x) for x in estimates],
        "timestamp": datetime.now().isoformat(),
        "note": "Cross-check against jackknife B=64 result"
    }
    
    # Save results
    output_file = "/home/ubuntu/RL_Practice/naive_variance_B64_results.jsonl"
    with open(output_file, "w") as f:
        f.write(json.dumps(result, indent=2))
    
    print(f"\n✅ Results saved to: {output_file}")
    print(f"\n=== NAIVE VARIANCE RESULTS (B={B}, N={N}) ===")
    print(f"Mean estimate: {mean_estimate:.6f}")
    print(f"Variance: {naive_variance:.8f}")
    print(f"Std dev: {naive_std:.6f}")
    print(f"Individual estimates: {[f'{x:.4f}' for x in estimates[:5]]}...")

if __name__ == "__main__":
    main()