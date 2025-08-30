#!/usr/bin/env python3
"""
Debug script to investigate entropy computation issues.
"""

import json
import numpy as np
import pandas as pd

def debug_entropy_data(results_file="entropy_experiments/estimation_experiments/data/entropy_study_1.json"):
    """Debug the entropy computation issues."""
    
    print("🔍 DEBUGGING ENTROPY COMPUTATION ISSUES")
    print("=" * 80)
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    per_seq_data = results['per_sequence_data']
    df = pd.DataFrame(per_seq_data)
    
    print(f"Total sequences: {len(df)}")
    print()
    
    # Compare the different entropy measures
    entropy_columns = [col for col in df.columns if 'entropy' in col.lower()]
    print("Available entropy columns:")
    for col in entropy_columns:
        print(f"  {col}")
    print()
    
    # Focus on the problematic ones
    basic_sum = df['basic_entropy_sum'].values
    rb_sum = df['rb_entropy_sum'].values  # From logprob_results 
    diag_rb_sum = df['diag_rb_entropy_sum'].values  # From diagnostics
    
    print("=== ENTROPY COMPUTATION COMPARISON ===")
    print(f"basic_entropy_sum:     zero={np.sum(basic_sum == 0):>4}, mean={np.mean(basic_sum):>8.4f}, std={np.std(basic_sum):>8.4f}")
    print(f"rb_entropy_sum:        zero={np.sum(rb_sum == 0):>4}, mean={np.mean(rb_sum):>8.4f}, std={np.std(rb_sum):>8.4f}")
    print(f"diag_rb_entropy_sum:   zero={np.sum(diag_rb_sum == 0):>4}, mean={np.mean(diag_rb_sum):>8.4f}, std={np.std(diag_rb_sum):>8.4f}")
    print()
    
    # Check correlations
    print("=== CORRELATIONS ===")
    print(f"basic_entropy_sum vs rb_entropy_sum:      {np.corrcoef(basic_sum, rb_sum)[0,1]:.4f}")
    print(f"basic_entropy_sum vs diag_rb_entropy_sum: {np.corrcoef(basic_sum, diag_rb_sum)[0,1]:.4f}")
    print(f"rb_entropy_sum vs diag_rb_entropy_sum:    {np.corrcoef(rb_sum, diag_rb_sum)[0,1]:.4f}")
    print()
    
    # Find cases where they disagree
    print("=== DISAGREEMENT ANALYSIS ===")
    both_zero = (basic_sum == 0) & (diag_rb_sum == 0)
    basic_zero_diag_nonzero = (basic_sum == 0) & (diag_rb_sum != 0)
    basic_nonzero_diag_zero = (basic_sum != 0) & (diag_rb_sum == 0)
    
    print(f"Both zero: {np.sum(both_zero)}")
    print(f"Basic zero, diag non-zero: {np.sum(basic_zero_diag_nonzero)}")  
    print(f"Basic non-zero, diag zero: {np.sum(basic_nonzero_diag_zero)}")
    print()
    
    # Examine some specific cases
    print("=== SPECIFIC CASES INVESTIGATION ===")
    
    # Case 1: Both are zero
    if np.sum(both_zero) > 0:
        print("Examples where both basic and diag RB entropy are zero:")
        zero_indices = np.where(both_zero)[0][:3]
        for idx in zero_indices:
            seq = per_seq_data[idx]
            print(f"  Index {idx}:")
            print(f"    Response length: {seq['response_length_tokens']} tokens")
            print(f"    Response text: '{seq['response_text'][:100]}...'")
            print(f"    All entropy values: basic_sum={seq['basic_entropy_sum']}, rb_sum={seq['rb_entropy_sum']}, diag_rb_sum={seq['diag_rb_entropy_sum']}")
            print(f"    Sequence logprob: {seq['sequence_logprob']}")
            print(f"    Reward: {seq['reward']}")
            print()
    
    # Case 2: Only one is zero
    if np.sum(basic_zero_diag_nonzero) > 0:
        print("Examples where basic entropy is zero but diag RB is not:")
        indices = np.where(basic_zero_diag_nonzero)[0][:2]
        for idx in indices:
            seq = per_seq_data[idx]
            print(f"  Index {idx}: basic_sum={seq['basic_entropy_sum']}, diag_rb_sum={seq['diag_rb_entropy_sum']}")
            print(f"    Response: '{seq['response_text'][:100]}...'")
            print()
    
    if np.sum(basic_nonzero_diag_zero) > 0:
        print("Examples where diag RB entropy is zero but basic is not:")
        indices = np.where(basic_nonzero_diag_zero)[0][:2]
        for idx in indices:
            seq = per_seq_data[idx]
            print(f"  Index {idx}: basic_sum={seq['basic_entropy_sum']}, diag_rb_sum={seq['diag_rb_entropy_sum']}")
            print(f"    Response: '{seq['response_text'][:100]}...'")
            print()
    
    # Check if zeros correspond to short responses
    response_lengths = df['response_length_tokens'].values
    zero_mask = both_zero
    
    print("=== ZERO ENTROPY vs RESPONSE LENGTH ===")
    if np.sum(zero_mask) > 0:
        zero_lengths = response_lengths[zero_mask]
        nonzero_lengths = response_lengths[~zero_mask]
        
        print(f"Response lengths for zero entropy cases: mean={np.mean(zero_lengths):.1f}, min={np.min(zero_lengths)}, max={np.max(zero_lengths)}")
        print(f"Response lengths for non-zero entropy: mean={np.mean(nonzero_lengths):.1f}, min={np.min(nonzero_lengths)}, max={np.max(nonzero_lengths)}")
        
        # Check if all zero-entropy cases have zero-length responses
        zero_length_responses = np.sum(zero_lengths == 0)
        print(f"Zero-entropy cases with zero-length responses: {zero_length_responses} / {len(zero_lengths)}")
    
    # Investigate the batch mean issue
    print("\n=== BATCH MEAN IDENTITY ISSUE ===")
    
    # Quick batch analysis to replicate the issue
    batch_size = 128
    n_batches = len(df) // batch_size
    
    basic_batch_means = []
    diag_batch_means = []
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        
        basic_batch_mean = np.mean(basic_sum[start_idx:end_idx])
        diag_batch_mean = np.mean(diag_rb_sum[start_idx:end_idx])
        
        basic_batch_means.append(basic_batch_mean)
        diag_batch_means.append(diag_batch_mean)
    
    print(f"Batch size {batch_size}, {n_batches} batches:")
    print(f"Basic entropy batch means: {basic_batch_means}")
    print(f"Diag RB entropy batch means: {diag_batch_means}")
    
    basic_mean_range = max(basic_batch_means) - min(basic_batch_means)
    diag_mean_range = max(diag_batch_means) - min(diag_batch_means)
    
    print(f"Range of basic batch means: {basic_mean_range}")
    print(f"Range of diag RB batch means: {diag_mean_range}")
    
    if basic_mean_range == 0.0:
        print("🚨 CONFIRMED: Basic entropy batch means are EXACTLY identical!")
    if diag_mean_range == 0.0:
        print("🚨 CONFIRMED: Diag RB entropy batch means are EXACTLY identical!")
    
    return df

if __name__ == "__main__":
    df = debug_entropy_data()