#!/usr/bin/env python3
"""
🔍 Multi-E-batch Analysis for Sign Error Investigation

This script runs the entropy probe with a single U batch and multiple E batches
to investigate the sign error between δH₁ prediction and ground truth ΔH.

Key idea:
1. Generate one U batch → compute Δθ (parameter update) once
2. Generate 8 different E batches → compute δH₁ for each using the same Δθ
3. Compare statistics across all 8 δH₁ estimates
4. Investigate if sign error is consistent across all E batches

Usage:
    python entropy_experiments/run_multi_e_analysis.py --config entropy_experiments/configs/lambda_multi_e_analysis.yaml --num-e-samples 8
"""

import sys
import argparse
import time
import torch
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from entropy_experiments.offline_entropy_probe import OfflineEntropyProbe


def run_multi_e_analysis(config_path: str, num_e_samples: int = 8) -> Dict[str, Any]:
    """Run entropy probe with single U batch and multiple E batches."""
    
    print(f"🔍 MULTI-E-BATCH ANALYSIS")
    print(f"📝 Config: {config_path}")
    print(f"🎯 E samples: {num_e_samples}, U samples: 1")
    print("=" * 60)
    
    # Initialize probe
    probe = OfflineEntropyProbe.from_config_file(config_path)
    
    # Extract configuration
    batch_config = probe.config.get('batch_config', {})
    computation_config = probe.config.get('computation_options', {})
    checkpoint_config = probe.config.get('checkpoint', {})
    B_E = batch_config.get('B_E', 512)
    B_U = batch_config.get('B_U', 64)
    G_U = batch_config.get('G', 8)
    mb_size_prompts = computation_config.get('mb_size_prompts', 2)
    weighting_mode = computation_config.get('weighting_mode', 'dr_grpo')
    
    print(f"📊 Configuration: B_E={B_E}, B_U={B_U}, G={G_U}, mb_size={mb_size_prompts}")
    
    # Load model and optimizer checkpoints
    print(f"🔧 Loading model and optimizer...")
    checkpoint_path = checkpoint_config.get('checkpoint_path')
    optimizer_path = checkpoint_config.get('optimizer_path')
    probe.load_checkpoint(checkpoint_path, optimizer_path)
    print(f"✅ Model loaded from {checkpoint_path}")
    print(f"✅ Optimizer loaded from {optimizer_path}")
    
    # === PHASE 1: Generate single U batch and compute Δθ ===
    print(f"\\n🎲 Phase 1: Generating single U batch...")
    start_time = time.time()
    
    U_batch = probe._get_or_sample_U(B_U, G_U)
    print(f"✅ Generated U batch: {U_batch['sequences'].shape[0]} prompts, {U_batch['sequences'].shape[1]} responses/prompt")
    
    print(f"⚙️  Computing parameter update buffer (Δθ)...")
    delta_theta_buf, delta_theta_norm, B_U_used = probe._compute_param_update_buffer(U_batch, mb_size_prompts)
    print(f"✅ Parameter update ready: ||Δθ||={delta_theta_norm:.3e} across {len(delta_theta_buf)} trainables")
    
    phase1_time = time.time() - start_time
    print(f"⏱️  Phase 1 completed in {phase1_time:.2f}s")
    
    # === PHASE 2: Generate multiple E batches and compute δH₁ for each ===
    print(f"\\n🔄 Phase 2: Generating {num_e_samples} E batches and computing δH₁...")
    
    delta_h1_estimates = []
    bars_dot_estimates = []
    
    for i in range(num_e_samples):
        print(f"\\n  📝 E-batch {i+1}/{num_e_samples}:")
        
        # Generate fresh E batch
        E_batch = probe._get_or_sample_E(B_E)
        print(f"    ✅ Generated {E_batch['sequences'].shape[0]} E prompts")
        
        # Compute δH₁ using the same Δθ
        compute_start = time.time()
        compute = probe.probe_components.compute_delta_h1_from_batches(
            E_batch=E_batch,
            U_batch=U_batch,  # Same U batch for metadata
            mb_size_prompts=mb_size_prompts,
            weighting_mode=weighting_mode,
            adam_preconditioner=probe.adam_preconditioner,
            optimizer=probe.optimizer,
            param_update_buf=delta_theta_buf,  # Same Δθ for all E batches
        )
        
        delta_h1 = compute['deltaH1']
        bars_dot = compute['bars_dot'] 
        compute_time = time.time() - compute_start
        
        delta_h1_estimates.append(delta_h1)
        bars_dot_estimates.append(bars_dot)
        
        print(f"    📊 δH₁ = {delta_h1:.6f}, bars_dot = {bars_dot:.6f} (computed in {compute_time:.1f}s)")
    
    # === PHASE 3: Compute ground truth ΔH for comparison ===
    print(f"\\n📈 Phase 3: Computing ground truth entropy change...")
    ground_truth_start = time.time()
    
    # Use the first E batch for ground truth computation  
    E_batch_for_gt = probe._get_or_sample_E(B_E)
    
    # Enable importance sampling
    importance_config = probe.config.get('true_delta_h', {})
    if importance_config.get('enabled', False):
        gt_results = probe._compute_ground_truth_delta_entropy(
            E_batch=E_batch_for_gt,
            U_batch=U_batch, 
            mb_size_prompts=mb_size_prompts,
            **importance_config
        )
        delta_h_true = gt_results['deltaH_true']
        H_orig = gt_results['H_orig']
        H_upd = gt_results['H_upd']
        ground_truth_time = time.time() - ground_truth_start
        
        print(f"✅ Ground truth: ΔH = {delta_h_true:.6f} (H_orig={H_orig:.3f} → H_upd={H_upd:.3f})")
        print(f"⏱️  Ground truth computed in {ground_truth_time:.2f}s")
    else:
        delta_h_true = None
        print("⚠️  Ground truth computation disabled in config")
    
    # === PHASE 4: Statistical Analysis ===
    print(f"\\n📊 STATISTICAL ANALYSIS")
    print("=" * 60)
    
    delta_h1_array = np.array(delta_h1_estimates)
    bars_dot_array = np.array(bars_dot_estimates)
    
    # Basic statistics
    delta_h1_stats = {
        'mean': np.mean(delta_h1_array),
        'std': np.std(delta_h1_array),
        'min': np.min(delta_h1_array),
        'max': np.max(delta_h1_array),
        'median': np.median(delta_h1_array),
    }
    
    print(f"δH₁ estimates across {num_e_samples} E batches:")
    print(f"  Mean:   {delta_h1_stats['mean']:.6f} ± {delta_h1_stats['std']:.6f}")
    print(f"  Range:  [{delta_h1_stats['min']:.6f}, {delta_h1_stats['max']:.6f}]")
    print(f"  Median: {delta_h1_stats['median']:.6f}")
    
    # Sign analysis
    positive_count = np.sum(delta_h1_array > 0)
    negative_count = np.sum(delta_h1_array < 0)
    zero_count = np.sum(delta_h1_array == 0)
    
    print(f"\\nSign distribution:")
    print(f"  Positive: {positive_count}/{num_e_samples} ({100*positive_count/num_e_samples:.1f}%)")
    print(f"  Negative: {negative_count}/{num_e_samples} ({100*negative_count/num_e_samples:.1f}%)")
    print(f"  Zero:     {zero_count}/{num_e_samples} ({100*zero_count/num_e_samples:.1f}%)")
    
    # Comparison with ground truth
    if delta_h_true is not None:
        print(f"\\nComparison with ground truth:")
        print(f"  Ground truth ΔH:     {delta_h_true:.6f}")
        print(f"  δH₁ mean prediction:  {delta_h1_stats['mean']:.6f}")
        
        error_abs = abs(delta_h1_stats['mean'] - delta_h_true)
        error_rel = 100 * error_abs / abs(delta_h_true) if delta_h_true != 0 else float('inf')
        
        print(f"  Absolute error:       {error_abs:.6f}")
        print(f"  Relative error:       {error_rel:.2f}%")
        
        # Sign consistency check
        gt_sign = "positive" if delta_h_true > 0 else "negative" if delta_h_true < 0 else "zero"
        pred_sign = "positive" if delta_h1_stats['mean'] > 0 else "negative" if delta_h1_stats['mean'] < 0 else "zero"
        
        if gt_sign == pred_sign:
            print(f"  ✅ Sign match: both {gt_sign}")
        else:
            print(f"  ❌ Sign mismatch: ground truth is {gt_sign}, prediction is {pred_sign}")
    
    # Individual estimates  
    print(f"\\nIndividual δH₁ estimates:")
    for i, (delta_h1, bars_dot) in enumerate(zip(delta_h1_estimates, bars_dot_estimates)):
        print(f"  E-batch {i+1:2d}: δH₁ = {delta_h1:8.6f}, bars_dot = {bars_dot:8.6f}")
    
    # === RESULTS SUMMARY ===
    total_time = time.time() - start_time
    results = {
        'config_path': config_path,
        'num_e_samples': num_e_samples,
        'B_E': B_E,
        'B_U': B_U,
        'delta_theta_norm': delta_theta_norm,
        'delta_h1_estimates': delta_h1_estimates,
        'bars_dot_estimates': bars_dot_estimates,
        'delta_h1_stats': delta_h1_stats,
        'delta_h_true': delta_h_true,
        'sign_distribution': {
            'positive': positive_count,
            'negative': negative_count, 
            'zero': zero_count
        },
        'timing': {
            'total_time': total_time,
            'phase1_time': phase1_time,
            'ground_truth_time': ground_truth_time if delta_h_true is not None else 0.0
        }
    }
    
    print(f"\\n🎯 ANALYSIS COMPLETE")
    print(f"Total runtime: {total_time:.2f}s")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Multi-E-batch entropy probe analysis")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--num-e-samples", type=int, default=8, help="Number of E batch samples")
    
    args = parser.parse_args()
    
    try:
        results = run_multi_e_analysis(args.config, args.num_e_samples)
        
        # Save results to file
        import json
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        results_file = f"entropy_experiments/logs/multi_e_analysis_{timestamp}.json"
        
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            return obj
        
        # Recursively convert numpy types
        import copy
        json_results = json.loads(json.dumps(results, default=convert_numpy))
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\\n📋 Results saved to: {results_file}")
        
        return 0
        
    except Exception as e:
        print(f"\\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(code=main())