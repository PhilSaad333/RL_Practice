#!/usr/bin/env python3
"""
Stage 3 Probe Sanity Check

Quick validation runs to check if the probe is working correctly:
1. Fractional variance (V_X + V_Y)/δH₁² should be reasonably small
2. δH₁ - ΔH_true should be ~ N(0, σ²) (unbiased estimator)

Usage:
    torchrun --nproc_per_node=2 run_probe_sanity_check.py --checkpoint /path/to/checkpoint --runs 15
"""

import argparse
import sys
import os
import yaml
import json
import torch
import torch.distributed as dist
import logging
import time
from pathlib import Path
from datetime import datetime
import numpy as np

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from entropy_experiments.offline_entropy_probe import OfflineEntropyProbe


def setup_logging(rank: int = 0):
    """Simple logging setup."""
    log_format = f'%(asctime)s - RANK{rank} - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, datefmt='%H:%M:%S')
    return logging.getLogger("sanity_check")


def setup_distributed():
    """Setup distributed training if available."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        return True, rank, world_size, local_rank
    else:
        return False, 0, 1, 0


def load_config(config_path: str):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def extract_key_metrics(results, config):
    """Extract the key metrics we care about for sanity checking."""
    metrics = {
        'deltaH1': results['deltaH1'],
        'deltaH_true': results.get('deltaH_true'),
        'V_X': results.get('V_X'),
        'V_Y': results.get('V_Y'),
        'SE_deltaH1': results.get('SE_deltaH1'),
        'ESS': results.get('diagnostics', {}).get('ESS'),
        'runtime': sum(results.get('timing', {}).values()),
    }
    
    # Extract importance sampling diagnostics
    diagnostics = results.get('diagnostics', {})
    if diagnostics:
        metrics['w_max'] = diagnostics.get('w_max')
        metrics['w_min'] = diagnostics.get('w_min')
        metrics['clipped_fraction'] = diagnostics.get('clipped_fraction')
        metrics['w_sum_global'] = diagnostics.get('w_sum_global')
    else:
        metrics['w_max'] = None
        metrics['w_min'] = None
        metrics['clipped_fraction'] = None
        metrics['w_sum_global'] = None
    
    # Compute fractional variance: (V_X + V_Y) / bars_dot²
    # Note: δH₁ = lr * bars_dot, so bars_dot = δH₁ / lr
    # Fractional variance = Var(δH₁) / δH₁² = lr² * (V_X + V_Y) / δH₁²
    # = lr² * (V_X + V_Y) / (lr * bars_dot)² = (V_X + V_Y) / bars_dot²
    if metrics['V_X'] is not None and metrics['V_Y'] is not None and metrics['deltaH1'] != 0:
        learning_rate = config.get('learning_rate', 2e-6)  # Use config learning rate
        bars_dot = metrics['deltaH1'] / learning_rate
        metrics['frac_var'] = (metrics['V_X'] + metrics['V_Y']) / (bars_dot ** 2)
    else:
        metrics['frac_var'] = None
    
    # Compute bias: δH₁ - ΔH_true
    if metrics['deltaH_true'] is not None:
        metrics['bias'] = metrics['deltaH1'] - metrics['deltaH_true']
    else:
        metrics['bias'] = None
    
    return metrics


def run_sanity_check(config_path: str, checkpoint_path: str, n_runs: int, rank: int):
    """Run the sanity check with n_runs."""
    logger = setup_logging(rank)
    
    # Load and update config
    config = load_config(config_path)
    config['checkpoint']['checkpoint_path'] = checkpoint_path + "/model" if not checkpoint_path.endswith("/model") else checkpoint_path
    config['checkpoint']['optimizer_path'] = checkpoint_path.replace("/model", "/optimizer.pt") if checkpoint_path.endswith("/model") else checkpoint_path + "/optimizer.pt"
    
    results = []
    
    if rank == 0:
        logger.info(f"🧪 Starting probe sanity check: {n_runs} runs")
        logger.info(f"Checkpoint: {checkpoint_path}")
        logger.info(f"Target metrics:")
        logger.info(f"  - Fractional variance should be < 0.1 (ideally < 0.05)")
        logger.info(f"  - Bias (δH₁ - ΔH_true) should be ~ N(0, σ²)")
        logger.info("-" * 50)
    
    start_time = time.time()
    
    for run_id in range(n_runs):
        if rank == 0:
            logger.info(f"Run {run_id + 1}/{n_runs}")
        
        # Reset memory tracking
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        # Run probe
        probe = OfflineEntropyProbe(config)
        run_results = probe.run_mixed_probe()
        
        # Extract key metrics
        metrics = extract_key_metrics(run_results, config)
        metrics['run_id'] = run_id
        results.append(metrics)
        
        if rank == 0:
            logger.info(f"  δH₁: {metrics['deltaH1']:.6f}")
            if metrics['deltaH_true'] is not None:
                logger.info(f"  ΔH_true: {metrics['deltaH_true']:.6f}")
                logger.info(f"  Bias: {metrics['bias']:.6f}")
            if metrics['frac_var'] is not None:
                logger.info(f"  Frac_var: {metrics['frac_var']:.6f}")
            if metrics['ESS'] is not None:
                logger.info(f"  ESS: {metrics['ESS']:.1f}")
            
            # Importance sampling ratio diagnostics  
            if metrics['w_max'] is not None and metrics['w_min'] is not None:
                logger.info(f"  IS ratios: [{metrics['w_min']:.3f}, {metrics['w_max']:.3f}]")
                ratio_range = metrics['w_max'] / metrics['w_min'] if metrics['w_min'] > 0 else float('inf')
                logger.info(f"  Ratio range: {ratio_range:.1f}x")
                
                if metrics['clipped_fraction'] is not None:
                    logger.info(f"  Clipped: {metrics['clipped_fraction']:.1%}")
                    
            logger.info(f"  Runtime: {metrics['runtime']:.1f}s")
            
            # Quick feedback on targets
            if metrics['frac_var'] is not None:
                if metrics['frac_var'] < 0.05:
                    logger.info(f"  ✅ Fractional variance looks good")
                elif metrics['frac_var'] < 0.1:
                    logger.info(f"  ⚠️  Fractional variance acceptable")
                else:
                    logger.info(f"  ❌ Fractional variance high - may need larger batch")
                    
            # IS ratio feedback
            if metrics['w_max'] is not None and metrics['w_min'] is not None:
                if metrics['w_max'] < 10 and metrics['w_min'] > 0.1:
                    logger.info(f"  ✅ IS ratios look healthy")
                elif metrics['w_max'] < 100:
                    logger.info(f"  ⚠️  IS ratios acceptable")
                else:
                    logger.info(f"  ❌ Extreme IS ratios - model may be unstable")
            
            logger.info("")
    
    total_time = time.time() - start_time
    
    if rank == 0:
        # Analyze results
        analyze_results(results, logger, total_time)
    
    return results


def analyze_results(results, logger, total_time):
    """Analyze and summarize the sanity check results."""
    logger.info("=" * 60)
    logger.info("SANITY CHECK ANALYSIS")
    logger.info("=" * 60)
    
    n_runs = len(results)
    
    # δH₁ statistics
    deltaH1_values = [r['deltaH1'] for r in results]
    deltaH1_mean = np.mean(deltaH1_values)
    deltaH1_std = np.std(deltaH1_values)
    
    logger.info(f"📊 δH₁ Statistics (n={n_runs}):")
    logger.info(f"  Mean: {deltaH1_mean:.6f}")
    logger.info(f"  Std:  {deltaH1_std:.6f}")
    logger.info(f"  CV:   {deltaH1_std / abs(deltaH1_mean):.4f}")
    logger.info(f"  Range: [{min(deltaH1_values):.6f}, {max(deltaH1_values):.6f}]")
    
    # Fractional variance analysis
    frac_vars = [r['frac_var'] for r in results if r['frac_var'] is not None]
    if frac_vars:
        frac_var_mean = np.mean(frac_vars)
        frac_var_median = np.median(frac_vars)
        logger.info(f"")
        logger.info(f"📈 Fractional Variance Analysis:")
        logger.info(f"  Mean: {frac_var_mean:.6f}")
        logger.info(f"  Median: {frac_var_median:.6f}")
        logger.info(f"  Range: [{min(frac_vars):.6f}, {max(frac_vars):.6f}]")
        
        if frac_var_median < 0.05:
            logger.info(f"  ✅ GOOD: Fractional variance is low (< 0.05)")
            logger.info(f"      → Batch size appears sufficient")
        elif frac_var_median < 0.1:
            logger.info(f"  ⚠️  OK: Fractional variance is acceptable (< 0.1)")
            logger.info(f"      → Could consider slightly larger batches")
        else:
            logger.info(f"  ❌ HIGH: Fractional variance is high (≥ 0.1)")
            logger.info(f"      → Should increase batch size for better precision")
    
    # Bias analysis (δH₁ vs ΔH_true)
    biases = [r['bias'] for r in results if r['bias'] is not None]
    if biases:
        bias_mean = np.mean(biases)
        bias_std = np.std(biases)
        
        logger.info(f"")
        logger.info(f"🎯 Bias Analysis (δH₁ - ΔH_true):")
        logger.info(f"  Mean: {bias_mean:.6f}")
        logger.info(f"  Std:  {bias_std:.6f}")
        logger.info(f"  Range: [{min(biases):.6f}, {max(biases):.6f}]")
        
        # Test if mean is close to zero (unbiased estimator)
        if abs(bias_mean) < 0.001:
            logger.info(f"  ✅ GOOD: Mean bias ≈ 0 (unbiased estimator)")
        elif abs(bias_mean) < 0.01:
            logger.info(f"  ⚠️  OK: Mean bias is small")
        else:
            logger.info(f"  ❌ HIGH: Mean bias is significant")
            logger.info(f"      → May indicate systematic error")
        
        # Check if distribution looks reasonable
        if len(biases) >= 10:
            # Simple normality check: most values within 2σ of mean
            within_2sigma = sum(1 for b in biases if abs(b - bias_mean) <= 2 * bias_std)
            pct_within_2sigma = within_2sigma / len(biases)
            
            if pct_within_2sigma >= 0.9:
                logger.info(f"  ✅ GOOD: Bias distribution looks normal-ish")
                logger.info(f"      → {within_2sigma}/{len(biases)} ({pct_within_2sigma:.1%}) within 2σ")
            else:
                logger.info(f"  ⚠️  Distribution may have outliers")
                logger.info(f"      → {within_2sigma}/{len(biases)} ({pct_within_2sigma:.1%}) within 2σ")
    
    # Importance sampling ratio analysis
    ess_values = [r['ESS'] for r in results if r['ESS'] is not None]
    w_max_values = [r['w_max'] for r in results if r['w_max'] is not None]
    w_min_values = [r['w_min'] for r in results if r['w_min'] is not None]
    
    if ess_values:
        logger.info(f"")
        logger.info(f"⚖️  Importance Sampling Analysis:")
        logger.info(f"  ESS: {np.mean(ess_values):.1f} ± {np.std(ess_values):.1f}")
        logger.info(f"  ESS range: [{min(ess_values):.1f}, {max(ess_values):.1f}]")
        
        if w_max_values and w_min_values:
            logger.info(f"  Max ratio: {np.mean(w_max_values):.3f} ± {np.std(w_max_values):.3f}")
            logger.info(f"  Min ratio: {np.mean(w_min_values):.3f} ± {np.std(w_min_values):.3f}")
            
            # Check for extreme ratios
            extreme_max = [w for w in w_max_values if w > 100]
            extreme_min = [w for w in w_min_values if w < 0.01]
            
            if extreme_max:
                logger.info(f"  ⚠️  {len(extreme_max)} runs with w_max > 100")
            if extreme_min:
                logger.info(f"  ⚠️  {len(extreme_min)} runs with w_min < 0.01")
        
        # ESS health check
        total_samples = 128 * 8  # B_E * G for current config
        avg_ess_frac = np.mean(ess_values) / total_samples
        if avg_ess_frac > 0.1:
            logger.info(f"  ✅ GOOD: ESS = {avg_ess_frac:.1%} of total samples")
        elif avg_ess_frac > 0.05:
            logger.info(f"  ⚠️  OK: ESS = {avg_ess_frac:.1%} of total samples")
        else:
            logger.info(f"  ❌ LOW: ESS = {avg_ess_frac:.1%} of total samples")
            logger.info(f"      → Poor importance sampling coverage")

    # Performance summary
    runtimes = [r['runtime'] for r in results]
    avg_runtime = np.mean(runtimes)
    
    logger.info(f"")
    logger.info(f"⏱️  Performance Summary:")
    logger.info(f"  Total time: {total_time:.1f}s")
    logger.info(f"  Avg per run: {avg_runtime:.1f}s")
    logger.info(f"  Throughput: {3600 / avg_runtime:.1f} runs/hour")
    
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        logger.info(f"  Peak GPU memory: {peak_memory:.1f}GB")
    
    # Overall verdict
    logger.info(f"")
    logger.info(f"🏁 OVERALL VERDICT:")
    
    all_good = True
    if frac_vars and np.median(frac_vars) >= 0.1:
        all_good = False
        logger.info(f"  ❌ Fractional variance too high - increase batch size")
    
    if biases and abs(np.mean(biases)) >= 0.01:
        all_good = False
        logger.info(f"  ❌ Significant bias detected - check implementation")
    
    if all_good:
        logger.info(f"  ✅ PROBE LOOKS HEALTHY!")
        logger.info(f"     → Fractional variance is reasonable")
        logger.info(f"     → δH₁ estimator appears unbiased")
        logger.info(f"     → Ready for production use")
    else:
        logger.info(f"  ⚠️  Issues detected - see analysis above")
    
    # Save results for further analysis
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"probe_sanity_check_{timestamp}.json"
    
    summary_data = {
        'timestamp': timestamp,
        'n_runs': n_runs,
        'total_time': total_time,
        'checkpoint_path': results[0].get('checkpoint_path', 'unknown'),
        'deltaH1_stats': {
            'mean': deltaH1_mean,
            'std': deltaH1_std,
            'cv': deltaH1_std / abs(deltaH1_mean),
            'values': deltaH1_values
        },
        'frac_var_stats': {
            'mean': np.mean(frac_vars) if frac_vars else None,
            'median': np.median(frac_vars) if frac_vars else None,
            'values': frac_vars
        },
        'bias_stats': {
            'mean': np.mean(biases) if biases else None,
            'std': np.std(biases) if biases else None,
            'values': biases
        },
        'detailed_results': results
    }
    
    with open(results_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    logger.info(f"")
    logger.info(f"📁 Detailed results saved to: {results_file}")


def main():
    parser = argparse.ArgumentParser(description="Run Stage 3 Probe Sanity Check")
    parser.add_argument("--config", 
                       default="entropy_experiments/configs/mixed_probe_stage3_multigpu_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--checkpoint", required=True,
                       help="Path to checkpoint directory")
    parser.add_argument("--runs", type=int, default=15,
                       help="Number of validation runs (default: 15)")
    
    args = parser.parse_args()
    
    try:
        # Setup distributed training
        is_distributed, rank, world_size, local_rank = setup_distributed()
        logger = setup_logging(rank)
        
        if rank == 0:
            logger.info("🧪 STAGE 3 PROBE SANITY CHECK")
            logger.info(f"Distributed: {is_distributed}, Rank: {rank}/{world_size}")
            if torch.cuda.is_available():
                logger.info(f"CUDA devices: {torch.cuda.device_count()}")
        
        # Run sanity check
        run_sanity_check(args.config, args.checkpoint, args.runs, rank)
        
        return 0
        
    except Exception as e:
        logger = setup_logging(0)
        logger.error(f"Sanity check failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    exit(main())