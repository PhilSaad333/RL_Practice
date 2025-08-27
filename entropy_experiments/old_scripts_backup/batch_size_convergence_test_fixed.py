#!/usr/bin/env python3
"""
Fixed Batch Size Convergence Test

Tests how δH₁ and conditional variance change with batch size using proper probe setup.
Tests B_E ∈ {256, 512, 1024} with B_U=32 fixed, 4 runs each.

This uses the same setup as the successful sanity check script:
- Proper LoRA checkpoint loading with /model and /optimizer.pt paths
- run_mixed_probe() method instead of subprocess calls
- Direct OfflineEntropyProbe import and usage
"""

import os
import sys
import yaml
import json
import time
from pathlib import Path
from typing import Dict, List
import logging

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from entropy_experiments.offline_entropy_probe import OfflineEntropyProbe


def setup_logging():
    """Setup logging for convergence test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger("batch_convergence")


def create_batch_size_config(base_config_path: str, B_E: int, B_U: int = 32, run_id: int = 1) -> str:
    """Create config with specified batch sizes."""
    
    # Load base config
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update batch sizes
    config['batch_config']['B_E'] = B_E
    config['batch_config']['B_U'] = B_U
    config['batch_config']['B'] = B_E + B_U  # Total batch size
    
    # Update output path to avoid conflicts
    timestamp = time.strftime('%H%M%S')
    config['output']['results_path'] = f"batch_test_BE{B_E}_run{run_id}_{timestamp}.json"
    
    # Ensure conditional variance estimator is enabled and others disabled for focused testing
    config['probe_rework']['compute_conditional_variance'] = True
    config['probe_rework']['compute_vx_vy_variance'] = False
    config['probe_rework']['compute_importance_sampling'] = False
    
    # Enable debug logging 
    config['output']['log_level'] = "DEBUG"
    
    # Save config
    config_name = f"batch_test_BE{B_E}_run{run_id}_{timestamp}.yaml"
    config_path = os.path.join(os.path.dirname(base_config_path), config_name)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return config_path


def run_single_probe_test(config_path: str, checkpoint_base_path: str) -> Dict:
    """Run single entropy probe test and extract key results."""
    
    print(f"Running probe with config: {os.path.basename(config_path)}")
    
    start_time = time.time()
    
    try:
        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Set up proper LoRA checkpoint paths (like sanity check script)
        model_path = checkpoint_base_path + "/model" if not checkpoint_base_path.endswith("/model") else checkpoint_base_path
        optimizer_path = checkpoint_base_path.replace("/model", "/optimizer.pt") if checkpoint_base_path.endswith("/model") else checkpoint_base_path + "/optimizer.pt"
        
        config['checkpoint']['checkpoint_path'] = model_path
        config['checkpoint']['optimizer_path'] = optimizer_path
        
        # Run probe using the working method from sanity check
        probe = OfflineEntropyProbe(config)
        results = probe.run_mixed_probe()
        
        runtime = time.time() - start_time
        
        # Extract key metrics
        delta_h1 = results.get('deltaH1')
        se_conditional = results.get('SE_conditional') 
        bars_dot = results.get('bars_dot')
        B_E = results.get('B_E')
        B_U = results.get('B_U')
        
        # Get results file path
        results_path = config.get('output', {}).get('results_path')
        
        return {
            "success": True,
            "delta_h1": delta_h1,
            "se_conditional": se_conditional, 
            "bars_dot": bars_dot,
            "B_E": B_E,
            "B_U": B_U,
            "runtime": runtime,
            "config_path": config_path,
            "results_path": results_path,
            "full_results": results
        }
        
    except Exception as e:
        runtime = time.time() - start_time
        print(f"❌ Probe failed: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "runtime": runtime,
            "config_path": config_path
        }


def main():
    """Run batch size convergence test."""
    
    if len(sys.argv) < 3:
        print("Usage: python batch_size_convergence_test_fixed.py <base_config.yaml> <checkpoint_base_path>")
        print("Example: python batch_size_convergence_test_fixed.py configs/batch_convergence_test_config.yaml /home/ubuntu/localfs/training_runs/run_2025-08-24_22-13-22/training_state/step_40")
        sys.exit(1)
    
    base_config_path = sys.argv[1]
    checkpoint_base_path = sys.argv[2]
    
    if not os.path.exists(base_config_path):
        print(f"❌ Config file not found: {base_config_path}")
        sys.exit(1)
        
    if not os.path.exists(checkpoint_base_path):
        print(f"❌ Checkpoint path not found: {checkpoint_base_path}")
        sys.exit(1)
    
    logger = setup_logging()
    
    logger.info("🚀 Starting Fixed Batch Size Convergence Test")
    logger.info(f"Base config: {base_config_path}")
    logger.info(f"Checkpoint: {checkpoint_base_path}")
    
    # Test parameters
    B_E_values = [256, 512, 1024]
    B_U = 32
    n_runs = 4
    
    logger.info(f"Testing B_E values: {B_E_values}")
    logger.info(f"Fixed B_U: {B_U}")
    logger.info(f"Runs per B_E: {n_runs}")
    logger.info("")
    
    all_results = []
    
    for B_E in B_E_values:
        logger.info(f"{'='*60}")
        logger.info(f"TESTING B_E = {B_E}")
        logger.info(f"{'='*60}")
        
        B_E_results = []
        
        for run_id in range(1, n_runs + 1):
            logger.info(f"\n--- Run {run_id}/{n_runs} ---")
            
            # Create config for this run
            config_path = create_batch_size_config(base_config_path, B_E, B_U, run_id)
            
            try:
                # Run probe
                result = run_single_probe_test(config_path, checkpoint_base_path)
                result['B_E_target'] = B_E
                result['B_U_target'] = B_U
                result['run_id'] = run_id
                
                B_E_results.append(result)
                all_results.append(result)
                
                # Print immediate results
                if result.get('success'):
                    delta_h1 = result.get('delta_h1')
                    se_cond = result.get('se_conditional') 
                    runtime = result.get('runtime', 0)
                    
                    logger.info(f"  ✅ δH₁ = {delta_h1}")
                    logger.info(f"     SE_conditional = {se_cond}")
                    logger.info(f"     Runtime: {runtime:.1f}s")
                    
                    if se_cond and delta_h1 and abs(delta_h1) > 0:
                        ratio = abs(se_cond / delta_h1)
                        logger.info(f"     |SE/δH₁| = {ratio:.2f}")
                        
                        if ratio < 0.1:
                            logger.info(f"     ✅ Good signal-to-noise ratio")
                        elif ratio < 0.2:
                            logger.info(f"     ⚠️  Acceptable signal-to-noise ratio") 
                        else:
                            logger.info(f"     ❌ Poor signal-to-noise ratio")
                else:
                    error = result.get('error', 'Unknown error')
                    logger.info(f"  ❌ Failed: {error}")
                    
            finally:
                # Clean up config file
                try:
                    if os.path.exists(config_path):
                        os.remove(config_path)
                except:
                    pass
        
        # Summary for this B_E
        successful_runs = [r for r in B_E_results if r.get('success')]
        if successful_runs:
            delta_h1_values = [r['delta_h1'] for r in successful_runs if r.get('delta_h1') is not None]
            se_values = [r['se_conditional'] for r in successful_runs if r.get('se_conditional') is not None]
            
            if delta_h1_values:
                mean_delta_h1 = sum(delta_h1_values) / len(delta_h1_values)
                logger.info(f"\n📊 B_E={B_E} Summary:")
                logger.info(f"   Successful runs: {len(successful_runs)}/{n_runs}")
                logger.info(f"   Mean δH₁: {mean_delta_h1:.6f}")
                if len(delta_h1_values) > 1:
                    std_delta_h1 = (sum((x - mean_delta_h1)**2 for x in delta_h1_values) / (len(delta_h1_values) - 1))**0.5
                    logger.info(f"   Std δH₁: {std_delta_h1:.6f}")
                
                if se_values:
                    mean_se = sum(se_values) / len(se_values)
                    logger.info(f"   Mean SE: {mean_se:.6f}")
                    if abs(mean_delta_h1) > 0:
                        logger.info(f"   Mean |SE/δH₁|: {abs(mean_se / mean_delta_h1):.2f}")
    
    # Final analysis
    logger.info(f"\n{'='*60}")
    logger.info("CONVERGENCE ANALYSIS")
    logger.info(f"{'='*60}")
    
    # Group results by B_E
    B_E_summary = {}
    for B_E in B_E_values:
        B_E_runs = [r for r in all_results if r.get('B_E_target') == B_E and r.get('success')]
        delta_h1_values = [r['delta_h1'] for r in B_E_runs if r.get('delta_h1') is not None]
        
        if delta_h1_values:
            mean_val = sum(delta_h1_values) / len(delta_h1_values)
            std_val = (sum((x - mean_val)**2 for x in delta_h1_values) / max(len(delta_h1_values) - 1, 1))**0.5
            B_E_summary[B_E] = {'mean': mean_val, 'std': std_val, 'n': len(delta_h1_values)}
    
    # Check for scaling patterns
    if len(B_E_summary) >= 2:
        logger.info("Batch Size Effects:")
        ref_B_E = min(B_E_values)
        ref_mean = B_E_summary.get(ref_B_E, {}).get('mean')
        
        if ref_mean and abs(ref_mean) > 0:
            for B_E in B_E_values:
                if B_E == ref_B_E:
                    continue
                    
                curr_mean = B_E_summary.get(B_E, {}).get('mean')
                if curr_mean:
                    ratio = curr_mean / ref_mean
                    B_E_ratio = B_E / ref_B_E
                    expected_1_over_B_E = 1.0 / B_E_ratio
                    
                    logger.info(f"  B_E {ref_B_E}→{B_E}: δH₁ ratio = {ratio:.3f}")
                    logger.info(f"    Expected if 1/B_E scaling: {expected_1_over_B_E:.3f}")
                    
                    if abs(ratio - 1.0) < 0.1:
                        logger.info(f"    → ✅ CONVERGENCE (values are similar)")
                    elif abs(ratio - expected_1_over_B_E) < 0.1:
                        logger.info(f"    → ❌ SCALING as 1/B_E (indicates bug)") 
                    else:
                        logger.info(f"    → ⚠️  Other behavior")
    
    # Save detailed results
    results_file = f"batch_convergence_results_fixed_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'test_config': {
                'B_E_values': B_E_values,
                'B_U': B_U,
                'n_runs': n_runs,
                'base_config': base_config_path,
                'checkpoint_path': checkpoint_base_path
            },
            'summary': B_E_summary,
            'all_results': all_results
        }, f, indent=2)
    
    logger.info(f"\n📁 Detailed results saved to: {results_file}")


if __name__ == "__main__":
    main()