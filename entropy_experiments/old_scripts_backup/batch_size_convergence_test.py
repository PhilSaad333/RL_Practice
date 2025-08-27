#!/usr/bin/env python3
"""
Batch Size Convergence Test

Simple test to understand how δH₁ changes with batch size using real checkpoint and data.
Tests B_E ∈ {256, 512, 1024} with B_U=32 fixed, 4 runs each.

This will reveal whether larger batch sizes cause:
1. Convergence to true δH₁ value (expected)
2. Systematic 1/B_E scaling (would indicate bug)
"""

import os
import sys
import yaml
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List


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
    
    # Ensure conditional variance estimator is enabled
    config['probe_rework']['compute_conditional_variance'] = True
    config['probe_rework']['compute_vx_vy_variance'] = False
    config['probe_rework']['compute_importance_sampling'] = False
    
    # Save config
    config_name = f"batch_test_BE{B_E}_run{run_id}_{timestamp}.yaml"
    with open(config_name, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return config_name


def run_single_probe(config_path: str) -> Dict:
    """Run single entropy probe and extract key results."""
    
    print(f"Running probe with config: {config_path}")
    
    # Run the probe
    cmd = [
        "python", "-m", "entropy_experiments.offline_entropy_probe", 
        "--config", config_path
    ]
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            cwd="/home/ubuntu/RL_Practice",
            timeout=300  # 5 minute timeout
        )
        
        runtime = time.time() - start_time
        
        if result.returncode != 0:
            print(f"❌ Probe failed with return code {result.returncode}")
            print(f"STDERR: {result.stderr[-1000:]}")  # Last 1000 chars
            return {"error": f"Return code {result.returncode}", "stderr": result.stderr[-500:]}
        
        # Parse results from stderr (since probe outputs go there)
        stderr_lines = result.stderr.split('\n')
        
        # Extract key metrics
        delta_h1 = None
        se_conditional = None
        bars_dot = None
        B_E = None
        B_U = None
        
        for line in stderr_lines:
            if 'δH₁=' in line or 'deltaH1=' in line:
                # Look for patterns like "δH₁=0.04123" or "deltaH1=0.04123"
                parts = line.split('=')
                if len(parts) >= 2:
                    try:
                        delta_h1 = float(parts[-1].strip().rstrip(','))
                    except:
                        pass
            elif 'SE_E(δH₁|U)=' in line:
                parts = line.split('=')
                if len(parts) >= 2:
                    try:
                        se_conditional = float(parts[-1].strip().rstrip(','))
                    except:
                        pass
            elif 'bars_dot=' in line:
                parts = line.split('=')
                if len(parts) >= 2:
                    try:
                        bars_dot = float(parts[-1].strip().rstrip(','))
                    except:
                        pass
            elif 'B_E:' in line or 'B_E=' in line:
                parts = line.replace(':', '=').split('=')
                if len(parts) >= 2:
                    try:
                        B_E = int(parts[-1].strip().rstrip(','))
                    except:
                        pass
            elif 'B_U:' in line or 'B_U=' in line:
                parts = line.replace(':', '=').split('=')
                if len(parts) >= 2:
                    try:
                        B_U = int(parts[-1].strip().rstrip(','))
                    except:
                        pass
        
        # Try to load results JSON if it exists
        config_data = yaml.safe_load(open(config_path, 'r'))
        results_path = config_data.get('output', {}).get('results_path')
        
        json_results = {}
        if results_path and os.path.exists(results_path):
            try:
                with open(results_path, 'r') as f:
                    json_results = json.load(f)
                    # Extract from JSON if parsing stderr failed
                    if delta_h1 is None:
                        delta_h1 = json_results.get('deltaH1')
                    if se_conditional is None:
                        se_conditional = json_results.get('SE_conditional')
                    if bars_dot is None:
                        bars_dot = json_results.get('bars_dot')
                    if B_E is None:
                        B_E = json_results.get('B_E')
                    if B_U is None:
                        B_U = json_results.get('B_U')
            except Exception as e:
                print(f"⚠️  Could not load JSON results: {e}")
        
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
            "json_results": json_results,
            "stderr_sample": result.stderr[-500:] if result.stderr else ""
        }
        
    except subprocess.TimeoutExpired:
        runtime = time.time() - start_time
        return {"error": "Timeout", "runtime": runtime}
    except Exception as e:
        runtime = time.time() - start_time
        return {"error": str(e), "runtime": runtime}


def main():
    """Run batch size convergence test."""
    
    if len(sys.argv) < 2:
        print("Usage: python batch_size_convergence_test.py <base_config.yaml>")
        sys.exit(1)
    
    base_config_path = sys.argv[1]
    
    if not os.path.exists(base_config_path):
        print(f"❌ Config file not found: {base_config_path}")
        sys.exit(1)
    
    print("🚀 Starting Batch Size Convergence Test")
    print(f"Base config: {base_config_path}")
    
    # Test parameters
    B_E_values = [256, 512, 1024]
    B_U = 32
    n_runs = 4
    
    print(f"Testing B_E values: {B_E_values}")
    print(f"Fixed B_U: {B_U}")
    print(f"Runs per B_E: {n_runs}")
    print()
    
    all_results = []
    
    for B_E in B_E_values:
        print(f"{'='*60}")
        print(f"TESTING B_E = {B_E}")
        print(f"{'='*60}")
        
        B_E_results = []
        
        for run_id in range(1, n_runs + 1):
            print(f"\n--- Run {run_id}/{n_runs} ---")
            
            # Create config for this run
            config_path = create_batch_size_config(base_config_path, B_E, B_U, run_id)
            
            # Run probe
            result = run_single_probe(config_path)
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
                
                print(f"  ✅ δH₁ = {delta_h1}")
                print(f"     SE_conditional = {se_cond}")
                print(f"     Runtime: {runtime:.1f}s")
                
                if se_cond and delta_h1 and abs(delta_h1) > 0:
                    ratio = abs(se_cond / delta_h1)
                    print(f"     |SE/δH₁| = {ratio:.2f}")
            else:
                error = result.get('error', 'Unknown error')
                print(f"  ❌ Failed: {error}")
            
            # Clean up config file
            try:
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
                print(f"\n📊 B_E={B_E} Summary:")
                print(f"   Successful runs: {len(successful_runs)}/{n_runs}")
                print(f"   Mean δH₁: {mean_delta_h1:.6f}")
                if len(delta_h1_values) > 1:
                    std_delta_h1 = (sum((x - mean_delta_h1)**2 for x in delta_h1_values) / (len(delta_h1_values) - 1))**0.5
                    print(f"   Std δH₁: {std_delta_h1:.6f}")
                
                if se_values:
                    mean_se = sum(se_values) / len(se_values)
                    print(f"   Mean SE: {mean_se:.6f}")
                    if abs(mean_delta_h1) > 0:
                        print(f"   Mean |SE/δH₁|: {abs(mean_se / mean_delta_h1):.2f}")
    
    # Final analysis
    print(f"\n{'='*60}")
    print("CONVERGENCE ANALYSIS")
    print(f"{'='*60}")
    
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
        print("Batch Size Effects:")
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
                    
                    print(f"  B_E {ref_B_E}→{B_E}: δH₁ ratio = {ratio:.3f}")
                    print(f"    Expected if 1/B_E scaling: {expected_1_over_B_E:.3f}")
                    
                    if abs(ratio - 1.0) < 0.1:
                        print(f"    → Suggests CONVERGENCE (values are similar)")
                    elif abs(ratio - expected_1_over_B_E) < 0.1:
                        print(f"    → Suggests SCALING as 1/B_E") 
                    else:
                        print(f"    → Other behavior")
    
    # Save detailed results
    results_file = f"batch_convergence_results_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'test_config': {
                'B_E_values': B_E_values,
                'B_U': B_U,
                'n_runs': n_runs,
                'base_config': base_config_path
            },
            'summary': B_E_summary,
            'all_results': all_results
        }, f, indent=2)
    
    print(f"\n📁 Detailed results saved to: {results_file}")


if __name__ == "__main__":
    main()