#!/usr/bin/env python3
"""
Flexible Batch Size Test for Entropy Probe - FIXED LOGGING

Features:
- NO timeouts - each run completes naturally  
- Timestamped experiment directories for isolation
- Captures ALL logging output to files (including debug gradient info)
- Manually saves probe results.json (run_mixed_probe doesn't auto-save)
- Each run isolated with full artifacts
- Graceful error handling - failed runs don't stop the test

Directory structure:
entropy_experiments/results/
├── flexible_batch_test_2025-08-27_22-16-26/     # One experiment
│   ├── BE016_run001/
│   │   ├── config.yaml       # Exact config used
│   │   ├── probe_log.txt     # ALL Python logging output  
│   │   └── results.json      # Probe's JSON results (manually saved)
│   ├── BE032_run001/
│   └── summary.json          # Experiment summary
├── flexible_batch_test_2025-08-28_10-30-15/     # Another experiment
│   └── ...

Usage:
    python flexible_batch_test_fixed.py --checkpoint /path/to/checkpoint
    python flexible_batch_test_fixed.py --be-values 16,32,64 --runs 3
"""

import os
import sys
import yaml
import json
import time
import logging
import traceback
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from entropy_experiments.offline_entropy_probe import OfflineEntropyProbe


def create_experiment_directory(results_base: Path, experiment_name: str = None) -> Path:
    """Create timestamped experiment directory."""
    if experiment_name is None:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        experiment_name = f"flexible_batch_test_{timestamp}"
    
    exp_dir = results_base / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    return exp_dir


def setup_run_directory(exp_dir: Path, B_E: int, run_id: int) -> Path:
    """Create isolated directory for a single run within experiment."""
    run_dir = exp_dir / f"BE{B_E:03d}_run{run_id:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def setup_probe_logging(probe: OfflineEntropyProbe, log_file: Path):
    """Add FileHandler to probe's logger to capture all output."""
    # Get the probe's logger
    logger = probe.logger
    
    # Create file handler
    file_handler = logging.FileHandler(log_file, mode='w')
    
    # Use same format as the probe's StreamHandler
    formatter = logging.Formatter(
        f'[Rank {probe.rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    
    # Add to probe's logger
    logger.addHandler(file_handler)
    
    # Return handler so we can remove it later
    return file_handler


def create_run_config(base_config_path: str, run_dir: Path, B_E: int, B_U: int) -> str:
    """Create config file for this specific run."""
    # Load base config
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update batch sizes
    config['batch_config']['B_E'] = B_E
    config['batch_config']['B_U'] = B_U  
    config['batch_config']['B'] = B_E + B_U
    
    # NOTE: We don't set results_path here because run_mixed_probe doesn't use it
    # We'll manually save the results returned by run_mixed_probe()
    config['output']['save_results'] = True  # Keep for potential future use
    config['output']['log_level'] = 'DEBUG'  # Always capture debug info
    
    # Enable conditional variance only (focus on the new estimator)
    config['probe_rework']['compute_conditional_variance'] = True
    config['probe_rework']['compute_vx_vy_variance'] = False
    config['probe_rework']['compute_importance_sampling'] = False
    
    # Save config to run directory
    config_path = run_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return str(config_path)


def run_single_test(run_dir: Path, config_path: str, checkpoint_base_path: str) -> Dict[str, Any]:
    """Run a single entropy probe test with full logging."""
    
    log_file = run_dir / 'probe_log.txt'
    results_file = run_dir / 'results.json'  # We'll manually save probe results here
    error_file = run_dir / 'error.txt'
    
    print(f"Running test in: {run_dir.name}")
    start_time = time.time()
    
    try:
        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Set up proper LoRA checkpoint paths
        model_path = checkpoint_base_path + "/model" if not checkpoint_base_path.endswith("/model") else checkpoint_base_path
        optimizer_path = checkpoint_base_path.replace("/model", "/optimizer.pt") if checkpoint_base_path.endswith("/model") else checkpoint_base_path + "/optimizer.pt"
        
        config['checkpoint']['checkpoint_path'] = model_path
        config['checkpoint']['optimizer_path'] = optimizer_path
        
        # Create probe
        probe = OfflineEntropyProbe(config)
        
        # Set up logging to capture ALL output to file
        file_handler = setup_probe_logging(probe, log_file)
        
        try:
            # Run the probe (this will log everything to our file)
            results = probe.run_mixed_probe()
            runtime = time.time() - start_time
            
            # MANUALLY SAVE RESULTS.JSON (since run_mixed_probe doesn't do it)
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Extract key metrics
            return {
                'success': True,
                'runtime': runtime,
                'deltaH1': results.get('deltaH1'),
                'SE_conditional': results.get('SE_conditional'),
                'B_E': results.get('B_E'),
                'B_U': results.get('B_U'),
                'bars_dot': results.get('bars_dot'),
                'full_results': results,
                'log_file': str(log_file),
                'results_json': str(results_file)  # Now this will exist!
            }
            
        finally:
            # Always clean up the file handler
            if file_handler:
                probe.logger.removeHandler(file_handler)
                file_handler.close()
                
    except Exception as e:
        runtime = time.time() - start_time
        error_msg = f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        
        # Save error info
        with open(error_file, 'w') as f:
            f.write(error_msg)
        
        print(f"  ❌ Failed after {runtime:.1f}s: {str(e)}")
        
        return {
            'success': False,
            'runtime': runtime,
            'error': str(e),
            'error_file': str(error_file)
        }


def analyze_results(all_results: List[Dict], B_E_values: List[int]) -> Dict[str, Any]:
    """Analyze convergence patterns across batch sizes."""
    
    # Group by B_E
    by_B_E = {}
    for result in all_results:
        if not result.get('success'):
            continue
            
        B_E = result.get('B_E') or result.get('B_E_target')
        if B_E is None:
            continue
            
        if B_E not in by_B_E:
            by_B_E[B_E] = []
        by_B_E[B_E].append(result)
    
    # Compute statistics for each B_E
    summary = {}
    for B_E in B_E_values:
        if B_E not in by_B_E:
            continue
            
        runs = by_B_E[B_E]
        delta_h1_values = [r['deltaH1'] for r in runs if r.get('deltaH1') is not None]
        se_values = [r['SE_conditional'] for r in runs if r.get('SE_conditional') is not None]
        
        if delta_h1_values:
            mean_delta = sum(delta_h1_values) / len(delta_h1_values)
            std_delta = (sum((x - mean_delta)**2 for x in delta_h1_values) / max(len(delta_h1_values) - 1, 1))**0.5
            
            summary[B_E] = {
                'successful_runs': len(runs),
                'mean_deltaH1': mean_delta,
                'std_deltaH1': std_delta,
                'delta_values': delta_h1_values,
                'mean_SE_conditional': sum(se_values) / len(se_values) if se_values else None,
                'mean_signal_to_noise': abs(sum(se_values) / len(se_values) / mean_delta) if se_values and mean_delta != 0 else None
            }
    
    # Analyze scaling patterns
    scaling_analysis = {}
    if len(summary) >= 2:
        sorted_B_E = sorted(summary.keys())
        ref_B_E = sorted_B_E[0]
        ref_mean = summary[ref_B_E]['mean_deltaH1']
        
        for B_E in sorted_B_E[1:]:
            curr_mean = summary[B_E]['mean_deltaH1']
            if ref_mean != 0:
                ratio = curr_mean / ref_mean
                expected_1_over_B_E = ref_B_E / B_E
                
                scaling_analysis[f"B_E_{ref_B_E}_to_{B_E}"] = {
                    'observed_ratio': ratio,
                    'expected_if_1_over_B_E': expected_1_over_B_E,
                    'suggests_convergence': abs(ratio - 1.0) < 0.1,
                    'suggests_1_over_B_E_scaling': abs(ratio - expected_1_over_B_E) < 0.1
                }
    
    return {
        'by_batch_size': summary,
        'scaling_analysis': scaling_analysis,
        'total_successful_runs': sum(s['successful_runs'] for s in summary.values()),
        'B_E_values_tested': list(summary.keys())
    }


def main():
    parser = argparse.ArgumentParser(description="Flexible Batch Size Test - FIXED LOGGING")
    parser.add_argument("--checkpoint", required=True,
                       help="Path to checkpoint directory")
    parser.add_argument("--config", 
                       default="entropy_experiments/configs/batch_convergence_test_config.yaml",
                       help="Base configuration file")
    parser.add_argument("--be-values", default="16,32,64",
                       help="Comma-separated B_E values to test (default: 16,32,64)")
    parser.add_argument("--bu", type=int, default=32,
                       help="Fixed B_U value (default: 32)")
    parser.add_argument("--runs", type=int, default=2,
                       help="Number of runs per B_E value (default: 2)")
    parser.add_argument("--results-dir", default="entropy_experiments/results",
                       help="Results directory (default: entropy_experiments/results)")
    parser.add_argument("--experiment-name", default=None,
                       help="Custom experiment name (default: timestamped)")
    
    args = parser.parse_args()
    
    # Parse B_E values
    B_E_values = [int(x.strip()) for x in args.be_values.split(',')]
    
    # Validate inputs
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint path not found: {args.checkpoint}")
        return 1
        
    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        return 1
    
    # Setup results directory and experiment
    results_base = Path(args.results_dir)
    results_base.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped experiment directory
    exp_dir = create_experiment_directory(results_base, args.experiment_name)
    
    print("🧪 FLEXIBLE BATCH SIZE TEST - FIXED LOGGING")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"B_E values: {B_E_values}")
    print(f"B_U: {args.bu}")
    print(f"Runs per B_E: {args.runs}")
    print(f"Experiment directory: {exp_dir}")
    print(f"Results will be saved with full logging - NO TIMEOUTS")
    print("="*60)
    
    all_results = []
    test_start_time = time.time()
    
    for B_E in B_E_values:
        print(f"\n🔍 TESTING B_E = {B_E}")
        print("-" * 40)
        
        for run_id in range(1, args.runs + 1):
            # Create run directory within experiment
            run_dir = setup_run_directory(exp_dir, B_E, run_id)
            
            # Create config for this run
            config_path = create_run_config(args.config, run_dir, B_E, args.bu)
            
            # Run the test
            result = run_single_test(run_dir, config_path, args.checkpoint)
            result['B_E_target'] = B_E
            result['B_U_target'] = args.bu
            result['run_id'] = run_id
            result['run_dir'] = str(run_dir)
            
            all_results.append(result)
            
            # Print immediate results
            if result['success']:
                delta_h1 = result.get('deltaH1')
                se_cond = result.get('SE_conditional')
                runtime = result.get('runtime', 0)
                
                print(f"  ✅ Run {run_id}: δH₁={delta_h1:.6f}, SE_conditional={se_cond:.6f}, time={runtime:.1f}s")
                
                if se_cond and delta_h1 and abs(delta_h1) > 0:
                    ratio = abs(se_cond / delta_h1)
                    status = "✅ Good" if ratio < 0.1 else "⚠️ OK" if ratio < 0.2 else "❌ Poor"
                    print(f"       Signal/noise |SE/δH₁|={ratio:.3f} {status}")
                    
            else:
                print(f"  ❌ Run {run_id}: Failed ({result.get('error', 'Unknown error')})")
    
    # Final analysis
    total_time = time.time() - test_start_time
    print(f"\n{'='*60}")
    print("📊 CONVERGENCE ANALYSIS")
    print(f"{'='*60}")
    
    analysis = analyze_results(all_results, B_E_values)
    
    print(f"Total time: {total_time:.1f}s")
    print(f"Successful runs: {analysis['total_successful_runs']}/{len(all_results)}")
    
    # Print batch size analysis
    for B_E in B_E_values:
        if B_E in analysis['by_batch_size']:
            stats = analysis['by_batch_size'][B_E]
            print(f"\nB_E = {B_E}:")
            print(f"  Successful runs: {stats['successful_runs']}")
            print(f"  Mean δH₁: {stats['mean_deltaH1']:.6f}")
            print(f"  Std δH₁: {stats['std_deltaH1']:.6f}")
            if stats['mean_signal_to_noise']:
                print(f"  Mean |SE/δH₁|: {stats['mean_signal_to_noise']:.3f}")
    
    # Print scaling analysis
    if analysis['scaling_analysis']:
        print(f"\n🔍 Scaling Analysis:")
        for comparison, data in analysis['scaling_analysis'].items():
            print(f"  {comparison}:")
            print(f"    Observed ratio: {data['observed_ratio']:.3f}")
            print(f"    Expected if 1/B_E: {data['expected_if_1_over_B_E']:.3f}")
            if data['suggests_convergence']:
                print(f"    → ✅ Suggests CONVERGENCE (good!)")
            elif data['suggests_1_over_B_E_scaling']:
                print(f"    → ❌ Suggests 1/B_E scaling (potential bug)")
            else:
                print(f"    → ⚠️ Other behavior")
    
    # Save comprehensive summary to experiment directory
    summary_data = {
        'test_config': {
            'B_E_values': B_E_values,
            'B_U': args.bu,
            'runs_per_B_E': args.runs,
            'checkpoint': args.checkpoint,
            'base_config': args.config,
            'total_time': total_time,
            'timestamp': datetime.now().isoformat(),
            'experiment_directory': str(exp_dir)
        },
        'analysis': analysis,
        'all_results': all_results
    }
    
    summary_file = exp_dir / 'summary.json'  # Save in experiment directory
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\n📁 Complete results saved to: {exp_dir}")
    print(f"📄 Summary: {summary_file}")
    print(f"\nEach run has:")
    print(f"  • config.yaml - exact config used")
    print(f"  • probe_log.txt - ALL debug logging (gradient sizes, etc.)")
    print(f"  • results.json - probe's JSON output (manually saved)")
    
    return 0


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        exit(130)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        traceback.print_exc()
        exit(1)