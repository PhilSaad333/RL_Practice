#!/usr/bin/env python3
"""
Entropy vs Learning Rate Linearity Test
=======================================

This script tests whether entropy changes are approximately linear in learning rate
for small learning rates. For each learning rate:

1. Load a checkpoint from a previous training run (model + optimizer state)
2. Run exactly 2 optimizer steps to get 2 entropy measurements  
3. Compute entropy change = entropy_after - entropy_before
4. Repeat N times for statistical reliability
5. Analyze linearity across learning rates

Usage:
    python scripts/entropy_vs_learning_rate.py \\
        --checkpoint /path/to/training_state/step_X \\
        --base-config rl_training/cfg/h100_dual_gns_64step.yaml \\
        --learning-rates 1e-8,1e-7,1e-6,1e-5,1e-4 \\
        --num-repeats 5 \\
        --output-dir entropy_lr_results

Requirements:
    - 2x H100 GPUs (uses DDP)
    - Checkpoint with model/ + optimizer.pt + scheduler.pt
    - Buffer size 64 (configurable)
    - Saves comprehensive experimental data and analysis results
"""

import os
import sys
import json
import yaml
import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def create_experiment_config(base_config_path: str, learning_rate: float, 
                           output_dir: Path, run_id: str) -> Path:
    """Create a temporary config file for this learning rate experiment."""
    
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Modify config for entropy measurement experiment
    config['lr'] = learning_rate
    config['total_steps'] = 2  # Exactly 2 steps for before/after entropy measurement
    config['save_every'] = 10   # High value so it never activates during 2 steps
    config['buffer_size'] = 64  # Reduced buffer size
    config['microbatch_size'] = 4  # Optimal for 2x H100
    
    # Enable simple entropy probe for measurements
    config['simple_entropy_probe'] = {
        'enabled': True,
        'debug': True,
        'preconditioning_mode': 'previous_step',
        'log_every': 1
    }
    
    # Disable other probes to focus on entropy
    config.setdefault('gns_probe', {})['enabled'] = False
    config.setdefault('entropy_probe', {})['enabled'] = False
    
    # Create run-specific config file
    config_path = output_dir / f"config_lr_{learning_rate:.0e}_{run_id}.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    return config_path

def run_training_experiment(config_path: Path, checkpoint_path: str, 
                           output_dir: Path, learning_rate: float, run_id: str) -> Dict[str, Any]:
    """Run a single 2-step training experiment and extract entropy values."""
    
    print(f"🔬 Running experiment: lr={learning_rate:.0e}, run={run_id}")
    
    # Prepare environment and command
    env = os.environ.copy()
    env.update({
        'CUDA_VISIBLE_DEVICES': '0,1',  # Use both H100s
        'WORLD_SIZE': '2',
        'MASTER_ADDR': 'localhost',
        'MASTER_PORT': '12345',
        'PYTHONPATH': str(project_root),  # Set PYTHONPATH to project root
    })
    
    # Prepare output directory for this run
    run_output_dir = output_dir / f"lr_{learning_rate:.0e}_{run_id}"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Command to run training with proper PYTHONPATH
    cmd = [
        'torchrun', '--nproc_per_node=2', 
        'rl_training/runners/rl_runner.py',
        '--cfg', str(config_path),
        '--ckpt', checkpoint_path
    ]
    
    try:
        # Run the training experiment
        result = subprocess.run(
            cmd, 
            cwd=str(project_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        if result.returncode != 0:
            print(f"❌ Training failed for lr={learning_rate:.0e}, run={run_id}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return {'success': False, 'error': result.stderr}
        
        # Parse entropy measurements from logs
        entropy_measurements = parse_entropy_from_logs(result.stdout)
        
        if len(entropy_measurements) < 2:
            print(f"⚠️  Insufficient entropy measurements for lr={learning_rate:.0e}, run={run_id}")
            return {'success': False, 'error': 'Insufficient entropy measurements'}
        
        # Calculate entropy change
        entropy_before = entropy_measurements[0]
        entropy_after = entropy_measurements[1] 
        entropy_change = entropy_after - entropy_before
        
        result_data = {
            'success': True,
            'learning_rate': learning_rate,
            'run_id': run_id,
            'entropy_before': entropy_before,
            'entropy_after': entropy_after,
            'entropy_change': entropy_change,
            'stdout': result.stdout[:1000] + '...' if len(result.stdout) > 1000 else result.stdout,
            'stderr': result.stderr[:1000] + '...' if len(result.stderr) > 1000 else result.stderr
        }
        
        # Save detailed results
        with open(run_output_dir / 'result.json', 'w') as f:
            json.dump(result_data, f, indent=2)
            
        print(f"✅ lr={learning_rate:.0e}, run={run_id}: ΔH={entropy_change:.6f}")
        return result_data
        
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout for lr={learning_rate:.0e}, run={run_id}")
        return {'success': False, 'error': 'Timeout'}
    except Exception as e:
        print(f"💥 Exception for lr={learning_rate:.0e}, run={run_id}: {e}")
        return {'success': False, 'error': str(e)}

def parse_entropy_from_logs(stdout: str) -> List[float]:
    """Extract entropy measurements from training logs."""
    entropy_values = []
    
    # Look for simple entropy probe output in logs
    for line in stdout.split('\\n'):
        if 'entropy:' in line.lower() or 'simple_entropy' in line.lower():
            # Try to extract numerical value
            # This is a simple parser - may need adjustment based on actual log format
            import re
            numbers = re.findall(r'[-+]?\\d*\\.?\\d+(?:[eE][-+]?\\d+)?', line)
            for num_str in numbers:
                try:
                    val = float(num_str)
                    if abs(val) > 0.001:  # Filter out obviously wrong values
                        entropy_values.append(val)
                        break
                except ValueError:
                    continue
    
    return entropy_values

def analyze_results(all_results: List[Dict[str, Any]], output_dir: Path):
    """Analyze results and test for linearity."""
    
    successful_results = [r for r in all_results if r['success']]
    
    if not successful_results:
        print("❌ No successful experiments!")
        return
    
    # Group by learning rate
    lr_groups = {}
    for result in successful_results:
        lr = result['learning_rate']
        if lr not in lr_groups:
            lr_groups[lr] = []
        lr_groups[lr].append(result['entropy_change'])
    
    # Compute statistics for each learning rate
    analysis = {}
    for lr, changes in lr_groups.items():
        analysis[lr] = {
            'learning_rate': lr,
            'n_samples': len(changes),
            'mean_entropy_change': np.mean(changes),
            'std_entropy_change': np.std(changes),
            'stderr_entropy_change': np.std(changes) / np.sqrt(len(changes)),
            'min_entropy_change': np.min(changes),
            'max_entropy_change': np.max(changes),
            'all_changes': changes
        }
    
    # Sort by learning rate for analysis
    sorted_analysis = sorted(analysis.values(), key=lambda x: x['learning_rate'])
    
    # Test for linearity
    learning_rates = [a['learning_rate'] for a in sorted_analysis]
    mean_changes = [a['mean_entropy_change'] for a in sorted_analysis]
    std_errors = [a['stderr_entropy_change'] for a in sorted_analysis]
    
    # Linear regression
    coeffs = np.polyfit(learning_rates, mean_changes, 1)
    slope, intercept = coeffs
    r_squared = np.corrcoef(learning_rates, mean_changes)[0, 1] ** 2
    
    print(f"\\n📊 LINEARITY ANALYSIS")
    print(f"{'='*50}")
    print(f"Linear fit: ΔH = {slope:.2e} * lr + {intercept:.6f}")
    print(f"R² = {r_squared:.4f}")
    print(f"Slope = {slope:.2e} (entropy change per unit learning rate)")
    
    # Create visualizations
    create_plots(sorted_analysis, slope, intercept, output_dir)
    
    # Save analysis results
    analysis_data = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_experiments': len(all_results),
            'successful_experiments': len(successful_results),
            'learning_rates_tested': len(lr_groups),
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared
        },
        'per_learning_rate': {str(a['learning_rate']): a for a in sorted_analysis}
    }
    
    with open(output_dir / 'analysis.json', 'w') as f:
        json.dump(analysis_data, f, indent=2, default=str)
    
    # Print detailed results table
    print(f"\\n📋 DETAILED RESULTS")
    print(f"{'Learning Rate':<12} {'N':<3} {'Mean ΔH':<12} {'Std Error':<12} {'Range':<20}")
    print(f"{'-'*60}")
    
    for a in sorted_analysis:
        lr_str = f"{a['learning_rate']:.0e}"
        mean_str = f"{a['mean_entropy_change']:.6f}"
        stderr_str = f"{a['stderr_entropy_change']:.6f}"
        range_str = f"[{a['min_entropy_change']:.4f}, {a['max_entropy_change']:.4f}]"
        
        print(f"{lr_str:<12} {a['n_samples']:<3} {mean_str:<12} {stderr_str:<12} {range_str:<20}")

def create_plots(sorted_analysis: List[Dict], slope: float, intercept: float, output_dir: Path):
    """Create visualization plots."""
    
    learning_rates = [a['learning_rate'] for a in sorted_analysis]
    mean_changes = [a['mean_entropy_change'] for a in sorted_analysis]
    std_errors = [a['stderr_entropy_change'] for a in sorted_analysis]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Entropy change vs Learning rate with error bars
    ax1.errorbar(learning_rates, mean_changes, yerr=std_errors, 
                 fmt='o-', capsize=5, capthick=2, markersize=8)
    
    # Add linear fit line
    fit_line = [slope * lr + intercept for lr in learning_rates]
    ax1.plot(learning_rates, fit_line, '--', color='red', alpha=0.7, 
             label=f'Linear fit: y = {slope:.2e}x + {intercept:.4f}')
    
    ax1.set_xlabel('Learning Rate')
    ax1.set_ylabel('Mean Entropy Change')
    ax1.set_title('Entropy Change vs Learning Rate')
    ax1.set_xscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Residuals from linear fit
    residuals = [mean_changes[i] - fit_line[i] for i in range(len(mean_changes))]
    ax2.scatter(learning_rates, residuals, s=60, alpha=0.7)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Learning Rate')
    ax2.set_ylabel('Residuals (Data - Fit)')
    ax2.set_title('Residuals from Linear Fit')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'entropy_vs_learning_rate.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create raw data scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_analysis)))
    
    for i, a in enumerate(sorted_analysis):
        lr = a['learning_rate'] 
        changes = a['all_changes']
        x_vals = [lr] * len(changes)
        ax.scatter(x_vals, changes, alpha=0.6, color=colors[i], 
                  s=50, label=f'lr={lr:.0e}')
    
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Entropy Change')
    ax.set_title('All Entropy Change Measurements')
    ax.set_xscale('log')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'raw_measurements.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Plots saved to {output_dir}/")

def main():
    parser = argparse.ArgumentParser(description='Test entropy vs learning rate linearity')
    parser.add_argument('--checkpoint', required=True, 
                       help='Path to checkpoint directory (containing model/, optimizer.pt, etc.)')
    parser.add_argument('--base-config', required=True,
                       help='Path to base YAML config file')
    parser.add_argument('--learning-rates', default='1e-8,1e-7,1e-6,1e-5,1e-4',
                       help='Comma-separated learning rates to test')
    parser.add_argument('--num-repeats', type=int, default=16,
                       help='Number of repeat experiments per learning rate')
    parser.add_argument('--output-dir', default='entropy_lr_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Parse learning rates
    learning_rates = [float(lr.strip()) for lr in args.learning_rates.split(',')]
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = Path(args.output_dir) / f'entropy_lr_experiment_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🧪 ENTROPY vs LEARNING RATE LINEARITY TEST")
    print(f"{'='*50}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Learning rates: {learning_rates}")
    print(f"Repeats per LR: {args.num_repeats}")
    print(f"Output: {output_dir}")
    print(f"{'='*50}")
    
    # Validate checkpoint path
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint path does not exist: {args.checkpoint}")
        return 1
    
    # Validate base config
    if not os.path.exists(args.base_config):
        print(f"❌ Base config does not exist: {args.base_config}")
        return 1
    
    # Run all experiments
    all_results = []
    total_experiments = len(learning_rates) * args.num_repeats
    experiment_count = 0
    
    for learning_rate in learning_rates:
        print(f"\\n🎯 Testing learning rate: {learning_rate:.0e}")
        
        for run_id in range(args.num_repeats):
            experiment_count += 1
            print(f"[{experiment_count}/{total_experiments}] ", end="")
            
            # Create experiment config
            config_path = create_experiment_config(
                args.base_config, learning_rate, output_dir, f"run_{run_id}"
            )
            
            # Run experiment
            result = run_training_experiment(
                config_path, args.checkpoint, output_dir, learning_rate, f"run_{run_id}"
            )
            
            all_results.append(result)
            
            # Clean up config file
            config_path.unlink()
    
    # Analyze results  
    print(f"\\n🔍 ANALYZING RESULTS...")
    analyze_results(all_results, output_dir)
    
    # Save complete experimental data
    with open(output_dir / 'raw_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\\n💾 SAVED DATA:")
    print(f"  - raw_results.json: All experimental measurements")
    print(f"  - analysis.json: Statistical analysis and linear fit")
    print(f"  - *.png: Visualization plots")
    print(f"  - lr_*/result.json: Individual experiment details")
    
    print(f"\\n✅ Experiment complete! Results saved to {output_dir}/")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())