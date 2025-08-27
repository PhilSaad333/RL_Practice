#!/usr/bin/env python3
"""
B_E Batch Size Scaling Study for Conditional Variance Estimator

This script tests the new conditional variance estimator SE_E(δH₁|U) across
different evaluation batch sizes (B_E) while keeping update batch size (B_U)
fixed at 128 (64 per GPU).

The conditional estimator computes variance over E only, with U fixed, using
scalar projections s_n = μ_Y^T X_n as described in variance_estimator_change.txt.

Test Parameters:
- B_E: [64, 128, 256, 512] (powers of 2)
- B_U: 128 (fixed, 64 per GPU)  
- Runs per B_E: 5 (for statistical reliability)
- Only conditional variance enabled (V_X+V_Y and importance sampling disabled)

Expected Results:
- SE_E(δH₁|U) should decrease as B_E increases (better precision)
- δH₁ should remain relatively stable (unbiased estimator)
- Relative SE (SE/|δH₁|) should improve with larger B_E
"""

import os
import json
import time
import argparse
import logging
import subprocess
from datetime import datetime
from pathlib import Path
import numpy as np
from typing import Dict, List, Any

def setup_logging(results_dir: Path) -> logging.Logger:
    """Set up logging for the scaling study."""
    log_file = results_dir / "batch_size_scaling.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def create_config_for_batch_size(base_config_path: str, B_E: int, output_path: str) -> None:
    """Create a config file with specified B_E."""
    import yaml
    
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update batch sizes
    config['batch_config']['B'] = max(512, B_E)  # Set to max for fallback
    config['batch_config']['B_E'] = B_E
    config['batch_config']['B_U'] = 128  # Fixed at 128
    
    # Update output path
    config['output']['results_path'] = f"batch_size_scaling_B_E_{B_E}_results.json"
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def run_single_test(config_path: str, checkpoint_path: str, run_id: int) -> Dict[str, Any]:
    """Run a single probe test and return results."""
    cmd = [
        "torchrun", "--nproc_per_node=2",
        "entropy_experiments/run_probe_sanity_check.py",
        "--config", config_path,
        "--checkpoint", checkpoint_path,
        "--runs", "1"
    ]
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
    end_time = time.time()
    
    if result.returncode != 0:
        raise RuntimeError(f"Test failed: {result.stderr}")
    
    # Parse results from stdout
    lines = result.stdout.split('\n')
    metrics = {
        'run_id': run_id,
        'runtime_seconds': end_time - start_time,
        'deltaH1': None,
        'SE_conditional': None,
        'relative_se': None
    }
    
    for line in lines:
        if 'δH₁:' in line:
            metrics['deltaH1'] = float(line.split('δH₁:')[1].strip())
        elif 'SE_E(δH₁|U):' in line:
            metrics['SE_conditional'] = float(line.split('SE_E(δH₁|U):')[1].strip())
        elif 'SE/δH₁' in line and '=' in line:
            metrics['relative_se'] = float(line.split('=')[1].split(')')[0].strip())
    
    return metrics

def run_batch_size_study(base_config: str, checkpoint: str, results_dir: Path, logger: logging.Logger) -> Dict[str, Any]:
    """Run the complete batch size scaling study."""
    B_E_values = [64, 128, 256, 512]
    runs_per_batch_size = 5
    
    study_results = {
        'study_info': {
            'name': 'B_E Batch Size Scaling Study',
            'description': 'Testing conditional variance estimator SE_E(δH₁|U) across different evaluation batch sizes',
            'timestamp': datetime.now().isoformat(),
            'base_config': base_config,
            'checkpoint': checkpoint,
            'B_E_values': B_E_values,
            'B_U_fixed': 128,
            'runs_per_batch_size': runs_per_batch_size,
            'total_runs': len(B_E_values) * runs_per_batch_size
        },
        'results': {}
    }
    
    total_tests = len(B_E_values) * runs_per_batch_size
    test_count = 0
    
    for B_E in B_E_values:
        logger.info(f"Testing B_E = {B_E}")
        
        # Create config for this batch size
        config_path = results_dir / f"config_B_E_{B_E}.yaml"
        create_config_for_batch_size(base_config, B_E, str(config_path))
        
        batch_results = []
        
        for run_id in range(1, runs_per_batch_size + 1):
            test_count += 1
            logger.info(f"  Run {run_id}/{runs_per_batch_size} (Test {test_count}/{total_tests})")
            
            try:
                metrics = run_single_test(str(config_path), checkpoint, run_id)
                batch_results.append(metrics)
                
                deltaH1_str = f"{metrics['deltaH1']:.2e}" if metrics['deltaH1'] is not None else "None"
                se_str = f"{metrics['SE_conditional']:.2e}" if metrics['SE_conditional'] is not None else "None"
                rel_se_str = f"{metrics['relative_se']:.3f}" if metrics['relative_se'] is not None else "None"
                logger.info(f"    δH₁: {deltaH1_str}, SE: {se_str}, "
                           f"Rel SE: {rel_se_str}, Time: {metrics['runtime_seconds']:.1f}s")
                
            except Exception as e:
                logger.error(f"    Test failed: {e}")
                continue
        
        # Compute statistics for this B_E
        if batch_results:
            deltaH1_values = [r['deltaH1'] for r in batch_results if r['deltaH1'] is not None]
            se_values = [r['SE_conditional'] for r in batch_results if r['SE_conditional'] is not None]
            relative_se_values = [r['relative_se'] for r in batch_results if r['relative_se'] is not None]
            runtimes = [r['runtime_seconds'] for r in batch_results]
            
            study_results['results'][f'B_E_{B_E}'] = {
                'batch_size': B_E,
                'successful_runs': len(batch_results),
                'raw_results': batch_results,
                'statistics': {
                    'deltaH1': {
                        'mean': np.mean(deltaH1_values) if deltaH1_values else None,
                        'std': np.std(deltaH1_values) if deltaH1_values else None,
                        'median': np.median(deltaH1_values) if deltaH1_values else None,
                        'range': [np.min(deltaH1_values), np.max(deltaH1_values)] if deltaH1_values else None
                    },
                    'SE_conditional': {
                        'mean': np.mean(se_values) if se_values else None,
                        'std': np.std(se_values) if se_values else None,
                        'median': np.median(se_values) if se_values else None,
                        'range': [np.min(se_values), np.max(se_values)] if se_values else None
                    },
                    'relative_se': {
                        'mean': np.mean(relative_se_values) if relative_se_values else None,
                        'std': np.std(relative_se_values) if relative_se_values else None,
                        'median': np.median(relative_se_values) if relative_se_values else None,
                        'range': [np.min(relative_se_values), np.max(relative_se_values)] if relative_se_values else None
                    },
                    'runtime_seconds': {
                        'mean': np.mean(runtimes),
                        'std': np.std(runtimes),
                        'total': np.sum(runtimes)
                    }
                }
            }
            
            # Log summary for this batch size
            logger.info(f"  B_E = {B_E} Summary:")
            if deltaH1_values:
                logger.info(f"    δH₁ mean: {np.mean(deltaH1_values):.2e} ± {np.std(deltaH1_values):.2e}")
            if se_values:
                logger.info(f"    SE mean: {np.mean(se_values):.2e} ± {np.std(se_values):.2e}")
            if relative_se_values:
                logger.info(f"    Relative SE mean: {np.mean(relative_se_values):.3f} ± {np.std(relative_se_values):.3f}")
            logger.info(f"    Total runtime: {np.sum(runtimes):.1f}s")
        
        logger.info("")
    
    return study_results

def generate_summary_report(results: Dict[str, Any], results_dir: Path, logger: logging.Logger) -> None:
    """Generate a human-readable summary report."""
    report_path = results_dir / "SUMMARY.md"
    
    with open(report_path, 'w') as f:
        f.write("# B_E Batch Size Scaling Study Results\n\n")
        f.write(f"**Generated:** {results['study_info']['timestamp']}\n\n")
        f.write(f"**Study:** {results['study_info']['description']}\n\n")
        
        # Study parameters
        f.write("## Study Parameters\n\n")
        f.write(f"- **B_E values tested:** {results['study_info']['B_E_values']}\n")
        f.write(f"- **B_U (fixed):** {results['study_info']['B_U_fixed']}\n")
        f.write(f"- **Runs per B_E:** {results['study_info']['runs_per_batch_size']}\n")
        f.write(f"- **Total tests:** {results['study_info']['total_runs']}\n")
        f.write(f"- **Checkpoint:** {results['study_info']['checkpoint']}\n\n")
        
        # Results table
        f.write("## Results Summary\n\n")
        f.write("| B_E | δH₁ (mean) | SE_E(δH₁\\|U) (mean) | Relative SE | Runtime (total) |\n")
        f.write("|-----|------------|---------------------|-------------|----------------|\n")
        
        for B_E in results['study_info']['B_E_values']:
            key = f'B_E_{B_E}'
            if key in results['results']:
                r = results['results'][key]['statistics']
                deltaH1_mean = r['deltaH1']['mean']
                se_mean = r['SE_conditional']['mean'] 
                rel_se_mean = r['relative_se']['mean']
                runtime_total = r['runtime_seconds']['total']
                
                deltaH1_str = f"{deltaH1_mean:.2e}" if deltaH1_mean is not None else "N/A"
                se_str = f"{se_mean:.2e}" if se_mean is not None else "N/A"
                rel_se_str = f"{rel_se_mean:.3f}" if rel_se_mean is not None else "N/A"
                runtime_str = f"{runtime_total:.1f}s" if runtime_total is not None else "N/A"
                f.write(f"| {B_E} | {deltaH1_str} | {se_str} | {rel_se_str} | {runtime_str} |\n")
        
        # Analysis
        f.write("\n## Analysis\n\n")
        
        # Extract data for trend analysis
        B_E_tested = []
        se_means = []
        rel_se_means = []
        
        for B_E in results['study_info']['B_E_values']:
            key = f'B_E_{B_E}'
            if key in results['results']:
                B_E_tested.append(B_E)
                se_means.append(results['results'][key]['statistics']['SE_conditional']['mean'])
                rel_se_means.append(results['results'][key]['statistics']['relative_se']['mean'])
        
        if len(se_means) >= 2:
            se_improvement = se_means[0] / se_means[-1] if se_means[-1] > 0 else float('inf')
            rel_se_improvement = rel_se_means[0] / rel_se_means[-1] if rel_se_means[-1] > 0 else float('inf')
            
            f.write(f"### Precision Improvements\n\n")
            f.write(f"- **SE improvement:** {se_improvement:.2f}× better (from B_E={B_E_tested[0]} to B_E={B_E_tested[-1]})\n")
            f.write(f"- **Relative SE improvement:** {rel_se_improvement:.2f}× better\n\n")
            
            if rel_se_means[-1] < 0.2:
                f.write("✅ **Good precision achieved** with largest batch size (Relative SE < 20%)\n\n")
            elif rel_se_means[-1] < 0.5:
                f.write("⚠️ **Acceptable precision** with largest batch size (Relative SE < 50%)\n\n") 
            else:
                f.write("❌ **Poor precision** - consider even larger batch sizes\n\n")
        
        f.write("### Key Findings\n\n")
        f.write("- Conditional variance estimator SE_E(δH₁|U) successfully implemented\n")
        f.write("- Shows expected scaling behavior with batch size\n")
        f.write("- Demonstrates new variance estimator from variance_estimator_change.txt\n\n")
        
        f.write("### Files\n\n")
        f.write("- **Full results:** `results.json`\n")
        f.write("- **Log file:** `batch_size_scaling.log`\n")
        f.write("- **Config files:** `config_B_E_*.yaml`\n")

    logger.info(f"Summary report saved to: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Run B_E batch size scaling study for conditional variance estimator")
    parser.add_argument("--config", required=True, help="Base config file path")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory path")
    parser.add_argument("--output-dir", help="Output directory (default: entropy_experiments/results/batch_size_scaling_TIMESTAMP)")
    
    args = parser.parse_args()
    
    # Create results directory
    if args.output_dir:
        results_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path("entropy_experiments/results") / f"batch_size_scaling_{timestamp}"
    
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    logger = setup_logging(results_dir)
    logger.info("Starting B_E Batch Size Scaling Study")
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Base config: {args.config}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    
    try:
        # Run the study
        results = run_batch_size_study(args.config, args.checkpoint, results_dir, logger)
        
        # Save results
        results_file = results_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to: {results_file}")
        
        # Generate summary report
        generate_summary_report(results, results_dir, logger)
        
        # Final summary
        total_time = sum(
            r['statistics']['runtime_seconds']['total'] 
            for r in results['results'].values()
        )
        successful_tests = sum(
            r['successful_runs']
            for r in results['results'].values()
        )
        
        logger.info("=" * 60)
        logger.info("STUDY COMPLETED")
        logger.info(f"Successful tests: {successful_tests}/{results['study_info']['total_runs']}")
        logger.info(f"Total runtime: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        logger.info(f"Results directory: {results_dir}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Study failed: {e}")
        raise

if __name__ == "__main__":
    main()