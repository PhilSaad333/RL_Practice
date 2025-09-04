#!/usr/bin/env python3
"""
Main entropy probe runner for A100 GPU with production settings.
Designed for Lord Krang's entropy experiments with RL-trained checkpoints.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import subprocess
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"📋 Loaded config from {config_path}")
    logger.info(f"  B_E={config['batch_config']['B_E']}, "
                f"B_U={config['batch_config']['B_U']}, "
                f"G={config['batch_config']['G']}")
    return config


def run_entropy_probe(
    config_path: str,
    output_dir: Optional[str] = None,
    num_iterations: int = 1,
    temperatures: Optional[List[float]] = None,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Run the offline entropy probe with specified configuration.
    
    Args:
        config_path: Path to YAML configuration file
        output_dir: Directory for results (auto-generated if None)
        num_iterations: Number of iterations to run
        temperatures: List of temperatures to test (overrides config if provided)
        dry_run: If True, only show what would be run without executing
        
    Returns:
        Dictionary with results from all runs
    """
    
    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = f"entropy_experiments/results/a100_run_{timestamp}"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Output directory: {output_path}")
    
    # Load config
    config = load_config(config_path)
    
    # Temperature override if specified
    if temperatures:
        logger.info(f"🌡️ Temperature override: {temperatures}")
        original_temp = config['generation']['temperature']
        
    results = {
        'config_path': config_path,
        'timestamp': datetime.now().isoformat(),
        'iterations': [],
        'summary': {}
    }
    
    # Run iterations
    for iteration in range(num_iterations):
        logger.info(f"\n{'='*60}")
        logger.info(f"🚀 Starting iteration {iteration + 1}/{num_iterations}")
        logger.info(f"{'='*60}")
        
        iteration_results = {}
        
        # Handle temperature variations
        temps_to_run = temperatures if temperatures else [config['generation']['temperature']]
        
        for temp in temps_to_run:
            logger.info(f"\n🌡️ Running with temperature={temp}")
            
            # Update config if needed
            if temperatures:
                config['generation']['temperature'] = temp
                # Save temporary config
                temp_config_path = output_path / f"config_temp_{temp:.2f}.yaml"
                with open(temp_config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
                config_to_use = str(temp_config_path)
            else:
                config_to_use = config_path
            
            # Construct command
            cmd = [
                sys.executable,
                "entropy_experiments/offline_entropy_probe.py",
                "--config", config_to_use
            ]
            
            # Add output path
            result_file = output_path / f"probe_iter{iteration}_temp{temp:.2f}.json"
            cmd.extend(["--output", str(result_file)])
            
            if dry_run:
                logger.info(f"🎯 [DRY RUN] Would execute: {' '.join(cmd)}")
                iteration_results[f"temp_{temp}"] = {"status": "dry_run"}
            else:
                logger.info(f"⚡ Executing entropy probe...")
                start_time = time.time()
                
                try:
                    # Run the probe
                    env = os.environ.copy()
                    env['PYTHONPATH'] = '.'
                    
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        env=env,
                        timeout=1800  # 30 minute timeout
                    )
                    
                    elapsed = time.time() - start_time
                    
                    if result.returncode == 0:
                        logger.info(f"✅ Completed successfully in {elapsed:.1f}s")
                        
                        # Try to load results
                        if result_file.exists():
                            with open(result_file, 'r') as f:
                                probe_results = json.load(f)
                            iteration_results[f"temp_{temp}"] = {
                                "status": "success",
                                "elapsed_time": elapsed,
                                "results": probe_results
                            }
                            
                            # Log key metrics
                            if 'true_delta_h' in probe_results:
                                delta_h = probe_results['true_delta_h'].get('delta_h', 'N/A')
                                ess_frac = probe_results['true_delta_h'].get('ESS_fraction', 'N/A')
                                logger.info(f"  📊 ΔH = {delta_h:.4f}, ESS = {ess_frac:.1%}")
                        else:
                            logger.warning(f"⚠️ Result file not found: {result_file}")
                            iteration_results[f"temp_{temp}"] = {
                                "status": "success_no_output",
                                "elapsed_time": elapsed
                            }
                    else:
                        logger.error(f"❌ Failed with return code {result.returncode}")
                        logger.error(f"stderr: {result.stderr[-1000:]}")  # Last 1000 chars
                        iteration_results[f"temp_{temp}"] = {
                            "status": "failed",
                            "return_code": result.returncode,
                            "elapsed_time": elapsed,
                            "error": result.stderr[-1000:]
                        }
                        
                except subprocess.TimeoutExpired:
                    logger.error(f"⏰ Timeout after 30 minutes")
                    iteration_results[f"temp_{temp}"] = {
                        "status": "timeout",
                        "elapsed_time": 1800
                    }
                except Exception as e:
                    logger.error(f"💥 Exception: {e}")
                    iteration_results[f"temp_{temp}"] = {
                        "status": "exception",
                        "error": str(e)
                    }
        
        results['iterations'].append(iteration_results)
    
    # Compute summary statistics
    logger.info(f"\n{'='*60}")
    logger.info(f"📈 Computing summary statistics...")
    
    all_delta_h = []
    all_ess = []
    
    for iteration in results['iterations']:
        for temp_key, temp_results in iteration.items():
            if temp_results.get('status') == 'success' and 'results' in temp_results:
                probe_res = temp_results['results']
                if 'true_delta_h' in probe_res:
                    if 'delta_h' in probe_res['true_delta_h']:
                        all_delta_h.append(probe_res['true_delta_h']['delta_h'])
                    if 'ESS_fraction' in probe_res['true_delta_h']:
                        all_ess.append(probe_res['true_delta_h']['ESS_fraction'])
    
    if all_delta_h:
        results['summary'] = {
            'delta_h_mean': sum(all_delta_h) / len(all_delta_h),
            'delta_h_min': min(all_delta_h),
            'delta_h_max': max(all_delta_h),
            'ess_mean': sum(all_ess) / len(all_ess) if all_ess else None,
            'ess_min': min(all_ess) if all_ess else None,
            'successful_runs': len(all_delta_h),
            'total_runs': num_iterations * len(temps_to_run)
        }
        
        logger.info(f"✅ Successful runs: {results['summary']['successful_runs']}/{results['summary']['total_runs']}")
        logger.info(f"📊 ΔH: mean={results['summary']['delta_h_mean']:.4f}, "
                   f"range=[{results['summary']['delta_h_min']:.4f}, "
                   f"{results['summary']['delta_h_max']:.4f}]")
        if results['summary']['ess_mean']:
            logger.info(f"📊 ESS: mean={results['summary']['ess_mean']:.1%}, "
                       f"min={results['summary']['ess_min']:.1%}")
    
    # Save full results
    full_results_path = output_path / "full_results.json"
    with open(full_results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"💾 Saved full results to {full_results_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run entropy probe with A100-optimized configuration"
    )
    parser.add_argument(
        "--config",
        default="entropy_experiments/configs/A100_config.yaml",
        help="Configuration file path"
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for results (auto-generated if not specified)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of iterations to run (default: 1)"
    )
    parser.add_argument(
        "--temps",
        type=str,
        help="Comma-separated list of temperatures to test (e.g., '0.7,1.0')"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without executing"
    )
    parser.add_argument(
        "--gpu-check",
        action="store_true",
        help="Check GPU availability before running"
    )
    
    args = parser.parse_args()
    
    # Parse temperatures if provided
    temperatures = None
    if args.temps:
        temperatures = [float(t.strip()) for t in args.temps.split(',')]
        logger.info(f"🌡️ Testing temperatures: {temperatures}")
    
    # GPU check if requested
    if args.gpu_check:
        try:
            import torch
            if torch.cuda.is_available():
                logger.info(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
                logger.info(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            else:
                logger.error("❌ No GPU available!")
                if not args.dry_run:
                    sys.exit(1)
        except ImportError:
            logger.warning("⚠️ PyTorch not available for GPU check")
    
    # Run the probe
    logger.info(f"\n🚀 Starting A100 Entropy Probe Runner")
    logger.info(f"📋 Config: {args.config}")
    logger.info(f"🔄 Iterations: {args.iterations}")
    
    results = run_entropy_probe(
        config_path=args.config,
        output_dir=args.output_dir,
        num_iterations=args.iterations,
        temperatures=temperatures,
        dry_run=args.dry_run
    )
    
    # Print final summary
    if results.get('summary'):
        logger.info(f"\n{'='*60}")
        logger.info(f"🏁 FINAL SUMMARY")
        logger.info(f"{'='*60}")
        summary = results['summary']
        logger.info(f"Success rate: {summary['successful_runs']}/{summary['total_runs']}")
        logger.info(f"Mean ΔH: {summary['delta_h_mean']:.4f}")
        if summary.get('ess_mean'):
            logger.info(f"Mean ESS: {summary['ess_mean']:.1%}")
    
    logger.info(f"\n✨ Done!")


if __name__ == "__main__":
    main()