#!/usr/bin/env python3
"""
Stage 3 Probe Benchmarking Suite

Runs the mixed E/U batch entropy probe N times on a checkpoint and collects 
comprehensive statistics for analysis. Designed for systematic evaluation
of probe performance, variance, and convergence properties.

Key metrics logged:
- δH₁ (Stage 1 entropy estimate)
- ΔH_true (Stage 2 ground-truth entropy via SNIS)
- Fractional variance: (V_X + V_Y) / δH₁²
- Statistical diagnostics (ESS, confidence intervals, etc.)

Usage:
    # Single GPU validation
    python run_probe_benchmark.py --checkpoint /path/to/checkpoint --runs 10
    
    # Multi-GPU production runs
    torchrun --nproc_per_node=2 run_probe_benchmark.py --checkpoint /path/to/checkpoint --runs 20
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
from typing import Dict, List, Any, Optional
import numpy as np
from dataclasses import dataclass, asdict

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from entropy_experiments.offline_entropy_probe import OfflineEntropyProbe
from entropy_experiments import distributed_helpers


@dataclass
class ProbeRunResults:
    """Results from a single probe run."""
    run_id: int
    timestamp: str
    
    # Stage 1 - Primary entropy estimate
    deltaH1: float
    bars_dot: float
    learning_rate: float
    B_E: int
    B_U: int
    weighting_mode: str
    
    # Stage 2 - Variance estimation
    V_X: Optional[float] = None
    V_Y: Optional[float] = None
    SE_deltaH1: Optional[float] = None
    frac_var: Optional[float] = None  # (V_X + V_Y) / δH₁²
    
    # Stage 2 - Ground-truth measurement
    H_orig: Optional[float] = None
    H_upd: Optional[float] = None
    deltaH_true: Optional[float] = None  # Ground-truth ΔH via SNIS
    ESS: Optional[float] = None
    
    # Performance metrics
    total_time: float = 0.0
    peak_memory_gb: float = 0.0
    
    # Timing breakdown
    phase_0_time: float = 0.0  # Sampling
    phase_1_time: float = 0.0  # X accumulation
    phase_2_time: float = 0.0  # Y accumulation
    phase_3_time: float = 0.0  # All-reduce & computation
    phase_4_time: float = 0.0  # Variance estimation
    phase_5_time: float = 0.0  # Importance sampling
    
    # Distributed info
    is_distributed: bool = False
    world_size: int = 1
    
    def compute_derived_metrics(self):
        """Compute derived metrics after basic fields are set."""
        if self.deltaH1 != 0 and self.V_X is not None and self.V_Y is not None:
            # Correct fractional variance: (V_X + V_Y) / bars_dot²
            # bars_dot = deltaH1 / learning_rate (assuming lr = 2e-6)
            learning_rate = 2e-6  # TODO: Extract from results if available
            bars_dot = self.deltaH1 / learning_rate
            self.frac_var = (self.V_X + self.V_Y) / (bars_dot ** 2)


@dataclass 
class BenchmarkSummary:
    """Summary statistics across all probe runs."""
    n_runs: int
    checkpoint_path: str
    config_path: str
    start_time: str
    end_time: str
    total_duration: float
    
    # δH₁ statistics
    deltaH1_mean: float
    deltaH1_std: float
    deltaH1_min: float
    deltaH1_max: float
    deltaH1_ci_lower: float  # 95% confidence interval
    deltaH1_ci_upper: float
    
    # ΔH_true statistics (if available)
    deltaH_true_mean: Optional[float] = None
    deltaH_true_std: Optional[float] = None
    deltaH_true_min: Optional[float] = None
    deltaH_true_max: Optional[float] = None
    
    # Fractional variance statistics
    frac_var_mean: Optional[float] = None
    frac_var_std: Optional[float] = None
    frac_var_median: Optional[float] = None
    
    # Performance statistics
    avg_runtime: float = 0.0
    avg_peak_memory: float = 0.0
    
    # Convergence diagnostics
    deltaH1_cv: float = 0.0  # Coefficient of variation
    est_bias: Optional[float] = None  # δH₁ - ΔH_true if available
    est_rmse: Optional[float] = None


def setup_logging(level: str = "INFO", rank: int = 0, output_dir: str = ".") -> logging.Logger:
    """Setup logging configuration with rank prefix and file output."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(output_dir) / f"probe_benchmark_rank{rank}_{timestamp}.log"
    
    # Create formatter with rank info
    formatter = logging.Formatter(
        f'%(asctime)s - RANK{rank} - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(getattr(logging, level))
    
    # Console handler (only for rank 0)
    handlers = [file_handler]
    if rank == 0:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(getattr(logging, level))
        handlers.append(console_handler)
    
    # Setup logger
    logger = logging.getLogger("probe_benchmark")
    logger.setLevel(getattr(logging, level))
    logger.handlers.clear()  # Remove any existing handlers
    for handler in handlers:
        logger.addHandler(handler)
    
    return logger


def setup_distributed() -> tuple[bool, int, int, int]:
    """Setup distributed training if available."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # torchrun sets these environment variables
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # Initialize distributed backend
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        return True, rank, world_size, local_rank
    else:
        return False, 0, 1, 0


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def extract_results_from_probe_output(results: Dict[str, Any]) -> ProbeRunResults:
    """Extract structured results from probe output."""
    run_results = ProbeRunResults(
        run_id=0,  # Will be set by caller
        timestamp=datetime.now().isoformat(),
        
        # Stage 1 results
        deltaH1=results['deltaH1'],
        bars_dot=results['bars_dot'],
        learning_rate=results['learning_rate'],
        B_E=results['B_E'],
        B_U=results['B_U'],
        weighting_mode=results['weighting_mode'],
        
        # Distributed info
        is_distributed=results.get('is_distributed', False),
        world_size=results.get('world_size', 1),
    )
    
    # Stage 2 variance results
    if results.get('variance_enabled'):
        run_results.V_X = results['V_X']
        run_results.V_Y = results['V_Y'] 
        run_results.SE_deltaH1 = results['SE_deltaH1']
    
    # Stage 2 ground-truth results
    if results.get('importance_enabled'):
        run_results.H_orig = results['H_orig']
        run_results.H_upd = results['H_upd']
        run_results.deltaH_true = results['deltaH_true']
        run_results.ESS = results.get('diagnostics', {}).get('ESS')
    
    # Timing information
    timing = results.get('timing', {})
    run_results.total_time = sum(timing.values()) if timing else 0.0
    run_results.phase_0_time = timing.get('Phase 0', 0.0)
    run_results.phase_1_time = timing.get('Phase 1', 0.0) 
    run_results.phase_2_time = timing.get('Phase 2', 0.0)
    run_results.phase_3_time = timing.get('Phase 3', 0.0)
    run_results.phase_4_time = timing.get('Phase 4', 0.0)
    run_results.phase_5_time = timing.get('Phase 5', 0.0)
    
    # Memory usage
    if torch.cuda.is_available():
        run_results.peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
        torch.cuda.reset_peak_memory_stats()  # Reset for next run
    
    # Compute derived metrics
    run_results.compute_derived_metrics()
    
    return run_results


def compute_summary_statistics(results: List[ProbeRunResults], 
                             config: Dict[str, Any],
                             start_time: datetime,
                             end_time: datetime) -> BenchmarkSummary:
    """Compute summary statistics across all runs."""
    n_runs = len(results)
    
    # Extract deltaH1 values for statistics
    deltaH1_values = [r.deltaH1 for r in results]
    deltaH1_array = np.array(deltaH1_values)
    
    # Confidence interval (95%)
    deltaH1_ci_lower = np.percentile(deltaH1_array, 2.5)
    deltaH1_ci_upper = np.percentile(deltaH1_array, 97.5)
    
    summary = BenchmarkSummary(
        n_runs=n_runs,
        checkpoint_path=config.get('checkpoint', {}).get('checkpoint_path', 'unknown'),
        config_path='from_memory',  # Updated by caller
        start_time=start_time.isoformat(),
        end_time=end_time.isoformat(),
        total_duration=(end_time - start_time).total_seconds(),
        
        # δH₁ statistics
        deltaH1_mean=float(np.mean(deltaH1_array)),
        deltaH1_std=float(np.std(deltaH1_array)),
        deltaH1_min=float(np.min(deltaH1_array)),
        deltaH1_max=float(np.max(deltaH1_array)),
        deltaH1_ci_lower=float(deltaH1_ci_lower),
        deltaH1_ci_upper=float(deltaH1_ci_upper),
        deltaH1_cv=float(np.std(deltaH1_array) / abs(np.mean(deltaH1_array))),
        
        # Performance statistics
        avg_runtime=float(np.mean([r.total_time for r in results])),
        avg_peak_memory=float(np.mean([r.peak_memory_gb for r in results])),
    )
    
    # ΔH_true statistics (if available)
    deltaH_true_values = [r.deltaH_true for r in results if r.deltaH_true is not None]
    if deltaH_true_values:
        deltaH_true_array = np.array(deltaH_true_values)
        summary.deltaH_true_mean = float(np.mean(deltaH_true_array))
        summary.deltaH_true_std = float(np.std(deltaH_true_array))
        summary.deltaH_true_min = float(np.min(deltaH_true_array))
        summary.deltaH_true_max = float(np.max(deltaH_true_array))
        
        # Bias and RMSE (δH₁ vs ΔH_true)
        if len(deltaH_true_values) == n_runs:  # All runs have ground truth
            bias_values = [r.deltaH1 - r.deltaH_true for r in results]
            summary.est_bias = float(np.mean(bias_values))
            summary.est_rmse = float(np.sqrt(np.mean(np.square(bias_values))))
    
    # Fractional variance statistics (if available)
    frac_var_values = [r.frac_var for r in results if r.frac_var is not None]
    if frac_var_values:
        frac_var_array = np.array(frac_var_values)
        summary.frac_var_mean = float(np.mean(frac_var_array))
        summary.frac_var_std = float(np.std(frac_var_array))
        summary.frac_var_median = float(np.median(frac_var_array))
    
    return summary


def save_results(results: List[ProbeRunResults], 
                summary: BenchmarkSummary,
                output_dir: str,
                rank: int = 0) -> tuple[str, str]:
    """Save detailed results and summary to JSON files."""
    if rank != 0:
        return "", ""  # Only rank 0 saves results
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save detailed results
    results_file = output_path / f"probe_benchmark_results_{timestamp}.json"
    results_data = {
        'results': [asdict(r) for r in results],
        'metadata': {
            'n_runs': len(results),
            'timestamp': timestamp,
            'distributed': results[0].is_distributed if results else False,
            'world_size': results[0].world_size if results else 1,
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    # Save summary
    summary_file = output_path / f"probe_benchmark_summary_{timestamp}.json"
    summary.config_path = str(results_file)  # Reference to detailed results
    
    with open(summary_file, 'w') as f:
        json.dump(asdict(summary), f, indent=2)
    
    return str(results_file), str(summary_file)


def run_probe_benchmark(config_path: str, 
                       checkpoint_path: str,
                       n_runs: int,
                       output_dir: str = "./probe_benchmark_results",
                       rank: int = 0) -> tuple[List[ProbeRunResults], BenchmarkSummary]:
    """Run the probe benchmark suite."""
    logger = logging.getLogger("probe_benchmark")
    
    # Load config
    config = load_config(config_path)
    
    # Override checkpoint path
    config['checkpoint']['checkpoint_path'] = checkpoint_path + "/model" if not checkpoint_path.endswith("/model") else checkpoint_path
    config['checkpoint']['optimizer_path'] = checkpoint_path.replace("/model", "/optimizer.pt") if checkpoint_path.endswith("/model") else checkpoint_path + "/optimizer.pt"
    
    results = []
    start_time = datetime.now()
    
    if rank == 0:
        logger.info(f"🚀 Starting probe benchmark: {n_runs} runs on {checkpoint_path}")
        logger.info(f"Configuration: {config_path}")
        logger.info(f"Results will be saved to: {output_dir}")
    
    try:
        for run_id in range(n_runs):
            if rank == 0:
                logger.info(f"📊 Run {run_id + 1}/{n_runs}")
            
            # Initialize probe for this run
            probe = OfflineEntropyProbe(config)
            
            # Run the probe
            run_start = time.time()
            probe_results = probe.run_mixed_probe()
            run_end = time.time()
            
            # Extract and structure results
            run_results = extract_results_from_probe_output(probe_results)
            run_results.run_id = run_id
            run_results.total_time = run_end - run_start
            
            results.append(run_results)
            
            if rank == 0:
                logger.info(f"  δH₁: {run_results.deltaH1:.6f}")
                if run_results.deltaH_true is not None:
                    logger.info(f"  ΔH_true: {run_results.deltaH_true:.6f}")
                if run_results.frac_var is not None:
                    logger.info(f"  Frac_var: {run_results.frac_var:.6f}")
                logger.info(f"  Runtime: {run_results.total_time:.1f}s")
    
    except Exception as e:
        logger.error(f"Benchmark failed on run {len(results) + 1}: {e}")
        if rank == 0:
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        raise
    
    end_time = datetime.now()
    
    # Compute summary statistics (only on rank 0)
    if rank == 0:
        summary = compute_summary_statistics(results, config, start_time, end_time)
        
        logger.info("🎯 BENCHMARK COMPLETE!")
        logger.info(f"Runs completed: {summary.n_runs}")
        logger.info(f"Total duration: {summary.total_duration:.1f}s")
        logger.info(f"δH₁: {summary.deltaH1_mean:.6f} ± {summary.deltaH1_std:.6f}")
        logger.info(f"95% CI: [{summary.deltaH1_ci_lower:.6f}, {summary.deltaH1_ci_upper:.6f}]")
        logger.info(f"CV: {summary.deltaH1_cv:.4f}")
        
        if summary.deltaH_true_mean is not None:
            logger.info(f"ΔH_true: {summary.deltaH_true_mean:.6f} ± {summary.deltaH_true_std:.6f}")
            logger.info(f"Bias: {summary.est_bias:.6f}")
            logger.info(f"RMSE: {summary.est_rmse:.6f}")
        
        if summary.frac_var_mean is not None:
            logger.info(f"Fractional variance: {summary.frac_var_mean:.6f} (median: {summary.frac_var_median:.6f})")
        
        logger.info(f"Avg runtime: {summary.avg_runtime:.1f}s")
        logger.info(f"Avg peak memory: {summary.avg_peak_memory:.1f}GB")
        
        return results, summary
    else:
        return results, None


def main():
    parser = argparse.ArgumentParser(description="Run Stage 3 Multi-GPU Probe Benchmark")
    parser.add_argument("--config", 
                       default="entropy_experiments/configs/mixed_probe_stage3_multigpu_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--checkpoint", required=True,
                       help="Path to checkpoint directory")
    parser.add_argument("--runs", type=int, default=10,
                       help="Number of probe runs to execute")
    parser.add_argument("--output-dir", default="./probe_benchmark_results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    try:
        # Setup distributed training
        is_distributed, rank, world_size, local_rank = setup_distributed()
        
        # Setup logging
        logger = setup_logging("INFO", rank, args.output_dir)
        
        if rank == 0:
            logger.info("=" * 60)
            logger.info("STAGE 3 MULTI-GPU PROBE BENCHMARK")
            logger.info("=" * 60)
            logger.info(f"Distributed: {is_distributed}, Rank: {rank}/{world_size}")
            if torch.cuda.is_available():
                logger.info(f"CUDA devices: {torch.cuda.device_count()}")
        
        # Run benchmark
        results, summary = run_probe_benchmark(
            config_path=args.config,
            checkpoint_path=args.checkpoint, 
            n_runs=args.runs,
            output_dir=args.output_dir,
            rank=rank
        )
        
        # Save results (only rank 0)
        if rank == 0 and summary is not None:
            results_file, summary_file = save_results(results, summary, args.output_dir, rank)
            logger.info(f"📁 Results saved:")
            logger.info(f"  Detailed: {results_file}")
            logger.info(f"  Summary: {summary_file}")
        
        return 0
        
    except Exception as e:
        logger = logging.getLogger("probe_benchmark")
        logger.error(f"Benchmark suite failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    exit(main())