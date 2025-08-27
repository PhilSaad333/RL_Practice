#!/usr/bin/env python3
"""
Stage 3 Multi-GPU Mixed E/U Batch Probe Runner

Complete test runner for the mixed batch entropy probe with Stage 3 multi-GPU enhancements:
- Stage 1: δH₁ = lr * (X̄ · Ȳ) using separate E/U batches
- Stage 2: Variance estimation (V_X, V_Y) and two-batch ground-truth ΔH_true
- Stage 3: Deterministic multi-GPU coordination with DDP

Usage:
    # Single GPU test (validation)
    python run_mixed_probe_stage3_multigpu.py [--config CONFIG_PATH]
    
    # Multi-GPU test (production)
    torchrun --nproc_per_node=2 run_mixed_probe_stage3_multigpu.py [--config CONFIG_PATH]
"""

import argparse
import sys
import yaml
import torch
import torch.distributed as dist
import logging
from pathlib import Path
import os

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from entropy_experiments.offline_entropy_probe import OfflineEntropyProbe
from entropy_experiments import distributed_helpers


def setup_logging(level="INFO", rank=0):
    """Setup logging configuration with rank prefix."""
    log_format = f'%(asctime)s - RANK{rank} - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=getattr(logging, level),
        format=log_format,
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def setup_distributed():
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


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def validate_stage3_results(results: dict, is_distributed: bool, rank: int):
    """Validate Stage 3 multi-GPU results."""
    logger = logging.getLogger("stage3_validation")
    
    validation_passed = True
    
    # Stage 1 validation
    if abs(results['deltaH1']) > 1e-12:
        logger.info("✓ Non-zero δH₁ achieved")
    else:
        logger.warning("✗ δH₁ is effectively zero - possible issue")
        validation_passed = False
    
    if abs(results['bars_dot']) > 1e-12:
        logger.info("✓ Non-zero bars_dot achieved")
    else:
        logger.warning("✗ bars_dot is effectively zero - possible issue")
        validation_passed = False
    
    # Batch sizes
    if results['B_E'] > 0 and results['B_U'] > 0:
        logger.info(f"✓ Valid batch sizes: B_E={results['B_E']}, B_U={results['B_U']}")
    else:
        logger.error("✗ Invalid batch sizes")
        validation_passed = False
    
    # Multi-GPU specific validation
    if is_distributed:
        expected_B_E = 32  # From config
        expected_B_U = 32
        if results['B_E'] == expected_B_E and results['B_U'] == expected_B_U:
            logger.info(f"✓ Correct global batch sizes: B_E={expected_B_E}, B_U={expected_B_U}")
        else:
            logger.warning(f"⚠ Unexpected global batch sizes: got B_E={results['B_E']}, B_U={results['B_U']}, expected {expected_B_E}, {expected_B_U}")
    
    # Stage 2 variance validation
    if results.get('variance_enabled'):
        V_X, V_Y = results['V_X'], results['V_Y']
        if V_X >= 0 and V_Y >= 0:
            logger.info(f"✓ Non-negative variances: V_X={V_X:.10f}, V_Y={V_Y:.10f}")
        else:
            logger.warning(f"✗ Negative variance detected: V_X={V_X:.10f}, V_Y={V_Y:.10f}")
            validation_passed = False
        
        if results['SE_deltaH1'] >= 0 and results['SE_deltaH1'] < float('inf'):
            logger.info(f"✓ Finite standard error: {results['SE_deltaH1']:.10f}")
        else:
            logger.warning(f"✗ Invalid standard error: {results['SE_deltaH1']}")
            validation_passed = False
    
    # Stage 2 ground-truth validation
    if results.get('importance_enabled'):
        deltaH_true = results['deltaH_true']
        if abs(deltaH_true) > 1e-12:
            logger.info(f"✓ Non-zero ground-truth entropy change: {deltaH_true:.10f}")
            
            # Check sign consistency
            deltaH1 = results['deltaH1']
            if (deltaH1 * deltaH_true > 0):
                logger.info("✓ Sign agreement between δH₁ and ΔH_true")
            else:
                logger.info("⚠ Sign disagreement (may be expected for large steps)")
        else:
            logger.warning("✗ Ground-truth entropy change is effectively zero")
    
    # Memory validation
    if torch.cuda.is_available():
        memory_used = torch.cuda.max_memory_allocated() / 1024**3
        logger.info(f"Peak GPU memory: {memory_used:.2f}GB")
        if memory_used < 75:  # Conservative for H100 80GB
            logger.info("✓ Memory usage within limits")
        else:
            logger.warning("✗ High memory usage detected")
    
    return validation_passed


def main():
    parser = argparse.ArgumentParser(description="Run Stage 3 Multi-GPU Mixed E/U Batch Probe")
    parser.add_argument("--config", 
                       default="configs/mixed_probe_stage3_multigpu_config.yaml",
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    try:
        # Setup distributed training
        is_distributed, rank, world_size, local_rank = setup_distributed()
        
        # Setup logging with rank
        setup_logging("INFO", rank)
        logger = logging.getLogger("mixed_probe_stage3")
        
        if rank == 0:  # Only rank 0 prints startup info
            logger.info("🚀 Starting Stage 3 Multi-GPU Mixed E/U Batch Probe")
            logger.info(f"Configuration: {args.config}")
            logger.info(f"Distributed: {is_distributed}, Rank: {rank}/{world_size}")
            
            if torch.cuda.is_available():
                logger.info(f"CUDA devices: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                    memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    logger.info(f"GPU {i} memory: {memory:.1f}GB")
        
        # Load configuration
        config = load_config(args.config)
        
        # Validate checkpoint paths are set
        if config['checkpoint']['checkpoint_path'] == "TBD_AFTER_EXTRACTION":
            logger.error("❌ Checkpoint paths not set in config! Please extract checkpoint archive first.")
            return 1
        
        # Initialize probe
        if rank == 0:
            logger.info("Initializing OfflineEntropyProbe...")
        
        probe = OfflineEntropyProbe(config)
        
        # Add distributed barrier if enabled
        if is_distributed and config.get('distributed', {}).get('barriers', False):
            dist.barrier()
            if rank == 0:
                logger.info("All ranks synchronized - starting mixed probe")
        
        # Run Stage 3 mixed probe analysis
        if rank == 0:
            logger.info("Running Stage 3 multi-GPU mixed probe analysis...")
        
        results = probe.run_mixed_probe()
        
        # Only rank 0 prints results and validates
        if rank == 0:
            logger.info("=== STAGE 3 MULTI-GPU RESULTS ===")
            
            # Stage 1 results
            logger.info(f"δH₁ (deltaH1): {results['deltaH1']:.10f}")
            logger.info(f"bars_dot (X̄ · Ȳ): {results['bars_dot']:.10f}")
            logger.info(f"learning_rate: {results['learning_rate']:.2e}")
            logger.info(f"B_E (global eval batch): {results['B_E']}")
            logger.info(f"B_U (global update batch): {results['B_U']}")
            logger.info(f"weighting_mode: {results['weighting_mode']}")
            
            # Stage 2 variance results (if enabled)
            if results.get('variance_enabled'):
                logger.info("--- VARIANCE RESULTS ---")
                logger.info(f"V_X: {results['V_X']:.10f}")
                logger.info(f"V_Y: {results['V_Y']:.10f}")
                logger.info(f"SE_deltaH1: {results['SE_deltaH1']:.10f}")
                logger.info(f"frac_var: {results['frac_var']:.6f}")
            
            # Stage 2 ground-truth results (if enabled)
            if results.get('importance_enabled'):
                logger.info("--- GROUND-TRUTH RESULTS ---")
                logger.info(f"H_orig: {results['H_orig']:.6f}")
                logger.info(f"H_upd: {results['H_upd']:.6f}")
                logger.info(f"deltaH_true: {results['deltaH_true']:.10f}")
                logger.info(f"ESS: {results.get('diagnostics', {}).get('ESS', 'N/A')}")
            
            # Timing breakdown
            logger.info("=== TIMING BREAKDOWN ===")
            timing = results['timing']
            for phase, time_val in timing.items():
                logger.info(f"{phase}: {time_val:.2f}s")
            
            # Stage 3 validation
            logger.info("=== STAGE 3 VALIDATION ===")
            validation_passed = validate_stage3_results(results, is_distributed, rank)
            
            if validation_passed:
                stage_info = []
                if results.get('variance_enabled'):
                    stage_info.append("variance estimation")
                if results.get('importance_enabled'):
                    stage_info.append("ground-truth measurement")
                if is_distributed:
                    stage_info.append("multi-GPU coordination")
                
                feature_desc = ' with ' + ' and '.join(stage_info) if stage_info else ''
                logger.info(f"🎉 Stage 3 Mixed Probe completed successfully{feature_desc}!")
                return 0
            else:
                logger.error("❌ Stage 3 validation failed - check results above")
                return 1
        
        # Non-rank-0 processes just need to participate in distributed computation
        if is_distributed and rank != 0:
            logger.info(f"Rank {rank}: Participating in distributed probe computation")
        
        return 0
        
    except Exception as e:
        logger = logging.getLogger("mixed_probe_stage3")
        logger.error(f"Stage 3 Multi-GPU Mixed Probe failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    exit(main())