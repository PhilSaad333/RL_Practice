#!/usr/bin/env python3
"""
Stage 1 Mixed E/U Batch Probe Runner

Simple test runner for the new mixed batch entropy probe approach.
This implements Stage 1 of the rework which computes δH₁ = lr * (X̄ · Ȳ)
using separate evaluation (E) and update (U) batches.

Usage:
    python run_mixed_probe_stage1.py [--config CONFIG_PATH]
"""

import argparse
import sys
import yaml
import torch
import logging
from pathlib import Path

# Add entropy_experiments to path
sys.path.insert(0, str(Path(__file__).parent))

from offline_entropy_probe import OfflineEntropyProbe


def setup_logging(level="INFO"):
    """Setup basic logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Run Stage 1 Mixed E/U Batch Probe")
    parser.add_argument("--config", 
                       default="configs/mixed_probe_stage1_config.yaml",
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        setup_logging(config['output']['log_level'])
        
        logger = logging.getLogger("mixed_probe_stage1")
        logger.info("🚀 Starting Stage 1 Mixed E/U Batch Probe")
        logger.info(f"Configuration: {args.config}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
        # Initialize probe
        logger.info("Initializing OfflineEntropyProbe...")
        probe = OfflineEntropyProbe(config)
        
        # Run Stage 1 mixed probe analysis
        logger.info("Running Stage 1 mixed probe analysis...")
        results = probe.run_mixed_probe()
        
        # Log key results
        logger.info("=== STAGE 1 RESULTS ===")
        logger.info(f"δH₁ (deltaH1): {results['deltaH1']:.10f}")
        logger.info(f"bars_dot (X̄ · Ȳ): {results['bars_dot']:.10f}")
        logger.info(f"learning_rate: {results['learning_rate']:.2e}")
        logger.info(f"B_E (eval batch): {results['B_E']}")
        logger.info(f"B_U (update batch): {results['B_U']}")
        logger.info(f"weighting_mode: {results['weighting_mode']}")
        logger.info(f"Total time: {results['timing']['total_time']:.2f}s")
        
        # Detailed timing
        logger.info("=== TIMING BREAKDOWN ===")
        for phase, time_val in results['timing'].items():
            if phase != 'total_time':
                logger.info(f"{phase}: {time_val:.2f}s")
        
        # Acceptance criteria check
        logger.info("=== ACCEPTANCE CRITERIA ===")
        
        # Check for non-zero results
        if abs(results['deltaH1']) > 1e-12:
            logger.info("✓ Non-zero δH₁ achieved")
        else:
            logger.warning("✗ δH₁ is effectively zero - possible issue")
        
        if abs(results['bars_dot']) > 1e-12:
            logger.info("✓ Non-zero bars_dot achieved")
        else:
            logger.warning("✗ bars_dot is effectively zero - possible issue")
        
        # Check batch sizes
        if results['B_E'] > 0 and results['B_U'] > 0:
            logger.info(f"✓ Valid batch sizes: B_E={results['B_E']}, B_U={results['B_U']}")
        else:
            logger.error("✗ Invalid batch sizes")
        
        # Memory check (approximate)
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / 1024**3
            logger.info(f"Peak GPU memory: {memory_used:.2f}GB")
            if memory_used < 35:  # Conservative for A100 40GB
                logger.info("✓ Memory usage within limits")
            else:
                logger.warning("✗ High memory usage detected")
        
        logger.info("🎉 Stage 1 Mixed Probe completed successfully!")
        
        return 0
        
    except Exception as e:
        logger = logging.getLogger("mixed_probe_stage1") 
        logger.error(f"Stage 1 Mixed Probe failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    exit(main())