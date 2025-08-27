#!/usr/bin/env python3
"""
Stage 2 Mixed E/U Batch Probe Runner

Complete test runner for the mixed batch entropy probe with Stage 2 enhancements:
- Stage 1: δH₁ = lr * (X̄ · Ȳ) using separate E/U batches
- Stage 2: Variance estimation (V_X, V_Y) and two-batch ground-truth ΔH_true

Usage:
    python run_mixed_probe_stage2.py [--config CONFIG_PATH]
"""

import argparse
import sys
import yaml
import torch
import logging
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from entropy_experiments.offline_entropy_probe import OfflineEntropyProbe


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
    parser = argparse.ArgumentParser(description="Run Stage 2 Mixed E/U Batch Probe")
    parser.add_argument("--config", 
                       default="configs/mixed_probe_stage2_config.yaml",
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        setup_logging(config['output']['log_level'])
        
        logger = logging.getLogger("mixed_probe_stage2")
        logger.info("🚀 Starting Stage 2 Mixed E/U Batch Probe")
        logger.info(f"Configuration: {args.config}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
        # Initialize probe
        logger.info("Initializing OfflineEntropyProbe...")
        probe = OfflineEntropyProbe(config)
        
        # Run Stage 2 mixed probe analysis (includes Stage 1 + variance + ground-truth)
        logger.info("Running Stage 2 mixed probe analysis...")
        results = probe.run_mixed_probe()
        
        # Log key results
        logger.info("=== STAGE 2 RESULTS ===")
        
        # Stage 1 results
        logger.info(f"δH₁ (deltaH1): {results['deltaH1']:.10f}")
        logger.info(f"bars_dot (X̄ · Ȳ): {results['bars_dot']:.10f}")
        logger.info(f"learning_rate: {results['learning_rate']:.2e}")
        logger.info(f"B_E (eval batch): {results['B_E']}")
        logger.info(f"B_U (update batch): {results['B_U']}")
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
        
        # Acceptance criteria check
        logger.info("=== ACCEPTANCE CRITERIA ===")
        
        # Stage 1 checks
        if abs(results['deltaH1']) > 1e-12:
            logger.info("✓ Non-zero δH₁ achieved")
        else:
            logger.warning("✗ δH₁ is effectively zero - possible issue")
        
        if abs(results['bars_dot']) > 1e-12:
            logger.info("✓ Non-zero bars_dot achieved")
        else:
            logger.warning("✗ bars_dot is effectively zero - possible issue")
        
        # Batch sizes
        if results['B_E'] > 0 and results['B_U'] > 0:
            logger.info(f"✓ Valid batch sizes: B_E={results['B_E']}, B_U={results['B_U']}")
        else:
            logger.error("✗ Invalid batch sizes")
        
        # Stage 2 variance checks
        if results.get('variance_enabled'):
            V_X, V_Y = results['V_X'], results['V_Y']
            if V_X >= 0 and V_Y >= 0:
                logger.info(f"✓ Non-negative variances: V_X={V_X:.10f}, V_Y={V_Y:.10f}")
            else:
                logger.warning(f"✗ Negative variance detected: V_X={V_X:.10f}, V_Y={V_Y:.10f}")
            
            if results['SE_deltaH1'] >= 0 and results['SE_deltaH1'] < float('inf'):
                logger.info(f"✓ Finite standard error: {results['SE_deltaH1']:.10f}")
            else:
                logger.warning(f"✗ Invalid standard error: {results['SE_deltaH1']}")
        
        # Stage 2 ground-truth checks
        if results.get('importance_enabled'):
            deltaH_true = results['deltaH_true']
            if abs(deltaH_true) > 1e-12:
                logger.info(f"✓ Non-zero ground-truth entropy change: {deltaH_true:.10f}")
                
                # Check sign consistency (for small learning rates, should often agree)
                deltaH1 = results['deltaH1']
                if (deltaH1 * deltaH_true > 0):
                    logger.info("✓ Sign agreement between δH₁ and ΔH_true")
                else:
                    logger.info("⚠ Sign disagreement (may be expected for large steps)")
            else:
                logger.warning("✗ Ground-truth entropy change is effectively zero")
        
        # Memory check (approximate)
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / 1024**3
            logger.info(f"Peak GPU memory: {memory_used:.2f}GB")
            if memory_used < 38:  # Conservative for A100 40GB
                logger.info("✓ Memory usage within limits")
            else:
                logger.warning("✗ High memory usage detected")
        
        # Final status
        stage_info = []
        if results.get('variance_enabled'):
            stage_info.append("variance estimation")
        if results.get('importance_enabled'):
            stage_info.append("ground-truth measurement")
        
        if stage_info:
            logger.info(f"🎉 Stage 2 Mixed Probe completed successfully with {' and '.join(stage_info)}!")
        else:
            logger.info("🎉 Stage 1 Mixed Probe completed successfully!")
        
        return 0
        
    except Exception as e:
        logger = logging.getLogger("mixed_probe_stage2") 
        logger.error(f"Stage 2 Mixed Probe failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    exit(main())