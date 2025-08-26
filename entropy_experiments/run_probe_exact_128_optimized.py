#!/usr/bin/env python3
"""
Run optimized entropy probe measurement with batched rollout generation.

This script uses the optimized configuration with:
- Batched rollout generation (16 prompts * 8 responses = 128 sequences per call)
- Increased microbatch_size for 40GB A100
- Maximum GPU utilization targeting >70% memory usage
"""

import torch
import yaml
import logging
import sys
import warnings
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Suppress the extremely verbose Qwen2 caching warnings
warnings.filterwarnings("ignore", message=".*Caching is incompatible with gradient checkpointing.*")

# Also suppress transformers logging warnings
import transformers
transformers.logging.set_verbosity_error()

# Set logging level to reduce model output
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from entropy_experiments import OfflineEntropyProbe


def setup_logging():
    """Setup logging for optimized entropy probe run."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger("entropy_probe_optimized")


def main():
    """Main function to run optimized entropy probe."""
    logger = setup_logging()
    logger.info("🚀 Starting OPTIMIZED entropy probe measurement")
    logger.info("OPTIMIZATIONS:")
    logger.info("  - Batched rollout generation (16 prompts × 8 responses = 128 sequences per call)")
    logger.info("  - Increased microbatch_size=2 for 40GB A100")
    logger.info("  - Target: >70% GPU memory utilization")
    
    # Set deterministic behavior
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    # Load optimized configuration
    config_path = Path(__file__).parent / "configs" / "probe_config_exact_128_optimized.yaml"
    logger.info(f"Loading optimized config from: {config_path}")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Print optimization details
    logger.info("OPTIMIZATION SUMMARY:")
    logger.info(f"  Mode: {config['probe_config']['mode']}")
    logger.info(f"  Total prompts (B): {config['batch_config']['B']}")
    logger.info(f"  Responses per prompt (G): {config['batch_config']['G']}")
    logger.info(f"  Rollout batch size: {config['batch_config']['rollout_batch_size']} prompts simultaneously")
    logger.info(f"  Sequences per generation call: {config['batch_config']['rollout_batch_size'] * config['batch_config']['G']}")
    logger.info(f"  Generation calls: {config['batch_config']['B'] // config['batch_config']['rollout_batch_size']}")
    logger.info(f"  Microbatch size: {config['memory_config']['microbatch_size']}")
    logger.info(f"  Expected speedup: ~{config['batch_config']['B'] // config['batch_config']['rollout_batch_size']}x faster rollout generation (vs sequential)")
    
    # Create and run entropy probe
    logger.info("Initializing OfflineEntropyProbe...")
    probe = OfflineEntropyProbe(config)
    
    # Run the full offline analysis with optimizations
    logger.info("Starting OPTIMIZED entropy probe analysis...")
    
    try:
        results = probe.run_offline_analysis(
            config['checkpoint']['checkpoint_path']
        )
        
        # Print results summary
        logger.info("🎉 OPTIMIZED entropy probe analysis completed successfully!")
        logger.info("=" * 60)
        logger.info("RESULTS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"δH₁ (predicted): {results['deltaH1']:.6f}")
        logger.info(f"ΔH (SNIS actual): {results['deltaH_snis']:.6f}")
        logger.info(f"Effective Sample Size: {results['ESS']:.2f}")
        
        # Standard errors if computed
        if 'se_plugin' in results:
            logger.info(f"SE (plug-in): {results['se_plugin']:.6f}")
        if 'se_jack' in results:
            logger.info(f"SE (jackknife): {results['se_jack']:.6f}")
            
        # Timing information - should be much faster!
        if 'timing' in results:
            logger.info(f"Total time: {results['timing']['total_time']:.2f}s")
            if 'components' in results['timing']:
                logger.info("Component timings:")
                for component, time_val in results['timing']['components'].items():
                    logger.info(f"  {component}: {time_val:.2f}s")
        
        # Save location
        if config['output']['save_results']:
            logger.info(f"Detailed results saved to: {config['output']['results_path']}")
            
        logger.info("=" * 60)
        logger.info("🚀 Optimization successful! Check GPU memory usage should be >70%")
        return True
        
    except Exception as e:
        logger.error(f"Optimized entropy probe analysis failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)