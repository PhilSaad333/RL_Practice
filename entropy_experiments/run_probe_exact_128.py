#!/usr/bin/env python3
"""
Run entropy probe measurement with exact mode and B=128 batch size.

This script runs the full entropy probe analysis using the real RL checkpoint
with proper optimizer states for meaningful δH₁ predictions.
"""

import torch
import yaml
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from entropy_experiments import OfflineEntropyProbe


def setup_logging():
    """Setup logging for entropy probe run."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger("entropy_probe_exact_128")


def main():
    """Main function to run entropy probe with exact mode B=128."""
    logger = setup_logging()
    logger.info("Starting entropy probe measurement: Exact mode, B=128")
    
    # Set deterministic behavior
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    # Load configuration
    config_path = Path(__file__).parent / "configs" / "probe_config_exact_128.yaml"
    logger.info(f"Loading config from: {config_path}")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Print key configuration details
    logger.info("Configuration summary:")
    logger.info(f"  Mode: {config['probe_config']['mode']}")
    logger.info(f"  Batch size (B): {config['batch_config']['B']}")
    logger.info(f"  Responses per prompt (G): {config['batch_config']['G']}")
    logger.info(f"  Microbatch size: {config['memory_config']['microbatch_size']}")
    logger.info(f"  Dataset: {config['batch_config']['dataset_name']}")
    logger.info(f"  Checkpoint: {config['checkpoint']['checkpoint_path']}")
    
    # Create and run entropy probe
    logger.info("Initializing OfflineEntropyProbe...")
    probe = OfflineEntropyProbe(config)
    
    # Run the full offline analysis
    logger.info("Starting full entropy probe analysis...")
    logger.info("This will:")
    logger.info("  1. Load model and optimizer with proper Adam states")
    logger.info("  2. Sample 128 prompts × 8 responses = 1024 sequences")
    logger.info("  3. Compute δH₁ using exact per-prompt U-statistic")
    logger.info("  4. Compute ΔH using importance sampling (SNIS)")
    logger.info("  5. Provide variance estimates and diagnostics")
    
    try:
        results = probe.run_offline_analysis(
            config['checkpoint']['checkpoint_path'],
            optimizer_path=config['checkpoint']['optimizer_path']
        )
        
        # Print results summary
        logger.info("🎉 Entropy probe analysis completed successfully!")
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
            
        # Timing information
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
        return True
        
    except Exception as e:
        logger.error(f"Entropy probe analysis failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)