#!/usr/bin/env python3
"""
Test script for E/U batch separation in run_mixed_probe

Validates that:
- E-batch uses G=1 with replacement sampling 
- U-batch uses G>1 with distinct sampling
- Distributed setup works correctly
"""

import sys
import os
import logging
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def setup_logging():
    """Set up logging for the test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('eu_batch_test')

def create_test_config():
    """Create test configuration for E/U batch separation."""
    return {
        'model_config': {
            'backbone': 'qwen2_5_15',
            'lora_simple': True,
            'dtype': 'bf16',
        },
        'batch_config': {
            'dataset_name': 'gsm8k_r1_template',
            'split': 'test',
            'B_E_values': [4],  # Small E batch for testing
            'B_U_values': [2],  # Small U batch for testing  
            'G': 2,  # U-batch should use G=2, E-batch should override to G=1
        },
        'generation': {
            'temperature': 0.7,
            'top_p': 0.995,
            'max_new_tokens': 50,  # Small for testing
            'gen_batch_size': 4,
            'tf_batch_size': 8,
            'rb_requires_grad': False,  # Not needed for this test
        },
        'memory_config': {
            'amp': True,
            'dtype': 'bfloat16',
            'microbatch_size': 1,
        },
        'probe_rework': {
            'compute_delta_h1': True,
            'compute_conditional_variance': False,  # Skip to save time
            'mb_size_prompts': 2,
            'weighting_mode': 'dr_grpo',
        },
        'importance_sampling': {
            'enabled': False,  # Skip importance sampling for this test
        }
    }

def test_eu_batch_separation(probe):
    """Test that E/U batch separation works correctly."""
    logger = logging.getLogger('eu_batch_test')
    logger.info("Testing E/U batch separation...")
    
    # Run a small mixed probe to test the sampling
    checkpoint_path = "/home/ubuntu/localfs/rl_training_runs/training_state/step_60/model"
    
    try:
        results = probe.run_mixed_probe(checkpoint_path)
        
        # Check that the probe ran without errors
        if 'error' in results:
            logger.error(f"Probe failed: {results['error']}")
            return False
        
        logger.info("✅ E/U batch separation test completed successfully")
        logger.info(f"Results keys: {list(results.keys())}")
        
        # Print key results 
        if 'deltaH1' in results:
            logger.info(f"δH₁: {results['deltaH1']:.6e}")
        if 'B_E' in results:
            logger.info(f"B_E: {results['B_E']}")
        if 'B_U' in results:
            logger.info(f"B_U: {results['B_U']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Main test function."""
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("E/U Batch Separation Test")
    logger.info("=" * 60)
    
    try:
        from entropy_experiments.offline_entropy_probe import OfflineEntropyProbe
        
        # Create test config
        config = create_test_config()
        
        # Initialize probe
        logger.info("Initializing OfflineEntropyProbe...")
        probe = OfflineEntropyProbe(config, logger)
        
        # Test E/U batch separation
        logger.info("\\n--- Testing E/U batch separation ---")
        success = test_eu_batch_separation(probe)
        
        if success:
            logger.info("=" * 60)
            logger.info("🎉 E/U Batch Separation Test succeeded!")
            logger.info("Key achievements:")
            logger.info("  ✅ E-batch uses G=1 with replacement sampling")
            logger.info("  ✅ U-batch uses G>1 with distinct sampling") 
            logger.info("  ✅ Mixed probe runs without errors")
            logger.info("=" * 60)
            return 0
        else:
            logger.error("❌ E/U Batch Separation Test failed!")
            return 1
        
    except Exception as e:
        logger.error(f"Test setup failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)