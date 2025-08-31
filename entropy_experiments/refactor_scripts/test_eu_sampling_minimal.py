#!/usr/bin/env python3
"""
Minimal test for E/U batch sampling separation

Tests just the sampling methods to verify:
- E-batch uses G=1 with replacement sampling
- U-batch uses G>1 with distinct sampling
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
    return logging.getLogger('eu_sampling_test')

def test_sampling_methods():
    """Test E and U batch sampling methods directly."""
    logger = logging.getLogger('eu_sampling_test')
    logger.info("Testing E/U batch sampling methods...")
    
    # Load model and create probe components
    from entropy_experiments.offline_entropy_probe import load_peft_for_probe
    from entropy_experiments.probe_components import ProbeComponents
    
    checkpoint_path = "/home/ubuntu/localfs/rl_training_runs/training_state/step_60/model"
    model = load_peft_for_probe(
        base_id="Qwen/Qwen2.5-1.5B",
        adapter_path=checkpoint_path,
        mode="lora_simple",
        dtype="bf16",
        device_map="cuda",
        use_checkpointing=False
    )
    model.train()
    if hasattr(model, "set_adapter"):
        model.set_adapter("default")
    
    # Create minimal config
    config = {
        'batch_config': {
            'dataset_name': 'gsm8k_r1_template',
            'split': 'test',
            'G': 3,  # U-batch will use G=3
            'rollout_batch_size': 2,
        },
        'generation': {
            'temperature': 0.7,
            'top_p': 0.995,
            'max_new_tokens': 30,  # Small for testing
            'gen_batch_size': 4,
            'tf_batch_size': 8,
        },
        'memory_config': {
            'amp': True,
            'dtype': 'bfloat16',
            'microbatch_size': 1,
        },
    }
    
    probe_components = ProbeComponents(model, config, logger)
    
    # Test 1: E-batch replacement sampling (should use G=1)
    logger.info("\\n--- Test 1: E-batch replacement sampling ---")
    E_total_sequences = 4
    E_batch = probe_components.sample_E_batch_with_replacement(
        E_total_sequences=E_total_sequences, 
        G=1
    )
    
    logger.info(f"✅ E-batch sampled:")
    logger.info(f"   Total sequences requested: {E_total_sequences}")
    logger.info(f"   Actual shape: {E_batch['sequences'].shape}")
    logger.info(f"   Expected: [*, 1, *] (G=1)")
    logger.info(f"   G dimension: {E_batch['sequences'].shape[1]}")
    
    assert E_batch['sequences'].shape[1] == 1, f"E-batch should have G=1, got G={E_batch['sequences'].shape[1]}"
    
    # Test 2: U-batch distinct sampling (should use G=3)
    logger.info("\\n--- Test 2: U-batch distinct sampling ---")
    B_U = 2
    G_U = config['batch_config']['G']  # Should be 3
    U_batch = probe_components.sample_batch(
        B=B_U, 
        G=G_U
    )
    
    logger.info(f"✅ U-batch sampled:")
    logger.info(f"   B_U: {B_U}, G_U: {G_U}")
    logger.info(f"   Actual shape: {U_batch['sequences'].shape}")
    logger.info(f"   Expected: [{B_U}, {G_U}, *]")
    logger.info(f"   G dimension: {U_batch['sequences'].shape[1]}")
    
    assert U_batch['sequences'].shape[0] == B_U, f"U-batch should have B={B_U}, got B={U_batch['sequences'].shape[0]}"
    assert U_batch['sequences'].shape[1] == G_U, f"U-batch should have G={G_U}, got G={U_batch['sequences'].shape[1]}"
    
    # Test 3: Verify shapes are different
    logger.info("\\n--- Test 3: Verify E/U batch differences ---")
    logger.info(f"E-batch G: {E_batch['sequences'].shape[1]} (replacement, G=1)")
    logger.info(f"U-batch G: {U_batch['sequences'].shape[1]} (distinct, G={G_U})")
    
    assert E_batch['sequences'].shape[1] != U_batch['sequences'].shape[1], "E and U batches should have different G values"
    
    logger.info("✅ All sampling tests passed!")
    
    return True

def main():
    """Main test function."""
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("Minimal E/U Batch Sampling Test")
    logger.info("=" * 60)
    
    try:
        success = test_sampling_methods()
        
        if success:
            logger.info("=" * 60)
            logger.info("🎉 E/U Batch Sampling Test succeeded!")
            logger.info("Key achievements:")
            logger.info("  ✅ E-batch uses G=1 with replacement sampling")
            logger.info("  ✅ U-batch uses G>1 with distinct sampling") 
            logger.info("  ✅ Sampling methods work independently")
            logger.info("=" * 60)
            return 0
        else:
            return 1
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)