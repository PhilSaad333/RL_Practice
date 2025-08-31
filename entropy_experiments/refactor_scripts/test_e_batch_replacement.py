#!/usr/bin/env python3
"""
Test script for E-batch replacement sampling (Phase 2 - small step)

Just tests the new sample_E_batch_with_replacement method in isolation.
"""

import sys
import os
import logging
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from entropy_experiments.probe_components import ProbeComponents

def setup_logging():
    """Set up logging for the test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('e_batch_test')

def create_test_config():
    """Create a minimal test configuration."""
    return {
        'batch_config': {
            'dataset_name': 'gsm8k_r1_template',
            'split': 'test',
            'G': 2,  # 2 responses per prompt
            'rollout_batch_size': 4,  # Small batch for testing
        },
        'generation': {
            'temperature': 0.7,
            'top_p': 1.0,
            'max_new_tokens': 150,
            'do_sample': True,
            'pad_token_id': 151643,  # Qwen2.5 EOS token
            'gen_batch_size': 8,
            'tf_batch_size': 16,
        },
        'memory_config': {
            'amp': True,
            'dtype': 'bfloat16',
            'microbatch_size': 1,  # Small for testing
        },
    }

def load_model_and_tokenizer(checkpoint_path: str):
    """Load the model and tokenizer from checkpoint."""
    logger = logging.getLogger('e_batch_test')
    
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    
    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-1.5B", 
        trust_remote_code=True
    )
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Model loaded on device: {next(model.parameters()).device}")
    logger.info(f"Model memory usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    return model, tokenizer

def test_e_batch_replacement(probe_components, E_total_sequences=16, G=2):
    """Test the E-batch replacement sampling."""
    logger = logging.getLogger('e_batch_test')
    
    logger.info(f"Testing E-batch replacement: {E_total_sequences} total sequences, G={G}")
    
    # Sample E-batch with replacement
    batch_data = probe_components.sample_E_batch_with_replacement(E_total_sequences, G)
    
    # Validate the structure
    sequences = batch_data['sequences']
    prompt_ids = batch_data['prompt_ids']
    advantages = batch_data['advantages']
    
    logger.info(f"E-batch results:")
    logger.info(f"  sequences shape: {sequences.shape}")
    logger.info(f"  prompt_ids: {prompt_ids}")
    logger.info(f"  advantages shape: {advantages.shape}")
    logger.info(f"  unique prompt_ids: {len(set(prompt_ids))} out of {len(prompt_ids)} total")
    
    # Check for duplicates (this is the key test - we SHOULD see duplicates with replacement)
    duplicates = len(prompt_ids) - len(set(prompt_ids))
    if duplicates > 0:
        logger.info(f"✅ SUCCESS: Found {duplicates} duplicate prompt_ids (replacement sampling working)")
    else:
        logger.warning(f"⚠️ No duplicates found - might be expected with small samples")
    
    # Validate shapes
    expected_B_E = int((E_total_sequences + G - 1) // G)  # Ceiling division
    actual_B_E = sequences.shape[0]
    
    assert actual_B_E == expected_B_E or actual_B_E == expected_B_E + 1, f"Expected ~{expected_B_E} prompts, got {actual_B_E}"
    assert sequences.shape[1] == G, f"Expected G={G}, got {sequences.shape[1]}"
    assert len(prompt_ids) == actual_B_E, f"prompt_ids length mismatch"
    assert advantages.shape == (actual_B_E, G), f"advantages shape mismatch"
    
    logger.info("✅ All shape validations passed!")
    
    return batch_data

def main():
    """Main test function."""
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("E-Batch Replacement Sampling Test")
    logger.info("=" * 60)
    
    # Checkpoint path
    checkpoint_path = "/home/ubuntu/localfs/rl_training_runs/training_state/step_60/model"
    
    # Check checkpoint exists
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return 1
    
    try:
        # Load model
        model, tokenizer = load_model_and_tokenizer(checkpoint_path)
        
        # Create test config
        config = create_test_config()
        
        # Initialize ProbeComponents
        logger.info("Initializing ProbeComponents...")
        probe_components = ProbeComponents(model, config, logger)
        
        # Test E-batch replacement sampling with different sizes
        logger.info("\n--- Test 1: Small batch (16 sequences, G=2) ---")
        test_e_batch_replacement(probe_components, E_total_sequences=16, G=2)
        
        logger.info("\n--- Test 2: Larger batch (24 sequences, G=3) ---")
        test_e_batch_replacement(probe_components, E_total_sequences=24, G=3)
        
        logger.info("=" * 60)
        logger.info("🎉 E-batch replacement test completed successfully!")
        logger.info(f"GPU memory usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)