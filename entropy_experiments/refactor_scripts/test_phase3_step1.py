#!/usr/bin/env python3
"""
Test script for Phase 3 Step 1: BaselineState and config initialization

Tests that the BaselineState class and Phase 3 config structure work correctly.
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
    return logging.getLogger('phase3_step1_test')

def test_baseline_state():
    """Test BaselineState class functionality."""
    from entropy_experiments.probe_components import BaselineState
    
    logger = logging.getLogger('phase3_step1_test')
    logger.info("Testing BaselineState class...")
    
    # Test initialization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    baseline = BaselineState(ema_decay=0.9, device=device)
    
    logger.info(f"✅ BaselineState initialized: ema_decay={baseline.ema_decay}, device={baseline.mu.device}")
    assert baseline.ema_decay == 0.9
    assert baseline.mu.device.type == device.type  # Compare device type, not exact device
    assert baseline.mu.numel() == 0
    
    # Test ensure_len
    baseline.ensure_len(5)
    logger.info(f"✅ ensure_len(5): mu.shape={baseline.mu.shape}")
    assert baseline.mu.shape == (5,)
    assert torch.all(baseline.mu == 0.0)
    
    # Test update_from_batch
    batch_size, seq_len = 3, 4
    residuals = torch.randn(batch_size, seq_len, device=device)
    lengths = torch.tensor([2, 4, 3], device=device)  # variable lengths
    
    baseline.update_from_batch(residuals, lengths)
    logger.info(f"✅ update_from_batch: mu after update = {baseline.mu[:seq_len].cpu().tolist()}")
    
    # Test get_mu_vector
    mu_vec = baseline.get_mu_vector(3)
    logger.info(f"✅ get_mu_vector(3): shape={mu_vec.shape}, requires_grad={mu_vec.requires_grad}")
    assert mu_vec.shape == (3,)
    assert not mu_vec.requires_grad  # Should be detached
    
    return baseline

def test_probe_components_initialization():
    """Test ProbeComponents initializes BaselineState correctly."""
    logger = logging.getLogger('phase3_step1_test')
    logger.info("Testing ProbeComponents Phase 3 initialization...")
    
    # Load model using proper LoRA loading
    from entropy_experiments.offline_entropy_probe import load_peft_for_probe
    
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
    
    # Create config with Phase 3 settings
    config = {
        'batch_config': {
            'G': 2,
            'rollout_batch_size': 4,
        },
        'generation': {
            'temperature': 0.7,
            'top_p': 0.995,
            'max_new_tokens': 150,
            'gen_batch_size': 4,
            'tf_batch_size': 8,
            'rb_requires_grad': True,  # Phase 3 requirement
        },
        'memory_config': {
            'amp': True,
            'dtype': 'bfloat16',
            'microbatch_size': 1,
        },
        'estimator': {  # Phase 3 config
            'x_estimator_mode': 'rb_residual',
            'rb_normalize_by_length': True,
            'baseline': {
                'mode': 'residual_mu',
                'ema_decay': 0.9,
            }
        }
    }
    
    # Initialize ProbeComponents
    from entropy_experiments.probe_components import ProbeComponents
    probe_components = ProbeComponents(model, config, logger)
    
    # Verify baseline state was created
    assert hasattr(probe_components, '_baseline_state_x')
    assert probe_components._baseline_state_x.ema_decay == 0.9
    assert probe_components._baseline_state_x.mu.device == probe_components.device
    
    logger.info(f"✅ ProbeComponents baseline state initialized correctly")
    logger.info(f"   ema_decay: {probe_components._baseline_state_x.ema_decay}")
    logger.info(f"   device: {probe_components._baseline_state_x.mu.device}")
    
    return probe_components

def test_batch_conversion_utility(probe_components):
    """Test the _to_batched_sequences_from_probe utility function."""
    logger = logging.getLogger('phase3_step1_test')
    logger.info("Testing batch conversion utility...")
    
    # Create a mock probe batch (similar to what sample_E_batch_with_replacement returns)
    device = probe_components.device
    B, G, T = 2, 2, 10
    
    # Create mock data
    sequences = torch.randint(0, 1000, (B, G, T), device=device)
    attention_masks = torch.ones((B, G, T), device=device)
    # Mask out some positions to simulate variable lengths
    attention_masks[0, 0, 8:] = 0  # first seq has 8 tokens
    attention_masks[0, 1, 6:] = 0  # second seq has 6 tokens  
    attention_masks[1, 0, 9:] = 0  # third seq has 9 tokens
    attention_masks[1, 1, 7:] = 0  # fourth seq has 7 tokens
    
    prompt_lens = [3, 4]  # prompt lengths for each of the B prompts
    
    batch = {
        'sequences': sequences,
        'attention_masks': attention_masks, 
        'prompt_lens': prompt_lens,
        'advantages': torch.randn(B, G, device=device),  # not used in conversion
        'max_lengths': [5, 3],  # not used in conversion
        'prompt_ids': [0, 1],   # not used in conversion
    }
    
    # Test the conversion
    batched_sequences = probe_components._to_batched_sequences_from_probe(batch)
    
    # Verify the conversion
    assert batched_sequences.sequences.shape == (B, G, T)
    assert batched_sequences.attention_masks.shape == (B, G, T)
    assert len(batched_sequences.prompt_lens) == B
    assert len(batched_sequences.gen_lens) == B
    assert len(batched_sequences.gen_lens[0]) == G
    assert len(batched_sequences.gen_lens[1]) == G
    
    # Check gen_lens calculation: gen_len = total_len - prompt_len
    expected_gen_lens = [
        [8 - 3, 6 - 3],  # first prompt: [5, 3]
        [9 - 4, 7 - 4]   # second prompt: [5, 3]
    ]
    
    logger.info(f"✅ BatchedSequences conversion successful")
    logger.info(f"   sequences shape: {batched_sequences.sequences.shape}")
    logger.info(f"   attention_masks shape: {batched_sequences.attention_masks.shape}")
    logger.info(f"   prompt_lens: {batched_sequences.prompt_lens}")
    logger.info(f"   gen_lens: {batched_sequences.gen_lens}")
    logger.info(f"   expected gen_lens: {expected_gen_lens}")
    
    assert batched_sequences.gen_lens == expected_gen_lens
    assert batched_sequences.responses_text is None
    
    return batched_sequences

def main():
    """Main test function."""
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("Phase 3 Steps 1-2 Test: BaselineState & Batch Conversion")
    logger.info("=" * 60)
    
    try:
        # Test 1: BaselineState class
        logger.info("\n--- Test 1: BaselineState class ---")
        baseline = test_baseline_state()
        
        # Test 2: ProbeComponents initialization 
        logger.info("\n--- Test 2: ProbeComponents Phase 3 initialization ---")
        probe_components = test_probe_components_initialization()
        
        # Test 3: Batch conversion utility (Step 2)
        logger.info("\n--- Test 3: Batch conversion utility (Step 2) ---")
        batched_sequences = test_batch_conversion_utility(probe_components)
        
        logger.info("=" * 60)
        logger.info("🎉 Phase 3 Steps 1-2 tests completed successfully!")
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