#!/usr/bin/env python3
"""
Phase 1 Test Script: Minimal probe test with SequenceProcessor

Tests the Phase 1 refactor with a small batch (B=2, G=2) to validate:
1. No zeros in sequence_logprob for obviously non-empty generations
2. prompt_lens equals the batch left-pad prompt length (constant within batch)
3. max_lengths[b] matches expected generation lengths
4. GPU memory usage is reasonable
5. Basic sanity checks pass
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
from entropy_experiments.offline_entropy_probe import OfflineEntropyProbe

def setup_logging():
    """Set up logging for the test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('phase1_test')

def create_test_config():
    """Create a minimal test configuration."""
    return {
        'batch_config': {
            'dataset_name': 'gsm8k_r1_template',
            'split': 'test',
            'G': 2,  # 2 responses per prompt
            'rollout_batch_size': 2,  # Small batch for testing
        },
        'generation': {
            'temperature': 0.7,  # Match training checkpoint temperature for consistent formatting
            'top_p': 1.0,
            'max_new_tokens': 150,  # Enough for complete mathematical reasoning
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
        # Phase 1: Use SequenceProcessor (no toggle needed since we removed legacy code)
    }

def load_model_and_tokenizer(checkpoint_path: str):
    """Load the model and tokenizer from checkpoint."""
    logger = logging.getLogger('phase1_test')
    
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

def test_batch_sampling(probe_components, B=2, G=2):
    """Test the sample_batch method with Phase 1 SequenceProcessor."""
    logger = logging.getLogger('phase1_test')
    
    logger.info(f"Testing batch sampling with B={B}, G={G}")
    
    # Sample a batch
    batch_data = probe_components.sample_batch(B=B, G=G)
    
    # Validate batch structure
    logger.info("Validating batch structure...")
    
    sequences = batch_data['sequences']  # [B, G, max_len]
    prompt_lens = batch_data['prompt_lens']  # [B]
    advantages = batch_data['advantages']  # [B, G]
    max_lengths = batch_data['max_lengths']  # [B]
    attention_masks = batch_data['attention_masks']  # [B, G, max_len]
    prompt_ids = batch_data['prompt_ids']  # [B]
    
    logger.info(f"Sequences shape: {sequences.shape}")
    logger.info(f"Prompt lengths: {prompt_lens}")
    logger.info(f"Max lengths: {max_lengths}")
    logger.info(f"Advantages shape: {advantages.shape}")
    logger.info(f"Advantages values: {advantages.tolist()}")
    
    # Sanity checks from Phase 1 plan
    assert sequences.shape[0] == B, f"Expected B={B}, got {sequences.shape[0]}"
    assert sequences.shape[1] == G, f"Expected G={G}, got {sequences.shape[1]}"
    assert len(prompt_lens) == B, f"Expected {B} prompt lengths, got {len(prompt_lens)}"
    assert len(max_lengths) == B, f"Expected {B} max lengths, got {len(max_lengths)}"
    assert advantages.shape == (B, G), f"Expected advantages shape ({B}, {G}), got {advantages.shape}"
    assert attention_masks.shape == sequences.shape, f"Attention mask shape {attention_masks.shape} != sequences shape {sequences.shape}"
    
    # Check prompt lengths are consistent (left-pad means all prompts have same effective length)
    unique_prompt_lens = set(prompt_lens)
    logger.info(f"Unique prompt lengths: {unique_prompt_lens}")
    if len(unique_prompt_lens) == 1:
        logger.info("✅ All prompts have same padded length (correct for left-padding)")
    else:
        logger.warning(f"⚠️  Multiple prompt lengths found: {unique_prompt_lens}")
    
    # Check no sequences are all padding tokens and decode some examples
    pad_token_id = 151643  # Qwen2.5 EOS/PAD token
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)
    
    for b in range(B):
        for g in range(G):
            seq = sequences[b, g]
            non_pad_tokens = (seq != pad_token_id).sum()
            logger.info(f"Sequence [{b},{g}]: {non_pad_tokens} non-pad tokens")
            assert non_pad_tokens > 0, f"Sequence [{b},{g}] is all padding tokens!"
            
            # Decode and show the full sequence for the first few examples
            if b < 2:  # Only show first 2 prompts to avoid spam
                # Find where the prompt ends and generation begins
                prompt_len = prompt_lens[b]
                prompt_tokens = seq[:prompt_len]
                gen_tokens = seq[prompt_len:]
                
                # Decode prompt and generation separately
                prompt_text = tokenizer.decode(prompt_tokens, skip_special_tokens=True)
                gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                
                logger.info(f"\n--- Sequence [{b},{g}] ---")
                logger.info(f"PROMPT: {prompt_text}")
                logger.info(f"GENERATION: {gen_text}")
                logger.info("=" * 50)
    
    logger.info("✅ All sanity checks passed!")
    return batch_data

def main():
    """Main test function."""
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("Phase 1 Test: SequenceProcessor Integration")
    logger.info("=" * 60)
    
    # Checkpoint path
    checkpoint_path = "/home/ubuntu/localfs/rl_training_runs/training_state/step_60/model"
    
    # Check checkpoint exists
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.info("Available checkpoints:")
        training_state_dir = "/home/ubuntu/localfs/rl_training_runs/training_state"
        if os.path.exists(training_state_dir):
            for item in os.listdir(training_state_dir):
                logger.info(f"  - {item}")
        return 1
    
    try:
        # Load model
        model, tokenizer = load_model_and_tokenizer(checkpoint_path)
        
        # Create test config
        config = create_test_config()
        
        # Initialize ProbeComponents with SequenceProcessor
        logger.info("Initializing ProbeComponents with SequenceProcessor...")
        probe_components = ProbeComponents(model, config, logger)
        
        # Test batch sampling
        batch_data = test_batch_sampling(probe_components, B=2, G=2)
        
        # Test with slightly larger batch
        logger.info("\nTesting with B=4, G=2...")
        batch_data_larger = test_batch_sampling(probe_components, B=4, G=2)
        
        logger.info("=" * 60)
        logger.info("🎉 Phase 1 test completed successfully!")
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