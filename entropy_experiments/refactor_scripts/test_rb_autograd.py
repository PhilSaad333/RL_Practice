#!/usr/bin/env python3
"""
Test script for RB autograd functionality (Phase 2 - small step)

Tests the rb_requires_grad flag in SequenceProcessor to enable/disable
differentiable RB entropy computation.
"""

import sys
import os
import logging
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from sequence_processing.sequence_processor import SequenceProcessor, GenerationConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

def setup_logging():
    """Set up logging for the test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('rb_autograd_test')

def load_model_and_tokenizer(checkpoint_path: str):
    """Load the model and tokenizer from checkpoint."""
    logger = logging.getLogger('rb_autograd_test')
    
    logger.info(f"Loading LoRA model from checkpoint: {checkpoint_path}")
    
    # Import the proper LoRA loader
    from entropy_experiments.offline_entropy_probe import load_peft_for_probe
    
    # Load base model + LoRA adapter (same approach as offline_entropy_probe)
    model = load_peft_for_probe(
        base_id="Qwen/Qwen2.5-1.5B",
        adapter_path=checkpoint_path,
        mode="lora_simple",
        dtype="bf16",
        device_map="cuda",
        use_checkpointing=False
    )
    
    # Set model to training mode for gradients
    model.train()
    
    # Ensure adapter is active
    if hasattr(model, "set_adapter"):
        model.set_adapter("default")
    
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-1.5B", 
        trust_remote_code=True
    )
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    
    # DIAGNOSTICS: Count trainable parameters (same as offline_entropy_probe)
    peft_model = model.module if hasattr(model, "module") else model
    trainable_named = [(n, p) for (n, p) in peft_model.named_parameters() if p.requires_grad]
    lora_named = [(n, p) for (n, p) in trainable_named 
                  if ("lora_a" in n.lower()) or ("lora_b" in n.lower()) or n.endswith("lm_head.weight")]
    
    logger.info(f"Trainable parameters after LoRA loading: {len(trainable_named)} total, {len(lora_named)} LoRA params")
    
    # Show first few LoRA parameter names for verification
    lora_names = [name for name, _ in lora_named[:5]]
    if lora_names:
        logger.info(f"Sample LoRA parameters: {lora_names}")
    else:
        logger.error("ERROR: No LoRA parameters found in trainable params!")
    
    return model, tokenizer

def test_rb_autograd_flag(model, tokenizer):
    """Test RB autograd flag functionality."""
    logger = logging.getLogger('rb_autograd_test')
    
    # Test data
    prompts = ["Solve: 2+2=", "What is 5*3?"]
    
    logger.info("=" * 50)
    logger.info("Test 1: rb_requires_grad=False (default)")
    
    # Config with RB autograd disabled
    config_no_grad = GenerationConfig(
        temperature=0.7,
        top_p=0.995,  # Add explicit top_p for RB computation
        max_new_tokens=50,
        rb_requires_grad=False,  # Disabled
        gen_batch_size=4,
        tf_batch_size=8
    )
    
    processor_no_grad = SequenceProcessor(model, tokenizer, config_no_grad)
    
    # Generate and compute teacher forcing
    sequences = processor_no_grad.generate_batched(prompts, G=2)
    logprob_results, _ = processor_no_grad.teacher_force_logprobs(sequences, with_grad=True, compute_rb=True)
    
    logger.info(f"With rb_requires_grad=False:")
    logger.info(f"  rb_entropies type: {type(logprob_results.rb_entropies[0][0]) if logprob_results.rb_entropies[0] else 'empty'}")
    logger.info(f"  rb_entropies_torch: {logprob_results.rb_entropies_torch}")
    logger.info(f"  config.rb_requires_grad: {processor_no_grad.config.rb_requires_grad}")
    logger.info(f"  config.top_p: {processor_no_grad.config.top_p}")
    logger.info(f"  config.temperature: {processor_no_grad.config.temperature}")
    
    logger.info("=" * 50)
    logger.info("Test 2: rb_requires_grad=True")
    
    # Config with RB autograd enabled
    config_with_grad = GenerationConfig(
        temperature=0.7,
        top_p=0.995,  # Add explicit top_p for RB computation
        max_new_tokens=50,
        rb_requires_grad=True,  # Enabled
        gen_batch_size=4,
        tf_batch_size=8
    )
    
    processor_with_grad = SequenceProcessor(model, tokenizer, config_with_grad)
    
    # Generate and compute teacher forcing
    sequences = processor_with_grad.generate_batched(prompts, G=2)
    logprob_results, _ = processor_with_grad.teacher_force_logprobs(sequences, with_grad=True, compute_rb=True)
    
    logger.info(f"With rb_requires_grad=True:")
    logger.info(f"  rb_entropies type: {type(logprob_results.rb_entropies[0][0]) if logprob_results.rb_entropies[0] else 'empty'}")
    logger.info(f"  rb_entropies_torch available: {logprob_results.rb_entropies_torch is not None}")
    logger.info(f"  config.rb_requires_grad: {processor_with_grad.config.rb_requires_grad}")
    logger.info(f"  config.top_p: {processor_with_grad.config.top_p}")
    logger.info(f"  config.temperature: {processor_with_grad.config.temperature}")
    
    if logprob_results.rb_entropies_torch is not None:
        logger.info(f"  rb_entropies_torch[0][0] type: {type(logprob_results.rb_entropies_torch[0][0])}")
        logger.info(f"  rb_entropies_torch[0][0] requires_grad: {logprob_results.rb_entropies_torch[0][0].requires_grad}")
        logger.info(f"  rb_entropies_torch[0][0] shape: {logprob_results.rb_entropies_torch[0][0].shape}")
        
        # Test gradient computation
        if len(logprob_results.rb_entropies_torch[0][0]) > 0:
            rb_sum = logprob_results.rb_entropies_torch[0][0].sum()
            logger.info(f"  Can compute gradients: {rb_sum.requires_grad}")
            
            # Try a simple backward pass
            try:
                rb_sum.backward(retain_graph=True)
                logger.info("  ✅ Backward pass successful - RB autograd working!")
            except Exception as e:
                logger.warning(f"  ⚠️ Backward pass failed: {e}")
        else:
            logger.info("  Empty RB tensor (no generation)")
    
    logger.info("=" * 50)
    logger.info("✅ RB autograd flag test completed!")

def main():
    """Main test function."""
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("RB Autograd Flag Test (Phase 2)")
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
        
        # Test RB autograd functionality
        test_rb_autograd_flag(model, tokenizer)
        
        logger.info("=" * 60)
        logger.info("🎉 RB autograd test completed successfully!")
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