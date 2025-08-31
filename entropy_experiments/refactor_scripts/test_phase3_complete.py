#!/usr/bin/env python3
"""
Test script for Phase 3 Complete: End-to-end RB-residual estimator

Tests the complete Phase 3 pipeline:
1. E-batch sampling with replacement
2. RB-residual X accumulation
3. Baseline EMA updates 
4. Gradient flow comparison with naive mode
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
    return logging.getLogger('phase3_complete_test')

def create_test_config(estimator_mode='rb_residual'):
    """Create test configuration for Phase 3."""
    return {
        'batch_config': {
            'dataset_name': 'gsm8k_r1_template',
            'split': 'test',
            'G': 2,
            'rollout_batch_size': 2,  # Small for testing
        },
        'generation': {
            'temperature': 0.7,
            'top_p': 0.995,
            'max_new_tokens': 50,  # Small for testing
            'gen_batch_size': 4,
            'tf_batch_size': 8,
            'rb_requires_grad': True,  # Required for Phase 3
        },
        'memory_config': {
            'amp': True,
            'dtype': 'bfloat16',
            'microbatch_size': 1,  # Small for testing
        },
        'estimator': {  # Phase 3 config
            'x_estimator_mode': estimator_mode,
            'rb_normalize_by_length': True,
            'baseline': {
                'mode': 'residual_mu',
                'ema_decay': 0.9,
            }
        }
    }

def test_e_batch_sampling(probe_components):
    """Test E-batch sampling with replacement."""
    logger = logging.getLogger('phase3_complete_test')
    logger.info("Testing E-batch sampling...")
    
    # Sample small E-batch
    E_total_sequences = 8  # Small for testing
    G = 2
    
    batch_E = probe_components.sample_E_batch_with_replacement(
        E_total_sequences=E_total_sequences, G=G
    )
    
    logger.info(f"✅ E-batch sampled successfully")
    logger.info(f"   sequences shape: {batch_E['sequences'].shape}")
    logger.info(f"   advantages shape: {batch_E['advantages'].shape}")
    logger.info(f"   num_prompts: {batch_E['num_prompts']}")
    logger.info(f"   prompt_ids: {batch_E['prompt_ids']}")
    
    # Check for replacement (duplicates)
    unique_prompts = len(set(batch_E['prompt_ids']))
    total_prompts = len(batch_E['prompt_ids'])
    duplicates = total_prompts - unique_prompts
    
    if duplicates > 0:
        logger.info(f"   ✅ Replacement sampling working: {duplicates} duplicate prompts")
    else:
        logger.info(f"   ⚠️ No duplicates found (may be expected with small sample)")
    
    return batch_E

def test_rb_x_accumulation(probe_components, E_batch):
    """Test RB-residual X accumulation."""
    logger = logging.getLogger('phase3_complete_test')
    logger.info("Testing RB-residual X accumulation...")
    
    # Check initial baseline state
    initial_mu = probe_components._baseline_state_x.mu.clone() if probe_components._baseline_state_x.mu.numel() > 0 else torch.tensor([])
    logger.info(f"Initial baseline mu: {initial_mu}")
    
    # Accumulate X gradients using RB estimator
    mb_size_prompts = 2  # Small microbatch
    sum_X_buf, B_local = probe_components.accumulate_sum_X(
        E_batch=E_batch, 
        mb_size_prompts=mb_size_prompts,
        weighting_mode="dr_grpo"  # Used only in naive mode
    )
    
    # Check updated baseline state
    updated_mu = probe_components._baseline_state_x.mu
    logger.info(f"Updated baseline mu: {updated_mu[:min(10, updated_mu.numel())]}")
    
    # Calculate gradient norms
    grad_norm = 0.0
    param_count = 0
    for param_id, grad_tensor in sum_X_buf.items():
        if grad_tensor is not None:
            norm = float(grad_tensor.norm())
            if norm > 0:
                grad_norm += norm ** 2
                param_count += 1
    grad_norm = grad_norm ** 0.5
    
    logger.info(f"✅ RB X accumulation complete")
    logger.info(f"   Prompts processed: {B_local}")
    logger.info(f"   Parameters with gradients: {param_count}")
    logger.info(f"   Total gradient norm: {grad_norm:.6e}")
    logger.info(f"   Baseline updated: {updated_mu.numel() > initial_mu.numel() or not torch.equal(initial_mu, updated_mu[:initial_mu.numel()]) if initial_mu.numel() > 0 else updated_mu.numel() > 0}")
    
    return sum_X_buf, grad_norm

def test_naive_comparison(probe_components, E_batch):
    """Test naive X accumulation for comparison."""
    logger = logging.getLogger('phase3_complete_test')
    logger.info("Testing naive X accumulation for comparison...")
    
    # Temporarily switch to naive mode
    original_mode = probe_components.config.get('estimator', {}).get('x_estimator_mode', 'naive')
    probe_components.config.setdefault('estimator', {})['x_estimator_mode'] = 'naive'
    
    try:
        # Clear gradients
        probe_components.model.zero_grad(set_to_none=True)
        
        # Accumulate X gradients using naive estimator
        mb_size_prompts = 2
        sum_X_buf_naive, B_local_naive = probe_components.accumulate_sum_X(
            E_batch=E_batch,
            mb_size_prompts=mb_size_prompts,
            weighting_mode="dr_grpo"
        )
        
        # Calculate naive gradient norm
        grad_norm_naive = 0.0
        param_count_naive = 0
        for param_id, grad_tensor in sum_X_buf_naive.items():
            if grad_tensor is not None:
                norm = float(grad_tensor.norm())
                if norm > 0:
                    grad_norm_naive += norm ** 2
                    param_count_naive += 1
        grad_norm_naive = grad_norm_naive ** 0.5
        
        logger.info(f"✅ Naive X accumulation complete")
        logger.info(f"   Prompts processed: {B_local_naive}")
        logger.info(f"   Parameters with gradients: {param_count_naive}")
        logger.info(f"   Total gradient norm: {grad_norm_naive:.6e}")
        
        return sum_X_buf_naive, grad_norm_naive
        
    finally:
        # Restore original mode
        probe_components.config['estimator']['x_estimator_mode'] = original_mode

def test_baseline_effect(probe_components):
    """Test baseline effect on variance reduction."""
    logger = logging.getLogger('phase3_complete_test')
    logger.info("Testing baseline effect...")
    
    # Sample a few E-batches to build up baseline
    for i in range(3):
        E_batch = probe_components.sample_E_batch_with_replacement(E_total_sequences=4, G=2)
        probe_components.accumulate_sum_X(E_batch, mb_size_prompts=2)
        logger.info(f"   Batch {i+1}: baseline mu = {probe_components._baseline_state_x.mu[:5]}")
    
    logger.info(f"✅ Baseline effect test complete")
    logger.info(f"   Final baseline state has {probe_components._baseline_state_x.mu.numel()} positions")

def main():
    """Main test function."""
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("Phase 3 Complete Test: End-to-End RB Estimator")
    logger.info("=" * 60)
    
    try:
        # Load model and initialize ProbeComponents
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
        
        # Initialize with RB-residual config
        config = create_test_config(estimator_mode='rb_residual')
        probe_components = ProbeComponents(model, config, logger)
        
        # Test 1: E-batch sampling
        logger.info("\n--- Test 1: E-batch sampling with replacement ---")
        E_batch = test_e_batch_sampling(probe_components)
        
        # Test 2: RB X accumulation
        logger.info("\n--- Test 2: RB-residual X accumulation ---")
        sum_X_buf_rb, grad_norm_rb = test_rb_x_accumulation(probe_components, E_batch)
        
        # Test 3: Naive comparison
        logger.info("\n--- Test 3: Naive X accumulation comparison ---")
        sum_X_buf_naive, grad_norm_naive = test_naive_comparison(probe_components, E_batch)
        
        # Test 4: Baseline effect
        logger.info("\n--- Test 4: Baseline EMA effect ---")
        test_baseline_effect(probe_components)
        
        # Summary comparison
        logger.info("\n--- Summary Comparison ---")
        logger.info(f"RB-residual gradient norm: {grad_norm_rb:.6e}")
        logger.info(f"Naive gradient norm: {grad_norm_naive:.6e}")
        ratio = grad_norm_rb / grad_norm_naive if grad_norm_naive > 0 else float('inf')
        logger.info(f"Ratio (RB/Naive): {ratio:.3f}")
        
        logger.info("=" * 60)
        logger.info("🎉 Phase 3 Complete Test succeeded!")
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