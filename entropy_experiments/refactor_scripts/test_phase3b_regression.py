#!/usr/bin/env python3
"""
Test script for Phase 3b: Regression baseline for RB-residual estimator

Tests the regression baseline that fits a ridge regression to predict 
residuals (G_j - H_j) from prefix-only features.
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
    return logging.getLogger('phase3b_regression_test')

def create_test_config(baseline_mode='regression', features=None, ridge_lambda=1e-3):
    """Create test configuration for Phase 3b."""
    if features is None:
        features = ["H", "top1", "margin", "head_mass", "two_point_entropy", "logit_var", "pos_frac"]
    
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
        'estimator': {  # Phase 3b config
            'x_estimator_mode': 'rb_residual',
            'rb_normalize_by_length': True,
            'baseline': {
                'mode': baseline_mode,
                'ema_decay': 0.9,  # kept for residual_mu fallback
                'ridge_lambda': ridge_lambda,
                'features': features,
            }
        }
    }

def test_regression_baseline(probe_components):
    """Test regression baseline functionality."""
    logger = logging.getLogger('phase3b_regression_test')
    logger.info("Testing regression baseline...")
    
    # Sample E-batch for regression baseline fitting
    E_total_sequences = 8  # Small for testing
    G = 2
    
    E_batch = probe_components.sample_E_batch_with_replacement(
        E_total_sequences=E_total_sequences, G=G
    )
    
    logger.info("✅ E-batch sampled successfully")
    logger.info(f"   sequences shape: {E_batch['sequences'].shape}")
    logger.info(f"   num_prompts: {E_batch['num_prompts']}")
    
    # Run RB X accumulation with regression baseline
    mb_size_prompts = 2  # Small microbatch
    sum_X_buf, B_local = probe_components.accumulate_sum_X(
        E_batch=E_batch, 
        mb_size_prompts=mb_size_prompts,
        weighting_mode="dr_grpo"
    )
    
    # Check gradient norm
    grad_norm = 0.0
    param_count = 0
    for param_id, grad_tensor in sum_X_buf.items():
        if grad_tensor is not None:
            norm = float(grad_tensor.norm())
            if norm > 0:
                grad_norm += norm ** 2
                param_count += 1
    grad_norm = grad_norm ** 0.5
    
    logger.info(f"✅ Regression baseline X accumulation complete")
    logger.info(f"   Prompts processed: {B_local}")
    logger.info(f"   Parameters with gradients: {param_count}")
    logger.info(f"   Total gradient norm: {grad_norm:.6e}")
    
    return grad_norm

def test_baseline_comparison(probe_components):
    """Compare regression vs residual_mu baseline."""
    logger = logging.getLogger('phase3b_regression_test')
    logger.info("Testing baseline comparison...")
    
    # Test with same E-batch for fair comparison
    E_total_sequences = 8
    G = 2
    E_batch = probe_components.sample_E_batch_with_replacement(
        E_total_sequences=E_total_sequences, G=G
    )
    
    results = {}
    
    # Test regression baseline
    probe_components.config['estimator']['baseline']['mode'] = 'regression'
    probe_components.model.zero_grad(set_to_none=True)
    sum_X_buf_reg, B_local_reg = probe_components.accumulate_sum_X(
        E_batch=E_batch, mb_size_prompts=2
    )
    
    grad_norm_reg = 0.0
    for param_id, grad_tensor in sum_X_buf_reg.items():
        if grad_tensor is not None:
            grad_norm_reg += float(grad_tensor.norm()) ** 2
    grad_norm_reg = grad_norm_reg ** 0.5
    results['regression'] = grad_norm_reg
    
    # Test residual_mu baseline for comparison
    probe_components.config['estimator']['baseline']['mode'] = 'residual_mu'
    probe_components.model.zero_grad(set_to_none=True)
    sum_X_buf_ema, B_local_ema = probe_components.accumulate_sum_X(
        E_batch=E_batch, mb_size_prompts=2
    )
    
    grad_norm_ema = 0.0
    for param_id, grad_tensor in sum_X_buf_ema.items():
        if grad_tensor is not None:
            grad_norm_ema += float(grad_tensor.norm()) ** 2
    grad_norm_ema = grad_norm_ema ** 0.5
    results['residual_mu'] = grad_norm_ema
    
    logger.info(f"✅ Baseline comparison complete")
    logger.info(f"   Regression gradient norm: {grad_norm_reg:.6e}")
    logger.info(f"   Residual_mu gradient norm: {grad_norm_ema:.6e}")
    ratio = grad_norm_reg / grad_norm_ema if grad_norm_ema > 0 else float('inf')
    logger.info(f"   Ratio (Regression/EMA): {ratio:.3f}")
    
    return results

def test_feature_configurations():
    """Test different feature configurations."""
    logger = logging.getLogger('phase3b_regression_test')
    logger.info("Testing different feature configurations...")
    
    # Test with different feature subsets
    feature_sets = [
        ["H", "top1"],  # minimal
        ["H", "top1", "margin", "pos_frac"],  # medium
        ["H", "top1", "margin", "head_mass", "two_point_entropy", "logit_var", "pos_frac"],  # full
    ]
    
    for i, features in enumerate(feature_sets):
        logger.info(f"   Testing feature set {i+1}: {features}")
        config = create_test_config(features=features)
        # Note: In a full test, we'd create new ProbeComponents with this config
        # For now, just log the configuration
        logger.info(f"   ✅ Feature set {i+1} configured: {len(features)} features")
    
    return feature_sets

def main():
    """Main test function."""
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("Phase 3b Test: Regression Baseline for RB Estimator")
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
        
        # Initialize with regression baseline config
        config = create_test_config(baseline_mode='regression')
        probe_components = ProbeComponents(model, config, logger)
        
        # Test 1: Basic regression baseline functionality
        logger.info("\\n--- Test 1: Basic regression baseline functionality ---")
        grad_norm_reg = test_regression_baseline(probe_components)
        
        # Test 2: Baseline comparison
        logger.info("\\n--- Test 2: Regression vs EMA baseline comparison ---")
        comparison_results = test_baseline_comparison(probe_components)
        
        # Test 3: Feature configuration testing
        logger.info("\\n--- Test 3: Feature configuration testing ---")
        feature_sets = test_feature_configurations()
        
        logger.info("=" * 60)
        logger.info("🎉 Phase 3b Regression Baseline Test succeeded!")
        logger.info(f"GPU memory usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        logger.info("Summary Results:")
        for baseline_type, grad_norm in comparison_results.items():
            logger.info(f"   {baseline_type}: {grad_norm:.6e}")
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