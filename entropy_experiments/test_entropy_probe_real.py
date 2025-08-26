#!/usr/bin/env python3
"""
Test script for offline entropy probe with real Qwen checkpoint.

This script uses the actual Qwen2.5-1.5B checkpoint from training
instead of mock models for realistic testing.
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
    """Setup logging for test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger("test_entropy_probe_real")


def create_real_checkpoint_config():
    """Create config using actual checkpoint paths."""
    return {
        # Core sampling parameters (smaller for testing)
        'batch_config': {
            'B': 2,  # Small batch for testing
            'G': 4,  # Small number of responses
            'dataset_name': 'gsm8k_r1_template',
            'split': 'train'
        },
        
        # Probe computation mode
        'probe_config': {
            'mode': 'blocks',  # Use blocks mode for memory efficiency
            'M': 2,  # Small number of blocks
            'pairs_per_block': 'all'
        },
        
        # Memory and performance (optimized for A100)
        'memory_config': {
            'microbatch_size': 2,  # 2 prompts per microbatch for A100
            'amp': True,
            'dtype': 'bfloat16'
        },
        
        # Variance estimation
        'stats_config': {
            'compute_plugin_se': True,
            'compute_jackknife_se': True
        },
        
        # Importance sampling
        'importance_sampling': {
            'use_snis': True,
            'use_psis': False,  # Keep simple for testing
            'ess_threshold': 0.5,
            'resample_on_low_ess': False  # Disable for testing
        },
        
        # Distributed settings
        'distributed': {
            'find_unused_parameters': True,
            'reduce_dtype': 'float32'
        },
        
        # Real checkpoint paths (using S3 checkpoint with proper optimizer states)
        'checkpoint': {
            'checkpoint_path': '/home/ubuntu/localfs/checkpoints/qwen2_05_finetuned/checkpoint-156',
            'optimizer_path': '/home/ubuntu/localfs/checkpoints/qwen2_05_finetuned/checkpoint-156/optimizer.pt',  # Real Adam states from S3
            'model_config_path': 'Qwen/Qwen2-0.5B'  # Base model for 0.5B checkpoint
        },
        
        # Generation settings (shorter for testing)
        'generation': {
            'max_new_tokens': 50,  # Shorter for testing
            'temperature': 0.7,
            'top_p': 1.0,
            'do_sample': True,
            'pad_token_id': None
        },
        
        # Output settings
        'output': {
            'save_results': True,  # Save test results
            'results_path': 'entropy_probe_test_results.json',
            'log_level': 'INFO',
            'save_samples': False
        },
        
        # Advanced options
        'advanced': {
            'cross_fitting': False,
            'force_recompute': True,
            'validation_mode': True,
            'profile_memory': True,  # Profile memory for diagnostics
            'profile_timing': True
        }
    }


def test_real_checkpoint_loading(config, logger):
    """Test loading actual checkpoint."""
    logger.info("Testing real checkpoint loading...")
    
    try:
        # Create probe and load checkpoint
        probe = OfflineEntropyProbe(config)
        probe.load_checkpoint(config['checkpoint']['checkpoint_path'])
        
        # Verify model is loaded
        if probe.model is None:
            raise RuntimeError("Model not loaded")
            
        if probe.optimizer is None:
            raise RuntimeError("Optimizer not loaded")
            
        logger.info("✓ Real checkpoint loaded successfully")
        logger.info(f"  Model device: {next(probe.model.parameters()).device}")
        logger.info(f"  Model dtype: {next(probe.model.parameters()).dtype}")
        
        return True, probe
        
    except Exception as e:
        logger.error(f"✗ Checkpoint loading failed: {e}")
        return False, None


def test_batch_sampling_real(probe, config, logger):
    """Test batch sampling with real model."""
    logger.info("Testing batch sampling with real model...")
    
    try:
        # Sample a small batch
        batch_data = probe.probe_components.sample_batch(
            B=config['batch_config']['B'],
            G=config['batch_config']['G']
        )
        
        # Validate batch structure
        required_keys = ['sequences', 'prompt_lens', 'advantages', 'max_lengths', 'attention_masks']
        for key in required_keys:
            if key not in batch_data:
                raise ValueError(f"Missing key in batch_data: {key}")
                
        B, G = config['batch_config']['B'], config['batch_config']['G']
        if batch_data['sequences'].shape[0] != B or batch_data['sequences'].shape[1] != G:
            raise ValueError(f"Wrong sequences shape: {batch_data['sequences'].shape}")
            
        logger.info("✓ Real batch sampling test passed")
        logger.info(f"  Batch shape: {batch_data['sequences'].shape}")
        logger.info(f"  Advantages shape: {batch_data['advantages'].shape}")
        logger.info(f"  Advantage range: [{batch_data['advantages'].min():.3f}, {batch_data['advantages'].max():.3f}]")
        
        return True, batch_data
        
    except Exception as e:
        logger.error(f"✗ Real batch sampling test failed: {e}")
        return False, None


def test_entropy_probe_computation(probe, batch_data, config, logger):
    """Test the core entropy probe computation."""
    logger.info("Testing entropy probe computation...")
    
    try:
        # Test δH₁ computation (predicted entropy change)
        delta_h1_results = probe.probe_components.compute_delta_h1(
            batch_data=batch_data,
            adam_preconditioner=probe.adam_preconditioner,
            u_statistics=probe.u_statistics,
            distributed_helpers=None  # Single GPU test
        )
        
        # Check results
        U_cross = delta_h1_results['U_cross']
        deltaH1 = delta_h1_results['deltaH1']
        
        logger.info(f"✓ δH₁ computation completed")
        logger.info(f"  U_cross = {U_cross:.6f}")
        logger.info(f"  δH₁ = {deltaH1:.6f}")
        
        # Check variance estimates
        se_plugin = delta_h1_results.get('se_plugin', float('inf'))
        se_jack = delta_h1_results.get('se_jack', float('inf'))
        
        if se_plugin != float('inf'):
            logger.info(f"  SE (plug-in) = {se_plugin:.6f}")
        if se_jack != float('inf'):
            logger.info(f"  SE (jackknife) = {se_jack:.6f}")
        
        return True, delta_h1_results
        
    except Exception as e:
        logger.error(f"✗ Entropy probe computation failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False, None


def test_importance_sampling(probe, batch_data, logger):
    """Test importance sampling for actual entropy change."""
    logger.info("Testing importance sampling...")
    
    try:
        # Test actual entropy change computation
        is_results = probe.importance_sampler.compute_entropy_change(
            batch_data=batch_data,
            optimizer=probe.optimizer
        )
        
        deltaH_snis = is_results['deltaH_snis']
        ESS = is_results['ESS']
        
        logger.info(f"✓ Importance sampling completed")
        logger.info(f"  ΔH (SNIS) = {deltaH_snis:.6f}")
        logger.info(f"  ESS = {ESS:.2f}")
        
        return True, is_results
        
    except Exception as e:
        logger.error(f"✗ Importance sampling failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False, None


def main():
    """Main test function with real checkpoint."""
    logger = setup_logging()
    logger.info("Starting entropy probe tests with real Qwen2.5-1.5B checkpoint...")
    
    # Set deterministic behavior
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    # Create configuration for real checkpoint
    config = create_real_checkpoint_config()
    
    # Test 1: Load real checkpoint
    success, probe = test_real_checkpoint_loading(config, logger)
    if not success:
        logger.error("Test 1 (checkpoint loading) failed - stopping")
        return False
    
    # Test 2: Sample batch with real model
    success, batch_data = test_batch_sampling_real(probe, config, logger)
    if not success:
        logger.error("Test 2 (real batch sampling) failed - stopping")
        return False
    
    # Test 3: Core entropy probe computation
    success, delta_h1_results = test_entropy_probe_computation(probe, batch_data, config, logger)
    if not success:
        logger.error("Test 3 (entropy probe computation) failed - stopping")
        return False
    
    # Test 4: Importance sampling
    success, is_results = test_importance_sampling(probe, batch_data, logger)
    if not success:
        logger.error("Test 4 (importance sampling) failed - stopping")
        return False
    
    # Test 5: Full pipeline
    logger.info("Testing full entropy probe pipeline...")
    try:
        full_results = probe.run_offline_analysis(config['checkpoint']['checkpoint_path'])
        
        logger.info("🎉 Full entropy probe pipeline completed successfully!")
        logger.info(f"  δH₁ = {full_results['deltaH1']:.6f}")
        logger.info(f"  ΔH (SNIS) = {full_results['deltaH_snis']:.6f}")
        logger.info(f"  ESS = {full_results['ESS']:.2f}")
        logger.info(f"  Total time = {full_results['timing']['total_time']:.2f}s")
        
        if config['output']['save_results']:
            logger.info(f"Results saved to: {config['output']['results_path']}")
            
    except Exception as e:
        logger.error(f"✗ Full pipeline test failed: {e}")
        return False
    
    logger.info("🎉 All entropy probe tests passed with real checkpoint!")
    logger.info("The entropy probe is ready for production use.")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)