#!/usr/bin/env python3
"""
Test script for Phase 3 Step 3: RB-residual X loss builder

Tests the core _build_X_loss_rb_residual method with mock data.
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
    return logging.getLogger('phase3_step3_test')

def create_mock_logprob_results(device, B=2, G=2, max_T=5):
    """Create mock LogprobResults for testing."""
    from sequence_processing.sequence_processor import LogprobResults
    
    # Create mock data with variable sequence lengths
    logprobs = []
    rb_entropies_torch = []
    
    for b in range(B):
        b_logprobs = []
        b_rb_torch = []
        for g in range(G):
            # Variable length sequences 
            L = max_T - g  # Different lengths per sequence
            if L > 0:
                # Mock logprobs (token log probabilities) - need gradients
                lp = torch.randn(L, device=device, requires_grad=True)
                b_logprobs.append(lp)
                
                # Mock RB entropies - need gradients for pathwise term
                rb = torch.rand(L, device=device, requires_grad=True) + 0.1  # positive entropies
                b_rb_torch.append(rb)
            else:
                b_logprobs.append(torch.tensor([], device=device))
                b_rb_torch.append(torch.tensor([], device=device))
        
        logprobs.append(b_logprobs)
        rb_entropies_torch.append(b_rb_torch)
    
    # Create mock entropies, sequence_logprobs, rb_entropies, and rewards
    entropies = []
    sequence_logprobs = []
    rb_entropies = []
    rewards = []
    
    for b in range(B):
        b_entropies = []
        b_seq_logprobs = []
        b_rb_entropies = []
        b_rewards = []
        
        for g in range(G):
            L = max_T - g
            if L > 0:
                b_entropies.append(torch.rand(L).numpy())  # mock numpy entropies
                b_seq_logprobs.append(float(torch.randn(1).item()))  # mock sequence logprob
                b_rb_entropies.append(torch.rand(L).numpy())  # mock numpy RB entropies
                b_rewards.append(float(torch.rand(1).item()))  # mock reward
            else:
                b_entropies.append(torch.tensor([]).numpy())
                b_seq_logprobs.append(0.0)
                b_rb_entropies.append(torch.tensor([]).numpy())
                b_rewards.append(0.0)
        
        entropies.append(b_entropies)
        sequence_logprobs.append(b_seq_logprobs)
        rb_entropies.append(b_rb_entropies)
        rewards.append(b_rewards)
    
    # Create LogprobResults with all required fields
    return LogprobResults(
        logprobs=logprobs,
        entropies=entropies,
        sequence_logprobs=sequence_logprobs,
        rb_entropies=rb_entropies,
        rewards=rewards,
        rb_entropies_torch=rb_entropies_torch
    )

def test_rb_loss_builder(probe_components):
    """Test the RB-residual X loss builder."""
    logger = logging.getLogger('phase3_step3_test')
    logger.info("Testing RB-residual X loss builder...")
    
    device = probe_components.device
    B, G = 2, 2
    
    # Create mock LogprobResults
    logprob_results = create_mock_logprob_results(device, B=B, G=G, max_T=6)
    prompt_lens = [3, 4]  # Mock prompt lengths
    
    # Log initial baseline state
    logger.info(f"Initial baseline mu: {probe_components._baseline_state_x.mu}")
    
    # Test the RB loss builder
    logger.info("Computing RB-residual loss...")
    loss = probe_components._build_X_loss_rb_residual(
        logprob_results=logprob_results,
        prompt_lens=prompt_lens,
        normalize_by_length=True
    )
    
    logger.info(f"✅ RB loss computed successfully")
    logger.info(f"   Loss value: {loss.item():.6f}")
    logger.info(f"   Loss requires_grad: {loss.requires_grad}")
    logger.info(f"   Loss shape: {loss.shape}")
    
    # Verify loss properties
    assert loss.requires_grad, "Loss should require gradients"
    assert loss.shape == torch.Size([]), "Loss should be scalar"
    assert not torch.isnan(loss), "Loss should not be NaN"
    assert not torch.isinf(loss), "Loss should not be infinite"
    
    # Log updated baseline state
    logger.info(f"Updated baseline mu: {probe_components._baseline_state_x.mu[:6]}")
    
    # Test backward pass
    logger.info("Testing backward pass...")
    loss.backward()
    
    # Check that gradients were computed
    # Note: With mock data not derived from model forward pass, gradients may be zero
    grad_norm = 0.0
    param_count = 0
    for p in probe_components.model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.data.norm().item() ** 2
            param_count += 1
    grad_norm = grad_norm ** 0.5
    
    logger.info(f"✅ Backward pass successful")
    logger.info(f"   Parameters with gradients: {param_count}")
    logger.info(f"   Total gradient norm: {grad_norm:.6e}")
    
    # With mock data, gradients may be zero since mock tensors aren't connected to model params
    # The important thing is that backward() succeeded without errors
    if grad_norm > 0:
        logger.info(f"   ✅ Non-zero gradients found (unexpected but good with mock data)")
    else:
        logger.info(f"   ⚠️ Zero gradients (expected with mock data not from model forward pass)")
    
    # Don't assert on gradient magnitude with mock data - the structure test is what matters
    
    return loss, grad_norm

def test_baseline_modes(probe_components):
    """Test different baseline modes."""
    logger = logging.getLogger('phase3_step3_test')
    logger.info("Testing different baseline modes...")
    
    device = probe_components.device
    logprob_results = create_mock_logprob_results(device, B=1, G=1, max_T=4)
    prompt_lens = [2]
    
    # Test 1: residual_mu mode (default)
    probe_components.config['estimator']['baseline']['mode'] = 'residual_mu'
    loss_residual = probe_components._build_X_loss_rb_residual(
        logprob_results, prompt_lens, normalize_by_length=True
    )
    logger.info(f"✅ residual_mu mode: loss={loss_residual.item():.6f}")
    
    # Clear gradients
    for p in probe_components.model.parameters():
        if p.grad is not None:
            p.grad.zero_()
    
    # Test 2: none mode (no baseline)
    probe_components.config['estimator']['baseline']['mode'] = 'none'
    loss_none = probe_components._build_X_loss_rb_residual(
        logprob_results, prompt_lens, normalize_by_length=True
    )
    logger.info(f"✅ none mode: loss={loss_none.item():.6f}")
    
    # Losses should be different (unless μ=0 by coincidence)
    logger.info(f"   Loss difference: {abs(loss_residual.item() - loss_none.item()):.6f}")
    
    # Reset to default
    probe_components.config['estimator']['baseline']['mode'] = 'residual_mu'
    
    return loss_residual, loss_none

def main():
    """Main test function."""
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("Phase 3 Step 3 Test: RB-residual X Loss Builder")
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
        
        # Phase 3 config
        config = {
            'batch_config': {'G': 2, 'rollout_batch_size': 4},
            'generation': {
                'temperature': 0.7, 'top_p': 0.995, 'max_new_tokens': 150,
                'gen_batch_size': 4, 'tf_batch_size': 8, 'rb_requires_grad': True,
            },
            'memory_config': {'amp': True, 'dtype': 'bfloat16', 'microbatch_size': 1},
            'estimator': {
                'x_estimator_mode': 'rb_residual', 'rb_normalize_by_length': True,
                'baseline': {'mode': 'residual_mu', 'ema_decay': 0.9}
            }
        }
        
        probe_components = ProbeComponents(model, config, logger)
        
        # Test 1: Basic RB loss builder
        logger.info("\n--- Test 1: RB-residual loss computation ---")
        loss, grad_norm = test_rb_loss_builder(probe_components)
        
        # Test 2: Different baseline modes
        logger.info("\n--- Test 2: Baseline mode testing ---")
        loss_residual, loss_none = test_baseline_modes(probe_components)
        
        logger.info("=" * 60)
        logger.info("🎉 Phase 3 Step 3 tests completed successfully!")
        logger.info(f"GPU memory usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        logger.info(f"Final gradient norm: {grad_norm:.6e}")
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