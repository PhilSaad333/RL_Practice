#!/usr/bin/env python3
"""
Test script for comprehensive LoRA gradient flow fixes.
This will run our PEFT adapter diagnostics and α-trick to verify fixes work.
"""

import sys
import os

# Ensure we can import our modules
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import logging

# Import from the parent package to avoid relative import issues
from entropy_experiments.offline_entropy_probe import OfflineEntropyProbe

# Configure logging to see our debug messages
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_lora_gradient_fixes():
    """Test the comprehensive LoRA gradient flow fixes."""
    
    print("🔧 Testing comprehensive LoRA gradient flow fixes...")
    
    # Use our debug config with small batches
    config_path = "entropy_experiments/configs/old_configs_backup/debug_scaling_config.yaml"
    checkpoint_path = "/home/ubuntu/localfs/checkpoints/qwen2_5_15_finetuned/qwen2_5_15_gsm8k_lora/checkpoint-156"
    
    print(f"Loading config from: {config_path}")
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    try:
        # Load config and set checkpoint path
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Set the checkpoint path in config
        config['checkpoint']['checkpoint_path'] = checkpoint_path
        config['checkpoint']['optimizer_path'] = checkpoint_path
        
        # Initialize probe with comprehensive diagnostics
        probe = OfflineEntropyProbe(config)
        
        print("✅ Probe initialized successfully")
        
        # Test with very small batch to focus on gradient flow diagnostics
        B_E = 2  # Just 2 prompts for testing
        B_U = 2
        
        print(f"🧪 Running gradient flow test with B_E={B_E}, B_U={B_U}")
        
        # This should trigger all our new diagnostics:
        # 1. PEFT adapter activation enforcement 
        # 2. LoRA on/off logits comparison
        # 3. S gradient path verification
        # 4. VJP S→LoRA parameter flow test
        result = probe.estimate_conditional_variance_E_U(B_E=B_E, B_U=B_U)
        
        print("✅ GRADIENT FLOW TEST PASSED!")
        print(f"Results: δH₁ = {result['delta_H1']:.6f} ± {result['se_delta_H1']:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_lora_gradient_fixes()
    sys.exit(0 if success else 1)