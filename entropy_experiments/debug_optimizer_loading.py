#!/usr/bin/env python3
"""
Debug script for optimizer state loading issues

This script identifies exactly why optimizer loading fails and shows how to fix it.
"""
import torch
import yaml
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def debug_optimizer_loading():
    """Debug the optimizer loading process step by step."""
    print("=" * 70)
    print("OPTIMIZER STATE LOADING DEBUG")
    print("=" * 70)
    
    # Load model the same way the probe does
    from entropy_experiments.offline_entropy_probe import load_peft_for_probe
    
    checkpoint_path = '/home/ubuntu/localfs/checkpoints/qwen2_5_15_finetuned/qwen2_5_15_gsm8k_lora/checkpoint-156'
    
    print("1. Loading model...")
    model = load_peft_for_probe(
        base_id='Qwen/Qwen2.5-1.5B',
        adapter_path=checkpoint_path,
        mode='lora_simple',
        dtype='bf16',
        device_map='cuda',
        use_checkpointing=False
    )
    model.train()
    if hasattr(model, 'set_adapter'):
        model.set_adapter('default')
    
    # Analyze model parameters
    all_params = list(model.parameters())
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    frozen_params = [p for p in model.parameters() if not p.requires_grad]
    
    print(f"   Total parameters: {len(all_params)}")
    print(f"   Trainable parameters: {len(trainable_params)}")  
    print(f"   Frozen parameters: {len(frozen_params)}")
    print()
    
    # Load saved optimizer state
    optimizer_path = f'{checkpoint_path}/optimizer.pt'
    print(f"2. Loading saved optimizer state from: {optimizer_path}")
    optimizer_state = torch.load(optimizer_path, map_location='cpu')
    
    saved_groups = optimizer_state.get('param_groups', [])
    saved_state = optimizer_state.get('state', {})
    saved_param_count = sum(len(group.get('params', [])) for group in saved_groups)
    
    print(f"   Saved parameter groups: {len(saved_groups)}")
    print(f"   Saved parameters total: {saved_param_count}")
    print(f"   Saved state entries: {len(saved_state)}")
    
    # Analyze parameter group structure
    print(f"   Parameter group details:")
    for i, group in enumerate(saved_groups):
        params_in_group = len(group.get('params', []))
        lr = group.get('lr', 'unknown')
        wd = group.get('weight_decay', 'unknown') 
        print(f"     Group {i}: {params_in_group} params, lr={lr}, weight_decay={wd}")
    print()
    
    # Current probe code BUG: Uses model.parameters() (includes frozen)
    print("3. CURRENT PROBE CODE (BUGGY):")
    print("   optimizer = AdamW(self.model.parameters(), ...)  # ← BUG!")
    buggy_optimizer = torch.optim.AdamW(model.parameters(), lr=0.0, weight_decay=0.0)
    buggy_param_count = sum(len(group['params']) for group in buggy_optimizer.param_groups)
    print(f"   Buggy optimizer sees: {buggy_param_count} parameters (includes frozen!)")
    print(f"   This causes: {saved_param_count} saved → {buggy_param_count} current mismatch")
    print()
    
    # Fixed approach: Use only trainable parameters
    print("4. FIXED APPROACH:")
    print("   optimizer = AdamW([p for p in model.parameters() if p.requires_grad], ...)")
    fixed_optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], 
        lr=0.0, weight_decay=0.0
    )
    fixed_param_count = sum(len(group['params']) for group in fixed_optimizer.param_groups)
    print(f"   Fixed optimizer sees: {fixed_param_count} parameters (trainable only)")
    print(f"   Perfect match: {saved_param_count} saved = {fixed_param_count} current")
    
    # Compare parameter group structures
    print(f"   Current parameter group details:")
    for i, group in enumerate(fixed_optimizer.param_groups):
        params_in_group = len(group['params'])
        lr = group.get('lr', 'unknown')
        wd = group.get('weight_decay', 'unknown')
        print(f"     Group {i}: {params_in_group} params, lr={lr}, weight_decay={wd}")
    print(f"   Group structure mismatch: {len(saved_groups)} saved vs {len(fixed_optimizer.param_groups)} current")
    print()
    
    # Test remapping with fixed optimizer
    print("5. TESTING PARAMETER REMAPPING:")
    from entropy_experiments.offline_entropy_probe import OfflineEntropyProbe
    
    # Create a dummy probe instance to access the remapping function
    config = {'importance': {'enabled': False}}
    probe = OfflineEntropyProbe.__new__(OfflineEntropyProbe)
    probe.logger = torch.multiprocessing.get_logger()
    
    try:
        print("   Running _remap_optimizer_state_ids with fixed optimizer...")
        remapped_state = probe._remap_optimizer_state_ids(optimizer_state, fixed_optimizer)
        
        # Test if the remapped state loads successfully
        fixed_optimizer.load_state_dict(remapped_state)
        print("   ✅ SUCCESS: Optimizer state loaded successfully!")
        
        # Verify Adam states are present
        state_count = len(fixed_optimizer.state)
        print(f"   ✅ Adam state loaded for {state_count}/{fixed_param_count} parameters")
        
        # Check for exp_avg_sq (needed for Adam preconditioning)  
        has_adam_state = False
        for param_id, state in fixed_optimizer.state.items():
            if 'exp_avg_sq' in state:
                has_adam_state = True
                break
                
        if has_adam_state:
            print("   ✅ exp_avg_sq found - Adam preconditioning will work!")
        else:
            print("   ❌ No exp_avg_sq found - Adam preconditioning may fail")
            
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        
    print()
    print("=" * 70)
    print("CONCLUSION:")
    print("The bug is in _initialize_optimizer_from_state line 378:")
    print("  OLD: optimizer = AdamW(self.model.parameters(), ...)")
    print("  NEW: optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad], ...)")
    print("=" * 70)

if __name__ == "__main__":
    debug_optimizer_loading()