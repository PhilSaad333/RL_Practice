#!/usr/bin/env python3
"""
🔍 Optimizer State Diagnostic Script

Determine if zero Adam states are due to:
1. Parameter-set mismatch when loading optimizer state
2. Genuine training issue (no optimizer steps occurred)
3. Early checkpoint before any training

Based on detailed analysis of the issue.
"""

import torch
import json
import sys
from pathlib import Path

def check_saved_optimizer_structure(checkpoint_path):
    """Check A: Inspect saved param-groups and counts (no model needed)"""
    print("=" * 60)
    print("🔍 CHECK A: Saved Optimizer Structure Analysis")
    print("=" * 60)
    
    opt_path = Path(checkpoint_path) / "optimizer.pt"
    if not opt_path.exists():
        print(f"❌ No optimizer.pt found at {opt_path}")
        return None
        
    print(f"Loading optimizer state from: {opt_path}")
    opt_sd = torch.load(opt_path, map_location="cpu", weights_only=False)
    
    print(f"Saved groups: {len(opt_sd['param_groups'])}")
    group_sizes = [len(g["params"]) for g in opt_sd["param_groups"]]
    print(f"Group sizes: {group_sizes}")
    print(f"States entries: {len(opt_sd['state'])}")  # how many tensors actually have state
    
    # Optional: inspect the very first state's keys
    if opt_sd["state"]:
        first_state = next(iter(opt_sd["state"].values()), {})
        print(f"A sample state's keys: {list(first_state.keys())}")
        
        # Check if any exp_avg_sq are nonzero
        nonzero_count = 0
        for state in opt_sd["state"].values():
            if "exp_avg_sq" in state:
                norm = float(state["exp_avg_sq"].norm().item())
                if norm > 1e-8:
                    nonzero_count += 1
                    
        print(f"States with nonzero exp_avg_sq: {nonzero_count}/{len(opt_sd['state'])}")
    else:
        print("⚠️  No states found - checkpoint saved before any optimizer step!")
        
    return opt_sd

def check_training_info(checkpoint_path):
    """Check C: Verify optimizer step should have occurred"""
    print("\n" + "=" * 60)
    print("🔍 CHECK C: Training Progress Analysis") 
    print("=" * 60)
    
    info_path = Path(checkpoint_path) / "training_info.json"
    if not info_path.exists():
        print(f"❌ No training_info.json found at {info_path}")
        return
        
    with open(info_path, 'r') as f:
        info = json.load(f)
        
    step = info.get("step", 0)
    global_step = info.get("global_step", step)
    
    print(f"Checkpoint step: {step}")
    print(f"Global step: {global_step}")
    
    # Extract key training parameters
    training_config = info.get("training_config", {})
    buffer_size = training_config.get("buffer_size", "unknown")
    microbatch_size = training_config.get("microbatch_size", "unknown") 
    world_size = info.get("distributed_info", {}).get("world_size", 1)
    
    print(f"Buffer size: {buffer_size}")
    print(f"Microbatch size: {microbatch_size}")
    print(f"World size: {world_size}")
    
    # Calculate expected grad accumulation steps
    if isinstance(buffer_size, int) and isinstance(microbatch_size, int):
        expected_grad_accum = buffer_size // (world_size * microbatch_size)
        print(f"Expected grad_accum_steps: {expected_grad_accum}")
        
        # Check if we should have had optimizer steps by this checkpoint
        if step > 0:
            print(f"✅ Step {step} > 0: Should have had optimizer updates")
        else:
            print(f"⚠️  Step 0: May be initial checkpoint before training")
    
    return info

def check_parameter_matching(checkpoint_path):
    """Check B: Test LoRA-only optimizer recreation"""
    print("\n" + "=" * 60)
    print("🔍 CHECK B: Parameter Matching Test")
    print("=" * 60)
    
    print("Loading model to test parameter matching...")
    
    # Import model loading code
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel, prepare_model_for_kbit_training
    
    # Load model (similar to gradient test)
    base_model_name = "Qwen/Qwen2.5-1.5B"
    
    try:
        print("Loading base model...")
        bnb = BitsAndBytesConfig(load_in_4bit=True,
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_compute_dtype=torch.float16)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        base_model = prepare_model_for_kbit_training(base_model)
        
        print("Loading PEFT model...")
        model_path = Path(checkpoint_path) / "model"
        peft_model = PeftModel.from_pretrained(base_model, str(model_path), is_trainable=True)
        peft_model.enable_input_require_grads()
        
        print("✅ Model loaded successfully")
        
        # Get LoRA parameters
        peft_unwrapped = peft_model.module if hasattr(peft_model, "module") else peft_model
        lora_params = [p for n, p in peft_unwrapped.named_parameters()
                      if p.requires_grad and ("lora_A" in n or "lora_B" in n)]
        
        print(f"Found {len(lora_params)} LoRA parameters")
        
        # Test both approaches
        print("\n--- Testing ALL parameters optimizer (current approach) ---")
        from torch.optim import AdamW
        all_params = list(peft_model.parameters())
        opt_all = AdamW(all_params, lr=1e-4, weight_decay=0.01)
        
        print(f"All params optimizer group sizes: {[len(g['params']) for g in opt_all.state_dict()['param_groups']]}")
        
        print("\n--- Testing LoRA-only optimizer ---")
        opt_lora = AdamW(lora_params, lr=1e-4, weight_decay=0.01) 
        
        print(f"LoRA-only optimizer group sizes: {[len(g['params']) for g in opt_lora.state_dict()['param_groups']]}")
        
        # Load saved optimizer state
        opt_path = Path(checkpoint_path) / "optimizer.pt"
        opt_sd = torch.load(opt_path, map_location="cpu", weights_only=False)
        saved_group_sizes = [len(g["params"]) for g in opt_sd["param_groups"]]
        print(f"Saved optimizer group sizes: {saved_group_sizes}")
        
        # Test which matches
        all_match = [len(g['params']) for g in opt_all.state_dict()['param_groups']] == saved_group_sizes
        lora_match = [len(g['params']) for g in opt_lora.state_dict()['param_groups']] == saved_group_sizes
        
        print(f"\n✅ All params approach matches saved: {all_match}")
        print(f"✅ LoRA-only approach matches saved: {lora_match}")
        
        # Try loading with the LoRA-only optimizer
        if lora_match:
            print("\n--- Testing LoRA-only optimizer loading ---")
            try:
                opt_lora.load_state_dict(opt_sd)
                print("✅ LoRA-only optimizer loaded successfully")
                
                # Check for nonzero states
                nonzero_count = 0
                total_states = 0
                for param in lora_params:
                    state = opt_lora.state.get(param, {})
                    if "exp_avg_sq" in state:
                        total_states += 1
                        norm = float(state["exp_avg_sq"].norm().item())
                        if norm > 1e-8:
                            nonzero_count += 1
                            
                print(f"LoRA params with nonzero exp_avg_sq: {nonzero_count}/{total_states}")
                
                return lora_match, nonzero_count > 0
                
            except Exception as e:
                print(f"❌ Failed to load LoRA-only optimizer: {e}")
                return False, False
        
        return all_match, False
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False, False

def main():
    if len(sys.argv) != 2:
        print("Usage: python optimizer_diagnostic.py <checkpoint_path>")
        sys.exit(1)
        
    checkpoint_path = sys.argv[1]
    
    print("🔬 Optimizer State Diagnostic Analysis")
    print(f"Checkpoint: {checkpoint_path}")
    
    # Run all three checks
    opt_sd = check_saved_optimizer_structure(checkpoint_path)
    info = check_training_info(checkpoint_path) 
    param_match, has_nonzero = check_parameter_matching(checkpoint_path)
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 DIAGNOSTIC SUMMARY")
    print("=" * 60)
    
    if opt_sd and len(opt_sd["state"]) == 0:
        print("🎯 ROOT CAUSE: Checkpoint saved before any optimizer step")
        print("   → This is step_0 or training never hit grad accumulation boundary")
    elif has_nonzero:
        print("🎯 ROOT CAUSE: Parameter-set mismatch on loading")
        print("   → Use LoRA-only optimizer for correct loading")
    elif param_match:
        print("🎯 ROOT CAUSE: Genuine training issue")
        print("   → RL training completed but never updated LoRA parameters")
    else:
        print("🤔 Unclear - need further investigation")
        
    print("\nRecommended next steps:")
    if has_nonzero:
        print("✅ Fix entropy probe to use LoRA-only optimizer")
    else:
        print("🔧 Investigate RL training pipeline for gradient flow issues")

if __name__ == "__main__":
    main()