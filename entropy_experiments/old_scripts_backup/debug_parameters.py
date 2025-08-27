#!/usr/bin/env python3
"""
Debug script to investigate the 392 trainable parameters issue.
This should help identify what's wrong with LoRA loading.
"""

import torch
import yaml
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from entropy_experiments import OfflineEntropyProbe

def analyze_model_parameters(model, checkpoint_path):
    """Comprehensive parameter analysis for debugging LoRA loading."""
    print("=" * 80)
    print("🔍 COMPREHENSIVE PARAMETER ANALYSIS")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print()
    
    # 1. Basic counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 PARAMETER COUNTS:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {total_params - trainable_params:,}")
    print(f"  Trainable ratio: {trainable_params/total_params:.2%}")
    print()
    
    # 2. Training mode status
    print(f"🎯 MODEL STATUS:")
    print(f"  Model in training mode: {model.training}")
    print(f"  Model type: {type(model).__name__}")
    print()
    
    # 3. Trainable parameter details
    print(f"🔧 TRAINABLE PARAMETERS (showing first 20):")
    trainable_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_count += 1
            if trainable_count <= 20:
                print(f"  {name}: {param.shape} ({param.numel():,} params)")
            elif trainable_count == 21:
                print(f"  ... (showing first 20 of {sum(1 for n, p in model.named_parameters() if p.requires_grad)} total trainable)")
    print()
    
    # 4. Check for LoRA-specific patterns
    print(f"🎨 LORA ANALYSIS:")
    lora_patterns = ['lora_A', 'lora_B', 'adapter', 'q_proj', 'k_proj', 'v_proj', 'o_proj']
    
    for pattern in lora_patterns:
        matching_params = [name for name, param in model.named_parameters() 
                          if pattern.lower() in name.lower()]
        trainable_matching = [name for name, param in model.named_parameters() 
                            if pattern.lower() in name.lower() and param.requires_grad]
        
        if matching_params:
            print(f"  {pattern} parameters: {len(matching_params)} total, {len(trainable_matching)} trainable")
            if trainable_matching:
                for name in trainable_matching[:3]:  # Show first 3
                    param = dict(model.named_parameters())[name]
                    print(f"    ✓ {name}: {param.shape}")
        else:
            print(f"  {pattern} parameters: None found ❌")
    print()
    
    # 5. All parameter names (grouped by layer type)
    print(f"🏗️  PARAMETER STRUCTURE:")
    all_params = list(model.named_parameters())
    layer_types = {}
    
    for name, param in all_params:
        # Extract layer type from parameter name
        if '.' in name:
            layer_type = name.split('.')[0] if not name.startswith('model.') else '.'.join(name.split('.')[:2])
        else:
            layer_type = 'root'
            
        if layer_type not in layer_types:
            layer_types[layer_type] = {'total': 0, 'trainable': 0, 'names': []}
        
        layer_types[layer_type]['total'] += param.numel()
        if param.requires_grad:
            layer_types[layer_type]['trainable'] += param.numel()
        layer_types[layer_type]['names'].append(name)
    
    for layer_type, info in sorted(layer_types.items()):
        trainable_pct = info['trainable'] / info['total'] * 100 if info['total'] > 0 else 0
        print(f"  {layer_type}:")
        print(f"    Total: {info['total']:,} params")
        print(f"    Trainable: {info['trainable']:,} params ({trainable_pct:.1f}%)")
        print(f"    Parameter names: {len(info['names'])} parameters")
    
    print()
    
    # 6. Check if model has PEFT/LoRA attributes
    print(f"🔌 ADAPTER STATUS:")
    peft_attributes = ['peft_config', 'base_model', 'active_adapters', 'adapter_config']
    for attr in peft_attributes:
        if hasattr(model, attr):
            value = getattr(model, attr)
            print(f"  {attr}: {type(value)} - {value}")
        else:
            print(f"  {attr}: Not found ❌")
    
    print("=" * 80)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'training_mode': model.training,
        'layer_types': layer_types
    }

def main():
    """Main debugging function."""
    print("🔍 Starting parameter count debugging...")
    
    # Load the same config used in the entropy probe
    config_path = Path(__file__).parent / "configs" / "probe_config_exact_128_optimized.yaml"
    print(f"Loading config from: {config_path}")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    checkpoint_path = config['checkpoint']['checkpoint_path']
    print(f"Checkpoint path: {checkpoint_path}")
    
    # Initialize the probe and load the checkpoint
    print("Initializing OfflineEntropyProbe...")
    probe = OfflineEntropyProbe(config)
    
    print("Loading checkpoint...")
    probe.load_checkpoint(checkpoint_path)
    
    if probe.model is None:
        print("❌ ERROR: Model failed to load!")
        return
    
    # Analyze the loaded model
    results = analyze_model_parameters(probe.model, checkpoint_path)
    
    # Expected vs actual analysis
    print("🎯 EXPECTED vs ACTUAL:")
    print(f"  Expected LoRA r=64 params: ~10,000-50,000 (typical range)")
    print(f"  Actual trainable params: {results['trainable_params']:,}")
    
    if results['trainable_params'] < 1000:
        print(f"  ❌ PROBLEM: Trainable parameter count is suspiciously low!")
        print(f"  🔧 This suggests LoRA adapter is not properly loaded or configured")
    else:
        print(f"  ✅ Parameter count looks reasonable")
    
    print("\n🚀 Debug analysis complete!")

if __name__ == "__main__":
    main()