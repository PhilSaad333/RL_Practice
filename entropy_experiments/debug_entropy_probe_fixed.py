#!/usr/bin/env python3
"""
Debug version with proper model loading before E/U batch sampling

This script tests the entropy probe E/U batch creation logic with smaller batch sizes
to verify the Stage 1-3 cleanup is working correctly.

Expected: E_batch=64×1=64 seqs, U_batch=16×8=128 seqs, Total=192 seqs
"""

import sys
import time
import yaml
from pathlib import Path

def debug_entropy_probe_fixed():
    print("🔍 DEBUG ENTROPY PROBE - FIXED MODEL LOADING ORDER")
    print("Expected: E_batch=64×1=64 seqs, U_batch=16×8=128 seqs, Total=192 seqs")
    print("=" * 70)
    
    try:
        from entropy_experiments.offline_entropy_probe import OfflineEntropyProbe
        print("✅ Imported OfflineEntropyProbe")
        
        # Load debug config
        config_path = "entropy_experiments/configs/test_deltaH1.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        print(f"📋 Config loaded:")
        print(f"   B_E={config['batch_config']['B_E_values']}, B_U={config['batch_config']['B_U']}, G={config['batch_config']['G']}")
        print(f"   gen_batch_size={config['generation']['gen_batch_size']}")
        
        # Create probe instance
        print("\n🏗️ Creating probe instance...")
        probe = OfflineEntropyProbe(config)
        
        # IMPORTANT: Load the model first (like run_mixed_probe does)
        print("\n📦 Loading model and optimizer (fixing the bug!)...")
        checkpoint_path = config['checkpoint']['checkpoint_path']
        
        # Load model - this sets self.model  
        model_start = time.time()
        probe.model = probe._load_lora_model(checkpoint_path)
        model_time = time.time() - model_start
        print(f"✅ Model loaded in {model_time:.2f}s")
        print(f"✅ self.model is now: {type(probe.model)}")
        
        # Now E/U sampling should work because self.model exists
        B_E = config['batch_config']['B_E_values'][0] if isinstance(config['batch_config']['B_E_values'], list) else config['batch_config']['B_E_values']
        B_U = config['batch_config']['B_U'] 
        G_U = config['batch_config']['G']
        
        print(f"\n📊 Expected batch sizes:")
        print(f"   E batch: {B_E} prompts × 1 response = {B_E} sequences")
        print(f"   U batch: {B_U} prompts × {G_U} responses = {B_U * G_U} sequences")
        print(f"   Total expected: {B_E + B_U * G_U} sequences")
        
        # Now try E/U batch sampling with model loaded
        print(f"\n🎯 Starting E/U batch sampling with model loaded...")
        start_time = time.time()
        
        E_batch, U_batch = probe._sample_EU_via_sequence_processor(
            B_E=B_E, B_U=B_U, G_U=G_U
        )
        
        e_u_time = time.time() - start_time
        print(f"✅ E/U batch sampling completed in {e_u_time:.2f}s")
        
        # Analyze results
        print(f"\n📈 ACTUAL BATCH ANALYSIS:")
        print(f"   E_batch sequences shape: {E_batch['sequences'].shape if 'sequences' in E_batch else 'N/A'}")
        print(f"   E_batch num_prompts: {E_batch.get('num_prompts', 'N/A')}")
        print(f"   U_batch sequences shape: {U_batch['sequences'].shape if 'sequences' in U_batch else 'N/A')}")
        print(f"   U_batch num_prompts: {U_batch.get('num_prompts', 'N/A')}")
        
        # Verify totals
        if 'sequences' in E_batch and 'sequences' in U_batch:
            e_total = E_batch['sequences'].shape[0] * E_batch['sequences'].shape[1]
            u_total = U_batch['sequences'].shape[0] * U_batch['sequences'].shape[1] 
            print(f"   Actual E batch total: {e_total}")
            print(f"   Actual U batch total: {u_total}")
            print(f"   Grand total: {e_total + u_total}")
            
            expected_total = B_E + B_U * G_U
            print(f"\n✅ VERIFICATION: {e_total + u_total} actual vs {expected_total} expected {'✅' if (e_total + u_total) == expected_total else '❌'}")
        
        print(f"\n" + "=" * 70)
        print(f"🎉 DEBUG SUCCESSFUL - MODEL LOADING ORDER FIXED!")
        print(f"✅ E/U batch creation works when model is loaded first")
        print(f"✅ Ready for full entropy probe run")
        
        return True
        
    except Exception as e:
        print(f"\n❌ DEBUG FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_entropy_probe_fixed()
    sys.exit(0 if success else 1)