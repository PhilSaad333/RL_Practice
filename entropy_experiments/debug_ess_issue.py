#!/usr/bin/env python3
"""
🔍 ESS Catastrophe Debugging Script

Systematically investigates why ESS is catastrophically low (~1%) even with Stage 2 fixes.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from entropy_experiments.offline_entropy_probe import OfflineEntropyProbe


def debug_ess_issue():
    """Run comprehensive ESS debugging."""
    
    print("=" * 80)
    print("🔍 ESS CATASTROPHE DEBUGGING")
    print("=" * 80)
    
    # Use tiny config for faster debugging
    config_path = "entropy_experiments/configs/lambda_tiny_test.yaml"
    probe = OfflineEntropyProbe.from_config_file(config_path)
    
    print("\n1️⃣ Checking configuration...")
    print(f"   Temperature: {probe.config['generation']['temperature']}")
    print(f"   Top-p: {probe.config['generation']['top_p']}")
    if 'true_delta_h' in probe.config:
        print(f"   Measure: {probe.config['true_delta_h'].get('measure', 'not set')}")
    
    # Initialize components
    probe._initialize_components()
    
    print("\n2️⃣ Sampling small E batch for debugging...")
    B_E = 8  # Very small for debugging
    E_batch = probe._sample_E_batch_via_datasets(B_E)
    
    print(f"   E batch shape: sequences={E_batch['sequences'].shape}")
    
    # Create a minimal U batch for the optimizer step
    print("\n3️⃣ Creating minimal U batch...")
    B_U = 2
    G = 2
    U_batch = probe._sample_U_batch_via_datasets(B_U, G)
    
    print("\n4️⃣ Computing logprobs BEFORE optimizer step...")
    
    # Get original logprobs (both p and q measures)
    if probe.delta_entropy_is and probe.delta_entropy_is.sequence_processor:
        sp = probe.delta_entropy_is.sequence_processor
        
        # Convert E_batch to BatchedSequences
        from sequence_processing.sequence_processor import BatchedSequences
        bs = BatchedSequences(
            sequences=E_batch['sequences'],
            prompt_lens=E_batch['prompt_lens'],
            gen_lens=E_batch['gen_lens'],
            attention_masks=E_batch['attention_masks'],
            responses_text=[[f"response_{b}_{g}" for g in range(E_batch['sequences'].shape[1])] 
                           for b in range(E_batch['sequences'].shape[0])]
        )
        
        # Get logprobs with diagnostics
        lp_orig, _ = sp.teacher_force_logprobs_with_diagnostics(
            bs, with_grad=False, tf_batch_size=4, compute_rb=True
        )
        
        print("\n   📊 Original model logprobs:")
        if lp_orig.sequence_logprobs:
            p_values = [val for sublist in lp_orig.sequence_logprobs for val in sublist]
            print(f"      sequence_logprobs (p): min={min(p_values):.2f}, max={max(p_values):.2f}, mean={np.mean(p_values):.2f}")
        
        if lp_orig.sequence_logqs:
            q_values = [val for sublist in lp_orig.sequence_logqs for val in sublist]
            print(f"      sequence_logqs (q):    min={min(q_values):.2f}, max={max(q_values):.2f}, mean={np.mean(q_values):.2f}")
            
            # Check if they're different
            if p_values and q_values:
                diff = np.array(q_values) - np.array(p_values)
                print(f"      q - p difference:      min={diff.min():.2f}, max={diff.max():.2f}, mean={diff.mean():.2f}")
                if np.allclose(p_values, q_values):
                    print("      ⚠️ WARNING: q and p are identical! Stage 2 not working!")
                else:
                    print("      ✅ q and p are different (Stage 2 is active)")
        else:
            print("      ❌ ERROR: sequence_logqs is None! Stage 2 not working!")
    
    print("\n5️⃣ Taking optimizer step...")
    
    # Store original model state for comparison
    orig_params = {name: param.clone() for name, param in probe.model.named_parameters() if param.requires_grad}
    
    # Take the optimizer step (minimal)
    cfg_importance = probe.config.get('true_delta_h', {})
    probe.delta_entropy_is._rl_update_streaming(
        U_batch, probe.optimizer, 
        rl_grad_accum=cfg_importance.get('rl_grad_accum', 1),
        importance_mb_size=cfg_importance.get('importance_microbatch_size', 1)
    )
    
    # Check how much parameters changed
    param_changes = []
    for name, param in probe.model.named_parameters():
        if param.requires_grad and name in orig_params:
            change = (param - orig_params[name]).abs().mean().item()
            param_changes.append(change)
    
    if param_changes:
        print(f"   Parameter changes: min={min(param_changes):.2e}, max={max(param_changes):.2e}, mean={np.mean(param_changes):.2e}")
    
    print("\n6️⃣ Computing logprobs AFTER optimizer step...")
    
    if probe.delta_entropy_is and probe.delta_entropy_is.sequence_processor:
        lp_upd, _ = sp.teacher_force_logprobs_with_diagnostics(
            bs, with_grad=False, tf_batch_size=4, compute_rb=True
        )
        
        print("\n   📊 Updated model logprobs:")
        if lp_upd.sequence_logprobs:
            p_values_upd = [val for sublist in lp_upd.sequence_logprobs for val in sublist]
            print(f"      sequence_logprobs (p): min={min(p_values_upd):.2f}, max={max(p_values_upd):.2f}, mean={np.mean(p_values_upd):.2f}")
        
        if lp_upd.sequence_logqs:
            q_values_upd = [val for sublist in lp_upd.sequence_logqs for val in sublist]
            print(f"      sequence_logqs (q):    min={min(q_values_upd):.2f}, max={max(q_values_upd):.2f}, mean={np.mean(q_values_upd):.2f}")
    
    print("\n7️⃣ Computing importance weights...")
    
    # Check what measure is being used
    use_q = cfg_importance.get('measure', 'p') == 'q' or sp.config.top_p < 1.0
    print(f"   Using measure: {'q' if use_q else 'p'}")
    
    if use_q and lp_orig.sequence_logqs and lp_upd.sequence_logqs:
        S_orig = torch.tensor([val for sublist in lp_orig.sequence_logqs for val in sublist])
        S_upd = torch.tensor([val for sublist in lp_upd.sequence_logqs for val in sublist])
        print("   ✅ Using q-measure for weights")
    else:
        S_orig = torch.tensor([val for sublist in lp_orig.sequence_logprobs for val in sublist])
        S_upd = torch.tensor([val for sublist in lp_upd.sequence_logprobs for val in sublist])
        print("   ⚠️ Using p-measure for weights")
    
    logw = S_upd - S_orig
    print(f"\n   📊 Log-weights (S_upd - S_orig):")
    print(f"      min={logw.min():.2f}, max={logw.max():.2f}, mean={logw.mean():.2f}, std={logw.std():.2f}")
    
    # Compute ESS
    logw_shifted = logw - logw.max()
    w = torch.exp(logw_shifted.to(torch.float64))
    w_sum = w.sum()
    w2_sum = (w * w).sum()
    ESS = (w_sum ** 2) / w2_sum if w2_sum > 0 else 0
    ESS_frac = ESS / len(logw)
    
    print(f"\n   📊 Importance Sampling Diagnostics:")
    print(f"      ESS: {ESS:.2f}")
    print(f"      ESS fraction: {ESS_frac:.1%}")
    print(f"      Max weight: {w.max() / w_sum:.1%} of total")
    
    # Show weight distribution
    normalized_w = w / w_sum
    top5_weights = normalized_w.sort(descending=True).values[:5]
    print(f"      Top 5 weights: {top5_weights.tolist()}")
    
    print("\n" + "=" * 80)
    print("🎯 DIAGNOSIS")
    print("=" * 80)
    
    if ESS_frac < 0.01:
        print("❌ CRITICAL: ESS < 1% - Importance sampling is completely broken")
        
        if not use_q:
            print("\n🔧 FIX: Not using q-measure! Check:")
            print("   1. Config has importance.measure = 'q'")
            print("   2. sequence_logqs is being computed correctly")
        elif logw.std() > 10:
            print("\n🔧 FIX: Extreme weight variance! Check:")
            print("   1. Optimizer step might be too large")
            print("   2. Model might be in wrong state")
            print("   3. Generation config might not match evaluation")
        else:
            print("\n🔧 FIX: Weights are degenerate but not extreme. Check:")
            print("   1. Top-p implementation in _compute_logq_top_p")
            print("   2. Whether sequences were generated with same model")
    elif ESS_frac < 0.05:
        print("⚠️ WARNING: ESS < 5% - High variance but might work")
    else:
        print("✅ ESS is healthy!")
    
    return {
        'ESS': ESS,
        'ESS_fraction': ESS_frac,
        'logw_stats': {
            'min': logw.min().item(),
            'max': logw.max().item(),
            'mean': logw.mean().item(),
            'std': logw.std().item()
        }
    }


if __name__ == "__main__":
    results = debug_ess_issue()
    print("\n🏁 Debug complete!")