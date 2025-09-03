#!/usr/bin/env python3
"""
🧪 Stage 2 Measure Mismatch Test Script

Tests the Stage 2 fix for importance sampling:
- Uses q (sampling) measure instead of p (raw model) measure
- Accounts for top-p truncation in importance weights

Usage:
    python entropy_experiments/run_stage2_test.py
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from entropy_experiments.offline_entropy_probe import OfflineEntropyProbe


def run_stage2_test():
    """Run entropy probe test with Stage 2 q-measure fix."""
    
    print("=" * 70)
    print("🧪 STAGE 2 MEASURE MISMATCH FIX TEST")
    print("=" * 70)
    print("Testing fixes:")
    print("  ✓ Stage 1: Float64 weights + global max shift")
    print("  ✓ Stage 2: Using q (sampling) measure for IS")
    print("    - Accounts for top_p=0.995 truncation")
    print("    - Should dramatically improve ESS")
    print("=" * 70)
    
    # Load medium-scale test config
    config_path = "entropy_experiments/configs/lambda_medium_test.yaml"
    print(f"\n📝 Loading config: {config_path}")
    print(f"   Batch sizes: B_E=256, B_U=32")
    print(f"   Temperature: 1.0")
    print(f"   Top-p: 0.995 (requires q-measure!)")
    
    # Initialize probe
    probe = OfflineEntropyProbe.from_config_file(config_path)
    
    # Force the use of q measure (Stage 2 fix) - add it to the existing config
    if 'true_delta_h' in probe.config and isinstance(probe.config['true_delta_h'], dict):
        probe.config['true_delta_h']['measure'] = 'q'
    else:
        # If true_delta_h doesn't exist or isn't a dict, skip (it's already configured in yaml)
        pass
    print(f"\n✅ Configured to use q-measure for importance sampling")
    
    # Run the standard mixed probe
    print(f"\n🚀 Running mixed probe analysis with Stage 2 fix...")
    start_time = time.time()
    
    try:
        results = probe.run_mixed_probe()
        elapsed = time.time() - start_time
        
        print(f"\n✅ Test completed in {elapsed:.1f} seconds")
        print("=" * 70)
        print("📊 RESULTS")
        print("=" * 70)
        
        # Display key metrics
        if 'deltaH1' in results:
            print(f"δH₁ prediction:        {results['deltaH1']:.6f}")
        
        if 'deltaH_true' in results:
            print(f"Ground truth ΔH:       {results['deltaH_true']:.6f}")
            
            # Check sign agreement
            if 'deltaH1' in results:
                sign_delta_h1 = "negative" if results['deltaH1'] < 0 else "positive"
                sign_delta_true = "negative" if results['deltaH_true'] < 0 else "positive"
                
                if sign_delta_h1 == sign_delta_true:
                    print(f"✅ Sign agreement:     Both {sign_delta_h1}")
                else:
                    print(f"❌ Sign mismatch:      δH₁ is {sign_delta_h1}, ground truth is {sign_delta_true}")
        
        # Display ground truth details
        if 'H_orig' in results and 'H_upd' in results:
            print(f"\nGround Truth Details:")
            print(f"  H_original:          {results['H_orig']:.6f}")
            print(f"  H_updated:           {results['H_upd']:.6f}")
            
            # Check for numerical explosion
            ratio = results['H_upd'] / results['H_orig'] if results['H_orig'] != 0 else float('inf')
            if ratio > 1.5:
                print(f"  ⚠️ WARNING: H_updated is {ratio:.1f}x H_original")
            else:
                print(f"  ✅ Ratio H_upd/H_orig: {ratio:.2f} (reasonable)")
        
        # Display ESS diagnostics
        if 'diagnostics' in results:
            diags = results['diagnostics']
            if 'ESS' in diags:
                print(f"\nImportance Sampling Diagnostics:")
                print(f"  ESS:                 {diags['ESS']:.1f}")
                
                if 'ESS_fraction' in diags:
                    ess_frac = diags['ESS_fraction']
                    print(f"  ESS fraction:        {ess_frac:.1%}")
                    
                    # Stage 2 should significantly improve ESS
                    if ess_frac < 0.01:
                        print(f"  ❌ Critical: ESS < 1% even with q-measure")
                    elif ess_frac < 0.05:
                        print(f"  ⚠️ Warning: ESS < 5% (Stage 2 helped but still low)")
                    elif ess_frac < 0.10:
                        print(f"  🔶 Acceptable: ESS between 5-10%")
                    else:
                        print(f"  ✅ Healthy: ESS > 10% (Stage 2 worked!)")
                
                if 'logw_max_global' in diags:
                    print(f"  logw_max:            {diags['logw_max_global']:.3f}")
                if 'logw_mean' in diags:
                    print(f"  logw_mean:           {diags['logw_mean']:.3f}")
        
        print("\n" + "=" * 70)
        print("🎯 STAGE 2 VALIDATION")
        print("=" * 70)
        
        # Compare with Stage 1 results
        print("\nComparison with Stage 1 only:")
        print("  Stage 1 ESS: ~2.04 (< 1%)")
        if 'diagnostics' in results and 'ESS' in results['diagnostics']:
            new_ess = results['diagnostics']['ESS']
            print(f"  Stage 2 ESS: {new_ess:.1f}", end="")
            if 'ESS_fraction' in results['diagnostics']:
                print(f" ({results['diagnostics']['ESS_fraction']:.1%})")
            else:
                print()
            
            # Calculate improvement
            old_ess = 2.04  # From Stage 1 test
            improvement = new_ess / old_ess
            print(f"  Improvement: {improvement:.1f}x")
            
            if improvement > 5:
                print("\n✅ Stage 2 fix is highly effective!")
            elif improvement > 2:
                print("\n🔶 Stage 2 fix helped but may need Stage 3/4")
            else:
                print("\n❌ Stage 2 fix had minimal impact - investigate further")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = run_stage2_test()
    
    if results is None:
        sys.exit(1)
    
    # Recommendations
    print("\n📋 Next Steps:")
    if results and 'diagnostics' in results:
        if 'ESS_fraction' in results['diagnostics']:
            ess_frac = results['diagnostics']['ESS_fraction']
            if ess_frac > 0.10:
                print("  ✓ Stage 2 successful - ESS is healthy")
                print("  → Consider testing with production batch sizes")
            elif ess_frac > 0.05:
                print("  ✓ Stage 2 improved ESS significantly")
                print("  → Proceed to Stage 3 (eval mode) for additional stability")
            else:
                print("  ⚠ ESS still critically low")
                print("  → May need Stage 4 (no-IS fallback) for this configuration")
    
    print("\nTest complete!")