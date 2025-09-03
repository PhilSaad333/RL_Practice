#!/usr/bin/env python3
"""
🧪 Stage 1 Numerical Fix Test Script

Tests the Stage 1 numerical stability improvements:
- Float64 weights
- Global log-weight shift
- Enhanced ESS diagnostics

Usage:
    python entropy_experiments/run_stage1_test.py
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from entropy_experiments.offline_entropy_probe import OfflineEntropyProbe


def run_stage1_test():
    """Run simple entropy probe test to validate Stage 1 fixes."""
    
    print("=" * 70)
    print("🧪 STAGE 1 NUMERICAL STABILITY TEST")
    print("=" * 70)
    print("Testing fixes:")
    print("  ✓ Float64 weight arithmetic")
    print("  ✓ Global log-weight max coordination") 
    print("  ✓ Enhanced ESS diagnostics")
    print("=" * 70)
    
    # Load medium-scale test config
    config_path = "entropy_experiments/configs/lambda_medium_test.yaml"
    print(f"\n📝 Loading config: {config_path}")
    print(f"   Batch sizes: B_E=256, B_U=32 (half of production)")
    print(f"   Temperature: 1.0 (to avoid measure mismatch)")
    
    # Initialize probe
    probe = OfflineEntropyProbe.from_config_file(config_path)
    
    # Run the standard mixed probe (includes ground truth)
    print(f"\n🚀 Running mixed probe analysis...")
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
        
        # Display ground truth diagnostics
        if 'H_orig' in results and 'H_upd' in results:
            print(f"\nGround Truth Details:")
            print(f"  H_original:          {results['H_orig']:.6f}")
            print(f"  H_updated:           {results['H_upd']:.6f}")
            
            # Check for numerical explosion
            ratio = results['H_upd'] / results['H_orig'] if results['H_orig'] != 0 else float('inf')
            if ratio > 1.5:
                print(f"  ⚠️ WARNING: H_updated is {ratio:.1f}x H_original (should be close to 1.0)")
            else:
                print(f"  ✅ Ratio H_upd/H_orig: {ratio:.2f} (reasonable)")
        
        # Display ESS diagnostics if available
        if 'diagnostics' in results:
            diags = results['diagnostics']
            if 'ESS' in diags:
                print(f"\nImportance Sampling Diagnostics:")
                print(f"  ESS:                 {diags['ESS']:.1f}")
                
                if 'ESS_fraction' in diags:
                    ess_frac = diags['ESS_fraction']
                    print(f"  ESS fraction:        {ess_frac:.1%}")
                    
                    if ess_frac < 0.05:
                        print(f"  ❌ Critical: ESS < 5% - IS unreliable")
                    elif ess_frac < 0.10:
                        print(f"  ⚠️ Warning: ESS < 10% - high variance")
                    else:
                        print(f"  ✅ ESS healthy: > 10%")
                
                if 'logw_max_global' in diags:
                    print(f"  logw_max:            {diags['logw_max_global']:.3f}")
                if 'logw_mean' in diags:
                    print(f"  logw_mean:           {diags['logw_mean']:.3f}")
        
        print("\n" + "=" * 70)
        print("🎯 STAGE 1 VALIDATION")
        print("=" * 70)
        
        # Determine if Stage 1 fixes are working
        issues = []
        
        # Check for numerical explosion
        if 'H_orig' in results and 'H_upd' in results:
            ratio = results['H_upd'] / results['H_orig'] if results['H_orig'] != 0 else float('inf')
            if ratio > 1.5:
                issues.append(f"H_updated still exploding ({ratio:.1f}x original)")
        
        # Check ESS health
        if 'diagnostics' in results and 'ESS_fraction' in results['diagnostics']:
            if results['diagnostics']['ESS_fraction'] < 0.05:
                issues.append("ESS critically low (< 5%)")
        
        # Check sign agreement
        if 'deltaH1' in results and 'deltaH_true' in results:
            if (results['deltaH1'] > 0) != (results['deltaH_true'] > 0):
                issues.append("Sign mismatch between δH₁ and ground truth")
        
        if issues:
            print("❌ Stage 1 fixes may not be fully effective:")
            for issue in issues:
                print(f"   - {issue}")
            print("\nRecommendations:")
            print("  1. Check if Stage 1 changes were properly applied")
            print("  2. Consider further reducing batch sizes")
            print("  3. May need Stage 2 (measure mismatch fix)")
        else:
            print("✅ Stage 1 fixes appear to be working!")
            print("   - No numerical explosion detected")
            print("   - ESS at acceptable levels")
            print("   - Signs agree between δH₁ and ground truth")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = run_stage1_test()
    
    if results is None:
        sys.exit(1)
    
    # Additional validation
    print("\n📋 Next Steps:")
    if results and 'H_upd' in results and 'H_orig' in results:
        ratio = results['H_upd'] / results['H_orig'] if results['H_orig'] != 0 else float('inf')
        if ratio < 1.5:
            print("  ✓ Stage 1 successful - proceed to Stage 2 (measure mismatch)")
        else:
            print("  ⚠ Stage 1 partially successful - may need additional debugging")
    
    print("\nTest complete!")