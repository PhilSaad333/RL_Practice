#!/usr/bin/env python3
"""
🧪 Test if top-p truncation is causing ESS collapse
"""

import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from entropy_experiments.offline_entropy_probe import OfflineEntropyProbe


def run_topp_test():
    """Test with top_p=0.9999 to see if it fixes ESS."""
    
    print("=" * 70)
    print("🧪 TOP-P HYPOTHESIS TEST")
    print("=" * 70)
    print("Hypothesis: top_p=0.995 causes tokens to fall outside")
    print("nucleus after update, collapsing importance weights.")
    print("Test: Use top_p=0.9999 (nearly 1.0)")
    print("=" * 70)
    
    config_path = "entropy_experiments/configs/lambda_test_p9999.yaml"
    print(f"\n📝 Loading config: {config_path}")
    print(f"   Key setting: top_p=0.9999 (was 0.995)")
    print(f"   Batch sizes: B_E=128, B_U=16")
    
    probe = OfflineEntropyProbe.from_config_file(config_path)
    
    print(f"\n🚀 Running mixed probe with top_p=0.9999...")
    start_time = time.time()
    
    try:
        results = probe.run_mixed_probe()
        elapsed = time.time() - start_time
        
        print(f"\n✅ Test completed in {elapsed:.1f} seconds")
        print("=" * 70)
        print("📊 RESULTS")
        print("=" * 70)
        
        if 'deltaH1' in results:
            print(f"δH₁ prediction:        {results['deltaH1']:.6f}")
        
        if 'deltaH_true' in results:
            print(f"Ground truth ΔH:       {results['deltaH_true']:.6f}")
        
        if 'H_orig' in results and 'H_upd' in results:
            print(f"\nEntropy values:")
            print(f"  H_original:          {results['H_orig']:.6f}")
            print(f"  H_updated:           {results['H_upd']:.6f}")
            
            ratio = results['H_upd'] / results['H_orig'] if results['H_orig'] != 0 else float('inf')
            print(f"  Ratio H_upd/H_orig:  {ratio:.2f}")
        
        # The critical metric: ESS
        if 'diagnostics' in results:
            diags = results['diagnostics']
            if 'ESS' in diags:
                print(f"\n🎯 IMPORTANCE SAMPLING DIAGNOSTICS:")
                print(f"  ESS:                 {diags['ESS']:.1f}")
                
                if 'ESS_fraction' in diags:
                    ess_frac = diags['ESS_fraction']
                    print(f"  ESS fraction:        {ess_frac:.1%}")
                    
                    # Compare with previous results
                    print(f"\n📈 COMPARISON:")
                    print(f"  With top_p=0.995:  ESS ≈ 2.6 (1.0%)")
                    print(f"  With top_p=0.9999: ESS = {diags['ESS']:.1f} ({ess_frac:.1%})")
                    
                    if ess_frac > 0.05:
                        improvement = ess_frac / 0.01  # Compare to 1% baseline
                        print(f"\n✅ SUCCESS! ESS improved {improvement:.0f}x")
                        print("The top-p truncation was indeed the problem!")
                    elif ess_frac > 0.02:
                        print(f"\n🔶 PARTIAL SUCCESS: Some improvement but still low")
                        print("Top-p helped but other issues remain")
                    else:
                        print(f"\n❌ NO IMPROVEMENT: Top-p was not the issue")
                        print("Need to investigate other causes")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = run_topp_test()
    
    if results and 'diagnostics' in results and 'ESS_fraction' in results['diagnostics']:
        ess_frac = results['diagnostics']['ESS_fraction']
        
        print("\n" + "=" * 70)
        print("📋 RECOMMENDATIONS")
        print("=" * 70)
        
        if ess_frac > 0.05:
            print("✓ Use top_p ≥ 0.9999 for all entropy experiments")
            print("✓ Consider fixing the top_p=1.0 crash for even better results")
            print("✓ The Stage 2 q-measure fix is working correctly")
        else:
            print("→ Top-p wasn't the main issue. Next steps:")
            print("  1. Try reducing learning rate (currently 1e-5)")
            print("  2. Use E_split='train' (same as U)")
            print("  3. Test with earlier checkpoint (step 0 or 10)")
            print("  4. Implement Stage 3 (eval mode)")
    
    print("\nTest complete!")