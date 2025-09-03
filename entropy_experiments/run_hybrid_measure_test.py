#!/usr/bin/env python3
"""
🧪 Hybrid Measure Test
Generate with top_p=0.995 but compute IS weights with p-measure (full distribution)
"""

import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from entropy_experiments.offline_entropy_probe import OfflineEntropyProbe


def run_hybrid_test():
    """Test hybrid approach: generate with top-p, compute IS with full p."""
    
    print("=" * 70)
    print("🧪 HYBRID MEASURE TEST")
    print("=" * 70)
    print("Strategy: Generate sequences with top_p=0.995")
    print("         But compute IS weights with p-measure (no truncation)")
    print("This avoids -inf logprobs while keeping generation quality")
    print("=" * 70)
    
    # Use the original config but force p-measure for IS
    config_path = "entropy_experiments/configs/lambda_medium_test.yaml"
    probe = OfflineEntropyProbe.from_config_file(config_path)
    
    # Override to use p-measure for IS computation
    if 'true_delta_h' in probe.config:
        probe.config['true_delta_h']['measure'] = 'p'  # Force p-measure
    
    print(f"\n📝 Configuration:")
    print(f"   Generation top_p: {probe.config['generation']['top_p']}")
    print(f"   IS measure: p (full distribution)")
    print(f"   This decouples generation quality from IS stability")
    
    print(f"\n🚀 Running mixed probe with hybrid measure...")
    start_time = time.time()
    
    try:
        results = probe.run_mixed_probe()
        elapsed = time.time() - start_time
        
        print(f"\n✅ Test completed in {elapsed:.1f} seconds")
        print("=" * 70)
        print("📊 RESULTS")
        print("=" * 70)
        
        if 'H_orig' in results and 'H_upd' in results:
            print(f"H_original:            {results['H_orig']:.6f}")
            print(f"H_updated:             {results['H_upd']:.6f}")
            
            if 'deltaH_true' in results:
                print(f"ΔH_true:              {results['deltaH_true']:.6f}")
            
            if 'deltaH1' in results:
                print(f"δH₁:                  {results['deltaH1']:.6f}")
        
        # The critical metric: ESS
        if 'diagnostics' in results and 'ESS' in results['diagnostics']:
            ess = results['diagnostics']['ESS']
            ess_frac = results['diagnostics'].get('ESS_fraction', ess/256)
            
            print(f"\n🎯 IMPORTANCE SAMPLING:")
            print(f"  ESS:                 {ess:.1f}")
            print(f"  ESS fraction:        {ess_frac:.1%}")
            
            print(f"\n📈 COMPARISON:")
            print(f"  top_p=0.995,  q-measure: ESS ≈ 2.6 (1.0%)")
            print(f"  top_p=0.9999, q-measure: ESS ≈ 6.5 (5.1%)")
            print(f"  top_p=0.995,  p-measure: ESS = {ess:.1f} ({ess_frac:.1%})")
            
            if ess_frac > 0.10:
                print(f"\n✅ SUCCESS! Using p-measure fixed the issue!")
                print("The q-measure (top-p truncation) was causing the problem")
            elif ess_frac > 0.05:
                print(f"\n🔶 IMPROVEMENT: Better but still not ideal")
            else:
                print(f"\n❌ Still problematic - the issue is deeper")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TEST 1: HYBRID MEASURE (top_p for generation, p for IS)")
    print("=" * 70)
    
    results_hybrid = run_hybrid_test()
    
    print("\n" + "=" * 70)
    print("TEST 2: FULL DISTRIBUTION (top_p=1.0)")
    print("=" * 70)
    
    # Now test with top_p=1.0
    print("\nTrying top_p=1.0 to see if it crashes...")
    
    from entropy_experiments.offline_entropy_probe import OfflineEntropyProbe
    config_path = "entropy_experiments/configs/lambda_test_p1.yaml"
    
    try:
        probe = OfflineEntropyProbe.from_config_file(config_path)
        print("Starting probe with top_p=1.0...")
        results_p1 = probe.run_mixed_probe()
        
        if results_p1 and 'diagnostics' in results_p1:
            ess = results_p1['diagnostics'].get('ESS', 0)
            ess_frac = results_p1['diagnostics'].get('ESS_fraction', 0)
            print(f"\n✅ top_p=1.0 worked! ESS = {ess:.1f} ({ess_frac:.1%})")
        
    except Exception as e:
        print(f"\n❌ top_p=1.0 crashed with: {e}")
        print("\nThis confirms we need the hybrid approach or to fix the crash")
    
    print("\n" + "=" * 70)
    print("📋 RECOMMENDATION")
    print("=" * 70)
    
    if results_hybrid and 'diagnostics' in results_hybrid:
        ess_frac = results_hybrid['diagnostics'].get('ESS_fraction', 0)
        if ess_frac > 0.10:
            print("✓ Use hybrid approach: generate with top_p, compute IS with p")
            print("✓ This gives both quality generation and stable IS")
        else:
            print("→ Even p-measure has low ESS. The issue is likely:")
            print("  1. Optimizer step too large")
            print("  2. Train/test distribution mismatch")
            print("  3. Need Stage 3 (eval mode) to reduce variance")
    
    print("\nTests complete!")