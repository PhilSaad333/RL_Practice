#!/usr/bin/env python3
"""
Minimal ESS debugging - isolate the core problem
"""

import torch
import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def main():
    """Test importance sampling with minimal setup."""
    
    print("=" * 80)
    print("MINIMAL ESS TEST - Isolating the core issue")
    print("=" * 80)
    
    # 1. Create toy data to test IS mechanics
    print("\n1️⃣ Testing IS with controlled data...")
    
    # Simulate logprobs before and after a small update
    n_samples = 100
    
    # Case A: Small uniform change (should work well)
    print("\nCase A: Small uniform change")
    S_orig = torch.randn(n_samples) * 10 - 50  # Random logprobs around -50
    S_upd = S_orig + 0.1  # Small uniform increase
    
    logw = S_upd - S_orig
    w = torch.exp(logw - logw.max())
    ESS = (w.sum() ** 2) / (w * w).sum()
    print(f"  ESS = {ESS:.1f} / {n_samples} = {ESS/n_samples:.1%}")
    
    # Case B: Large uniform change (should still work)
    print("\nCase B: Large uniform change")
    S_upd = S_orig + 2.0  # Larger but uniform
    
    logw = S_upd - S_orig
    w = torch.exp(logw - logw.max())
    ESS = (w.sum() ** 2) / (w * w).sum()
    print(f"  ESS = {ESS:.1f} / {n_samples} = {ESS/n_samples:.1%}")
    
    # Case C: Non-uniform change (problematic)
    print("\nCase C: Non-uniform change (like your case)")
    S_upd = S_orig.clone()
    # Make a few sequences much more likely
    S_upd[:5] += 10.0  # 5 sequences become much more likely
    S_upd[5:] -= 1.0   # Rest become less likely
    
    logw = S_upd - S_orig
    w = torch.exp(logw - logw.max())
    ESS = (w.sum() ** 2) / (w * w).sum()
    print(f"  ESS = {ESS:.1f} / {n_samples} = {ESS/n_samples:.1%}")
    print(f"  Top 5 weights account for {w[:5].sum() / w.sum():.1%} of mass")
    
    # Case D: Extreme non-uniform (catastrophic, like your data)
    print("\nCase D: Extreme non-uniform (catastrophic)")
    S_upd = S_orig.clone()
    S_upd[:3] += 20.0  # 3 sequences become MUCH more likely
    S_upd[3:] -= 5.0   # Rest become much less likely
    
    logw = S_upd - S_orig
    w = torch.exp(logw - logw.max())
    ESS = (w.sum() ** 2) / (w * w).sum()
    print(f"  ESS = {ESS:.1f} / {n_samples} = {ESS/n_samples:.1%}")
    print(f"  Top 3 weights account for {w[:3].sum() / w.sum():.1%} of mass")
    
    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    print("\nYour ESS of ~1% matches Case D: a few sequences have")
    print("drastically increased probability after the update.")
    print("\nPossible causes:")
    print("1. The RL update strongly favors certain patterns in test data")
    print("2. The optimizer step size is too large") 
    print("3. Train/test distribution mismatch amplified by RL")
    
    # 2. Check actual logprob changes
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    print("\n1. Reduce learning rate by 10x or 100x")
    print("2. Use E_split='train' (same as U_split)")
    print("3. Try with base model (step 0) instead of step 40")
    print("4. Implement Stage 3: eval mode to reduce variance")
    print("5. As last resort, use Stage 4: no-IS fallback")
    
    return True

if __name__ == "__main__":
    main()