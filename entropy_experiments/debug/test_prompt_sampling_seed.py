"""
Test script to demonstrate the configurable prompt sampling seed behavior.

This shows how the new prompt_sampling_seed config parameter controls
whether you get the same prompts each run or different ones.
"""

import numpy as np

def test_seed_behavior():
    """Demonstrate the difference between seeded and unseeded sampling."""
    print("=" * 60)
    print("PROMPT SAMPLING SEED BEHAVIOR TEST")
    print("=" * 60)
    
    # Simulate dataset size
    N = 1000  # Dataset size
    B_E = 10  # Number of prompts to sample
    
    print(f"Dataset size: {N}")
    print(f"Sampling {B_E} prompts\n")
    
    # Test 1: Fixed seed (reproducible)
    print("TEST 1: Fixed seed (prompt_sampling_seed: 42)")
    print("Running 3 times with seed=42:")
    
    for run in range(3):
        rng = np.random.default_rng(42)  # Fixed seed
        idx = rng.integers(low=0, high=N, size=B_E, endpoint=False)
        print(f"  Run {run+1}: {idx.tolist()}")
    
    print("→ Same prompts every time (good for reproducible experiments)\n")
    
    # Test 2: No seed (random)
    print("TEST 2: No seed (prompt_sampling_seed: null)")
    print("Running 3 times with seed=None:")
    
    for run in range(3):
        rng = np.random.default_rng(None)  # No seed = random
        idx = rng.integers(low=0, high=N, size=B_E, endpoint=False)
        print(f"  Run {run+1}: {idx.tolist()}")
    
    print("→ Different prompts every time (good for diverse data collection)\n")
    
    # Test 3: Config simulation
    print("TEST 3: Config-based behavior")
    
    configs = [
        {"probe_rework": {"prompt_sampling_seed": 42}},
        {"probe_rework": {"prompt_sampling_seed": None}},
        {"probe_rework": {}},  # No config
    ]
    
    for i, config in enumerate(configs):
        prompt_seed = config.get('probe_rework', {}).get('prompt_sampling_seed', None)
        rng = np.random.default_rng(prompt_seed)
        idx = rng.integers(low=0, high=N, size=5, endpoint=False)
        
        seed_desc = "random" if prompt_seed is None else f"fixed ({prompt_seed})"
        print(f"  Config {i+1}: prompt_sampling_seed={prompt_seed} → {seed_desc}")
        print(f"    Sample: {idx.tolist()}")
    
    print("\n" + "=" * 60)
    print("CONFIGURATION GUIDE")
    print("=" * 60)
    print("For your entropy studies, use:")
    print()
    print("# Random prompts each run (recommended for data collection)")
    print("probe_rework:")
    print("  prompt_sampling_seed: null")
    print()
    print("# Fixed prompts for reproducibility (debugging/comparison)")
    print("probe_rework:")
    print("  prompt_sampling_seed: 42")
    print()
    print("# Default behavior (random if not specified)")
    print("probe_rework:")
    print("  # prompt_sampling_seed not specified → random")

if __name__ == "__main__":
    test_seed_behavior()