#!/usr/bin/env python3
"""
🚀 Quick Gradient Test Runner

Simple wrapper to run gradient isolation tests on Lambda with RL checkpoint.
"""

import sys
import os
from pathlib import Path

def find_rl_checkpoint():
    """Find RL checkpoint step_40 in localfs."""
    possible_paths = [
        "/home/ubuntu/localfs/training_runs/run_*/training_state/step_40",
        "/home/ubuntu/localfs/training_state/step_40",
        "/home/ubuntu/localfs/rl_training_runs/training_state/step_40"
    ]
    
    for pattern in possible_paths:
        import glob
        matches = glob.glob(pattern)
        if matches:
            return matches[0]  # Return first match
            
    return None

def main():
    print("🔬 Gradient Isolation Test Runner")
    print("Searching for RL checkpoint step_40...")
    
    checkpoint_path = find_rl_checkpoint()
    if checkpoint_path:
        print(f"✅ Found checkpoint: {checkpoint_path}")
    else:
        print("❌ No checkpoint found. Please specify manually:")
        print("python entropy_experiments/gradient_isolation_test.py --checkpoint /path/to/step_40")
        return 1
        
    # Run the gradient isolation test
    cmd = f"python entropy_experiments/gradient_isolation_test.py --checkpoint {checkpoint_path} --verbose"
    print(f"🚀 Running: {cmd}")
    
    os.system(cmd)
    
    print("\n📄 Check gradient_isolation_report.json for detailed results!")
    return 0

if __name__ == "__main__":
    sys.exit(main())