#!/usr/bin/env python3
"""
🚀 Lambda H100 Entropy Probe Test Script

Simple test script for running entropy probe on Lambda H100 with the optimized config.
This script uses the new streamlined config and checkpoint from CLAUDE.md.

Usage:
    python entropy_experiments/run_lambda_h100_test.py

Prerequisites:
    - Lambda H100 instance with RL_Practice repo pulled  
    - RL environment activated: conda activate rl
    - RL-trained checkpoint at localfs/rl_training_runs/training_state/step_60/
"""

import sys
import os
import time
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def check_environment():
    """Check that we're in the right environment with required resources."""
    print("🔍 Checking Lambda H100 environment...")
    
    # Check GPU
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA not available - ensure you're on a GPU instance")
    
    gpu_name = torch.cuda.get_device_name(0)
    print(f"✅ GPU detected: {gpu_name}")
    
    if "H100" not in gpu_name:
        print(f"⚠️  GPU is {gpu_name}, expected H100 - continuing anyway")
    
    # Check checkpoint paths  
    checkpoint_path = "/lambda/nfs/localfs/training_runs/run_2025-09-03_03-59-57/training_state/step_40/model"
    optimizer_path = "/lambda/nfs/localfs/training_runs/run_2025-09-03_03-59-57/training_state/step_40/optimizer.pt"
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"❌ Checkpoint not found: {checkpoint_path}")
    
    if not os.path.exists(optimizer_path):
        raise FileNotFoundError(f"❌ Optimizer not found: {optimizer_path}")
    
    print(f"✅ Checkpoint found: {checkpoint_path}")
    print(f"✅ Optimizer found: {optimizer_path}")
    
    # Check conda environment
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'unknown')
    print(f"📦 Conda environment: {conda_env}")
    
    if conda_env != 'rl':
        print("⚠️  Expected 'rl' conda environment, please run: conda activate rl")
    
    return True

def run_entropy_probe():
    """Run the entropy probe with optimized config."""
    print("\n🧪 Starting Entropy Probe Analysis...")
    
    try:
        from entropy_experiments.offline_entropy_probe import OfflineEntropyProbe
        
        # Load optimized config
        config_path = "entropy_experiments/configs/lambda_h100_optimized.yaml"
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"❌ Config not found: {config_path}")
        
        print(f"📝 Loading config: {config_path}")
        probe = OfflineEntropyProbe.from_config_file(config_path)
        
        # Run the probe
        print("🚀 Running mixed probe analysis...")
        start_time = time.time()
        
        results = probe.run_mixed_probe()
        
        elapsed_time = time.time() - start_time
        
        # Display results
        print(f"\n🎯 ENTROPY PROBE RESULTS (completed in {elapsed_time:.1f}s)")
        print("=" * 60)
        
        if 'deltaH1' in results:
            print(f"δH₁ prediction:     {results['deltaH1']:.8f}")
            print(f"bars_dot:           {results['bars_dot']:.8f}")  
            print(f"Learning rate:      {results['learning_rate']:.2e}")
        
        if 'deltaH_true' in results:
            print(f"Ground truth ΔH:    {results['deltaH_true']:.8f}")
            
            if 'deltaH1' in results:
                error = abs(results['deltaH1'] - results['deltaH_true'])
                relative_error = error / abs(results['deltaH_true']) * 100
                print(f"Prediction error:   {error:.8f} ({relative_error:.2f}%)")
        
        print(f"Batch sizes:        B_E={results.get('B_E', 'N/A')}, B_U={results.get('B_U', 'N/A')}")
        
        # Timing breakdown
        if 'timing' in results:
            timing = results['timing']
            print(f"\n⏱️  TIMING BREAKDOWN:")
            for phase, time_val in timing.items():
                if isinstance(time_val, (int, float)) and time_val > 0:
                    print(f"  {phase}: {time_val:.2f}s")
        
        # Ground truth diagnostics
        if 'H_orig' in results and 'H_upd' in results:
            print(f"\n📊 GROUND TRUTH DIAGNOSTICS:")
            print(f"  H_original:       {results['H_orig']:.6f}")
            print(f"  H_updated:        {results['H_upd']:.6f}")
            if 'ESS' in results:
                print(f"  ESS:              {results['ESS']:.2f}")
            if 'w_max' in results and 'w_min' in results:
                print(f"  Weight range:     [{results['w_min']:.4f}, {results['w_max']:.4f}]")
        
        print("\n🎉 Entropy probe test completed successfully!")
        
        # Check if detailed logs were generated
        log_dir = "entropy_experiments/logs"
        if os.path.exists(log_dir):
            from datetime import datetime
            today = datetime.now().strftime("%Y-%m-%d")
            today_logs = Path(log_dir) / today
            if today_logs.exists():
                log_files = list(today_logs.glob("*.json*"))
                if log_files:
                    print(f"📋 Detailed logs saved: {len(log_files)} files in {today_logs}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ ERROR during entropy probe execution:")
        print(f"   {type(e).__name__}: {e}")
        print(f"\n🔧 TROUBLESHOOTING:")
        print("   1. Ensure you're in the RL_Practice directory")  
        print("   2. Check that conda environment 'rl' is activated")
        print("   3. Verify checkpoint exists: /lambda/nfs/localfs/training_runs/run_2025-09-03_03-59-57/training_state/step_40/")
        print("   4. Check CUDA/GPU availability")
        raise

def main():
    """Main execution."""
    print("🚀 Lambda H100 Entropy Probe Test")
    print("=" * 50)
    
    try:
        # Environment checks
        check_environment()
        
        # Run probe
        results = run_entropy_probe()
        
        print(f"\n✅ ALL TESTS PASSED - Ready for production runs!")
        
        return 0
        
    except Exception as e:
        print(f"\n💥 Test failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)