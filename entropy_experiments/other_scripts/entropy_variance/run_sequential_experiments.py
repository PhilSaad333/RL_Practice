#!/usr/bin/env python3
"""
Sequential Entropy Variance Experiments

Runs entropy variance experiments with B=64, 128, 256 sequentially.
This allows running all experiments unattended over ~1+ hours.

Usage on Lambda:
    cd ~/RL_Practice
    python entropy_experiments/other_scripts/entropy_variance/run_sequential_experiments.py
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

def run_experiment(B: int, G: int = 8, N: int = 16):
    """Run a single entropy variance experiment with the given parameters."""
    
    # Checkpoint path on Lambda
    adapter_path = "/home/ubuntu/localfs/training_runs/run_2025-08-24_22-13-22/training_state/step_40/model"
    
    cmd = [
        sys.executable, 
        "entropy_experiments/other_scripts/entropy_variance/entropy_variance.py",
        "--adapter", adapter_path,
        "--backbone", "Qwen/Qwen2.5-1.5B",
        "--dataset", "gsm8k_r1_template",
        "--split", "test",
        "--B", str(B),
        "--G", str(G),
        "--N", str(N),
        "--method", "repeats",
        "--same-prompts",  # Fix prompts for consistency
        "--max-new-tokens", "256",  # Allow longer generations
        "--temperature", "0.8",
        "--seed", "42",
        "--out-prefix", f"entropy_variance_B{B}"
    ]
    
    print(f"\n{'='*60}")
    print(f"STARTING EXPERIMENT: B={B}, G={G}, N={N}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        duration = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"COMPLETED EXPERIMENT: B={B}")
        print(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        # Print stdout (should contain the summary)
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            
        return True, duration
        
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"FAILED EXPERIMENT: B={B}")
        print(f"Duration: {duration:.1f} seconds")
        print(f"Return code: {e.returncode}")
        print(f"{'='*60}")
        
        print("STDOUT:")
        print(e.stdout if e.stdout else "(empty)")
        print("\nSTDERR:")
        print(e.stderr if e.stderr else "(empty)")
        
        return False, duration

def main():
    """Run sequential experiments with B=64, 128, 256."""
    
    print("Sequential Entropy Variance Experiments")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Will run experiments with B=64, 128, 256")
    
    # Experiment parameters
    experiments = [64, 128, 256]
    G = 8  # generations per prompt
    N = 16 # number of repeated runs
    
    results = []
    total_start = time.time()
    
    for B in experiments:
        success, duration = run_experiment(B, G, N)
        results.append({
            'B': B,
            'success': success, 
            'duration': duration
        })
        
        if not success:
            print(f"\nWARNING: Experiment B={B} failed, but continuing with remaining experiments...")
        
        # Brief pause between experiments
        time.sleep(10)
    
    # Summary
    total_duration = time.time() - total_start
    
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Total runtime: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    for result in results:
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        duration_min = result['duration'] / 60
        print(f"B={result['B']:3d}: {status} ({duration_min:.1f} min)")
    
    successful = sum(1 for r in results if r['success'])
    print(f"\nOverall: {successful}/{len(results)} experiments completed successfully")
    
    # List output files
    output_dir = Path("entropy_experiments/other_scripts/entropy_variance")
    json_files = list(output_dir.glob("entropy_variance_B*.json"))
    if json_files:
        print(f"\nOutput files ({len(json_files)}):")
        for f in sorted(json_files):
            print(f"  {f}")
    
    print(f"\n{'='*80}")

if __name__ == "__main__":
    main()