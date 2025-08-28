#!/usr/bin/env python3
"""Quick debug script to test random advantages"""

import os
import sys
import logging

# Add the root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from entropy_experiments.offline_entropy_probe import OfflineEntropyProbe

def main():
    # Enable debug logging
    logging.basicConfig(level=logging.INFO)
    
    # Set the environment variable for random advantages
    os.environ['DEBUG_RANDOM_ADVANTAGES'] = '1'
    os.environ['ENTROPY_PROBE_SINGLE_GPU'] = '1'
    
    print("🧪 Testing random advantages debug mode")
    
    # Initialize probe with minimal settings
    config = {
        'checkpoint': {
            'model_config_path': "Qwen/Qwen2.5-1.5B",
            'checkpoint_path': "/home/ubuntu/localfs/rl_training_runs/training_state/step_60"
        },
        'distributed': {
            'find_unused_parameters': False
        },
        'memory_config': {
            'microbatch_size': 2,
            'teacher_force_microbatch_size': 2,
            'amp': True,
            'dtype': 'bfloat16'
        },
        'generation': {
            'max_new_tokens': 50,
            'temperature': 1.0,
            'top_p': 1.0,
            'do_sample': True
        }
    }
    
    try:
        probe = OfflineEntropyProbe(config)
        
        # Run a very small probe test with B_E=4, B_U=4
        print("📊 Running minimal probe test...")
        results = probe.run_mixed_probe_analysis(
            B_E_global=4,
            B_U_global=4,
            G=4,
            rollout_batch_size=2,
            dataset_name="gsm8k_r1_template",
            split="train",
            mode="exact"
        )
        
        print(f"✅ Results: bars_dot = {results.get('bars_dot', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()