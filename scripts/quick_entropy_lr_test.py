#!/usr/bin/env python3
"""
Quick Entropy vs Learning Rate Test
==================================

Simplified version for quick testing with fewer learning rates and repeats.
Good for initial validation before running the full experiment.

Usage:
    python scripts/quick_entropy_lr_test.py \\
        --checkpoint /path/to/training_state/step_60 \\
        --output-dir quick_entropy_test

This will test 3 learning rates (1e-7, 1e-6, 1e-5) with 3 repeats each.
"""

import sys
import subprocess
from pathlib import Path

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Quick entropy vs learning rate test')
    parser.add_argument('--checkpoint', required=True,
                       help='Path to checkpoint directory')
    parser.add_argument('--output-dir', default='quick_entropy_test',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Use the main script with updated test parameters
    main_script = Path(__file__).parent / 'entropy_vs_learning_rate.py'
    base_config = Path(__file__).parent.parent / 'rl_training/cfg/h100_dual_gns_64step.yaml'
    
    cmd = [
        'python', str(main_script),
        '--checkpoint', args.checkpoint,
        '--base-config', str(base_config),
        '--learning-rates', '1e-10,1e-9,1e-8,1e-7',  # Ultra-low learning rates
        '--num-repeats', '16',                        # 16 repeats per LR
        '--output-dir', args.output_dir
    ]
    
    print("🚀 Running quick entropy vs learning rate test...")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd)
    return result.returncode

if __name__ == '__main__':
    sys.exit(main())