#!/usr/bin/env python3
"""
Script to run entropy experiments using the EntropyMeasurements class.
"""

import yaml
import json
from pathlib import Path
from entropy_experiment_runner import EntropyMeasurements

def main():
    # Load config
    config_path = Path("entropy_experiments/configs/config_template.yaml")
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    print(f"Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create entropy measurements instance
    print("Initializing entropy measurements...")
    entropy_probe = EntropyMeasurements(config)
    
    # Run experiments
    print("Running entropy experiments...")
    results = entropy_probe.run_experiments()
    
    # Save results
    output_path = "entropy_experiment_results.json"
    print(f"Saving results to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("Experiments completed successfully!")
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main()