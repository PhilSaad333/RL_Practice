#!/usr/bin/env python3
"""
Helper script to extract Stage 3 checkpoint and update config file.

This script:
1. Extracts the rl_run_complete_with_optimizers.tar.gz archive
2. Finds the extracted run directory
3. Updates the Stage 3 config file with correct checkpoint paths
4. Validates the checkpoint structure

Usage:
    python setup_stage3_checkpoint.py
"""

import os
import tarfile
import yaml
import json
from pathlib import Path


def main():
    print("🔧 Setting up Stage 3 checkpoint for multi-GPU testing...")
    
    # Paths
    base_dir = Path("/home/ubuntu/localfs/stage3_checkpoints")
    archive_path = base_dir / "rl_run_complete_with_optimizers.tar.gz"
    config_path = Path("~/RL_Practice/entropy_experiments/configs/mixed_probe_stage3_multigpu_config.yaml").expanduser()
    
    # Check if archive exists
    if not archive_path.exists():
        print(f"❌ Archive not found: {archive_path}")
        return 1
    
    print(f"📦 Found archive: {archive_path} ({archive_path.stat().st_size / 1024**3:.1f}GB)")
    
    # Extract archive
    print("📂 Extracting checkpoint archive...")
    with tarfile.open(archive_path, 'r:gz') as tar:
        tar.extractall(path=base_dir)
    print("✅ Archive extracted")
    
    # Find the training run directory
    training_runs_dir = base_dir / "training_runs"
    if not training_runs_dir.exists():
        print("❌ training_runs directory not found after extraction")
        return 1
    
    run_dirs = [d for d in training_runs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    if not run_dirs:
        print("❌ No run directories found")
        return 1
    
    # Use the first (should be only) run directory
    run_dir = run_dirs[0]
    print(f"🎯 Found run directory: {run_dir.name}")
    
    # Find the latest checkpoint
    training_state_dir = run_dir / "training_state"
    if not training_state_dir.exists():
        print("❌ training_state directory not found")
        return 1
    
    # Look for step_latest or step_final
    checkpoint_dir = None
    for step_name in ["step_latest", "step_final"]:
        potential_dir = training_state_dir / step_name
        if potential_dir.exists():
            checkpoint_dir = potential_dir
            print(f"📍 Found checkpoint: {step_name}")
            break
    
    if checkpoint_dir is None:
        # Fall back to highest numbered step
        step_dirs = [d for d in training_state_dir.iterdir() if d.is_dir() and d.name.startswith("step_")]
        if step_dirs:
            # Sort by step number
            step_dirs.sort(key=lambda d: int(d.name.split("_")[1]) if d.name.split("_")[1].isdigit() else -1)
            checkpoint_dir = step_dirs[-1]
            print(f"📍 Using latest numbered checkpoint: {checkpoint_dir.name}")
        else:
            print("❌ No checkpoint directories found")
            return 1
    
    # Validate checkpoint structure
    model_dir = checkpoint_dir / "model"
    optimizer_path = checkpoint_dir / "optimizer.pt"
    training_info_path = checkpoint_dir / "training_info.json"
    
    if not model_dir.exists():
        print(f"❌ Model directory not found: {model_dir}")
        return 1
    if not optimizer_path.exists():
        print(f"❌ Optimizer file not found: {optimizer_path}")
        return 1
    if not training_info_path.exists():
        print(f"❌ Training info not found: {training_info_path}")
        return 1
    
    print("✅ Checkpoint structure validated")
    
    # Read training info for context
    with open(training_info_path, 'r') as f:
        training_info = json.load(f)
    
    print(f"📊 Training info: Step {training_info.get('step', 'unknown')}, "
          f"Loss: {training_info.get('train_loss', 'unknown')}")
    
    # Update config file
    print(f"⚙️ Updating config file: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update checkpoint paths
    config['checkpoint']['checkpoint_path'] = str(model_dir)
    config['checkpoint']['optimizer_path'] = str(optimizer_path)
    
    # Write updated config
    with open(config_path, 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False)
    
    print("✅ Config file updated with checkpoint paths:")
    print(f"   Model: {model_dir}")
    print(f"   Optimizer: {optimizer_path}")
    
    # Final validation test
    print("🔍 Testing config loading...")
    try:
        with open(config_path, 'r') as f:
            test_config = yaml.safe_load(f)
        if test_config['checkpoint']['checkpoint_path'] != "TBD_AFTER_EXTRACTION":
            print("✅ Config file is ready for Stage 3 testing!")
            return 0
        else:
            print("❌ Config file was not properly updated")
            return 1
    except Exception as e:
        print(f"❌ Config validation failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())