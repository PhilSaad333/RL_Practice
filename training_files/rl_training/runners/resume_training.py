#!/usr/bin/env python3
"""
Resume RL training from saved checkpoint with full state restoration.

Usage:
    python resume_training.py --training-run run_2025-08-19_12-34-56 --from-step latest
    python resume_training.py --training-run run_2025-08-19_12-34-56 --from-step 10 --additional-steps 50
"""

import os
import sys
import json
import torch
import argparse
import pathlib
from typing import Optional
import subprocess

# Add project root to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

LOCALFS_ROOT = pathlib.Path(os.environ.get("LOCALFS_ROOT", "/lambda/nfs/localfs"))
TRAINING_ROOT = LOCALFS_ROOT / "training_runs"


def find_training_run(run_name: str) -> pathlib.Path:
    """Find training run directory."""
    if run_name.startswith("run_"):
        run_dir = TRAINING_ROOT / run_name
    else:
        run_dir = TRAINING_ROOT / f"run_{run_name}"
    
    if not run_dir.exists():
        raise FileNotFoundError(f"Training run not found: {run_dir}")
    return run_dir


def find_checkpoint_to_resume(training_run_dir: pathlib.Path, from_step: str) -> pathlib.Path:
    """Find the checkpoint to resume from."""
    training_state_dir = training_run_dir / "training_state"
    
    if not training_state_dir.exists():
        raise FileNotFoundError(f"No training_state directory found in {training_run_dir}")
    
    if from_step == "latest":
        latest_link = training_state_dir / "step_latest"
        if latest_link.is_symlink():
            return latest_link.resolve()
        else:
            # Find highest numbered step
            steps = []
            for step_dir in training_state_dir.iterdir():
                if step_dir.is_dir() and step_dir.name.startswith("step_"):
                    step_num_str = step_dir.name.replace("step_", "")
                    if step_num_str == "final":
                        steps.append((float('inf'), step_dir))
                    else:
                        try:
                            step_num = int(step_num_str)
                            steps.append((step_num, step_dir))
                        except ValueError:
                            continue
            if not steps:
                raise FileNotFoundError("No valid checkpoints found")
            steps.sort(key=lambda x: x[0])
            return steps[-1][1]  # Return highest step
    else:
        # Specific step
        if from_step == "final":
            checkpoint_dir = training_state_dir / "step_final"
        else:
            try:
                step_num = int(from_step)
                checkpoint_dir = training_state_dir / f"step_{step_num}"
            except ValueError:
                raise ValueError(f"Invalid step specification: {from_step}")
        
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")
        return checkpoint_dir


def load_training_info(checkpoint_dir: pathlib.Path) -> dict:
    """Load training info from checkpoint."""
    training_info_path = checkpoint_dir / "training_info.json"
    if not training_info_path.exists():
        raise FileNotFoundError(f"No training_info.json found in {checkpoint_dir}")
    
    with open(training_info_path, 'r') as f:
        return json.load(f)


def create_resume_config(original_config: dict, resume_info: dict, additional_steps: Optional[int] = None) -> dict:
    """Create modified config for resumption."""
    resume_config = original_config.copy()
    
    # Update step count if additional steps specified
    if additional_steps is not None:
        current_step = resume_info["step"]
        new_total_steps = current_step + additional_steps
        resume_config["total_steps"] = new_total_steps
        print(f"📈 Resuming from step {current_step}, will train for {additional_steps} more steps (total: {new_total_steps})")
    else:
        print(f"📈 Resuming from step {resume_info['step']}, using original total_steps: {resume_config.get('total_steps', 'not specified')}")
    
    return resume_config


def run_resumed_training(
    training_run_dir: pathlib.Path, 
    checkpoint_dir: pathlib.Path, 
    resume_config: dict,
    use_torchrun: bool = True,
    num_gpus: int = 2
) -> bool:
    """Run the resumed training with restored state."""
    
    # Save resume config
    resume_config_path = training_run_dir / "resume_config.yaml"
    import yaml
    with open(resume_config_path, 'w') as f:
        yaml.dump(resume_config, f, indent=2)
    
    # Add resume parameters to config
    model_dir = checkpoint_dir / "model"
    optimizer_path = checkpoint_dir / "optimizer.pt"
    scheduler_path = checkpoint_dir / "scheduler.pt"
    training_info_path = checkpoint_dir / "training_info.json"
    
    # Build command
    script_path = pathlib.Path(__file__).parent / "rl_runner.py"
    
    if use_torchrun and num_gpus > 1:
        cmd = [
            "torchrun", f"--nproc_per_node={num_gpus}",
            str(script_path),
            "--cfg", str(resume_config_path),
            "--ckpt", str(model_dir),
            "--resume-optimizer", str(optimizer_path),
            "--resume-scheduler", str(scheduler_path),
            "--resume-training-info", str(training_info_path)
        ]
    else:
        cmd = [
            "python", str(script_path),
            "--cfg", str(resume_config_path),
            "--ckpt", str(model_dir),
            "--resume-optimizer", str(optimizer_path),
            "--resume-scheduler", str(scheduler_path),
            "--resume-training-info", str(training_info_path)
        ]
    
    print(f"🚀 Starting resumed training...")
    print(f"   Command: {' '.join(cmd)}")
    print(f"   Working directory: {pathlib.Path.cwd()}")
    print(f"   Training run directory: {training_run_dir}")
    
    # Set environment
    env = os.environ.copy()
    env['PYTHONPATH'] = '.'
    
    try:
        # Run training
        result = subprocess.run(
            cmd,
            env=env,
            cwd=str(pathlib.Path.cwd())
        )
        
        if result.returncode == 0:
            print(f"✅ Resumed training completed successfully")
            return True
        else:
            print(f"❌ Resumed training failed with return code {result.returncode}")
            return False
            
    except KeyboardInterrupt:
        print(f"⏹️ Training interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Resume RL training from checkpoint")
    parser.add_argument("--training-run", required=True, help="Training run name (e.g., run_2025-08-19_12-34-56)")
    parser.add_argument("--from-step", default="latest", help="Step to resume from (latest/final/step_number)")
    parser.add_argument("--additional-steps", type=int, help="Additional steps to train (if not specified, uses original total_steps)")
    parser.add_argument("--num-gpus", type=int, default=2, help="Number of GPUs to use")
    parser.add_argument("--no-torchrun", action="store_true", help="Don't use torchrun (single GPU mode)")
    
    args = parser.parse_args()
    
    try:
        # Find training run
        training_run_dir = find_training_run(args.training_run)
        print(f"📁 Found training run: {training_run_dir}")
        
        # Find checkpoint to resume from
        checkpoint_dir = find_checkpoint_to_resume(training_run_dir, args.from_step)
        print(f"🔄 Resuming from checkpoint: {checkpoint_dir}")
        
        # Load training info
        training_info = load_training_info(checkpoint_dir)
        print(f"📊 Checkpoint info: Step {training_info['step']}, Model: {training_info['model_config'].get('backbone', 'unknown')}")
        
        # Validate checkpoint completeness
        required_files = ["model", "optimizer.pt", "scheduler.pt", "training_info.json"]
        missing_files = []
        for file in required_files:
            file_path = checkpoint_dir / file
            if not file_path.exists():
                missing_files.append(file)
        
        if missing_files:
            print(f"❌ Incomplete checkpoint - missing files: {missing_files}")
            return 1
        
        print(f"✅ Checkpoint validation passed")
        
        # Optional metadata: ordered optimizer param names for robust reload
        names_json = checkpoint_dir / "optimizer_param_names.json"
        if names_json.exists():
            print(f"Found optimizer_param_names.json: {names_json}")
        else:
            print("Warning: optimizer_param_names.json not found; default optimizer state load will be used")

        # Load original config
        original_config_path = training_run_dir / "config.yaml"
        if not original_config_path.exists():
            print(f"❌ Original config not found: {original_config_path}")
            return 1
        
        import yaml
        with open(original_config_path, 'r') as f:
            original_config = yaml.safe_load(f)
        
        # Create resume config
        resume_config = create_resume_config(original_config, training_info, args.additional_steps)
        
        # Run resumed training
        use_torchrun = not args.no_torchrun and args.num_gpus > 1
        success = run_resumed_training(
            training_run_dir, 
            checkpoint_dir, 
            resume_config,
            use_torchrun=use_torchrun,
            num_gpus=args.num_gpus
        )
        
        return 0 if success else 1
        
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
