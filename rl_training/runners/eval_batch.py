#!/usr/bin/env python3
"""
Batch evaluation script for RL training checkpoints.

Usage:
    python eval_batch.py --training-run run_2025-08-19_12-34-56 --eval-dataset gsm8k_r1_template
    python eval_batch.py --training-run run_2025-08-19_12-34-56 --steps 1,5,10 --subset-frac 0.1
"""

import os
import sys
import json
import argparse
import pathlib
import subprocess
from typing import List, Optional
import pandas as pd

# Add project root to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

LOCALFS_ROOT = pathlib.Path(os.environ.get("LOCALFS_ROOT", "/lambda/nfs/localfs"))
TRAINING_ROOT = LOCALFS_ROOT / "training_runs"
EVAL_ROOT = LOCALFS_ROOT / "eval_runs"


def find_training_run(run_name: str) -> pathlib.Path:
    """Find training run directory."""
    if run_name.startswith("run_"):
        run_dir = TRAINING_ROOT / run_name
    else:
        run_dir = TRAINING_ROOT / f"run_{run_name}"
    
    if not run_dir.exists():
        raise FileNotFoundError(f"Training run not found: {run_dir}")
    return run_dir


def find_checkpoints(training_run_dir: pathlib.Path, steps: Optional[List[int]] = None) -> List[pathlib.Path]:
    """Find available checkpoints in training run."""
    training_state_dir = training_run_dir / "training_state"
    
    if not training_state_dir.exists():
        raise FileNotFoundError(f"No training_state directory found in {training_run_dir}")
    
    checkpoints = []
    for step_dir in training_state_dir.iterdir():
        if step_dir.is_dir() and step_dir.name.startswith("step_") and step_dir.name != "step_latest":
            step_num_str = step_dir.name.replace("step_", "")
            if step_num_str == "final":
                checkpoints.append((float('inf'), step_dir))  # Final step at end
            else:
                try:
                    step_num = int(step_num_str)
                    if steps is None or step_num in steps:
                        checkpoints.append((step_num, step_dir))
                except ValueError:
                    continue
    
    # Sort by step number
    checkpoints.sort(key=lambda x: x[0])
    return [ckpt_dir for _, ckpt_dir in checkpoints]


def run_evaluation(checkpoint_dir: pathlib.Path, eval_config: dict, eval_run_dir: pathlib.Path) -> bool:
    """Run evaluation on a single checkpoint."""
    # Extract step number for eval_runner
    step_name = checkpoint_dir.name.replace("step_", "")
    if step_name == "final":
        step_num = "final"
    else:
        step_num = step_name
    
    model_dir = checkpoint_dir / "model"
    if not model_dir.exists():
        print(f"❌ No model directory found in {checkpoint_dir}")
        return False
    
    # Build eval command
    cmd = [
        "python", "-m", "evals.eval_runner",
        "--backbone", eval_config["backbone"],
        "--eval-dataset", eval_config["eval_dataset"],
        "--ckpt-path", str(model_dir),
        "--ckpt-step", str(step_num),
        "--batch-size", str(eval_config["batch_size"]),
        "--subset-frac", str(eval_config["subset_frac"]),
        "--temperature", str(eval_config["temperature"]),
        "--top-p", str(eval_config["top_p"]),
        "--num-return-sequences", str(eval_config["num_return_sequences"]),
        "--max-new-tokens", str(eval_config["max_new_tokens"]),
        "--runs-root", str(eval_run_dir)
    ]
    
    print(f"🔄 Evaluating step {step_num}...")
    print(f"   Command: {' '.join(cmd)}")
    
    try:
        # Set environment for clean evaluation
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '0'  # Use single GPU
        env['PYTHONPATH'] = '.'
        
        result = subprocess.run(
            cmd,
            env=env,
            cwd=str(pathlib.Path.cwd()),
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout per eval
        )
        
        if result.returncode == 0:
            print(f"✅ Step {step_num} evaluation completed")
            return True
        else:
            print(f"❌ Step {step_num} evaluation failed:")
            print(f"   stderr: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Step {step_num} evaluation timed out")
        return False
    except Exception as e:
        print(f"❌ Step {step_num} evaluation error: {e}")
        return False


def consolidate_metrics(eval_run_dir: pathlib.Path, output_file: str = "consolidated_metrics.csv"):
    """Consolidate metrics from all evaluations into a single CSV."""
    all_metrics = []
    
    for step_dir in eval_run_dir.iterdir():
        if step_dir.is_dir() and step_dir.name.startswith("step_"):
            metrics_file = None
            # Find metrics.csv in subdirectories
            for subdir in step_dir.iterdir():
                if subdir.is_dir():
                    potential_metrics = subdir / "metrics.csv"
                    if potential_metrics.exists():
                        metrics_file = potential_metrics
                        break
            
            if metrics_file:
                try:
                    df = pd.read_csv(metrics_file)
                    df['step'] = step_dir.name.replace("step_", "")
                    df['eval_dir'] = str(step_dir)
                    all_metrics.append(df)
                except Exception as e:
                    print(f"⚠️ Failed to read metrics from {metrics_file}: {e}")
    
    if all_metrics:
        consolidated_df = pd.concat(all_metrics, ignore_index=True)
        output_path = eval_run_dir / output_file
        consolidated_df.to_csv(output_path, index=False)
        print(f"📊 Consolidated metrics saved to {output_path}")
        
        # Print summary
        print(f"📈 Evaluation Summary:")
        for step in consolidated_df['step'].unique():
            step_data = consolidated_df[consolidated_df['step'] == step]
            if 'pass_rate' in step_data.columns:
                pass_rate = step_data['pass_rate'].mean()
                print(f"   Step {step}: Pass Rate = {pass_rate:.3f}")
        
        return output_path
    else:
        print("⚠️ No metrics found to consolidate")
        return None


def main():
    parser = argparse.ArgumentParser(description="Batch evaluate RL training checkpoints")
    parser.add_argument("--training-run", required=True, help="Training run name (e.g., run_2025-08-19_12-34-56)")
    parser.add_argument("--steps", help="Comma-separated list of steps to evaluate (e.g., 1,5,10). If not specified, evaluates all steps.")
    parser.add_argument("--eval-dataset", default="gsm8k_r1_template", help="Evaluation dataset")
    parser.add_argument("--backbone", help="Model backbone (auto-detected from training config if not specified)")
    parser.add_argument("--batch-size", type=int, default=8, help="Evaluation batch size")
    parser.add_argument("--subset-frac", type=float, default=0.01, help="Fraction of eval dataset to use")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument("--num-return-sequences", type=int, default=8, help="Number of sequences per prompt")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max tokens to generate")
    
    args = parser.parse_args()
    
    # Find training run
    training_run_dir = find_training_run(args.training_run)
    print(f"📁 Found training run: {training_run_dir}")
    
    # Parse steps if specified
    steps = None
    if args.steps:
        try:
            steps = [int(s.strip()) for s in args.steps.split(",")]
            print(f"🎯 Evaluating specific steps: {steps}")
        except ValueError:
            print(f"❌ Invalid steps format: {args.steps}")
            return 1
    
    # Find checkpoints
    checkpoints = find_checkpoints(training_run_dir, steps)
    if not checkpoints:
        print(f"❌ No checkpoints found in {training_run_dir}")
        return 1
    
    print(f"🔍 Found {len(checkpoints)} checkpoints to evaluate")
    
    # Load training config to get backbone if not specified
    config_path = training_run_dir / "config.yaml"
    if config_path.exists() and not args.backbone:
        import yaml
        with open(config_path) as f:
            training_config = yaml.safe_load(f)
        backbone = training_config.get("eval_backbone") or training_config.get("backbone")
        if not backbone:
            print(f"❌ Cannot determine backbone from config and none specified")
            return 1
    else:
        backbone = args.backbone
        if not backbone:
            print(f"❌ No backbone specified and no config found")
            return 1
    
    # Create eval run directory
    eval_run_name = f"{args.training_run}_{args.eval_dataset}"
    eval_run_dir = EVAL_ROOT / eval_run_name
    eval_run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save eval config
    eval_config = {
        "backbone": backbone,
        "eval_dataset": args.eval_dataset,
        "batch_size": args.batch_size,
        "subset_frac": args.subset_frac,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "num_return_sequences": args.num_return_sequences,
        "max_new_tokens": args.max_new_tokens,
        "training_run": args.training_run,
        "training_run_dir": str(training_run_dir),
        "steps_evaluated": steps or "all"
    }
    
    with open(eval_run_dir / "eval_config.yaml", 'w') as f:
        import yaml
        yaml.dump(eval_config, f, indent=2)
    
    print(f"📝 Evaluation config saved to {eval_run_dir / 'eval_config.yaml'}")
    print(f"🎯 Using backbone: {backbone}")
    print(f"📊 Evaluation results will be saved to: {eval_run_dir}")
    
    # Run evaluations
    successful_evals = 0
    total_evals = len(checkpoints)
    
    for checkpoint_dir in checkpoints:
        success = run_evaluation(checkpoint_dir, eval_config, eval_run_dir)
        if success:
            successful_evals += 1
    
    print(f"\n🎉 Evaluation completed: {successful_evals}/{total_evals} successful")
    
    # Consolidate metrics
    if successful_evals > 0:
        consolidate_metrics(eval_run_dir)
    
    return 0 if successful_evals > 0 else 1


if __name__ == "__main__":
    sys.exit(main())