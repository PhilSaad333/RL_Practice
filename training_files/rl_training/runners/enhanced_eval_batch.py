#!/usr/bin/env python3
"""
Enhanced batch evaluation script with improved result organization and progress monitoring.

Usage:
    python enhanced_eval_batch.py --training-run run_2025-08-19_12-34-56 --eval-dataset gsm8k_r1_template
    python enhanced_eval_batch.py --training-run run_2025-08-19_12-34-56 --steps 1,5,10 --profile quick_test
"""

import os
import sys
import json
import argparse
import pathlib
import yaml
from typing import List, Optional, Dict, Any
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from evals.enhanced_evaluator import evaluate_checkpoint
from evals.result_organizer import EvaluationResultOrganizer
from evals.profile_loader import get_profile, list_profiles

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
                step_num = "final"
            else:
                try:
                    step_num = int(step_num_str)
                except ValueError:
                    continue
            
            # Filter by requested steps if specified
            if steps is not None:
                if step_num == "final" and "final" not in steps:
                    continue
                if isinstance(step_num, int) and step_num not in steps:
                    continue
            
            checkpoints.append((step_num, step_dir))
    
    # Sort checkpoints (final comes last)
    checkpoints.sort(key=lambda x: (x[0] == "final", x[0] if x[0] != "final" else float('inf')))
    return checkpoints


def load_training_config(training_run_dir: pathlib.Path) -> Dict[str, Any]:
    """Load training configuration."""
    config_path = training_run_dir / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    else:
        return {"backbone": "unknown"}


def evaluate_step_enhanced(
    step_num: str, 
    checkpoint_dir: pathlib.Path,
    eval_config: Dict[str, Any],
    training_run_name: str,
    organizer: EvaluationResultOrganizer
) -> bool:
    """
    Evaluate a single step using the enhanced evaluator.
    
    Args:
        step_num: Step number or "final"
        checkpoint_dir: Path to checkpoint directory
        eval_config: Evaluation configuration
        training_run_name: Name of the training run
        organizer: Result organizer for this evaluation
    
    Returns:
        True if evaluation succeeded, False otherwise
    """
    print(f"🔄 Evaluating step {step_num}...")
    start_time = time.time()
    
    try:
        # Progress callback for monitoring
        def progress_callback(current_batch, total_batches, batch_time):
            percent = (current_batch / total_batches) * 100
            eta = batch_time * (total_batches - current_batch)
            eta_str = f"{eta/60:.1f}m" if eta > 60 else f"{eta:.1f}s"
            print(f"      Progress: {current_batch}/{total_batches} ({percent:.1f}%) - ETA: {eta_str}")
        
        # Run evaluation
        result = evaluate_checkpoint(
            training_run_dir=checkpoint_dir.parent.parent,  # Go up from training_state/step_X
            step=step_num,
            eval_dataset=eval_config["eval_dataset"],
            subset_frac=eval_config["subset_frac"],
            batch_size=eval_config["batch_size"],
            temperature=eval_config["temperature"],
            top_p=eval_config["top_p"],
            num_return_sequences=eval_config["num_return_sequences"],
            max_new_tokens=eval_config["max_new_tokens"],
            progress_callback=progress_callback
        )
        
        elapsed = time.time() - start_time
        print(f"✅ Step {step_num} completed in {elapsed:.1f}s")
        
        # Print key metrics
        if "pass_rate" in result:
            print(f"   Pass rate: {result['pass_rate']:.3f}")
        if "avg_entropy" in result:
            print(f"   Avg entropy: {result['avg_entropy']:.3f}")
        
        return True
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Step {step_num} failed after {elapsed:.1f}s: {e}")
        return False


def run_batch_evaluation(
    training_run_name: str,
    eval_config: Dict[str, Any],
    steps: Optional[List[str]] = None
) -> pathlib.Path:
    """
    Run batch evaluation on multiple checkpoints with enhanced organization.
    
    Args:
        training_run_name: Name of training run to evaluate
        eval_config: Evaluation configuration
        steps: List of specific steps to evaluate (None = all available)
    
    Returns:
        Path to evaluation results directory
    """
    print(f"🚀 Starting enhanced batch evaluation")
    print(f"   Training run: {training_run_name}")
    print(f"   Dataset: {eval_config['eval_dataset']}")
    print(f"   Subset fraction: {eval_config['subset_frac']}")
    print(f"   Batch size: {eval_config['batch_size']}")
    
    # Find training run
    training_run_dir = find_training_run(training_run_name)
    print(f"📁 Found training run: {training_run_dir}")
    
    # Find checkpoints
    if steps:
        # Convert string steps to appropriate types
        parsed_steps = []
        for step in steps:
            if step == "final":
                parsed_steps.append("final")
            else:
                try:
                    parsed_steps.append(int(step))
                except ValueError:
                    print(f"⚠️ Invalid step: {step}")
        checkpoints = find_checkpoints(training_run_dir, parsed_steps)
    else:
        checkpoints = find_checkpoints(training_run_dir)
    
    if not checkpoints:
        raise ValueError("No checkpoints found to evaluate")
    
    print(f"🎯 Found {len(checkpoints)} checkpoints to evaluate:")
    for step_num, _ in checkpoints:
        print(f"   • Step {step_num}")
    
    # Initialize result organizer
    organizer = EvaluationResultOrganizer(
        eval_run_name=training_run_name,
        eval_dataset=eval_config["eval_dataset"],
        runs_root=EVAL_ROOT
    )
    
    # Save evaluation configuration
    full_config = {
        "training_run": training_run_name,
        "evaluation_config": eval_config,
        "checkpoints_to_evaluate": [str(step) for step, _ in checkpoints],
        "started_at": datetime.now().isoformat()
    }
    organizer.save_evaluation_config(full_config)
    
    # Evaluate each checkpoint
    successful_evaluations = 0
    total_start_time = time.time()
    
    for i, (step_num, checkpoint_dir) in enumerate(checkpoints):
        print(f"\n📊 Evaluating checkpoint {i+1}/{len(checkpoints)}: step {step_num}")
        
        success = evaluate_step_enhanced(
            step_num=step_num,
            checkpoint_dir=checkpoint_dir,
            eval_config=eval_config,
            training_run_name=training_run_name,
            organizer=organizer
        )
        
        if success:
            successful_evaluations += 1
    
    total_time = time.time() - total_start_time
    
    print(f"\n🎉 Batch evaluation completed!")
    print(f"   Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
    print(f"   Successful evaluations: {successful_evaluations}/{len(checkpoints)}")
    
    # Finalize evaluation (consolidate metrics, create summary)
    organizer.finalize_evaluation()
    
    # Print final summary
    if successful_evaluations > 0:
        consolidated_path = organizer.results_dir / "consolidated_metrics.csv"
        if consolidated_path.exists():
            print(f"📊 Consolidated metrics: {consolidated_path}")
        
        summary_path = organizer.results_dir / "consolidated_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            
            print(f"📈 Results Summary:")
            step_comparison = summary.get("step_comparison", {})
            for step, metrics in step_comparison.items():
                pass_rate = metrics.get("pass_rate", "N/A")
                print(f"   Step {step}: Pass Rate = {pass_rate}")
    
    return organizer.run_dir


def main():
    parser = argparse.ArgumentParser(description="Enhanced batch evaluate RL training checkpoints")
    parser.add_argument("--training-run", required=True, help="Training run name (e.g., run_2025-08-19_12-34-56)")
    parser.add_argument("--steps", help="Comma-separated list of steps to evaluate (e.g., 1,5,10). If not specified, evaluates all steps.")
    parser.add_argument("--eval-dataset", default="gsm8k_r1_template", help="Evaluation dataset")
    parser.add_argument("--backbone", help="Model backbone (auto-detected from training config if not specified)")
    
    # Batch size parsing
    def parse_batch_size(value):
        if value.lower() in ["auto", "conservative", "aggressive"]:
            return value.lower()
        try:
            return int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"batch_size must be an integer or 'auto'/'conservative'/'aggressive', got: {value}")
    
    parser.add_argument("--batch-size", type=parse_batch_size, default="auto", 
                        help="Batch size: integer or 'auto'/'conservative'/'aggressive' for auto-detection")
    parser.add_argument("--subset-frac", type=float, default=0.01, help="Fraction of eval dataset to use")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument("--num-return-sequences", type=int, default=8, help="Number of sequences per prompt")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--profile", type=str, help="Evaluation profile name (e.g., 'quick_test', 'full_evaluation')")
    parser.add_argument("--list-profiles", action="store_true", help="List available evaluation profiles and exit")
    
    args = parser.parse_args()
    
    if args.list_profiles:
        from evals.profile_loader import print_profile_info
        print_profile_info()
        return
    
    # Apply profile if specified
    if args.profile:
        try:
            profile_config = get_profile(args.profile)
            print(f"📋 Applying evaluation profile: {args.profile}")
            print(f"   Description: {profile_config.get('_description', 'No description')}")
            
            # Apply profile settings to args
            for key, value in profile_config.items():
                if key.startswith('_'):  # Skip metadata
                    continue
                if hasattr(args, key):
                    setattr(args, key, value)
                    print(f"   {key} = {value}")
            
        except ValueError as e:
            print(f"❌ Profile error: {e}")
            available_profiles = list(list_profiles().keys())
            print(f"   Available profiles: {', '.join(available_profiles)}")
            return
    
    # Find training run and load config
    training_run_dir = find_training_run(args.training_run)
    training_config = load_training_config(training_run_dir)
    
    # Auto-detect backbone if not specified
    if not args.backbone:
        args.backbone = training_config.get("backbone", "unknown")
        print(f"🔍 Auto-detected backbone: {args.backbone}")
    
    # Parse steps if specified
    steps = None
    if args.steps:
        steps = [s.strip() for s in args.steps.split(",")]
        print(f"🎯 Evaluating specific steps: {steps}")
    
    # Create evaluation config
    eval_config = {
        "backbone": args.backbone,
        "eval_dataset": args.eval_dataset,
        "batch_size": args.batch_size,
        "subset_frac": args.subset_frac,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "num_return_sequences": args.num_return_sequences,
        "max_new_tokens": args.max_new_tokens
    }
    
    # Run batch evaluation
    try:
        result_dir = run_batch_evaluation(
            training_run_name=args.training_run,
            eval_config=eval_config,
            steps=steps
        )
        print(f"\n🎉 Evaluation complete! Results saved to: {result_dir}")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()