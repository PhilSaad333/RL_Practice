# evals/enhanced_evaluator.py
# ──────────────────────────────────────────────────────────────────────────────
"""
Enhanced evaluator with improved result organization, progress monitoring,
and integration with the new file structure.
"""

import json
import pandas as pd
import time
from pathlib import Path
from typing import Union, List, Dict, Any, Optional
from datetime import datetime
import logging
import torch

from evals.records import EvalRecord
from evals.result_organizer import EvaluationResultOrganizer
from evals.utils_io import load_everything, generate_with_logprobs
from evals.auto_batch import get_recommended_batch_sizes
from transformers import GenerationConfig

logger = logging.getLogger(__name__)


class EnhancedEvaluator:
    """
    Enhanced evaluator with improved result organization and monitoring.
    """
    
    def __init__(
        self,
        backbone: str,
        eval_dataset: str,
        ckpt_path: str,
        step: Union[int, str],
        eval_run_name: str,
        subset_frac: float = 1.0,
        batch_size: Union[int, str] = "auto",
        runs_root: Union[str, Path] = "eval_runs",
        **generation_kwargs
    ):
        """
        Initialize enhanced evaluator.
        
        Args:
            backbone: Model backbone name
            eval_dataset: Dataset to evaluate on
            ckpt_path: Path to model checkpoint
            step: Training step being evaluated
            eval_run_name: Name of the evaluation run (e.g., "run_2025-08-20_03-31-43")
            subset_frac: Fraction of dataset to use
            batch_size: Batch size (int or "auto"/"conservative"/"aggressive")
            runs_root: Root directory for evaluation results
            **generation_kwargs: Generation parameters (temperature, top_p, etc.)
        """
        self.backbone = backbone
        self.eval_dataset = eval_dataset
        self.ckpt_path = ckpt_path
        self.step = step
        self.subset_frac = subset_frac
        self.batch_size = batch_size
        self.generation_kwargs = generation_kwargs
        
        # Initialize result organizer
        self.organizer = EvaluationResultOrganizer(
            eval_run_name=eval_run_name,
            eval_dataset=eval_dataset,
            runs_root=runs_root
        )
        
        # Load model and data
        self.model, self.tokenizer, self.prompts, self.golds, self.stopper = load_everything(
            backbone=backbone,
            eval_dataset=eval_dataset,
            ckpt_path=ckpt_path
        )
        
        # Apply subset sampling
        if subset_frac < 1.0:
            keep = int(len(self.prompts) * subset_frac)
            self.prompts = self.prompts[:keep]
            self.golds = self.golds[:keep]
        
        # Auto-detect batch size if needed
        if isinstance(batch_size, str):
            print(f"🔍 Auto-detecting batch size (mode: {batch_size})...")
            rollout_batch_size, tf_micro_batch = get_recommended_batch_sizes(
                self.model, self.tokenizer,
                max_tokens=generation_kwargs.get('max_new_tokens', 200),
                num_sequences=generation_kwargs.get('num_return_sequences', 8),
                mode=batch_size
            )
            self.batch_size = rollout_batch_size
            self.tf_micro_batch = tf_micro_batch
            print(f"✅ Using batch_size={self.batch_size}, tf_micro_batch={self.tf_micro_batch}")
        else:
            self.tf_micro_batch = batch_size
        
        # Initialize metric functions
        self._init_metric_functions()
        
        # Evaluation metadata
        self.eval_metadata = {
            "backbone": backbone,
            "eval_dataset": eval_dataset,
            "ckpt_path": str(ckpt_path),
            "step": step,
            "subset_frac": subset_frac,
            "batch_size": self.batch_size,
            "tf_micro_batch": self.tf_micro_batch,
            "generation_config": generation_kwargs,
            "num_prompts": len(self.prompts),
            "started_at": datetime.now().isoformat()
        }
    
    def _init_metric_functions(self):
        """Initialize metric computation functions."""
        from evals.metrics.tag_format import tag_format_metrics
        from evals.metrics.passk import passk_metrics
        from evals.metrics.response_len import response_len_metrics
        from evals.metrics.entropy import entropy_metrics
        
        self.metric_functions = [
            tag_format_metrics,
            passk_metrics,
            response_len_metrics,
            entropy_metrics,
        ]
    
    def evaluate(self, progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Run evaluation with progress monitoring and organized result saving.
        
        Args:
            progress_callback: Optional callback function for progress updates
                               Signature: callback(current_batch, total_batches, batch_time)
        
        Returns:
            Dictionary with evaluation results and metadata
        """
        print(f"🚀 Starting evaluation: {self.backbone} step {self.step}")
        print(f"   Dataset: {self.eval_dataset} ({len(self.prompts)} samples)")
        print(f"   Batch size: {self.batch_size}, TF micro batch: {self.tf_micro_batch}")
        
        start_time = time.time()
        
        # Create generation config
        gen_config = GenerationConfig(
            num_return_sequences=self.generation_kwargs.get('num_return_sequences', 8),
            temperature=self.generation_kwargs.get('temperature', 0.7),
            top_p=self.generation_kwargs.get('top_p', 1.0),
            max_new_tokens=self.generation_kwargs.get('max_new_tokens', 200),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
        )
        
        # Generate predictions batch by batch
        records = []
        total_batches = (len(self.prompts) + self.batch_size - 1) // self.batch_size
        
        for batch_idx, start in enumerate(range(0, len(self.prompts), self.batch_size)):
            batch_start_time = time.time()
            
            batch_prompts = self.prompts[start:start + self.batch_size]
            batch_golds = self.golds[start:start + self.batch_size]
            
            # Generate with logprobs
            gens, lps, ents = generate_with_logprobs(
                self.model, self.tokenizer, batch_prompts, gen_config, self.stopper,
                tf_micro_batch=self.tf_micro_batch
            )
            
            # Create records
            for i, prompt in enumerate(batch_prompts):
                records.append(EvalRecord(
                    step=self.step,
                    q_idx=start + i,
                    prompt=prompt,
                    generations=gens[i],
                    logprobs=lps[i],
                    entropies=ents[i],
                    cfg=dict(
                        temperature=gen_config.temperature,
                        top_p=gen_config.top_p,
                        num_return_sequences=gen_config.num_return_sequences
                    ),
                    gold=batch_golds[i]
                ))
            
            batch_time = time.time() - batch_start_time
            
            # Progress callback
            if progress_callback:
                progress_callback(batch_idx + 1, total_batches, batch_time)
            
            # Print progress
            completed = batch_idx + 1
            eta_seconds = (batch_time * (total_batches - completed))
            eta_str = f"{eta_seconds/60:.1f}m" if eta_seconds > 60 else f"{eta_seconds:.1f}s"
            
            print(f"   Batch {completed}/{total_batches} "
                  f"({completed/total_batches*100:.1f}%) - "
                  f"{batch_time:.1f}s/batch - ETA: {eta_str}")
            
            # Free GPU memory
            torch.cuda.empty_cache()
        
        total_time = time.time() - start_time
        
        # Compute metrics
        print("📊 Computing metrics...")
        metrics_dfs = [pd.DataFrame(fn(records)) for fn in self.metric_functions]
        metrics_df = metrics_dfs[0]
        for df in metrics_dfs[1:]:
            metrics_df = metrics_df.merge(df, on="q_idx", how="left")
        
        # Update evaluation metadata
        self.eval_metadata.update({
            "completed_at": datetime.now().isoformat(),
            "total_time_seconds": total_time,
            "batches_processed": total_batches,
            "avg_batch_time": total_time / total_batches,
            "samples_per_second": len(self.prompts) / total_time
        })
        
        # Add GPU info if available
        if torch.cuda.is_available():
            self.eval_metadata["gpu_info"] = {
                "device_name": torch.cuda.get_device_name(0),
                "memory_allocated_gb": torch.cuda.memory_allocated(0) / 1024**3,
                "memory_reserved_gb": torch.cuda.memory_reserved(0) / 1024**3
            }
        
        # Save results using new organized structure
        self.organizer.save_step_results(
            step=self.step,
            metrics_df=metrics_df,
            records=records,
            eval_config=self.eval_metadata,
            step_metadata={
                "evaluation_time_seconds": total_time,
                "gpu_memory_used_gb": torch.cuda.memory_allocated(0) / 1024**3 if torch.cuda.is_available() else None
            }
        )
        
        # Save evaluation configuration
        self.organizer.save_evaluation_config(self.eval_metadata)
        
        print(f"✅ Evaluation completed in {total_time:.1f}s")
        print(f"   Results saved to: {self.organizer.run_dir}")
        
        # Return summary
        summary = {
            "step": self.step,
            "num_samples": len(records),
            "total_time": total_time,
            "metrics_summary": {}
        }
        
        # Add key metrics to summary
        if 'pass_rate' in metrics_df.columns:
            summary["pass_rate"] = float(metrics_df['pass_rate'].mean())
        if 'entropy_mean' in metrics_df.columns:
            summary["avg_entropy"] = float(metrics_df['entropy_mean'].mean())
        
        return summary


def evaluate_checkpoint(
    training_run_dir: Path,
    step: Union[int, str],
    eval_dataset: str = "gsm8k_r1_template",
    **eval_kwargs
) -> Dict[str, Any]:
    """
    Convenience function to evaluate a single checkpoint from a training run.
    
    Args:
        training_run_dir: Path to training run directory
        step: Step to evaluate
        eval_dataset: Dataset to evaluate on
        **eval_kwargs: Additional evaluation parameters
    
    Returns:
        Evaluation summary
    """
    # Load training config to get backbone
    config_path = training_run_dir / "config.yaml"
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            training_config = yaml.safe_load(f)
        backbone = training_config.get('backbone', 'unknown')
    else:
        backbone = 'unknown'
    
    # Find checkpoint path
    ckpt_path = training_run_dir / "training_state" / f"step_{step}"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    # Extract run name from directory
    eval_run_name = training_run_dir.name
    
    # Create evaluator
    evaluator = EnhancedEvaluator(
        backbone=backbone,
        eval_dataset=eval_dataset,
        ckpt_path=str(ckpt_path),
        step=step,
        eval_run_name=eval_run_name,
        **eval_kwargs
    )
    
    # Run evaluation
    return evaluator.evaluate()


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced evaluation with organized results")
    parser.add_argument("--training-run", required=True, help="Training run directory")
    parser.add_argument("--step", required=True, help="Step to evaluate")
    parser.add_argument("--eval-dataset", default="gsm8k_r1_template", help="Evaluation dataset")
    parser.add_argument("--subset-frac", type=float, default=1.0, help="Fraction of dataset")
    parser.add_argument("--batch-size", default="auto", help="Batch size")
    
    args = parser.parse_args()
    
    training_run_dir = Path(args.training_run)
    
    result = evaluate_checkpoint(
        training_run_dir=training_run_dir,
        step=args.step,
        eval_dataset=args.eval_dataset,
        subset_frac=args.subset_frac,
        batch_size=args.batch_size
    )
    
    print(f"Evaluation result: {result}")