# evals/result_organizer.py
# ──────────────────────────────────────────────────────────────────────────────
"""
Improved result organization for evaluation runs.

New file structure:
eval_runs/
├── run_YYYY-MM-DD_HH-MM-SS_DATASET/
│   ├── config.yaml           # Evaluation configuration
│   ├── metadata.json         # Model info, timing, resource usage  
│   ├── results/              # All results organized by step
│   │   ├── step_10/
│   │   │   ├── metrics.csv   # Clean tabular format
│   │   │   ├── predictions.jsonl  # Raw predictions
│   │   │   └── summary.json  # Quick stats
│   │   ├── step_20/
│   │   └── consolidated_metrics.csv  # Cross-step comparison
│   └── logs/                 # Evaluation logs
│       ├── step_10.log
│       └── evaluation.log
"""

import json
import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import logging
import shutil
import gzip

logger = logging.getLogger(__name__)


class EvaluationResultOrganizer:
    """
    Manages evaluation result organization with improved file structure.
    """
    
    def __init__(self, eval_run_name: str, eval_dataset: str, runs_root: Union[str, Path] = "eval_runs"):
        """
        Initialize result organizer for an evaluation run.
        
        Args:
            eval_run_name: Name like "run_2025-08-20_03-31-43" 
            eval_dataset: Dataset being evaluated (e.g., "gsm8k_r1_template")
            runs_root: Root directory for evaluation runs
        """
        self.runs_root = Path(runs_root)
        self.eval_run_name = eval_run_name
        self.eval_dataset = eval_dataset
        
        # Create main run directory with timestamp and dataset
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = self.runs_root / f"{eval_run_name}_{eval_dataset}"
        
        # Create organized subdirectories
        self.results_dir = self.run_dir / "results"
        self.logs_dir = self.run_dir / "logs"
        
        # Ensure directories exist
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Track metadata
        self.metadata = {
            "eval_run_name": eval_run_name,
            "eval_dataset": eval_dataset,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "steps_evaluated": [],
            "total_steps": 0,
            "status": "in_progress"
        }
        
    def create_step_directory(self, step: Union[int, str]) -> Path:
        """Create and return directory for a specific step."""
        step_dir = self.results_dir / f"step_{step}"
        step_dir.mkdir(exist_ok=True)
        return step_dir
    
    def save_step_results(
        self, 
        step: Union[int, str],
        metrics_df: pd.DataFrame,
        records: List[Any],
        eval_config: Dict[str, Any],
        step_metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Save results for a single evaluation step with improved organization.
        
        Args:
            step: Step number or identifier
            metrics_df: Computed metrics dataframe
            records: Raw evaluation records  
            eval_config: Configuration used for this evaluation
            step_metadata: Additional metadata about this step
        """
        step_dir = self.create_step_directory(step)
        
        # 1. Save metrics in clean CSV format
        metrics_path = step_dir / "metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        
        # 2. Save raw predictions in JSONL format for easy inspection
        predictions_path = step_dir / "predictions.jsonl"
        with open(predictions_path, 'w', encoding='utf-8') as f:
            for record in records:
                prediction_data = {
                    "q_idx": record.q_idx,
                    "prompt": record.prompt,
                    "generations": record.generations,
                    "gold": record.gold,
                    "step": record.step
                }
                f.write(json.dumps(prediction_data, ensure_ascii=False) + '\n')
        
        # 3. Create step summary with key metrics
        summary = self._create_step_summary(step, metrics_df, eval_config, step_metadata)
        summary_path = step_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # 4. Save compressed raw records for completeness
        records_path = step_dir / "records.jsonl.gz"
        self._save_compressed_records(records, records_path)
        
        # 5. Update step in metadata
        self._update_step_metadata(step, eval_config)
        
        logger.info(f"📁 Saved step {step} results to {step_dir}")
    
    def _create_step_summary(
        self, 
        step: Union[int, str], 
        metrics_df: pd.DataFrame,
        eval_config: Dict[str, Any],
        step_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create summary statistics for a step."""
        summary = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "config": eval_config,
            "num_samples": len(metrics_df),
            "metrics_summary": {}
        }
        
        # Add step-specific metadata
        if step_metadata:
            summary.update(step_metadata)
        
        # Compute summary statistics for numeric columns
        numeric_cols = metrics_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if col != 'q_idx':  # Skip index column
                summary["metrics_summary"][col] = {
                    "mean": float(metrics_df[col].mean()),
                    "std": float(metrics_df[col].std()),
                    "min": float(metrics_df[col].min()),
                    "max": float(metrics_df[col].max()),
                    "median": float(metrics_df[col].median())
                }
        
        # Add key performance indicators
        if 'pass_rate' in metrics_df.columns:
            summary["pass_rate"] = float(metrics_df['pass_rate'].mean())
        
        return summary
    
    def _save_compressed_records(self, records: List[Any], output_path: Path):
        """Save records in compressed JSONL format."""
        with gzip.open(output_path, 'wt', encoding='utf-8') as f:
            for record in records:
                # Convert record to dict for JSON serialization
                record_dict = {
                    "step": record.step,
                    "q_idx": record.q_idx,
                    "prompt": record.prompt,
                    "generations": record.generations,
                    "logprobs": [arr.tolist() for arr in record.logprobs],  # Convert numpy arrays
                    "entropies": [arr.tolist() for arr in record.entropies],
                    "cfg": record.cfg,
                    "gold": record.gold
                }
                f.write(json.dumps(record_dict, ensure_ascii=False) + '\n')
    
    def _update_step_metadata(self, step: Union[int, str], eval_config: Dict[str, Any]):
        """Update metadata with new step information."""
        step_str = str(step)
        if step_str not in self.metadata["steps_evaluated"]:
            self.metadata["steps_evaluated"].append(step_str)
        
        self.metadata["total_steps"] = len(self.metadata["steps_evaluated"])
        self.metadata["updated_at"] = datetime.now().isoformat()
        self.metadata["last_step_config"] = eval_config
        
        # Save updated metadata
        self.save_metadata()
    
    def consolidate_metrics(self) -> Optional[Path]:
        """
        Consolidate metrics across all evaluated steps into a single file.
        
        Returns:
            Path to consolidated metrics file, or None if no metrics found
        """
        all_metrics = []
        
        # Collect metrics from all step directories
        for step_dir in sorted(self.results_dir.glob("step_*")):
            if step_dir.is_dir():
                metrics_file = step_dir / "metrics.csv"
                summary_file = step_dir / "summary.json"
                
                if metrics_file.exists():
                    try:
                        df = pd.read_csv(metrics_file)
                        df['step'] = step_dir.name.replace("step_", "")
                        
                        # Add summary metrics as new columns
                        if summary_file.exists():
                            with open(summary_file) as f:
                                summary = json.load(f)
                            if 'pass_rate' in summary:
                                df['step_pass_rate'] = summary['pass_rate']
                        
                        all_metrics.append(df)
                    except Exception as e:
                        logger.warning(f"Failed to read metrics from {metrics_file}: {e}")
        
        if not all_metrics:
            logger.warning("No metrics found to consolidate")
            return None
        
        # Combine all metrics
        consolidated_df = pd.concat(all_metrics, ignore_index=True)
        
        # Save consolidated metrics
        consolidated_path = self.results_dir / "consolidated_metrics.csv"
        consolidated_df.to_csv(consolidated_path, index=False)
        
        # Create consolidated summary
        self._create_consolidated_summary(consolidated_df)
        
        logger.info(f"📊 Consolidated {len(all_metrics)} steps into {consolidated_path}")
        return consolidated_path
    
    def _create_consolidated_summary(self, consolidated_df: pd.DataFrame):
        """Create summary across all steps."""
        steps = consolidated_df['step'].unique()
        
        summary = {
            "evaluation_summary": {
                "total_steps": len(steps),
                "steps_evaluated": sorted(steps, key=lambda x: int(x) if x.isdigit() else float('inf')),
                "total_samples": len(consolidated_df),
                "created_at": datetime.now().isoformat()
            },
            "step_comparison": {}
        }
        
        # Per-step summaries
        for step in steps:
            step_data = consolidated_df[consolidated_df['step'] == step]
            step_summary = {}
            
            # Key metrics
            if 'pass_rate' in step_data.columns:
                step_summary['pass_rate'] = float(step_data['pass_rate'].mean())
            if 'entropy_mean' in step_data.columns:
                step_summary['avg_entropy'] = float(step_data['entropy_mean'].mean())
            if 'response_len_mean' in step_data.columns:
                step_summary['avg_response_length'] = float(step_data['response_len_mean'].mean())
            
            summary["step_comparison"][step] = step_summary
        
        # Save consolidated summary
        summary_path = self.results_dir / "consolidated_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def save_evaluation_config(self, config: Dict[str, Any]):
        """Save the evaluation configuration."""
        config_path = self.run_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=True)
        logger.info(f"💾 Saved evaluation config to {config_path}")
    
    def save_metadata(self):
        """Save evaluation metadata."""
        metadata_path = self.run_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def finalize_evaluation(self):
        """Finalize the evaluation run."""
        self.metadata["status"] = "completed"
        self.metadata["completed_at"] = datetime.now().isoformat()
        
        # Consolidate all metrics
        consolidated_path = self.consolidate_metrics()
        if consolidated_path:
            self.metadata["consolidated_metrics_path"] = str(consolidated_path)
        
        # Save final metadata
        self.save_metadata()
        
        logger.info(f"🎉 Evaluation run finalized: {self.run_dir}")
    
    def get_step_summary(self, step: Union[int, str]) -> Optional[Dict[str, Any]]:
        """Get summary for a specific step."""
        step_dir = self.results_dir / f"step_{step}"
        summary_file = step_dir / "summary.json"
        
        if summary_file.exists():
            with open(summary_file) as f:
                return json.load(f)
        return None
    
    def list_evaluated_steps(self) -> List[str]:
        """List all evaluated steps."""
        return sorted([
            d.name.replace("step_", "") 
            for d in self.results_dir.glob("step_*") 
            if d.is_dir()
        ], key=lambda x: int(x) if x.isdigit() else float('inf'))
    
    def get_run_info(self) -> Dict[str, Any]:
        """Get comprehensive information about this evaluation run."""
        return {
            "run_directory": str(self.run_dir),
            "eval_run_name": self.eval_run_name,
            "eval_dataset": self.eval_dataset,
            "metadata": self.metadata,
            "evaluated_steps": self.list_evaluated_steps(),
            "has_consolidated_metrics": (self.results_dir / "consolidated_metrics.csv").exists()
        }


def migrate_old_results(old_eval_runs_dir: Path, new_eval_runs_dir: Path):
    """
    Migrate results from old evaluation structure to new organized structure.
    """
    logger.info(f"Migrating evaluation results from {old_eval_runs_dir} to {new_eval_runs_dir}")
    
    # This would implement migration logic for existing evaluation results
    # For now, we'll just log that it's not implemented
    logger.warning("Result migration not yet implemented - new evaluations will use improved structure")


if __name__ == "__main__":
    # Example usage
    organizer = EvaluationResultOrganizer("run_2025-08-20_03-31-43", "gsm8k_r1_template")
    print(f"Created organizer for: {organizer.run_dir}")
    print(f"Evaluated steps: {organizer.list_evaluated_steps()}")