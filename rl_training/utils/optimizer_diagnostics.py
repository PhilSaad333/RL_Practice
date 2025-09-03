"""
Optimizer State Diagnostics for RL Training

Tracks evolution of optimizer states (exp_avg, exp_avg_sq, step counts)
during training to understand parameter update patterns and identify issues
like tiny second moment estimates that can cause scaling problems.
"""

import json
import numpy as np
import torch
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import re


class OptimizerDiagnostics:
    """
    Comprehensive optimizer state statistics tracker.
    
    Monitors evolution of Adam optimizer states (exp_avg, exp_avg_sq)
    with detailed breakdowns by parameter type and layer.
    """
    
    def __init__(
        self, 
        output_dir: Path,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize optimizer diagnostics.
        
        Args:
            output_dir: Directory to save diagnostics files
            config: Configuration dictionary with diagnostics settings
            logger: Logger for status messages
        """
        self.output_dir = Path(output_dir)
        self.stats_dir = self.output_dir / "optimizer_stats"
        self.stats_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logger or logging.getLogger(__name__)
        
        # Configuration
        default_config = {
            'enabled': True,
            'frequency': 1,
            'tiny_thresholds': [1e-12, 1e-10, 1e-8, 1e-6],
            'histogram_bins': 50,
            'track_gradients': True,
            'param_patterns': ['lora_A', 'lora_B', 'q_proj', 'k_proj', 'v_proj', 'o_proj']
        }
        self.config = {**default_config, **(config or {})}
        
        # Summary data for CSV export
        self.summary_data = []
        
        self.logger.info(f"OptimizerDiagnostics initialized: {self.stats_dir}")
        
    def should_log_step(self, step: int) -> bool:
        """Check if this step should be logged based on frequency."""
        return self.config['enabled'] and (step % self.config['frequency'] == 0)
    
    def _get_param_category(self, param_name: str) -> Tuple[str, str, str]:
        """
        Categorize parameter by type, attention component, and layer.
        
        Returns:
            (param_type, attention_type, layer_id)
        """
        param_type = "unknown"
        attention_type = "unknown" 
        layer_id = "unknown"
        
        # Extract layer number
        layer_match = re.search(r'layer[._](\d+)', param_name, re.IGNORECASE)
        if layer_match:
            layer_id = f"layer_{layer_match.group(1)}"
        
        # Parameter type (LoRA A/B matrices)
        if 'lora_A' in param_name:
            param_type = "lora_A"
        elif 'lora_B' in param_name:
            param_type = "lora_B"
        elif 'lm_head' in param_name:
            param_type = "lm_head"
        else:
            param_type = "other"
            
        # Attention component
        for component in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            if component in param_name:
                attention_type = component
                break
                
        return param_type, attention_type, layer_id
    
    def _compute_tensor_stats(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """Compute comprehensive statistics for a tensor."""
        if tensor is None or tensor.numel() == 0:
            return {"count": 0, "all_zero": True}
            
        # Convert to CPU for numpy operations
        t = tensor.detach().cpu().float()
        t_flat = t.flatten()
        
        # Basic statistics
        stats = {
            "count": int(t.numel()),
            "min": float(t.min().item()),
            "max": float(t.max().item()),
            "mean": float(t.mean().item()),
            "std": float(t.std().item()),
            "all_zero": bool((t == 0).all().item())
        }
        
        # Percentiles
        percentiles = [5, 25, 50, 75, 95]
        np_tensor = t_flat.numpy()
        stats["percentiles"] = {
            str(p): float(np.percentile(np_tensor, p)) for p in percentiles
        }
        
        # Count tiny values at different thresholds
        stats["tiny_counts"] = {}
        for threshold in self.config['tiny_thresholds']:
            tiny_count = int((torch.abs(t) < threshold).sum().item())
            stats["tiny_counts"][str(threshold)] = tiny_count
            
        # Zero count
        stats["zero_count"] = int((t == 0).sum().item())
        
        # Histogram
        try:
            # Use log scale for exp_avg_sq since values span many orders of magnitude
            log_values = torch.log10(torch.abs(t_flat) + 1e-20)  # Add small epsilon to avoid log(0)
            hist_counts, bin_edges = np.histogram(
                log_values.numpy(), 
                bins=self.config['histogram_bins']
            )
            stats["histogram"] = {
                "counts": hist_counts.tolist(),
                "bin_edges": bin_edges.tolist(),
                "log_scale": True
            }
        except Exception as e:
            self.logger.warning(f"Failed to compute histogram: {e}")
            stats["histogram"] = {"error": str(e)}
            
        return stats
    
    def _aggregate_by_category(
        self, 
        param_stats: Dict[str, Any],
        param_names: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Aggregate statistics by parameter categories."""
        categories = {
            'by_param_type': {},
            'by_attention': {},  
            'by_layer': {}
        }
        
        for param_name in param_names:
            param_type, attention_type, layer_id = self._get_param_category(param_name)
            
            # Initialize category dictionaries
            for cat_name, cat_key in [
                ('by_param_type', param_type),
                ('by_attention', attention_type),
                ('by_layer', layer_id)
            ]:
                if cat_key not in categories[cat_name]:
                    categories[cat_name][cat_key] = {
                        'params': [],
                        'combined_tensor': []
                    }
                categories[cat_name][cat_key]['params'].append(param_name)
        
        return categories
    
    def analyze_optimizer_state(
        self, 
        optimizer: torch.optim.Optimizer, 
        step: int,
        model: Optional[torch.nn.Module] = None
    ) -> Dict[str, Any]:
        """
        Analyze current optimizer state and return comprehensive statistics.
        
        Args:
            optimizer: PyTorch optimizer (typically AdamW)
            step: Current training step
            model: Model (for gradient analysis if enabled)
            
        Returns:
            Dictionary with detailed optimizer statistics
        """
        timestamp = datetime.now().isoformat()
        
        # Get parameter names and states
        param_names = []
        exp_avg_tensors = []
        exp_avg_sq_tensors = []
        step_values = []
        
        for group in optimizer.param_groups:
            for param in group['params']:
                # Find parameter name from model
                param_name = f"param_{id(param)}"  # Fallback
                if model is not None:
                    for name, model_param in model.named_parameters():
                        if model_param is param:
                            param_name = name
                            break
                            
                param_names.append(param_name)
                
                # Extract optimizer states
                state = optimizer.state.get(param, {})
                exp_avg = state.get('exp_avg', None)
                exp_avg_sq = state.get('exp_avg_sq', None) 
                param_step = state.get('step', 0)
                
                exp_avg_tensors.append(exp_avg)
                exp_avg_sq_tensors.append(exp_avg_sq)
                step_values.append(param_step)
        
        # Combine all tensors for global statistics
        def combine_tensors(tensor_list):
            valid_tensors = [t for t in tensor_list if t is not None and t.numel() > 0]
            if not valid_tensors:
                return None
            return torch.cat([t.flatten() for t in valid_tensors])
        
        combined_exp_avg = combine_tensors(exp_avg_tensors)
        combined_exp_avg_sq = combine_tensors(exp_avg_sq_tensors)
        
        # Compute global statistics
        stats = {
            "step": step,
            "timestamp": timestamp,
            "total_params": len(param_names),
            "exp_avg": {
                "global": self._compute_tensor_stats(combined_exp_avg) if combined_exp_avg is not None else {}
            },
            "exp_avg_sq": {
                "global": self._compute_tensor_stats(combined_exp_avg_sq) if combined_exp_avg_sq is not None else {}
            },
            "step_values": {
                "values": step_values,
                "min": min(step_values) if step_values else 0,
                "max": max(step_values) if step_values else 0,
                "unique_count": len(set(step_values)) if step_values else 0
            }
        }
        
        # Learning rates from parameter groups
        stats["learning_rates"] = [group.get('lr', 0.0) for group in optimizer.param_groups]
        
        # Category breakdowns (simplified for now - can be expanded)
        param_type_counts = {}
        for param_name in param_names:
            param_type, _, _ = self._get_param_category(param_name)
            param_type_counts[param_type] = param_type_counts.get(param_type, 0) + 1
        stats["param_type_distribution"] = param_type_counts
        
        return stats
    
    def log_step(
        self, 
        optimizer: torch.optim.Optimizer, 
        step: int,
        model: Optional[torch.nn.Module] = None
    ) -> Optional[str]:
        """
        Log optimizer statistics for current step.
        
        Returns:
            Path to saved statistics file, or None if not logged
        """
        if not self.should_log_step(step):
            return None
            
        try:
            # Analyze optimizer state
            stats = self.analyze_optimizer_state(optimizer, step, model)
            
            # Save to JSON file
            step_file = self.stats_dir / f"step_{step:03d}.json"
            with open(step_file, 'w') as f:
                json.dump(stats, f, indent=2)
            
            # Add to summary data
            summary_row = {
                'step': step,
                'timestamp': stats['timestamp'],
                'total_params': stats['total_params'],
                'exp_avg_sq_min': stats['exp_avg_sq']['global'].get('min', 0),
                'exp_avg_sq_max': stats['exp_avg_sq']['global'].get('max', 0),
                'exp_avg_sq_mean': stats['exp_avg_sq']['global'].get('mean', 0),
                'tiny_1e-12': stats['exp_avg_sq']['global'].get('tiny_counts', {}).get('1e-12', 0),
                'tiny_1e-8': stats['exp_avg_sq']['global'].get('tiny_counts', {}).get('1e-8', 0),
                'zero_count': stats['exp_avg_sq']['global'].get('zero_count', 0)
            }
            self.summary_data.append(summary_row)
            
            # Save summary CSV
            self._save_summary_csv()
            
            self.logger.info(f"[OptimDiag] Step {step}: {stats['total_params']} params, "
                           f"tiny_exp_avg_sq(1e-8): {summary_row['tiny_1e-8']}, "
                           f"saved: {step_file.name}")
                           
            return str(step_file)
            
        except Exception as e:
            self.logger.error(f"Failed to log optimizer stats for step {step}: {e}")
            return None
    
    def _save_summary_csv(self):
        """Save summary statistics as CSV for easy plotting."""
        if not self.summary_data:
            return
            
        import csv
        summary_file = self.stats_dir / "summary.csv"
        
        with open(summary_file, 'w', newline='') as f:
            if self.summary_data:
                writer = csv.DictWriter(f, fieldnames=self.summary_data[0].keys())
                writer.writeheader()
                writer.writerows(self.summary_data)
    
    def get_latest_stats(self) -> Optional[Dict[str, Any]]:
        """Get the most recent statistics."""
        if not self.summary_data:
            return None
        return self.summary_data[-1]