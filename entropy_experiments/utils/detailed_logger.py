"""
Detailed Logger for Offline Entropy Probe

Provides comprehensive logging of entropy probe runs with configurable detail levels.
Captures all relevant quantities, sequence data, and diagnostics for debugging and analysis.
"""

import json
import gzip
import time
import os
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import logging

import torch
import numpy as np
from sequence_processing.sequence_processor import BatchedSequences


class DetailedLogger:
    """
    Comprehensive logging system for entropy probe runs.
    
    Supports multiple logging levels:
    - minimal: Core results + timing only
    - standard: + batch statistics + ground truth results  
    - detailed: + individual sequence data with text
    - debug: + token-level data + raw tensors
    """
    
    def __init__(self, config: Dict[str, Any], base_logger: logging.Logger):
        """
        Initialize the detailed logger.
        
        Args:
            config: Full probe configuration
            base_logger: Base logger instance for status messages
        """
        self.config = config
        self.logger = base_logger
        
        # Extract logging configuration
        self.log_config = config.get('detailed_logging', {})
        self.enabled = self.log_config.get('enabled', False)
        self.level = self.log_config.get('level', 'standard')
        self.log_sequences = self.log_config.get('log_sequences', True)
        self.log_tokens = self.log_config.get('log_tokens', False)
        self.log_raw_tensors = self.log_config.get('log_raw_tensors', False)
        self.output_dir = Path(self.log_config.get('output_directory', 'entropy_experiments/logs'))
        self.compress = self.log_config.get('compress', True)
        
        # Initialize log data structure
        self.log_data = {
            'run_metadata': {},
            'core_results': {},
            'timing': {},
            'ground_truth': {},
            'batch_statistics': {},
            'sequences': {},
            'token_data': {},
            'raw_tensors': {},
            'debug_info': {}
        }
        
        # Timing tracking
        self.phase_timers = {}
        self.start_time = None
        
        # File paths
        self.log_file_path = None
        self.summary_file_path = None
        self.config_file_path = None
    
    def log_run_start(self, checkpoint_path: str, run_config: Dict[str, Any]) -> str:
        """
        Log the start of a probe run and initialize metadata.
        
        Args:
            checkpoint_path: Path to the checkpoint being analyzed
            run_config: Configuration for this run
            
        Returns:
            Path to the main log file
        """
        if not self.enabled:
            return ""
            
        self.start_time = time.time()
        timestamp = datetime.now()
        
        # Create output directory structure
        day_dir = self.output_dir / timestamp.strftime('%Y-%m-%d')
        day_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filenames
        checkpoint_name = Path(checkpoint_path).name.replace('/', '_')
        time_str = timestamp.strftime('%H-%M-%S')
        base_name = f"entropy_probe_{time_str}_{checkpoint_name}"
        
        if self.compress:
            self.log_file_path = day_dir / f"{base_name}.json.gz"
        else:
            self.log_file_path = day_dir / f"{base_name}.json"
            
        self.summary_file_path = day_dir / f"{base_name}_summary.json"
        self.config_file_path = day_dir / f"{base_name}_config.yaml"
        
        # Initialize run metadata
        self.log_data['run_metadata'] = {
            'timestamp': timestamp.isoformat(),
            'checkpoint_path': checkpoint_path,
            'probe_version': '2.0',  # Could be extracted from git or version file
            'config_hash': self._hash_dict(run_config),
            'logging_level': self.level,
            'environment': self._get_environment_info()
        }
        
        # Save configuration file
        self._save_config(run_config)
        
        self.logger.info(f"Detailed logging started (level: {self.level})")
        self.logger.info(f"Log file: {self.log_file_path}")
        
        return str(self.log_file_path)
    
    def log_phase_start(self, phase_name: str) -> None:
        """Start timing a phase."""
        if not self.enabled:
            return
        self.phase_timers[phase_name] = time.time()
    
    def log_phase_end(self, phase_name: str, results: Optional[Dict[str, Any]] = None) -> None:
        """End timing a phase and optionally log results."""
        if not self.enabled:
            return
            
        if phase_name in self.phase_timers:
            phase_time = time.time() - self.phase_timers[phase_name]
            self.log_data['timing'][f"{phase_name}_time"] = phase_time
            del self.phase_timers[phase_name]
            
        if results and self.level in ['detailed', 'debug']:
            self.log_data['debug_info'][f"{phase_name}_results"] = self._sanitize_for_json(results)
    
    def log_core_results(self, results: Dict[str, Any]) -> None:
        """Log core probe results."""
        if not self.enabled:
            return
            
        # Extract core quantities
        core_data = {
            'bars_dot': results.get('bars_dot', 0.0),
            'deltaH1': results.get('deltaH1', 0.0),
            'deltaH_true': results.get('deltaH_true'),
            'learning_rate': results.get('learning_rate', 0.0),
            'B_E': results.get('B_E', 0),
            'B_U': results.get('B_U', 0),
            'G': self.config.get('batch_config', {}).get('G', 1),
            'mb_size_prompts': results.get('mb_size_prompts', 1),
            'weighting_mode': results.get('weighting_mode', 'uniform')
        }
        
        self.log_data['core_results'].update(core_data)
    
    def log_batch_data(self, batch_type: str, batch: Dict[str, Any], 
                      sequences: Optional[BatchedSequences] = None) -> None:
        """
        Log batch data with appropriate detail level.
        
        Args:
            batch_type: 'E_batch' or 'U_batch'
            batch: Batch dictionary from probe
            sequences: Original BatchedSequences object (if available)
        """
        if not self.enabled or self.level == 'minimal':
            return
            
        batch_stats = self._compute_batch_statistics(batch, sequences)
        self.log_data['batch_statistics'][batch_type] = batch_stats
        
        # Log sequence-level data for detailed/debug levels
        if self.level in ['detailed', 'debug'] and self.log_sequences:
            sequence_data = self._extract_sequence_data(batch_type, batch, sequences)
            self.log_data['sequences'][batch_type] = sequence_data
            
        # Log token-level data for debug level
        if self.level == 'debug' and self.log_tokens:
            token_data = self._extract_token_data(batch_type, batch, sequences)
            if token_data:
                self.log_data['token_data'][batch_type] = token_data
                
        # Log raw tensors for debug level  
        if self.level == 'debug' and self.log_raw_tensors:
            tensor_data = self._extract_tensor_data(batch_type, batch)
            if tensor_data:
                self.log_data['raw_tensors'][batch_type] = tensor_data
    
    def log_ground_truth_results(self, gt_results: Dict[str, Any]) -> None:
        """Log ground truth results from DeltaEntropyIS."""
        if not self.enabled or self.level == 'minimal':
            return
            
        # Extract relevant ground truth data
        gt_data = {
            'H_orig': gt_results.get('H_orig'),
            'H_upd': gt_results.get('H_upd'),
            'deltaH_true': gt_results.get('deltaH_true'),
            'diagnostics': gt_results.get('diagnostics', {})
        }
        
        # Add per-token results if available
        if 'H_orig_tok' in gt_results:
            gt_data['H_orig_tok'] = gt_results['H_orig_tok']
            gt_data['H_upd_tok'] = gt_results['H_upd_tok']
            gt_data['deltaH_true_tok'] = gt_results['deltaH_true_tok']
            
        self.log_data['ground_truth'] = gt_data
    
    def log_importance_sampling_details(self, S_orig: torch.Tensor, S_upd: torch.Tensor,
                                      RB_orig: torch.Tensor, RB_upd: torch.Tensor) -> None:
        """Log detailed importance sampling intermediate results."""
        if not self.enabled or self.level not in ['detailed', 'debug']:
            return
            
        logw = S_upd - S_orig
        importance_weights = torch.exp(logw - logw.max())
        
        is_details = {
            'logw_stats': {
                'mean': float(logw.mean().item()),
                'std': float(logw.std().item()),
                'min': float(logw.min().item()),
                'max': float(logw.max().item())
            },
            'weight_stats': {
                'mean': float(importance_weights.mean().item()),
                'std': float(importance_weights.std().item()),
                'min': float(importance_weights.min().item()),
                'max': float(importance_weights.max().item())
            },
            'S_orig_stats': self._tensor_stats(S_orig),
            'S_upd_stats': self._tensor_stats(S_upd),
            'RB_orig_stats': self._tensor_stats(RB_orig),
            'RB_upd_stats': self._tensor_stats(RB_upd)
        }
        
        self.log_data['ground_truth']['importance_sampling_details'] = is_details
        
        # Log raw values for debug level
        if self.level == 'debug' and self.log_raw_tensors:
            self.log_data['raw_tensors']['importance_sampling'] = {
                'S_orig': S_orig.cpu().tolist(),
                'S_upd': S_upd.cpu().tolist(), 
                'RB_orig': RB_orig.cpu().tolist(),
                'RB_upd': RB_upd.cpu().tolist(),
                'logw': logw.cpu().tolist(),
                'importance_weights': importance_weights.cpu().tolist()
            }
    
    def finalize_log(self, final_results: Dict[str, Any]) -> str:
        """Finalize and save the log file."""
        if not self.enabled or not self.log_file_path:
            return ""
            
        # Complete timing information
        if self.start_time:
            self.log_data['timing']['total_time'] = time.time() - self.start_time
            
        # Add final results to core_results
        self.log_core_results(final_results)
        
        # Add timing from final results if available
        if 'timing' in final_results:
            self.log_data['timing'].update(final_results['timing'])
        
        # Save main log file
        self._save_log_file()
        
        # Save summary file
        self._save_summary_file()
        
        self.logger.info(f"Detailed log saved: {self.log_file_path}")
        self.logger.info(f"Summary saved: {self.summary_file_path}")
        
        return str(self.log_file_path)
    
    def _compute_batch_statistics(self, batch: Dict[str, Any], 
                                sequences: Optional[BatchedSequences] = None) -> Dict[str, Any]:
        """Compute statistical summary of a batch."""
        sequences_tensor = batch.get('sequences')
        if sequences_tensor is None:
            return {}
            
        B, G, T = sequences_tensor.shape
        prompt_lens = batch.get('prompt_lens', [])
        
        # Generation lengths
        if 'gen_lengths' in batch:
            gen_lengths = batch['gen_lengths']
        else:
            gen_lengths = self._compute_generation_lengths(batch)
            
        stats = {
            'num_prompts': B,
            'num_sequences': B * G,
            'G': G,
            'avg_prompt_length': float(np.mean(prompt_lens)) if prompt_lens else 0.0,
            'max_sequence_length': T
        }
        
        if gen_lengths is not None:
            if isinstance(gen_lengths, torch.Tensor):
                gen_lengths = gen_lengths.cpu().numpy()
            flat_lengths = gen_lengths.flatten() if len(gen_lengths.shape) > 1 else gen_lengths
            stats.update({
                'avg_generation_length': float(np.mean(flat_lengths)),
                'std_generation_length': float(np.std(flat_lengths)),
                'min_generation_length': float(np.min(flat_lengths)),
                'max_generation_length': float(np.max(flat_lengths)),
                'generation_length_percentiles': [
                    float(np.percentile(flat_lengths, p)) for p in [10, 25, 50, 75, 90]
                ]
            })
        
        # Advantages and rewards (for U_batch)
        if 'advantages' in batch:
            advantages = batch['advantages']
            if isinstance(advantages, torch.Tensor):
                advantages = advantages.cpu().numpy()
            stats.update({
                'avg_advantages': float(np.mean(advantages)),
                'std_advantages': float(np.std(advantages))
            })
            
        if 'rewards' in batch:
            rewards = batch['rewards']
            if isinstance(rewards, torch.Tensor):
                rewards = rewards.cpu().numpy()
            flat_rewards = rewards.flatten() if len(rewards.shape) > 1 else rewards
            stats.update({
                'avg_reward': float(np.mean(flat_rewards)),
                'std_reward': float(np.std(flat_rewards)),
                'min_reward': float(np.min(flat_rewards)),
                'max_reward': float(np.max(flat_rewards))
            })
            
        return stats
    
    def _extract_sequence_data(self, batch_type: str, batch: Dict[str, Any],
                             sequences: Optional[BatchedSequences] = None) -> List[Dict[str, Any]]:
        """Extract individual sequence data with text."""
        # This is a simplified version - would need to be enhanced to extract
        # actual text data from the sequences and integrate with tokenizer
        sequence_data = []
        
        sequences_tensor = batch.get('sequences')
        if sequences_tensor is None:
            return sequence_data
            
        B, G, T = sequences_tensor.shape
        
        # For now, just log metadata - text extraction would require tokenizer integration
        for b in range(min(B, 100)):  # Limit to first 100 for performance
            if batch_type == 'E_batch':
                # E batch has G=1
                sequence_data.append({
                    'prompt_id': b,
                    'prompt_text': f"[PROMPT_{b}]",  # Would extract actual text
                    'response_text': f"[RESPONSE_{b}]",  # Would extract actual text
                    'prompt_length': int(batch.get('prompt_lens', [0])[b] if b < len(batch.get('prompt_lens', [])) else 0),
                    'sequence_logprob': None,  # Would extract from S tensor
                    'rb_entropy': None,  # Would extract from RB tensor
                    'advantage': 0.0
                })
            else:
                # U batch has multiple responses per prompt
                responses = []
                for g in range(G):
                    responses.append({
                        'response_id': g,
                        'response_text': f"[RESPONSE_{b}_{g}]",  # Would extract actual text
                        'sequence_logprob': None,  # Would extract from batch data
                        'reward': None,  # Would extract from rewards tensor
                        'advantage': None  # Would extract from advantages tensor
                    })
                sequence_data.append({
                    'prompt_id': b,
                    'prompt_text': f"[PROMPT_{b}]",  # Would extract actual text
                    'responses': responses
                })
                
        return sequence_data
    
    def _extract_token_data(self, batch_type: str, batch: Dict[str, Any],
                          sequences: Optional[BatchedSequences] = None) -> Optional[List[Dict[str, Any]]]:
        """Extract token-level data (debug level only)."""
        # This would require deeper integration with sequence processing
        # For now, return None - would be implemented in phase 2
        return None
    
    def _extract_tensor_data(self, batch_type: str, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Extract raw tensor data (debug level only)."""
        tensor_data = {}
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and value.numel() < 10000:  # Limit size
                tensor_data[key] = value.cpu().tolist()
        
        return tensor_data
    
    def _compute_generation_lengths(self, batch: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Compute generation lengths from attention masks and prompt lengths."""
        attention_masks = batch.get('attention_masks')
        prompt_lens = batch.get('prompt_lens')
        
        if attention_masks is None or prompt_lens is None:
            return None
            
        B, G, T = attention_masks.shape
        gen_lengths = torch.zeros(B, G, dtype=torch.long)
        
        for b in range(B):
            prompt_len = int(prompt_lens[b])
            for g in range(G):
                gen_lengths[b, g] = attention_masks[b, g, prompt_len:].long().sum()
        
        return gen_lengths
    
    def _tensor_stats(self, tensor: torch.Tensor) -> Dict[str, float]:
        """Compute statistics for a tensor."""
        return {
            'mean': float(tensor.mean().item()),
            'std': float(tensor.std().item()),
            'min': float(tensor.min().item()),
            'max': float(tensor.max().item()),
            'shape': list(tensor.shape)
        }
    
    def _get_environment_info(self) -> Dict[str, Any]:
        """Get environment information."""
        env_info = {
            'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            'torch_version': torch.__version__ if hasattr(torch, '__version__') else 'unknown',
        }
        
        try:
            if torch.cuda.is_available():
                env_info['cuda_available'] = True
                env_info['gpu_count'] = torch.cuda.device_count()
                if torch.cuda.device_count() > 0:
                    env_info['gpu_name'] = torch.cuda.get_device_name(0)
                    env_info['gpu_memory_gb'] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
            else:
                env_info['cuda_available'] = False
        except:
            env_info['cuda_available'] = False
            
        return env_info
    
    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Create hash of dictionary for config fingerprinting."""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]
    
    def _sanitize_for_json(self, obj: Any) -> Any:
        """Sanitize object for JSON serialization."""
        if isinstance(obj, torch.Tensor):
            return obj.cpu().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: self._sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize_for_json(v) for v in obj]
        else:
            return obj
    
    def _save_config(self, config: Dict[str, Any]) -> None:
        """Save configuration to YAML file."""
        try:
            import yaml
            with open(self.config_file_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        except Exception as e:
            self.logger.warning(f"Could not save config file: {e}")
    
    def _save_log_file(self) -> None:
        """Save the main log file."""
        sanitized_data = self._sanitize_for_json(self.log_data)
        
        if self.compress:
            with gzip.open(self.log_file_path, 'wt', encoding='utf-8') as f:
                json.dump(sanitized_data, f, indent=2)
        else:
            with open(self.log_file_path, 'w') as f:
                json.dump(sanitized_data, f, indent=2)
    
    def _save_summary_file(self) -> None:
        """Save a summary file with just core results."""
        summary = {
            'run_metadata': self.log_data['run_metadata'],
            'core_results': self.log_data['core_results'],
            'timing': self.log_data['timing'],
            'ground_truth': {
                'H_orig': self.log_data['ground_truth'].get('H_orig'),
                'H_upd': self.log_data['ground_truth'].get('H_upd'),
                'deltaH_true': self.log_data['ground_truth'].get('deltaH_true'),
                'ESS': self.log_data['ground_truth'].get('diagnostics', {}).get('ESS')
            },
            'batch_statistics': self.log_data['batch_statistics']
        }
        
        with open(self.summary_file_path, 'w') as f:
            json.dump(self._sanitize_for_json(summary), f, indent=2)