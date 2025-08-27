#!/usr/bin/env python3
"""
🎯 Unified Entropy Probe Test Runner

Replaces ALL 11 overlapping test scripts with a single, clean interface.
Supports all test types: batch scaling, sanity checks, single runs, debugging.

Features:
- Auto-categorized results organization
- Flexible logging modes (verbose, minimal, quiet)  
- Config templates with B_E_values lists
- NO timeouts, graceful error handling
- Complete debug traceability

Usage:
    # Batch scaling test (replaces batch_size_convergence_*, flexible_batch_*)
    python run_entropy_test.py batch-scaling \\
      --checkpoint /path/to/checkpoint \\
      --be-values 16,32,64 --runs 3 --verbose

    # Sanity check (replaces run_probe_sanity_check.py)
    python run_entropy_test.py sanity-check \\
      --checkpoint /path/to/checkpoint --runs 15

    # Single test (clean individual runs)
    python run_entropy_test.py single \\
      --checkpoint /path/to/checkpoint \\
      --be 256 --bu 32 --minimal

    # Debug mode (replaces debug_batch_scaling_test.py etc.)
    python run_entropy_test.py debug \\
      --checkpoint /path/to/checkpoint \\
      --be-values 16,32 --verbose --save-gradients
"""

import os
import sys
import yaml
import json
import time
import logging
import traceback
import argparse
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from entropy_experiments.offline_entropy_probe import OfflineEntropyProbe


class EntropyTestRunner:
    """Unified entropy probe test runner with auto-organized results."""
    
    def __init__(self, test_type: str, checkpoint_path: str, 
                 log_level: str = "INFO", save_gradients: bool = False):
        self.test_type = test_type
        self.checkpoint_path = checkpoint_path
        self.log_level = log_level
        self.save_gradients = save_gradients
        
        # Auto-create results directory structure
        self.results_base = Path("entropy_experiments/results")
        self.exp_dir = self._create_experiment_directory()
        
        # Setup logging
        self.logger = self._setup_logging()
        
    def _create_experiment_directory(self) -> Path:
        """Create auto-categorized experiment directory."""
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        # Auto-categorize by test type
        category_dir = self.results_base / self.test_type.replace('-', '_')
        category_dir.mkdir(parents=True, exist_ok=True)
        
        # Create descriptive experiment name
        if self.test_type == "batch-scaling":
            exp_name = f"δH₁_convergence_{timestamp}"
        elif self.test_type == "sanity-check":
            exp_name = f"validation_{timestamp}"
        elif self.test_type == "single":
            exp_name = f"single_run_{timestamp}"
        elif self.test_type == "debug":
            exp_name = f"debug_analysis_{timestamp}"
        else:
            exp_name = f"experiment_{timestamp}"
            
        exp_dir = category_dir / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        return exp_dir
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the test runner."""
        logger = logging.getLogger("entropy_test_runner")
        logger.setLevel(getattr(logging, self.log_level))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    def _setup_run_directory(self, run_name: str) -> Path:
        """Create directory for individual run."""
        run_dir = self.exp_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
        
    def _create_run_config(self, template_config: Dict, run_dir: Path, 
                          B_E: int, B_U: int = 32) -> str:
        """Create config file for specific run."""
        config = template_config.copy()
        
        # Update batch sizes (remove confusing "B" entirely!)
        config['batch_config']['B_E_values'] = [B_E]  # Always use list format
        config['batch_config']['B_U'] = B_U
        
        # Remove old confusing "B" parameter if it exists
        if 'B' in config['batch_config']:
            del config['batch_config']['B']
            
        # Set debug logging level
        config['output'] = config.get('output', {})
        config['output']['log_level'] = 'DEBUG' if self.log_level == 'DEBUG' else 'INFO'
        config['output']['save_results'] = True
        
        # Conditional variance focus (the new estimator we're testing)
        config['probe_rework']['compute_conditional_variance'] = True
        config['probe_rework']['compute_vx_vy_variance'] = False
        config['probe_rework']['compute_importance_sampling'] = False
        
        # Save config to run directory
        config_path = run_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
        return str(config_path)
        
    def _setup_probe_logging(self, probe: OfflineEntropyProbe, log_file: Path):
        """Add FileHandler to probe's logger."""
        file_handler = logging.FileHandler(log_file, mode='w')
        formatter = logging.Formatter(
            f'[Rank {probe.rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        probe.logger.addHandler(file_handler)
        return file_handler
        
    def _run_single_probe(self, run_dir: Path, config_path: str) -> Dict[str, Any]:
        """Run single probe with full logging and error handling."""
        log_file = run_dir / 'probe_log.txt'
        results_file = run_dir / 'results.json'
        error_file = run_dir / 'error.txt'
        
        start_time = time.time()
        
        try:
            # Load config and set checkpoint paths
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            # Setup LoRA checkpoint paths
            model_path = self.checkpoint_path + "/model" if not self.checkpoint_path.endswith("/model") else self.checkpoint_path
            optimizer_path = self.checkpoint_path.replace("/model", "/optimizer.pt") if self.checkpoint_path.endswith("/model") else self.checkpoint_path + "/optimizer.pt"
            
            config['checkpoint']['checkpoint_path'] = model_path
            config['checkpoint']['optimizer_path'] = optimizer_path
            
            # Create and run probe
            probe = OfflineEntropyProbe(config)
            
            # Setup logging capture
            file_handler = None
            if self.log_level in ['DEBUG', 'INFO']:
                file_handler = self._setup_probe_logging(probe, log_file)
                
            try:
                # Run probe
                results = probe.run_mixed_probe()
                runtime = time.time() - start_time
                
                # Manually save results.json (run_mixed_probe doesn't auto-save)
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2)
                    
                return {
                    'success': True,
                    'runtime': runtime,
                    'deltaH1': results.get('deltaH1'),
                    'SE_conditional': results.get('SE_conditional'),
                    'B_E': results.get('B_E'),
                    'B_U': results.get('B_U'),
                    'bars_dot': results.get('bars_dot'),
                    'full_results': results,
                    'files': {
                        'config': str(config_path),
                        'log': str(log_file) if log_file.exists() else None,
                        'results': str(results_file)
                    }
                }
                
            finally:
                if file_handler:
                    probe.logger.removeHandler(file_handler)
                    file_handler.close()
                    
        except Exception as e:
            runtime = time.time() - start_time
            error_msg = f"Error: {str(e)}\\n\\nTraceback:\\n{traceback.format_exc()}"
            
            with open(error_file, 'w') as f:
                f.write(error_msg)
                
            return {
                'success': False,
                'runtime': runtime,
                'error': str(e),
                'files': {'error': str(error_file)}
            }
            
    def batch_scaling_test(self, B_E_values: List[int], B_U: int = 32, 
                          runs_per_batch: int = 3, template_config: Dict = None) -> Dict:
        """Run batch scaling convergence test."""
        self.logger.info(f"🔍 BATCH SCALING TEST: δH₁ Convergence Analysis")
        self.logger.info(f"B_E values: {B_E_values}, B_U: {B_U}, runs per batch: {runs_per_batch}")
        self.logger.info(f"Experiment: {self.exp_dir}")
        
        all_results = []
        start_time = time.time()
        
        for B_E in B_E_values:
            self.logger.info(f"\\n{'='*50}")
            self.logger.info(f"TESTING B_E = {B_E}")
            self.logger.info(f"{'='*50}")
            
            for run_id in range(1, runs_per_batch + 1):
                run_name = f"BE{B_E:03d}_run{run_id:03d}"
                run_dir = self._setup_run_directory(run_name)
                
                # Create config for this run
                config_path = self._create_run_config(template_config, run_dir, B_E, B_U)
                
                if self.log_level != 'QUIET':
                    self.logger.info(f"  Running {run_name}...")
                    
                # Run probe
                result = self._run_single_probe(run_dir, config_path)
                result.update({
                    'B_E_target': B_E,
                    'B_U_target': B_U,
                    'run_id': run_id,
                    'run_name': run_name
                })
                
                all_results.append(result)
                
                # Print immediate results
                if result['success'] and self.log_level != 'QUIET':
                    delta_h1 = result.get('deltaH1')
                    se_cond = result.get('SE_conditional')
                    runtime = result.get('runtime', 0)
                    
                    self.logger.info(f"    ✅ δH₁={delta_h1:.6f}, SE_conditional={se_cond:.6f}, time={runtime:.1f}s")
                    
                    if se_cond and delta_h1 and abs(delta_h1) > 0:
                        ratio = abs(se_cond / delta_h1)
                        status = "✅ Good" if ratio < 0.1 else "⚠️ OK" if ratio < 0.2 else "❌ Poor"
                        self.logger.info(f"         Signal/noise |SE/δH₁|={ratio:.3f} {status}")
                        
                elif not result['success']:
                    self.logger.warning(f"    ❌ {run_name} failed: {result.get('error', 'Unknown error')}")
        
        # Analyze convergence patterns
        analysis = self._analyze_batch_scaling(all_results, B_E_values)
        
        # Save comprehensive results
        summary = {
            'test_config': {
                'test_type': 'batch_scaling',
                'B_E_values': B_E_values,
                'B_U': B_U,
                'runs_per_batch': runs_per_batch,
                'checkpoint': self.checkpoint_path,
                'total_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat(),
                'experiment_directory': str(self.exp_dir)
            },
            'analysis': analysis,
            'all_results': all_results
        }
        
        summary_file = self.exp_dir / 'convergence_analysis.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        # Print final analysis
        if self.log_level != 'QUIET':
            self._print_scaling_analysis(analysis, B_E_values)
            
        self.logger.info(f"\\n📁 Results: {self.exp_dir}")
        self.logger.info(f"📄 Analysis: {summary_file}")
        
        return summary
        
    def sanity_check_test(self, runs: int = 15, B_E: int = 256, B_U: int = 32,
                         template_config: Dict = None) -> Dict:
        """Run sanity check validation test."""
        self.logger.info(f"🧪 SANITY CHECK: Validation Test")
        self.logger.info(f"Runs: {runs}, B_E: {B_E}, B_U: {B_U}")
        self.logger.info(f"Experiment: {self.exp_dir}")
        
        all_results = []
        start_time = time.time()
        
        for run_id in range(1, runs + 1):
            run_name = f"run{run_id:03d}"
            run_dir = self._setup_run_directory(run_name)
            
            config_path = self._create_run_config(template_config, run_dir, B_E, B_U)
            
            if self.log_level != 'QUIET':
                self.logger.info(f"  Running validation {run_id}/{runs}...")
                
            result = self._run_single_probe(run_dir, config_path)
            result.update({
                'run_id': run_id,
                'run_name': run_name
            })
            
            all_results.append(result)
            
            if result['success'] and self.log_level not in ['QUIET', 'MINIMAL']:
                delta_h1 = result.get('deltaH1')
                se_cond = result.get('SE_conditional')
                runtime = result.get('runtime', 0)
                self.logger.info(f"    δH₁={delta_h1:.6f}, SE={se_cond:.6f}, time={runtime:.1f}s")
                
        # Analyze validation results
        analysis = self._analyze_sanity_check(all_results)
        
        summary = {
            'test_config': {
                'test_type': 'sanity_check',
                'runs': runs,
                'B_E': B_E,
                'B_U': B_U,
                'checkpoint': self.checkpoint_path,
                'total_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat(),
                'experiment_directory': str(self.exp_dir)
            },
            'analysis': analysis,
            'all_results': all_results
        }
        
        summary_file = self.exp_dir / 'validation_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        if self.log_level != 'QUIET':
            self._print_sanity_analysis(analysis)
            
        self.logger.info(f"\\n📁 Results: {self.exp_dir}")
        
        return summary
        
    def single_test(self, B_E: int, B_U: int = 32, template_config: Dict = None) -> Dict:
        """Run single entropy probe test."""
        self.logger.info(f"🎯 SINGLE TEST: B_E={B_E}, B_U={B_U}")
        self.logger.info(f"Experiment: {self.exp_dir}")
        
        run_name = f"BE{B_E:03d}_BU{B_U:03d}"
        run_dir = self._setup_run_directory(run_name)
        
        config_path = self._create_run_config(template_config, run_dir, B_E, B_U)
        
        start_time = time.time()
        result = self._run_single_probe(run_dir, config_path)
        
        if result['success']:
            delta_h1 = result.get('deltaH1')
            se_cond = result.get('SE_conditional')
            runtime = result.get('runtime', 0)
            
            self.logger.info(f"✅ δH₁ = {delta_h1:.6f}")
            self.logger.info(f"   SE_conditional = {se_cond:.6f}")
            self.logger.info(f"   Runtime: {runtime:.1f}s")
            
            if se_cond and delta_h1 and abs(delta_h1) > 0:
                ratio = abs(se_cond / delta_h1)
                self.logger.info(f"   |SE/δH₁| = {ratio:.3f}")
        else:
            self.logger.error(f"❌ Test failed: {result.get('error')}")
            
        summary = {
            'test_config': {
                'test_type': 'single',
                'B_E': B_E,
                'B_U': B_U,
                'checkpoint': self.checkpoint_path,
                'total_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            },
            'result': result
        }
        
        summary_file = self.exp_dir / 'single_test_results.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        self.logger.info(f"\\n📁 Results: {self.exp_dir}")
        
        return summary
        
    def debug_test(self, B_E_values: List[int], B_U: int = 32, 
                  template_config: Dict = None) -> Dict:
        """Run debug analysis with detailed logging."""
        self.logger.info(f"🔍 DEBUG MODE: Detailed Analysis")
        self.logger.info(f"B_E values: {B_E_values}, B_U: {B_U}")
        self.logger.info(f"Save gradients: {self.save_gradients}")
        
        # Debug mode always uses verbose logging
        self.log_level = 'DEBUG'
        
        # Run with detailed logging for each B_E
        all_results = []
        
        for B_E in B_E_values:
            run_name = f"debug_BE{B_E:03d}"
            run_dir = self._setup_run_directory(run_name)
            
            config_path = self._create_run_config(template_config, run_dir, B_E, B_U)
            
            self.logger.info(f"\\n🔍 Debug analysis for B_E={B_E}")
            
            result = self._run_single_probe(run_dir, config_path)
            result['B_E_target'] = B_E
            all_results.append(result)
            
            if self.save_gradients:
                # TODO: Add gradient saving functionality
                pass
                
        summary = {
            'test_config': {
                'test_type': 'debug',
                'B_E_values': B_E_values,
                'B_U': B_U,
                'save_gradients': self.save_gradients,
                'checkpoint': self.checkpoint_path,
                'timestamp': datetime.now().isoformat()
            },
            'all_results': all_results
        }
        
        summary_file = self.exp_dir / 'debug_report.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        self.logger.info(f"\\n📁 Debug results: {self.exp_dir}")
        
        return summary
        
    def _analyze_batch_scaling(self, results: List[Dict], B_E_values: List[int]) -> Dict:
        """Analyze batch scaling convergence patterns."""
        # Group by B_E
        by_B_E = {}
        for result in results:
            if not result.get('success'):
                continue
            B_E = result.get('B_E_target')
            if B_E not in by_B_E:
                by_B_E[B_E] = []
            by_B_E[B_E].append(result)
            
        # Compute statistics
        summary = {}
        for B_E in B_E_values:
            if B_E not in by_B_E:
                continue
                
            runs = by_B_E[B_E]
            delta_h1_values = [r['deltaH1'] for r in runs if r.get('deltaH1') is not None]
            se_values = [r['SE_conditional'] for r in runs if r.get('SE_conditional') is not None]
            
            if delta_h1_values:
                mean_delta = sum(delta_h1_values) / len(delta_h1_values)
                std_delta = (sum((x - mean_delta)**2 for x in delta_h1_values) / max(len(delta_h1_values) - 1, 1))**0.5
                
                summary[B_E] = {
                    'successful_runs': len(runs),
                    'mean_deltaH1': mean_delta,
                    'std_deltaH1': std_delta,
                    'delta_values': delta_h1_values,
                    'mean_SE_conditional': sum(se_values) / len(se_values) if se_values else None,
                }
                
        # Analyze scaling patterns
        scaling_analysis = {}
        if len(summary) >= 2:
            sorted_B_E = sorted(summary.keys())
            ref_B_E = sorted_B_E[0]
            ref_mean = summary[ref_B_E]['mean_deltaH1']
            
            for B_E in sorted_B_E[1:]:
                curr_mean = summary[B_E]['mean_deltaH1']
                if ref_mean != 0:
                    ratio = curr_mean / ref_mean
                    expected_1_over_B_E = ref_B_E / B_E
                    
                    scaling_analysis[f"B_E_{ref_B_E}_to_{B_E}"] = {
                        'observed_ratio': ratio,
                        'expected_if_1_over_B_E': expected_1_over_B_E,
                        'suggests_convergence': abs(ratio - 1.0) < 0.1,
                        'suggests_1_over_B_E_scaling': abs(ratio - expected_1_over_B_E) < 0.1
                    }
                    
        return {
            'by_batch_size': summary,
            'scaling_analysis': scaling_analysis,
            'total_successful_runs': sum(s['successful_runs'] for s in summary.values())
        }
        
    def _analyze_sanity_check(self, results: List[Dict]) -> Dict:
        """Analyze sanity check validation results."""
        successful = [r for r in results if r.get('success')]
        
        if not successful:
            return {'error': 'No successful runs'}
            
        delta_h1_values = [r['deltaH1'] for r in successful if r.get('deltaH1') is not None]
        se_values = [r['SE_conditional'] for r in successful if r.get('SE_conditional') is not None]
        
        if not delta_h1_values:
            return {'error': 'No valid δH₁ values'}
            
        mean_delta = sum(delta_h1_values) / len(delta_h1_values)
        std_delta = (sum((x - mean_delta)**2 for x in delta_h1_values) / max(len(delta_h1_values) - 1, 1))**0.5
        
        analysis = {
            'successful_runs': len(successful),
            'total_runs': len(results),
            'success_rate': len(successful) / len(results),
            'deltaH1_stats': {
                'mean': mean_delta,
                'std': std_delta,
                'min': min(delta_h1_values),
                'max': max(delta_h1_values),
                'values': delta_h1_values
            }
        }
        
        if se_values:
            mean_se = sum(se_values) / len(se_values)
            analysis['SE_conditional_stats'] = {
                'mean': mean_se,
                'mean_signal_to_noise': abs(mean_se / mean_delta) if mean_delta != 0 else None
            }
            
        return analysis
        
    def _print_scaling_analysis(self, analysis: Dict, B_E_values: List[int]):
        """Print batch scaling analysis."""
        self.logger.info(f"\\n{'='*60}")
        self.logger.info("📊 CONVERGENCE ANALYSIS") 
        self.logger.info(f"{'='*60}")
        
        successful_runs = analysis.get('total_successful_runs', 0)
        total_runs = len(B_E_values) * 3  # Assuming 3 runs per batch
        self.logger.info(f"Successful runs: {successful_runs}/{total_runs}")
        
        # Print per-batch statistics
        for B_E in B_E_values:
            if B_E in analysis['by_batch_size']:
                stats = analysis['by_batch_size'][B_E]
                self.logger.info(f"\\nB_E = {B_E}:")
                self.logger.info(f"  Mean δH₁: {stats['mean_deltaH1']:.6f}")
                self.logger.info(f"  Std δH₁: {stats['std_deltaH1']:.6f}")
                if stats.get('mean_SE_conditional'):
                    mean_se = stats['mean_SE_conditional']
                    mean_delta = stats['mean_deltaH1']
                    if abs(mean_delta) > 0:
                        snr = abs(mean_se / mean_delta)
                        self.logger.info(f"  Mean |SE/δH₁|: {snr:.3f}")
                        
        # Print scaling analysis
        if analysis.get('scaling_analysis'):
            self.logger.info(f"\\n🔍 Scaling Analysis:")
            for comparison, data in analysis['scaling_analysis'].items():
                self.logger.info(f"  {comparison}:")
                self.logger.info(f"    Observed ratio: {data['observed_ratio']:.3f}")
                self.logger.info(f"    Expected if 1/B_E: {data['expected_if_1_over_B_E']:.3f}")
                if data['suggests_convergence']:
                    self.logger.info(f"    → ✅ Suggests CONVERGENCE (good!)")
                elif data['suggests_1_over_B_E_scaling']:
                    self.logger.info(f"    → ❌ Suggests 1/B_E scaling (potential bug)")
                else:
                    self.logger.info(f"    → ⚠️ Other behavior")
                    
    def _print_sanity_analysis(self, analysis: Dict):
        """Print sanity check analysis."""
        self.logger.info(f"\\n{'='*60}")
        self.logger.info("📊 SANITY CHECK ANALYSIS")
        self.logger.info(f"{'='*60}")
        
        self.logger.info(f"Success rate: {analysis['success_rate']:.1%} ({analysis['successful_runs']}/{analysis['total_runs']})")
        
        if 'deltaH1_stats' in analysis:
            stats = analysis['deltaH1_stats']
            self.logger.info(f"δH₁ statistics:")
            self.logger.info(f"  Mean: {stats['mean']:.6f}")
            self.logger.info(f"  Std: {stats['std']:.6f}")
            self.logger.info(f"  Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
            
        if 'SE_conditional_stats' in analysis:
            se_stats = analysis['SE_conditional_stats'] 
            if se_stats.get('mean_signal_to_noise'):
                snr = se_stats['mean_signal_to_noise']
                status = "✅ Good" if snr < 0.1 else "⚠️ OK" if snr < 0.2 else "❌ Poor"
                self.logger.info(f"Mean |SE/δH₁|: {snr:.3f} {status}")


def load_config_template(config_path: str) -> Dict:
    """Load configuration template."""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        # Return minimal default config
        return {
            'batch_config': {
                'B_E_values': [256],
                'B_U': 32,
                'G': 8,
                'rollout_batch_size': 32,
                'dataset_name': 'gsm8k_r1_template',
                'split': 'train'
            },
            'probe_config': {'mode': 'exact', 'M': None},
            'probe_rework': {
                'mb_size_prompts': 2,
                'buffers_dtype': 'float32',
                'weighting_mode': 'dr_grpo',
                'conditional_variance_batch_size': 6,
                'compute_conditional_variance': True,
                'compute_vx_vy_variance': False,
                'compute_importance_sampling': False,
                'ddp_allreduce': False,
                'master_seed': 42
            },
            'distributed': {'find_unused_parameters': False, 'reduce_dtype': 'float32', 'barriers': False},
            'importance': {'enabled': False},
            'importance_sampling': {'use_snis': True, 'use_psis': False, 'ess_threshold': 0.5, 'resample_on_low_ess': False},
            'stats_config': {'compute_plugin_se': True, 'compute_jackknife_se': False},
            'memory_config': {'microbatch_size': 2, 'teacher_force_microbatch_size': 2, 'amp': True, 'dtype': 'bfloat16'},
            'learning_rate': 2e-6,
            'checkpoint': {'checkpoint_path': '', 'optimizer_path': '', 'model_config_path': 'Qwen/Qwen2.5-1.5B'},
            'generation': {'max_new_tokens': 50, 'temperature': 0.7, 'top_p': 1.0, 'do_sample': True, 'pad_token_id': None},
            'output': {'save_results': True, 'results_path': '', 'log_level': 'INFO', 'save_samples': False}
        }


def main():
    parser = argparse.ArgumentParser(description="🎯 Unified Entropy Probe Test Runner")
    
    # Test type (required positional argument)
    parser.add_argument("test_type", choices=['batch-scaling', 'sanity-check', 'single', 'debug'],
                       help="Type of test to run")
    
    # Common arguments
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint directory")
    parser.add_argument("--config", default=None, help="Configuration template file")
    
    # Logging control
    parser.add_argument("--verbose", action="store_true", help="Enable verbose DEBUG logging")
    parser.add_argument("--minimal", action="store_true", help="Minimal INFO logging")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode - final results only")
    
    # Batch scaling arguments
    parser.add_argument("--be-values", default="16,32,64", help="Comma-separated B_E values (default: 16,32,64)")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per batch size (default: 3)")
    
    # Single test arguments
    parser.add_argument("--be", type=int, default=256, help="B_E value for single test (default: 256)")
    parser.add_argument("--bu", type=int, default=32, help="B_U value (default: 32)")
    
    # Debug arguments
    parser.add_argument("--save-gradients", action="store_true", help="Save gradient information")
    
    args = parser.parse_args()
    
    # Determine log level
    if args.verbose:
        log_level = "DEBUG"
    elif args.minimal:
        log_level = "INFO"
    elif args.quiet:
        log_level = "QUIET"
    else:
        log_level = "INFO"
        
    # Validate checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint path not found: {args.checkpoint}")
        return 1
        
    # Load config template
    if args.config and os.path.exists(args.config):
        template_config = load_config_template(args.config)
    else:
        # Use default config template
        template_config = load_config_template("")
        
    try:
        # Create runner
        runner = EntropyTestRunner(
            test_type=args.test_type,
            checkpoint_path=args.checkpoint,
            log_level=log_level,
            save_gradients=args.save_gradients
        )
        
        # Run appropriate test
        if args.test_type == "batch-scaling":
            B_E_values = [int(x.strip()) for x in args.be_values.split(',')]
            result = runner.batch_scaling_test(
                B_E_values=B_E_values,
                B_U=args.bu,
                runs_per_batch=args.runs,
                template_config=template_config
            )
            
        elif args.test_type == "sanity-check":
            result = runner.sanity_check_test(
                runs=args.runs,
                B_E=args.be,
                B_U=args.bu,
                template_config=template_config
            )
            
        elif args.test_type == "single":
            result = runner.single_test(
                B_E=args.be,
                B_U=args.bu,
                template_config=template_config
            )
            
        elif args.test_type == "debug":
            B_E_values = [int(x.strip()) for x in args.be_values.split(',')]
            result = runner.debug_test(
                B_E_values=B_E_values,
                B_U=args.bu,
                template_config=template_config
            )
            
        return 0
        
    except KeyboardInterrupt:
        print("\\n⚠️ Test interrupted by user")
        return 130
    except Exception as e:
        print(f"\\n❌ Test failed: {e}")
        if log_level == "DEBUG":
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())