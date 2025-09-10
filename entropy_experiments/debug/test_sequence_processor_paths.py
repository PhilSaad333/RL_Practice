#!/usr/bin/env python3
"""
Test script to compare no-grad vs with-grad teacher forcing paths in SequenceProcessor.

This script ensures that both paths produce numerically identical results when
given the same input sequences, which is critical for entropy measurement reliability.
"""

import os
import sys
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from entropy_experiments.entropy_experiment_runner import EntropyMeasurements
from entropy_experiments.utils.sequence_processor import BatchedSequences


def setup_logging() -> logging.Logger:
    """Setup logging for the test script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_test_config() -> Dict[str, Any]:
    """Load and modify config for testing."""
    config_path = project_root / "entropy_experiments" / "configs" / "config_template.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override batch sizes for testing
    config['batch_config']['B_E'] = 8  # 8 prompts  
    config['batch_config']['B_U'] = 8  # Not used in this test but keeping consistent
    config['batch_config']['G'] = 1   # 1 response per prompt
    
    # Ensure deterministic generation
    config['generation']['temperature'] = 1.0
    config['generation']['top_p'] = 1.0
    config['generation']['max_new_tokens'] = 50  # Keep generations short for testing
    
    return config


def detach_logprob_results(results):
    """
    Detach tensors from LogprobResults for numerical comparison.
    
    The with-grad path returns tensors with gradients attached. We need to detach
    them and move to CPU for fair comparison with the no-grad path.
    """
    # Handle case where results might be a tuple (from teacher_force_logprobs_with_diagnostics)
    if isinstance(results, tuple):
        logprob_results = results[0]  # First element is LogprobResults
    else:
        logprob_results = results  # Direct LogprobResults
    
    detached_results = type(logprob_results)(
        logprobs=[],
        entropies=logprob_results.entropies,  # These are numpy arrays, already detached
        sequence_logprobs=logprob_results.sequence_logprobs,  # These are Python floats
        rb_entropies=logprob_results.rb_entropies,  # These are numpy arrays  
        rewards=getattr(logprob_results, 'rewards', []),  # Skip rewards as requested
        rb_entropies_torch=None,  # We don't compare the torch version
        baseline_feats_torch=None,
        token_logqs=None,  # We'll handle this separately if needed
        sequence_logqs=getattr(logprob_results, 'sequence_logqs', [])
    )
    
    # Detach logprobs: List[List[Tensor]] -> List[List[Tensor]] (detached, CPU)
    for b_results in logprob_results.logprobs:
        detached_b = []
        for g_tensor in b_results:
            if torch.is_tensor(g_tensor):
                detached_b.append(g_tensor.detach().cpu())
            else:
                detached_b.append(g_tensor)  # Should not happen, but be safe
        detached_results.logprobs.append(detached_b)
    
    return detached_results


def compare_sequence_logprobs(no_grad_seq_lp: List[List[float]], 
                            with_grad_seq_lp: List[List[float]], 
                            tolerance: float = 1e-6) -> Dict[str, Any]:
    """Compare sequence-level logprobs between the two paths."""
    results = {
        'match': True,
        'max_diff': 0.0,
        'mean_abs_diff': 0.0,
        'mismatches': []
    }
    
    if len(no_grad_seq_lp) != len(with_grad_seq_lp):
        results['match'] = False
        results['mismatches'].append(f"Different batch sizes: {len(no_grad_seq_lp)} vs {len(with_grad_seq_lp)}")
        return results
    
    all_diffs = []
    for b in range(len(no_grad_seq_lp)):
        if len(no_grad_seq_lp[b]) != len(with_grad_seq_lp[b]):
            results['match'] = False
            results['mismatches'].append(f"Batch {b}: Different G sizes: {len(no_grad_seq_lp[b])} vs {len(with_grad_seq_lp[b])}")
            continue
            
        for g in range(len(no_grad_seq_lp[b])):
            val_no_grad = float(no_grad_seq_lp[b][g])
            val_with_grad = float(with_grad_seq_lp[b][g])
            diff = abs(val_no_grad - val_with_grad)
            all_diffs.append(diff)
            
            if diff > tolerance:
                results['match'] = False
                results['mismatches'].append(
                    f"Batch {b}, G {g}: {val_no_grad:.8e} vs {val_with_grad:.8e} (diff: {diff:.8e})"
                )
    
    if all_diffs:
        results['max_diff'] = max(all_diffs)
        results['mean_abs_diff'] = sum(all_diffs) / len(all_diffs)
    
    return results


def compare_per_token_logprobs(no_grad_logprobs: List[List[torch.Tensor]], 
                              with_grad_logprobs: List[List[torch.Tensor]], 
                              tolerance: float = 1e-6) -> Dict[str, Any]:
    """Compare per-token logprobs between the two paths."""
    results = {
        'match': True,
        'max_diff': 0.0,
        'mean_abs_diff': 0.0,
        'mismatches': []
    }
    
    if len(no_grad_logprobs) != len(with_grad_logprobs):
        results['match'] = False
        results['mismatches'].append(f"Different batch sizes: {len(no_grad_logprobs)} vs {len(with_grad_logprobs)}")
        return results
    
    all_diffs = []
    for b in range(len(no_grad_logprobs)):
        if len(no_grad_logprobs[b]) != len(with_grad_logprobs[b]):
            results['match'] = False
            results['mismatches'].append(f"Batch {b}: Different G sizes")
            continue
            
        for g in range(len(no_grad_logprobs[b])):
            tensor_no_grad = no_grad_logprobs[b][g]
            tensor_with_grad = with_grad_logprobs[b][g]
            
            if tensor_no_grad.shape != tensor_with_grad.shape:
                results['match'] = False
                results['mismatches'].append(
                    f"Batch {b}, G {g}: Shape mismatch {tensor_no_grad.shape} vs {tensor_with_grad.shape}"
                )
                continue
            
            # Compute element-wise differences
            diff_tensor = torch.abs(tensor_no_grad - tensor_with_grad)
            max_diff = torch.max(diff_tensor).item()
            mean_diff = torch.mean(diff_tensor).item()
            
            all_diffs.extend(diff_tensor.flatten().tolist())
            
            if max_diff > tolerance:
                results['match'] = False
                results['mismatches'].append(
                    f"Batch {b}, G {g}: Max token diff {max_diff:.8e} > tolerance {tolerance:.8e}"
                )
    
    if all_diffs:
        results['max_diff'] = max(all_diffs)
        results['mean_abs_diff'] = sum(all_diffs) / len(all_diffs)
    
    return results


def compare_entropies(no_grad_entropies: List[List[np.ndarray]], 
                     with_grad_entropies: List[List[np.ndarray]], 
                     tolerance: float = 1e-6) -> Dict[str, Any]:
    """Compare entropy arrays between the two paths."""
    results = {
        'match': True,
        'max_diff': 0.0,
        'mean_abs_diff': 0.0,
        'mismatches': []
    }
    
    if len(no_grad_entropies) != len(with_grad_entropies):
        results['match'] = False
        results['mismatches'].append(f"Different batch sizes: {len(no_grad_entropies)} vs {len(with_grad_entropies)}")
        return results
    
    all_diffs = []
    for b in range(len(no_grad_entropies)):
        if len(no_grad_entropies[b]) != len(with_grad_entropies[b]):
            results['match'] = False
            results['mismatches'].append(f"Batch {b}: Different G sizes")
            continue
            
        for g in range(len(no_grad_entropies[b])):
            arr_no_grad = no_grad_entropies[b][g]
            arr_with_grad = with_grad_entropies[b][g]
            
            if arr_no_grad.shape != arr_with_grad.shape:
                results['match'] = False
                results['mismatches'].append(
                    f"Batch {b}, G {g}: Shape mismatch {arr_no_grad.shape} vs {arr_with_grad.shape}"
                )
                continue
            
            # Compute element-wise differences
            diff_arr = np.abs(arr_no_grad - arr_with_grad)
            max_diff = np.max(diff_arr)
            mean_diff = np.mean(diff_arr)
            
            all_diffs.extend(diff_arr.flatten().tolist())
            
            if max_diff > tolerance:
                results['match'] = False
                results['mismatches'].append(
                    f"Batch {b}, G {g}: Max entropy diff {max_diff:.8e} > tolerance {tolerance:.8e}"
                )
    
    if all_diffs:
        results['max_diff'] = max(all_diffs)
        results['mean_abs_diff'] = sum(all_diffs) / len(all_diffs)
    
    return results


def compare_rb_entropies(no_grad_rb: List[List[np.ndarray]], 
                        with_grad_rb: List[List[np.ndarray]], 
                        tolerance: float = 1e-6) -> Dict[str, Any]:
    """Compare RB entropy arrays between the two paths."""
    # This is identical to compare_entropies, but keeping separate for clarity
    return compare_entropies(no_grad_rb, with_grad_rb, tolerance)


def verify_precision_settings(sp, logger):
    """Verify that precision settings are consistent between paths."""
    logger.info("=== Precision Settings Verification ===")
    
    # Check the precision configs that control the two paths
    # Use getattr with defaults to avoid AttributeError
    tf_cfg = getattr(sp, '_tf_cfg', {})
    fo_cfg = getattr(sp, '_fo_cfg', {})
    
    logger.info(f"No-grad TF config: autocast={tf_cfg.get('autocast', 'N/A')}, dtype={tf_cfg.get('dtype', 'N/A')}")
    logger.info(f"Functional override config: autocast={fo_cfg.get('autocast', 'N/A')}, dtype={fo_cfg.get('dtype', 'N/A')}")
    
    # Check global entropy precision setting
    entropy_fp64 = getattr(sp, '_entropy_fp64', False)
    logger.info(f"Entropy FP64: {entropy_fp64}")
    
    # Check if there's a config object we can inspect
    sp_config = getattr(sp, 'config', None)
    if sp_config:
        logger.info(f"SP config type: {type(sp_config)}")
        if hasattr(sp_config, 'temperature'):
            logger.info(f"SP generation config - temperature: {sp_config.temperature}, top_p: {getattr(sp_config, 'top_p', 'N/A')}")
    
    return {
        'tf_cfg': tf_cfg,
        'fo_cfg': fo_cfg, 
        'entropy_fp64': entropy_fp64,
        'sp_config_type': str(type(sp_config)) if sp_config else None
    }


def run_comparison_test(logger) -> Dict[str, Any]:
    """Run the main comparison test."""
    logger.info("=== Starting Sequence Processor Path Comparison ===")
    
    # 1. Setup Phase
    logger.info("1. Loading configuration and initializing components...")
    config = load_test_config()
    
    probe = EntropyMeasurements(config)
    
    # Load checkpoint - using paths from config
    checkpoint_path = config['checkpoint'].get('checkpoint_path')
    optimizer_path = config['checkpoint'].get('optimizer_path')
    
    if not checkpoint_path or not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    probe.load_checkpoint(checkpoint_path, optimizer_path)
    
    # Ensure sequence processor is initialized
    probe._ensure_sequence_processor()
    
    # Extract key objects
    model = probe.model
    tokenizer = probe.tokenizer
    sp = probe._sequence_processor
    
    logger.info(f"Model dtype: {next(model.parameters()).dtype}")
    logger.info(f"Model device: {next(model.parameters()).device}")
    logger.info(f"Model training mode: {model.training}")
    
    # Verify precision settings
    precision_info = verify_precision_settings(sp, logger)
    
    # 2. Generate Test Sequences
    logger.info("2. Generating test sequences...")
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate test sequences - returns (BatchedSequences, LogprobResults, DiagnosticsResults)
    test_sequences, test_logprobs, test_diagnostics = sp.generate_with_logprobs(
        prompts=None,  # Sample from dataset
        G=1,
        dataset_name="gsm8k_r1_template",
        split="test", 
        num_prompts=8,
        seed=42,
        with_grad=False,  # We just need the sequences
        compute_rb=True
    )
    
    logger.info(f"Generated {len(test_sequences.sequences)} sequences")
    logger.info(f"Sequence shape: {test_sequences.sequences.shape}")
    
    # 3. Test Both Paths
    logger.info("3. Testing both teacher forcing paths...")
    
    # Ensure model is in same state for both tests
    model.eval()
    
    # Path 1: No-grad (functional_call)
    logger.info("3a. Running no-grad path...")
    logprob_results_no_grad = sp.teacher_force_logprobs(
        sequences=test_sequences,
        with_grad=False,
        compute_rb=True,
        params_override=None  # Use default cached zero-eta mapping
    )
    
    # Path 2: With-grad (live module)
    logger.info("3b. Running with-grad path...")
    logprob_results_with_grad = sp.teacher_force_logprobs(
        sequences=test_sequences,
        with_grad=True,
        compute_rb=True
    )
    
    # Detach the with-grad results for comparison
    logger.info("3c. Detaching with-grad results...")
    logprob_results_with_grad_detached = detach_logprob_results(logprob_results_with_grad)
    
    # 4. Detailed Numerical Comparison
    logger.info("4. Performing detailed numerical comparisons...")
    
    tolerance = 1e-6  # Reasonable tolerance for FP32 math
    
    # Compare sequence logprobs
    seq_lp_comparison = compare_sequence_logprobs(
        logprob_results_no_grad.sequence_logprobs,
        logprob_results_with_grad_detached.sequence_logprobs,
        tolerance
    )
    logger.info(f"Sequence logprobs match: {seq_lp_comparison['match']}")
    if not seq_lp_comparison['match']:
        logger.warning(f"Sequence logprobs mismatches: {len(seq_lp_comparison['mismatches'])}")
        for mismatch in seq_lp_comparison['mismatches'][:5]:  # Show first 5
            logger.warning(f"  {mismatch}")
    
    # Compare per-token logprobs
    token_lp_comparison = compare_per_token_logprobs(
        logprob_results_no_grad.logprobs,
        logprob_results_with_grad_detached.logprobs,
        tolerance
    )
    logger.info(f"Per-token logprobs match: {token_lp_comparison['match']}")
    if not token_lp_comparison['match']:
        logger.warning(f"Per-token logprobs mismatches: {len(token_lp_comparison['mismatches'])}")
        for mismatch in token_lp_comparison['mismatches'][:5]:
            logger.warning(f"  {mismatch}")
    
    # Compare entropies
    entropies_comparison = compare_entropies(
        logprob_results_no_grad.entropies,
        logprob_results_with_grad_detached.entropies,
        tolerance
    )
    logger.info(f"Entropies match: {entropies_comparison['match']}")
    if not entropies_comparison['match']:
        logger.warning(f"Entropies mismatches: {len(entropies_comparison['mismatches'])}")
        for mismatch in entropies_comparison['mismatches'][:5]:
            logger.warning(f"  {mismatch}")
    
    # Compare RB entropies
    rb_entropies_comparison = compare_rb_entropies(
        logprob_results_no_grad.rb_entropies,
        logprob_results_with_grad_detached.rb_entropies,
        tolerance
    )
    logger.info(f"RB entropies match: {rb_entropies_comparison['match']}")
    if not rb_entropies_comparison['match']:
        logger.warning(f"RB entropies mismatches: {len(rb_entropies_comparison['mismatches'])}")
        for mismatch in rb_entropies_comparison['mismatches'][:5]:
            logger.warning(f"  {mismatch}")
    
    # 5. Generate Final Report
    report = {
        'test_info': {
            'num_prompts': 8,
            'num_responses_per_prompt': 1,
            'sequences_shape': list(test_sequences.sequences.shape),
            'model_dtype': str(next(model.parameters()).dtype),
            'model_device': str(next(model.parameters()).device),
            'precision_settings': precision_info,
            'tolerance_used': tolerance
        },
        'comparison_results': {
            'sequence_logprobs': seq_lp_comparison,
            'per_token_logprobs': token_lp_comparison,
            'entropies': entropies_comparison,
            'rb_entropies': rb_entropies_comparison
        },
        'overall_match': (
            seq_lp_comparison['match'] and 
            token_lp_comparison['match'] and 
            entropies_comparison['match'] and 
            rb_entropies_comparison['match']
        )
    }
    
    return report


def main():
    """Main function to run the comparison test."""
    logger = setup_logging()
    
    try:
        report = run_comparison_test(logger)
        
        # Print final results
        logger.info("=== FINAL RESULTS ===")
        logger.info(f"Overall paths match: {report['overall_match']}")
        
        if report['overall_match']:
            logger.info("✅ SUCCESS: Both teacher forcing paths produce identical numerical results!")
        else:
            logger.error("❌ FAILURE: Paths produce different results!")
            
            # Show summary of differences
            for key, comparison in report['comparison_results'].items():
                if not comparison['match']:
                    logger.error(f"  {key}: max_diff={comparison['max_diff']:.8e}, mean_abs_diff={comparison['mean_abs_diff']:.8e}")
        
        # Save detailed report
        report_path = Path(__file__).parent / "sequence_processor_path_comparison_report.json"
        import json
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Detailed report saved to: {report_path}")
        
        return 0 if report['overall_match'] else 1
        
    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())