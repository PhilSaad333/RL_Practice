#!/usr/bin/env python3
"""
Parameter Override Pipeline Test
===============================

Tests the complete modernized parameter override pipeline:
1. Update vector computation with compute_update_vector
2. Parameter-only override mechanism using build_functional_params_named  
3. New entropy_change_with_param_overrides method
4. Precision handling throughout the pipeline
5. Mathematical continuity: tiny η → nearly identical entropies

This test verifies that the new pipeline produces mathematically consistent results
and handles precision correctly for tiny learning rates.
"""

import os
import sys
import torch
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any

# Configuration - Adjust these paths for your environment
CONFIG_PATH = "entropy_experiments/configs/A100_config.yaml"  # Relative to project root
PROJECT_ROOT = os.getcwd()  # Assumes we're running from project root
ETA = 1e-5  # Main learning rate for update vector computation
TINY_ETA = 1e-10  # Tiny learning rate for continuity test
B_U_SIZE = 8  # U batch size for update vector computation
G_U_SIZE = 4  # Generations per prompt for U batch
B_E_SIZE = 4  # E batch size for entropy computation  
G_E_SIZE = 4  # Generations per prompt for E batch

def setup_environment():
    """Ensure project imports work correctly"""
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    
    # Set PYTHONPATH environment variable
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    if PROJECT_ROOT not in current_pythonpath:
        os.environ["PYTHONPATH"] = PROJECT_ROOT + (":" + current_pythonpath if current_pythonpath else "")

def load_config(config_path: str) -> Dict[str, Any]:
    """Load and return YAML configuration"""
    full_path = os.path.join(PROJECT_ROOT, config_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Config file not found: {full_path}")
    
    with open(full_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    print(f"✓ Loaded config from {config_path}")
    return cfg

def test_parameter_override_pipeline():
    """
    Test the complete parameter override pipeline from update vector computation
    to entropy evaluation with precision handling.
    """
    print("=" * 80)
    print("PARAMETER OVERRIDE PIPELINE TEST")
    print("=" * 80)
    
    # Setup
    setup_environment()
    cfg = load_config(CONFIG_PATH)
    
    # Import after environment setup
    from entropy_experiments.offline_entropy_probe import OfflineEntropyProbe
    from entropy_experiments.update_vector import compute_update_vector
    from entropy_experiments.utils.param_overrides import build_functional_params_named
    from entropy_experiments.utils.precision_utils import str_to_dtype
    
    print("\n--- Phase 1: Initialize Probe and Model ---")
    
    # Create probe (this applies global precision settings)
    probe = OfflineEntropyProbe(cfg)
    print(f"✓ Probe initialized with precision config:")
    prec_cfg = cfg.get('precision', {})
    for profile, settings in prec_cfg.items():
        if isinstance(settings, dict):
            print(f"  {profile}: {settings}")
    
    print(f"✓ Model loaded: {probe.model}")
    print(f"✓ Sequence processor: {probe._sequence_processor}")
    
    print("\n--- Phase 2: Generate Test Batches ---")
    
    # Create test batches
    print("Generating U batch for update vector computation...")
    U_batch = probe._generate_batch(
        split_name='train',  # U batch from training split
        B=B_U_SIZE,
        G=G_U_SIZE,
        seed=42
    )
    print(f"✓ U batch: {U_batch['sequences'].shape}")
    
    print("Generating E batch for entropy evaluation...")
    E_batch = probe._generate_batch(
        split_name='test',   # E batch from test split  
        B=B_E_SIZE,
        G=G_E_SIZE,
        seed=43
    )
    print(f"✓ E batch: {E_batch['sequences'].shape}")
    
    print("\n--- Phase 3: Compute Update Vector ---")
    
    # Compute update vector using the modern approach
    print(f"Computing update vector with η = {ETA:.0e}...")
    update_vector_named, update_stats = compute_update_vector(
        model=probe.model,
        optimizer=probe.optimizer,
        U_batch=U_batch,
        config=probe.config,
        logger=probe.logger,
    )
    
    print(f"✓ Update vector computed: {len(update_vector_named)} parameters")
    print(f"  L2 norm: {update_stats['param_l2']:.6f}")
    print(f"  Avg loss: {update_stats['avg_loss']:.6f}")
    
    # Verify update vector has reasonable magnitude
    total_norm = 0.0
    for name, v in update_vector_named.items():
        total_norm += float((v.double() * v.double()).sum().item())
    total_norm = total_norm ** 0.5
    print(f"  Verification L2 norm: {total_norm:.6f}")
    
    print("\n--- Phase 4: Test Parameter Override Creation ---")
    
    # Test parameter override creation with different learning rates
    test_etas = [1e-4, 1e-6, 1e-8, TINY_ETA]
    
    for test_eta in test_etas:
        print(f"\nTesting parameter overrides with η = {test_eta:.0e}...")
        
        # Get precision config for func_override profile
        fo_cfg = cfg.get('precision', {}).get('func_override', {})
        force_dtype = str_to_dtype(fo_cfg.get('dtype', 'float32')) if fo_cfg.get('cast_params', False) else None
        
        params_override, _ = build_functional_params_named(
            probe.model, update_vector_named, test_eta,
            force_param_dtype=force_dtype,
            detach_params=True, detach_buffers=True,
        )
        
        print(f"✓ Parameter overrides created: {len(params_override)} parameters")
        
        # Verify overrides have expected magnitude
        max_diff = 0.0
        orig_params = dict(probe.model.named_parameters())
        for name, override_param in params_override.items():
            if name in orig_params:
                orig_param = orig_params[name]
                diff = (override_param - orig_param).abs().max().item()
                max_diff = max(max_diff, diff)
        
        print(f"  Max parameter difference: {max_diff:.2e}")
        expected_max_diff = test_eta * total_norm / len(params_override) ** 0.5
        print(f"  Expected order of magnitude: ~{expected_max_diff:.2e}")
    
    print("\n--- Phase 5: Test New Entropy Change Method ---")
    
    # Test the new entropy_change_with_param_overrides method
    print(f"Testing entropy_change_with_param_overrides with η = {ETA:.0e}...")
    
    # Configure importance sampling
    cfg_importance = {
        'report_per_token': cfg.get('true_delta_h', {}).get('report_per_token', False),
        'measure': cfg.get('true_delta_h', {}).get('measure', 'p'),
    }
    
    # Run the new method
    results = probe._delta_entropy_is.entropy_change_with_param_overrides(
        model=probe.model,
        E_batch=E_batch,
        update_vector_named=update_vector_named,
        eta=ETA,
        cfg_importance=cfg_importance,
    )
    
    print(f"✓ Entropy change computation completed:")
    print(f"  H_orig: {results['H_orig']:.6f}")
    print(f"  H_upd: {results['H_upd']:.6f}")
    print(f"  ΔH: {results['deltaH_true']:.8f}")
    print(f"  ESS fraction: {results['diagnostics']['ESS_fraction']:.2%}")
    print(f"  Compute time: {results['timing']['total_time']:.2f}s")
    
    print("\n--- Phase 6: Continuity Test ---")
    
    # Test mathematical continuity with tiny learning rate
    print(f"Testing continuity with tiny η = {TINY_ETA:.0e}...")
    
    tiny_results = probe._delta_entropy_is.entropy_change_with_param_overrides(
        model=probe.model,
        E_batch=E_batch,
        update_vector_named=update_vector_named,
        eta=TINY_ETA,
        cfg_importance=cfg_importance,
    )
    
    print(f"✓ Tiny η entropy change computation completed:")
    print(f"  H_orig: {tiny_results['H_orig']:.6f}")
    print(f"  H_upd: {tiny_results['H_upd']:.6f}")
    print(f"  ΔH: {tiny_results['deltaH_true']:.12f}")
    
    # Verify continuity: tiny eta should give nearly zero delta entropy
    delta_h_main = results['deltaH_true']
    delta_h_tiny = tiny_results['deltaH_true']
    continuity_ratio = abs(delta_h_tiny) / (abs(delta_h_main) + 1e-12)
    
    print(f"\n--- Continuity Analysis ---")
    print(f"ΔH (η = {ETA:.0e}):     {delta_h_main:.10f}")
    print(f"ΔH (η = {TINY_ETA:.0e}): {delta_h_tiny:.12f}")
    print(f"Ratio |ΔH_tiny| / |ΔH_main|: {continuity_ratio:.2e}")
    
    # Check if continuity test passes
    continuity_threshold = 1e-3  # Tiny eta should give <0.1% of main eta effect
    continuity_passed = continuity_ratio < continuity_threshold
    
    print(f"Continuity test: {'✓ PASSED' if continuity_passed else '✗ FAILED'}")
    if not continuity_passed:
        print(f"  Expected ratio < {continuity_threshold:.0e}, got {continuity_ratio:.2e}")
        print("  This may indicate precision issues in the parameter override pipeline")
    else:
        print(f"  Excellent! Tiny learning rate gives minimal entropy change as expected")
    
    print("\n--- Phase 7: Comparison with Original Method ---")
    
    # For comparison, also run the original method (if we want to compare)
    print("Note: Original RL method would require real optimizer steps and snapshot/restore")
    print("The new parameter override method avoids this complexity while maintaining precision")
    
    print("\n" + "=" * 80)
    print("PARAMETER OVERRIDE PIPELINE TEST COMPLETE")
    print("=" * 80)
    
    # Return results for any additional analysis
    return {
        'main_results': results,
        'tiny_results': tiny_results,
        'continuity_passed': continuity_passed,
        'continuity_ratio': continuity_ratio,
        'update_stats': update_stats,
    }

if __name__ == "__main__":
    """
    Run the parameter override pipeline test.
    
    Usage:
        cd /path/to/RL_Practice
        python entropy_experiments/debug/parameter_override_pipeline_test.py
    """
    
    try:
        results = test_parameter_override_pipeline()
        
        if results['continuity_passed']:
            print("🎉 All tests passed! Parameter override pipeline is working correctly.")
        else:
            print("⚠️ Continuity test failed. Check precision settings and pipeline implementation.")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)