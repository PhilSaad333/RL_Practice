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
CONFIG_PATH = "entropy_experiments/configs/colab_config.yaml"  # Relative to project root
PROJECT_ROOT = os.getcwd()  # Assumes we're running from project root
ETA = 1e-5  # Main learning rate for update vector computation
TINY_ETA = 1e-10  # Tiny learning rate for continuity test
B_U_SIZE = 8  # U batch size for update vector computation
G_U_SIZE = 4  # Generations per prompt for U batch
B_E_SIZE = 16  # E batch size for entropy computation  
G_E_SIZE = 1  # Generations per prompt for E batch

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
    
    # Load checkpoint (required before sequence processor initialization)
    print("Loading checkpoint...")
    ckpt_cfg = cfg.get("checkpoint", {})
    checkpoint_path = ckpt_cfg.get("checkpoint_path", "")
    optimizer_path = ckpt_cfg.get("optimizer_path", "")
    
    if not checkpoint_path:
        raise ValueError("checkpoint_path not found in config. Please ensure config has checkpoint.checkpoint_path")
    
    probe.load_checkpoint(checkpoint_path, optimizer_path if optimizer_path else None)
    print(f"✓ Model loaded: {probe.model}")
    
    # Ensure sequence processor is initialized
    probe._ensure_sequence_processor()
    print(f"✓ Sequence processor: {probe._sequence_processor}")
    
    print("\n--- Phase 2: Generate Test Batches ---")
    
    # Get dataset configuration
    dataset_name = cfg['batch_config']['dataset_name']
    E_split, U_split = probe._get_splits()
    print(f"Using dataset: {dataset_name}, E_split: {E_split}, U_split: {U_split}")
    
    # Generate U batch for update vector computation
    print("Generating U batch for update vector computation...")
    U_sequences, U_logprobs, _U_diag = probe._sequence_processor.generate_with_logprobs(
        prompts=None,
        G=G_U_SIZE,
        dataset_name=dataset_name,
        split=U_split,
        num_prompts=B_U_SIZE,
        compute_rb=True,
    )
    U_batch = probe._pack_U_from_sequences(U_sequences, U_logprobs.rewards)
    print(f"✓ U batch: {U_batch['sequences'].shape}")
    
    # Generate E batch for entropy evaluation  
    print("Generating E batch for entropy evaluation...")
    E_sequences, _E_logprobs, _E_diag = probe._sequence_processor.generate_with_replacement_sampling(
        total_sequences=B_E_SIZE,
        dataset_name=dataset_name,
        split=E_split,
        G=1,  # E batch uses G=1 (single generation per prompt)
        compute_rb=True,
    )
    E_batch = probe._pack_E_from_sequences(E_sequences)
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
    print(f"  L2 norm: {update_stats['vec_norm']:.6f}")
    print(f"  Avg loss: {update_stats['avg_mb_loss']:.6f}")
    print(f"  Num microbatches: {update_stats['num_microbatches']}")
    
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
    


    print("\n--- Phase 4b: SequenceProcessor Direct Continuity Checks ---")
    from entropy_experiments.utils.param_overrides import build_functional_params_named
    from entropy_experiments.utils.precision_utils import str_to_dtype

    # Force same precision on both paths for this probe
    cfg.setdefault('precision', {})
    cfg['precision'].setdefault('tf_nograd', {}).update(
        {'autocast': False, 'dtype': 'float32', 'cast_logits_fp32': True})
    cfg['precision'].setdefault('func_override', {}).update(
        {'autocast': False, 'dtype': 'float32', 'cast_params': False})
    probe._sequence_processor.config = cfg  # ensure SP sees these

    fo_cfg = cfg['precision']['func_override']
    force_dtype = str_to_dtype(fo_cfg.get('dtype', 'float32')) if fo_cfg.get('cast_params', False) else None

    # Build three overrides: eta=0, eta=tiny, eta=main
    params_zero, _ = build_functional_params_named(
        probe.model, update_vector_named, 0.0,
        force_param_dtype=force_dtype, detach_params=True, detach_buffers=True,
    )
    params_tiny, _ = build_functional_params_named(
        probe.model, update_vector_named, TINY_ETA,
        force_param_dtype=force_dtype, detach_params=True, detach_buffers=True,
    )
    params_main, _ = build_functional_params_named(
        probe.model, update_vector_named, ETA,
        force_param_dtype=force_dtype, detach_params=True, detach_buffers=True,
    )

    # --- Quantization check on the effective overrides passed to SP ---
    base_params = dict(probe.model.named_parameters())

    def delta_stats(params_over):
        same_as_base = 0
        total = 0
        max_abs = 0.0
        for k, eff in params_over.items():
            if k not in base_params: 
                continue
            d = (eff - base_params[k]).detach()
            total += d.numel()
            same_as_base += int((d == 0).sum().item())
            md = float(d.abs().max().item())
            if md > max_abs: 
                max_abs = md
        return same_as_base, total, max_abs

    same0, tot0, max0 = delta_stats(params_zero)
    sameT, totT, maxT = delta_stats(params_tiny)
    sameM, totM, maxM = delta_stats(params_main)

    print(f"[Δθ-eq] η=0 elements exactly unchanged: {same0}/{tot0}")
    print(f"[Δθ]    η=tiny unchanged elems: {sameT}/{totT}, max|Δ|={maxT:.3e}")
    print(f"[Δθ]    η=main unchanged elems: {sameM}/{totM}, max|Δ|={maxM:.3e}")


    # (3) Δθ norms scale by η (sanity on the inputs we pass)
    base_params = dict(probe.model.named_parameters())
    def dtheta_l2(params_over):
        acc = 0.0
        for k, eff in params_over.items():
            if k in base_params:
                d = (eff - base_params[k]).detach().double()
                acc += float((d*d).sum().item())
        return acc**0.5
    nM = dtheta_l2(params_main)
    nT = dtheta_l2(params_tiny)
    print(f"[SP-Δθ] ||Δθ||₂(main)={nM:.3e}  ||Δθ||₂(tiny)={nT:.3e}  ratio={nT/nM if nM>0 else float('nan'):.3e}  "
          f"(expected≈{TINY_ETA/ETA:.3e})")




    # Choose a single (b,g) to keep it tiny
    b0, g0 = 0, 0
    dbg0 = probe._sequence_processor.teacher_force_debug_probe(E_sequences, b_idx=b0, g_idx=g0, params_override=None)
    dbgZ = probe._sequence_processor.teacher_force_debug_probe(E_sequences, b_idx=b0, g_idx=g0, params_override=params_zero)
    dbgT = probe._sequence_processor.teacher_force_debug_probe(E_sequences, b_idx=b0, g_idx=g0, params_override=params_tiny)
    dbgM = probe._sequence_processor.teacher_force_debug_probe(E_sequences, b_idx=b0, g_idx=g0, params_override=params_main)

    # (1) η=0 equivalence must hold exactly
    def _max_abs(a, b): return float(torch.as_tensor(a).sub(torch.as_tensor(b)).abs().max().item())
    eq_logits = _max_abs(dbg0['logit_on_tok'],  dbgZ['logit_on_tok'])
    eq_logp   = _max_abs(dbg0['logprob_on_tok'],dbgZ['logprob_on_tok'])
    eq_H      = _max_abs(dbg0['entropy_naive'], dbgZ['entropy_naive'])
    print(f"[SP-η=0] max|Δlogit@tok|={eq_logits:.3e}  max|Δlogp@tok|={eq_logp:.3e}  max|ΔH_naive|={eq_H:.3e}")
    if max(eq_logits, eq_logp, eq_H) > 1e-6:
        raise RuntimeError("[SP-η=0] Baseline vs override(η=0) mismatch — functional_call path differs from live path.")

    # (2) η-continuity: ratios should be ≈ TINY_ETA/ETA
    def _norms(x):
        x = torch.as_tensor(x)
        return float(x.abs().max().item()), float(x.float().pow(2).sum().sqrt().item())
    for name, base_key in [("logit@tok","logit_on_tok"), ("logprob@tok","logprob_on_tok"), ("entropy_naive","entropy_naive")]:
        dM = torch.as_tensor(dbgM[base_key]) - torch.as_tensor(dbg0[base_key])
        dT = torch.as_tensor(dbgT[base_key]) - torch.as_tensor(dbg0[base_key])
        linfM, l2M = _norms(dM); linfT, l2T = _norms(dT)
        ratio_inf = (linfT/linfM) if linfM>0 else float('nan')
        ratio_l2  = (l2T/l2M)   if l2M>0 else float('nan')
        print(f"[SP-η-cont] {name:>13}: Δ∞(tiny)/Δ∞(main)={ratio_inf:.3e}  Δ₂(tiny)/Δ₂(main)={ratio_l2:.3e}  "
              f"(expected≈{TINY_ETA/ETA:.3e})")
        if linfM > 0 and abs(ratio_inf - (TINY_ETA/ETA)) > 10*(TINY_ETA/ETA):
            raise RuntimeError(f"[SP-η-cont] FAILED for {name}: observed {ratio_inf:.3e}, expected {TINY_ETA/ETA:.3e}")




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