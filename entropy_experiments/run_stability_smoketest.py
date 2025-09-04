#!/usr/bin/env python3
"""
Stability Smoke Test for SNIS and Temperature/Q-Measure

Runs small, fast checks to validate recent stability changes:
- Enforces top_p=1.0 for E sampling (full support)
- Uses eval-mode for TF/RB during SNIS evaluation
- Uses global MAX shift for log-weights (DDP-safe)
- Computes SNIS weights in float64
- Auto-selects q-measure when temperature != 1.0 (or top_p < 1.0)
- Verifies SequenceProcessor provides temperature-aware sequence_logqs

Usage (on GPU):
  python entropy_experiments/run_stability_smoketest.py \
    --config entropy_experiments/configs/lambda_test_p1.yaml \
    --temps 1.0,0.7 \
    --B_E 16 --B_U 8 --G_U 4

Notes:
- Expects config to point to a valid checkpoint and optimizer state.
- Keeps batch sizes small by default for speed; adjust as needed.
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from entropy_experiments.offline_entropy_probe import OfflineEntropyProbe
from entropy_experiments.delta_entropy_is import DeltaEntropyIS


def _ensure_sp_for_temp(probe: OfflineEntropyProbe, temperature: float) -> None:
    # Update config temperature and rebuild SP so it picks up the new value
    probe.config.setdefault('generation', {})['temperature'] = float(temperature)
    if hasattr(probe, '_sequence_processor'):
        probe._sequence_processor = None
    probe._ensure_sequence_processor()


def _sample_EU(probe: OfflineEntropyProbe, B_E: int, B_U: int, G_U: int):
    # Use probe helpers to sample E/U with the active SequenceProcessor
    E_batch = probe._get_or_sample_E(B_E)
    U_batch = probe._get_or_sample_U(B_U, G_U)
    return E_batch, U_batch


def _run_delta_is(probe: OfflineEntropyProbe, E_batch, U_batch, measure: str | None, lr_override: float | None):
    # Build minimal cfg_importance matching probe conventions
    cfg_importance = {
        'training_loss': probe.config.get('true_delta_h', {}).get('training_loss', 'rl'),
        'importance_microbatch_size': probe.config.get('true_delta_h', {}).get('microbatch_size', 1),
        'is_mode': probe.config.get('true_delta_h', {}).get('is_mode', 'snis'),
        'clip_c': probe.config.get('true_delta_h', {}).get('clip_c', 10.0),
        'report_per_token': probe.config.get('true_delta_h', {}).get('report_per_token', True),
        'snapshot_device': probe.config.get('true_delta_h', {}).get('snapshot_device', 'cpu'),
    }
    if measure is not None:
        cfg_importance['measure'] = measure  # 'p' or 'q' (auto if None)
    if lr_override is not None:
        cfg_importance['lr_override'] = float(lr_override)

    de = DeltaEntropyIS(
        model=probe.model,
        config=probe.config,
        logger=probe.logger,
        sequence_processor=getattr(probe, '_sequence_processor', None),
    )
    results = de.entropy_change_two_batch(
        probe.model, E_batch, U_batch, probe.optimizer, cfg_importance
    )
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='entropy_experiments/configs/lambda_test_p1.yaml')
    ap.add_argument('--temps', type=str, default='1.0,0.7')
    ap.add_argument('--B_E', type=int, default=16)
    ap.add_argument('--B_U', type=int, default=8)
    ap.add_argument('--G_U', type=int, default=4)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--results_path', type=str, default='entropy_experiments/results/stability_smoketest.json')
    ap.add_argument('--lr_override', type=float, default=None)
    args = ap.parse_args()

    temps = [float(x.strip()) for x in args.temps.split(',') if x.strip()]

    # Load probe from config
    probe = OfflineEntropyProbe.from_config_file(args.config)
    ckpt_path = probe.config['checkpoint'].get('checkpoint_path', '')
    opt_path = probe.config['checkpoint'].get('optimizer_path', None)
    if not ckpt_path:
        raise RuntimeError('Config must set checkpoint.checkpoint_path')

    probe.load_checkpoint(ckpt_path, opt_path)

    all_runs = []
    for t in temps:
        print(f"\n=== Temperature {t} ===")
        _ensure_sp_for_temp(probe, t)
        sp = getattr(probe, '_sequence_processor', None)
        assert sp is not None, 'SequenceProcessor not initialized'
        print(f"SequenceProcessor config: top_p={sp.config.top_p}, temperature={sp.config.temperature}")
        if abs(sp.config.top_p - 1.0) > 1e-12:
            print("WARNING: top_p not enforced to 1.0")

        # Sample batches once per temperature to ensure identical E/U across measures
        E_batch, U_batch = _sample_EU(probe, args.B_E, args.B_U, args.G_U)

        # Run three variants: measure='p', measure='q', and auto (None)
        out_p = _run_delta_is(probe, E_batch, U_batch, measure='p', lr_override=args.lr_override)
        out_q = _run_delta_is(probe, E_batch, U_batch, measure='q', lr_override=args.lr_override)
        out_auto = _run_delta_is(probe, E_batch, U_batch, measure=None, lr_override=args.lr_override)

        def pick(o):
            di = o.get('diagnostics', {})
            return {
                'H_orig': o.get('H_orig'),
                'H_upd': o.get('H_upd'),
                'deltaH_true': o.get('deltaH_true'),
                'H_orig_tok': o.get('H_orig_tok'),
                'H_upd_tok': o.get('H_upd_tok'),
                'deltaH_true_tok': o.get('deltaH_true_tok'),
                'ESS': di.get('ESS'),
                'ESS_fraction': di.get('ESS_fraction'),
                'logw_max_global': di.get('logw_max_global'),
                'logw_mean': di.get('logw_mean'),
            }

        rec = {
            'temperature': t,
            'top_p': float(sp.config.top_p),
            'B_E': args.B_E,
            'B_U': args.B_U,
            'G_U': args.G_U,
            'p': pick(out_p),
            'q': pick(out_q),
            'auto': pick(out_auto),
        }

        # Simple checks
        def _fmt(x):
            return 'nan' if x is None else f"{x:.8f}"

        dpq = None
        if rec['p']['deltaH_true'] is not None and rec['q']['deltaH_true'] is not None:
            dpq = abs(rec['p']['deltaH_true'] - rec['q']['deltaH_true'])

        print("Results (deltaH_true):")
        print(f"  p:    {_fmt(rec['p']['deltaH_true'])}")
        print(f"  q:    {_fmt(rec['q']['deltaH_true'])}")
        print(f"  auto: {_fmt(rec['auto']['deltaH_true'])}")
        if dpq is not None:
            print(f"  |p-q|: {dpq:.8e}")

        # Expectations
        if abs(t - 1.0) < 1e-12:
            # p and q should match closely at tau=1, top_p=1
            if dpq is not None and dpq > 1e-5:
                print("WARNING: At temperature=1, p and q differ noticeably.")
        else:
            # auto should match q (we auto-select q when tau != 1)
            if (
                rec['q']['deltaH_true'] is not None
                and rec['auto']['deltaH_true'] is not None
                and abs(rec['q']['deltaH_true'] - rec['auto']['deltaH_true']) > 1e-6
            ):
                print("WARNING: auto measure does not match explicit q result.")

        # Print ESS fraction
        ef = rec['auto']['ESS_fraction']
        if ef is not None:
            print(f"  ESS_fraction (auto): {ef:.2%}")
            if ef < 0.05:
                print("  NOTE: ESS is quite low; expect high variance.")

        all_runs.append(rec)

    # Save results JSON
    out_path = Path(args.results_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(all_runs, f, indent=2)
    print(f"\nSaved smoke test results to {out_path}")


if __name__ == '__main__':
    main()
