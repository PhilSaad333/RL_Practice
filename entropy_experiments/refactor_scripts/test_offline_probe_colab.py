"""
Colab sanity test for OfflineEntropyProbe with DeltaEntropyIS (RB + SNIS) and LoRA checkpoint.

Usage (in Colab):

    import sys
    sys.path.append('/content/RL_Practice')  # adjust to repo root if needed

    from entropy_experiments.refactor_scripts.test_offline_probe_colab import run_test
    results = run_test(
        adapter_dir='/content/drive/MyDrive/RL_Practice_Files/new_rl_checkpoint/step_60/model',
        optimizer_path='/content/drive/MyDrive/RL_Practice_Files/new_rl_checkpoint/step_60/optimizer.pt',
        dataset_name='gsm8k_r1_template',  # adjust if different in your environment
        split='test',
        B_E=16,
        B_U=8,
        G=4,
        max_new_tokens=128,
        rb_entropy=True,
    )
    print(results)

Notes:
- Single-GPU (A100) path only.
- Uses SequenceProcessor sampling (no DDP).
- Importance sampler path uses DeltaEntropyIS (RB + SNIS only).
"""

from __future__ import annotations

import os
import json
from typing import Any, Dict, Optional

import torch

from entropy_experiments.offline_entropy_probe import OfflineEntropyProbe


def _default_base_model_id() -> str:
    # Match defaults used elsewhere; adjust if your adapter expects a different base
    return os.environ.get('BASE_MODEL_ID', 'Qwen/Qwen2.5-1.5B')


def build_config(
    *,
    adapter_dir: str,
    optimizer_path: str,
    dataset_name: str,
    split: str = 'test',
    B_E: int = 16,
    B_U: int = 8,
    G: int = 4,
    max_new_tokens: int = 128,
    rb_entropy: bool = True,
) -> Dict[str, Any]:
    """Build a minimal config dict for the probe run.

    rb_entropy: if True, use RB payload for importance sampling (recommended).
    """
    cfg: Dict[str, Any] = {
        'output': {
            'log_level': 'INFO',
            'save_results': False,
            'results_path': '',
        },
        'checkpoint': {
            'checkpoint_path': adapter_dir,          # LoRA adapter dir
            'optimizer_path': optimizer_path,        # optimizer.pt
            'model_config_path': _default_base_model_id(),
            'use_qlora': False,                      # set True if your adapter needs QLoRA
            'dtype': 'bfloat16',
            'device_map': 'cuda',
        },
        'batch_config': {
            'dataset_name': dataset_name,
            'split': split,
            'G': int(G),
            'B_E': int(B_E),
            'B_U': int(B_U),
            'rollout_batch_size': 8,
        },
        'generation': {
            'temperature': 1.0,
            'top_p': 1.0,
            'max_new_tokens': int(max_new_tokens),
            'gen_batch_size': 8,
            'tf_batch_size': 16,
            'rb_requires_grad': True,  # enable RB grad path for X estimator if used
        },
        'memory_config': {
            'amp': True,
            'dtype': 'bfloat16',  # picked up by various components
            'microbatch_size': 1,
        },
        'probe_rework': {
            'compute_delta_h1': True,
            'mb_size_prompts': 2,
            'weighting_mode': 'dr_grpo',
            'use_sequence_processor_sampling': True,
        },
        'importance': {
            'enabled': True,
            'training_loss': 'rl',
            'importance_microbatch_size': 1,
            'report_per_token': False,
            'snapshot_device': 'cpu',
            'entropy_mode': 'rb' if rb_entropy else 'naive',
            'is_mode': 'snis',
        },
        'distributed': {
            'find_unused_parameters': False,
        },
        # reuse disabled by default; toggle in Colab if you want to cache batches
        'probe_reuse': {
            'reuse_E': False,
            'reuse_U': False,
            # 'e_batch_cache_path': '/content/E_batch.pt',
            # 'u_batch_cache_path': '/content/U_batch.pt',
            # 'lr_sweep': [1e-6, 2e-6, 5e-6],  # optional
        },
    }
    return cfg


def run_test(
    *,
    adapter_dir: str,
    optimizer_path: str,
    dataset_name: str,
    split: str = 'test',
    B_E: int = 16,
    B_U: int = 8,
    G: int = 4,
    max_new_tokens: int = 128,
    rb_entropy: bool = True,
) -> Dict[str, Any]:
    """Run a single mixed-probe test (Stage 1 + Stage 2).

    Returns the results dict produced by run_mixed_probe().
    """
    assert torch.cuda.is_available(), 'CUDA required for this test (A100)'
    cfg = build_config(
        adapter_dir=adapter_dir,
        optimizer_path=optimizer_path,
        dataset_name=dataset_name,
        split=split,
        B_E=B_E,
        B_U=B_U,
        G=G,
        max_new_tokens=max_new_tokens,
        rb_entropy=rb_entropy,
    )

    probe = OfflineEntropyProbe(cfg)
    # The orchestrator loads the adapter internally; pass checkpoint_path to be explicit
    results = probe.run_mixed_probe(checkpoint_path=adapter_dir)

    # Pretty-print key fields
    summary = {
        'deltaH1': results.get('deltaH1'),
        'learning_rate': results.get('learning_rate'),
        'bars_dot': results.get('bars_dot'),
        'B_E': results.get('B_E'),
        'B_U': results.get('B_U'),
        'timing': results.get('timing', {}),
    }
    # If ground-truth present
    for key in ('H_orig', 'H_upd', 'deltaH_true', 'ESS'):
        if key in results:
            summary[key] = results[key]

    print('[SUMMARY]', json.dumps(summary, indent=2))
    return results


if __name__ == '__main__':
    # Default paths for quick manual execution; override in Colab as needed
    adapter_dir = os.environ.get('ADAPTER_DIR', '/content/drive/MyDrive/RL_Practice_Files/new_rl_checkpoint/step_60/model')
    optimizer_path = os.environ.get('OPT_PATH', '/content/drive/MyDrive/RL_Practice_Files/new_rl_checkpoint/step_60/optimizer.pt')
    dataset_name = os.environ.get('DATASET_NAME', 'gsm8k_r1_template')
    split = os.environ.get('DATASET_SPLIT', 'test')
    run_test(
        adapter_dir=adapter_dir,
        optimizer_path=optimizer_path,
        dataset_name=dataset_name,
        split=split,
    )

