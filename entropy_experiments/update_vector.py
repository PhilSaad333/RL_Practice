"""
Update vector builders (Option A: AdamW math from grads; Option B: single optimizer step Δθ/lr).

The update vector is name-keyed and normalized by learning rate: update_vector = Δθ / lr_used.
All outputs are CPU fp32 tensors for numerical stability and portability.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

from .param_registry import get_trainable_named, to_cpu_fp32_named
from .delta_entropy_is import DeltaEntropyIS, rl_loss_naive


def _infer_group_hparams(optimizer: torch.optim.Optimizer) -> Dict[int, Dict[str, Any]]:
    """Build a map: param_id -> hyperparams (betas, eps, weight_decay, lr, step).

    Note: step is read from optimizer.state[p]['step'] when present.
    """
    param_hparams: Dict[int, Dict[str, Any]] = {}
    for group in optimizer.param_groups:
        betas = group.get("betas", (0.9, 0.999))
        eps = float(group.get("eps", 1e-8))
        weight_decay = float(group.get("weight_decay", 0.0))
        lr = float(group.get("lr", 1.0))
        for p in group["params"]:
            if p is None:
                continue
            st = optimizer.state.get(p, {})
            step = int(st.get("step", 0))
            param_hparams[id(p)] = {
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
                "lr": lr,
                "step": step,
            }
    return param_hparams


def _adamw_direction_from_grads(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> Dict[str, torch.Tensor]:
    """Compute per-parameter AdamW direction (per unit lr), using current grads and optimizer state.

    Matches torch.optim.AdamW update rule (decoupled weight decay) excluding lr:
        step_size = sqrt(bc2) / bc1
        p <- p - lr * [ step_size * exp_avg / (sqrt(exp_avg_sq) + eps) + weight_decay * p ]
    We form dir_per_lr = - [ step_size * exp_avg_t / (sqrt(exp_avg_sq_t)+eps) + weight_decay * p ].
    """
    hparams = _infer_group_hparams(optimizer)
    trainable = get_trainable_named(model)
    dir_named: Dict[str, torch.Tensor] = {}
    for name, p in trainable.items():
        pid = id(p)
        if p.grad is None:
            # No grad -> zero direction
            dir_named[name] = torch.zeros_like(p.detach()).to("cpu", torch.float32)
            continue
        hp = hparams.get(pid, None) or {}
        beta1, beta2 = map(float, hp.get("betas", (0.9, 0.999)))
        eps = float(hp.get("eps", 1e-8))
        weight_decay = float(hp.get("weight_decay", 0.0))
        step = int(hp.get("step", 0))

        st = optimizer.state.get(p, {})
        exp_avg = st.get("exp_avg", torch.zeros_like(p))
        exp_avg_sq = st.get("exp_avg_sq", torch.zeros_like(p))

        # Update moments with current grad (like AdamW.step)
        g = p.grad
        exp_avg_t = exp_avg.mul(beta1).add(g, alpha=(1.0 - beta1))
        exp_avg_sq_t = exp_avg_sq.mul(beta2).addcmul(g, g, value=(1.0 - beta2))

        # Bias correction factors for the step being formed
        t_eff = step + 1
        bc1 = 1.0 - (beta1 ** t_eff)
        bc2 = 1.0 - (beta2 ** t_eff)
        step_factor = (bc2 ** 0.5) / max(bc1, 1e-16)

        denom = exp_avg_sq_t.sqrt().add(eps)
        adam_dir = -step_factor * (exp_avg_t / denom)
        wd_dir = -weight_decay * p.detach()
        direction = (adam_dir + wd_dir)
        dir_named[name] = direction.detach().to("cpu", torch.float32)
    return dir_named


def compute_update_vector_step(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    U_batch: Dict[str, Any],
    config: Dict[str, Any],
    logger=None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Option B: Build update_vector by performing a single real RL step and normalizing Δθ by lr.

    Uses DeltaEntropyIS._snapshot_model_optimizer and _rl_update_streaming to ensure parity with existing code.
    """
    de = DeltaEntropyIS(model=model, config=config, logger=logger or _NullLogger())
    
    # Get model device
    device = next(model.parameters()).device
    
    # Move U_batch tensors to the same device as model
    if "sequences" in U_batch and U_batch["sequences"].device != device:
        U_batch["sequences"] = U_batch["sequences"].to(device)
    if "attention_masks" in U_batch and U_batch["attention_masks"].device != device:
        U_batch["attention_masks"] = U_batch["attention_masks"].to(device)
    if "advantages" in U_batch and U_batch["advantages"].device != device:
        U_batch["advantages"] = U_batch["advantages"].to(device)
    
    # Ensure model parameters require gradients (may have been disabled)
    for p in model.parameters():
        if hasattr(p, 'requires_grad'):
            p.requires_grad_(True)
    
    # Record before params first (before snapshot which detaches)
    trainable = get_trainable_named(model)
    before = {name: p.detach().clone() for name, p in trainable.items()}
    
    # Now snapshot for restoration later
    cpu_snaps, opt_state_snapshot = de._snapshot_model_optimizer(model, optimizer, snapshot_device="cpu")

    # Execute one RL-aligned step on U
    rl_grad_accum = int(config.get("computation_options", {}).get("rl_grad_accum", 1))
    imp_mb = int(config.get("true_delta_h", {}).get("microbatch_size", 1))
    de._rl_update_streaming(U_batch, optimizer, rl_grad_accum, imp_mb)

    # Compute Δθ and normalize by lr (use group 0 lr by default)
    lr_used = float(optimizer.param_groups[0].get("lr", 1.0))
    delta_over_lr: Dict[str, torch.Tensor] = {}
    for name, p in trainable.items():
        after = p.detach()
        dtheta = (after - before[name])
        vec = (dtheta / max(lr_used, 1e-38)).to("cpu", torch.float32)
        delta_over_lr[name] = vec

    # Restore snapshot
    de._restore_model_optimizer(model, optimizer, cpu_snaps, opt_state_snapshot)

    vec_norm = torch.sqrt(sum((v.to(torch.float64) ** 2).sum() for v in delta_over_lr.values())).item()
    stats = {
        "method": "single_step",
        "lr_used": lr_used,
        "vec_norm": vec_norm,
        "num_params": len(delta_over_lr),
    }
    return delta_over_lr, stats


def compute_update_vector_adamw(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    U_batch: Dict[str, Any],
    config: Dict[str, Any],
    logger=None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Option A: Build update_vector using AdamW math from current grads and optimizer state.

    Matches _rl_update_streaming slicing/masking/normalization (minus PPO/KL):
    - temperature scaling
    - token logp clamp/sanitize
    - generation-only slice per prompt (prompt_len − 1, masked by attention)
    - per-prompt loss: − sum_t,g (A_g * mask * logp) / (G * L_max_b)
    - microbatch scaling: (B_mb/B_U) before backward
    - optional grad clipping via config['max_grad_norm']
    """
    device = next(model.parameters()).device
    sequences = U_batch["sequences"]          # [B, G, L]
    attention_masks = U_batch["attention_masks"]
    advantages = U_batch["advantages"].to(device)  # [B, G]
    prompt_lens = U_batch["prompt_lens"]          # [B]
    max_lengths = U_batch["max_lengths"]          # [B]
    B, G, Lmax = sequences.shape

    # Zero grads
    model.zero_grad(set_to_none=True)

    # Use same knobs as training/update
    temp = float(config.get("generation", {}).get("temperature", 1.0))
    importance_mb_size = int(config.get("true_delta_h", {}).get("microbatch_size", 1))

    # Ensure train mode for grad path
    was_training = model.training
    model.train()
    try:
        loss = rl_loss_naive(
            U_batch,
            model,
            temp=temp,
            mb_size=importance_mb_size,
            amp_dtype=getattr(torch, config.get("memory_config", {}).get("dtype", "bfloat16"), torch.bfloat16)
            if hasattr(torch, str(config.get("memory_config", {}).get("dtype", "bfloat16")))
            else torch.bfloat16,
            use_amp=bool(config.get("memory_config", {}).get("amp", False)),
        )
        loss.backward()
        total_loss_val = float(loss.item())
        num_microbatches = (B + importance_mb_size - 1) // importance_mb_size
    finally:
        model.train(was_training)

    # Optional grad clipping
    max_norm = float(config.get("max_grad_norm", 0.0))
    if max_norm and max_norm > 0.0:
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm)

    # Build direction using AdamW math (per unit lr)
    dir_named = _adamw_direction_from_grads(model, optimizer)

    # Clear grads to be polite
    model.zero_grad(set_to_none=True)

    vec_norm = torch.sqrt(sum((v.to(torch.float64) ** 2).sum() for v in dir_named.values())).item()
    stats = {
        "method": "adamw_from_grads",
        "vec_norm": vec_norm,
        "num_params": len(dir_named),
        "avg_mb_loss": (total_loss_val / B) if B > 0 else 0.0,
        "num_microbatches": num_microbatches,
    }
    return dir_named, stats


def compute_update_vector_adamw_manual(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    U_batch: Dict[str, Any],
    config: Dict[str, Any],
    logger=None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Option C: Manual AdamW math to produce Δθ/lr without mutating optimizer or params.

    1) Compute grads via shared RL loss (naive) on U
    2) For each param, form exp_avg_t/exp_avg_sq_t, bias-corrections, and direction per unit lr
    3) Return vector and simple component norms for inspection
    """
    device = next(model.parameters()).device
    
    # Ensure grads and training mode
    model.zero_grad(set_to_none=True)
    was_training = model.training
    model.train()
    
    temp = float(config.get("generation", {}).get("temperature", 1.0))
    importance_mb_size = int(config.get("true_delta_h", {}).get("microbatch_size", 1))
    
    try:
        loss = rl_loss_naive(
            U_batch,
            model,
            temp=temp,
            mb_size=importance_mb_size,
            amp_dtype=getattr(torch, config.get("memory_config", {}).get("dtype", "bfloat16"), torch.bfloat16)
            if hasattr(torch, str(config.get("memory_config", {}).get("dtype", "bfloat16")))
            else torch.bfloat16,
            use_amp=bool(config.get("memory_config", {}).get("amp", False)),
        )
        loss.backward()
    finally:
        model.train(was_training)

    # Optional grad clipping
    max_norm = float(config.get("max_grad_norm", 0.0))
    if max_norm and max_norm > 0.0:
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm)

    # Manual AdamW direction (per unit lr)
    hparams = _infer_group_hparams(optimizer)
    trainable = get_trainable_named(model)
    vec_named: Dict[str, torch.Tensor] = {}
    adam_norm = torch.zeros((), dtype=torch.float64)
    wd_norm = torch.zeros((), dtype=torch.float64)
    for name, p in trainable.items():
        pid = id(p)
        hp = hparams.get(pid, None) or {}
        beta1, beta2 = map(float, hp.get("betas", (0.9, 0.999)))
        eps = float(hp.get("eps", 1e-8))
        weight_decay = float(hp.get("weight_decay", 0.0))
        step = int(hp.get("step", 0))
        st = optimizer.state.get(p, {})
        exp_avg = st.get("exp_avg", torch.zeros_like(p))
        exp_avg_sq = st.get("exp_avg_sq", torch.zeros_like(p))

        g = p.grad if p.grad is not None else torch.zeros_like(p)
        exp_avg_t = exp_avg.mul(beta1).add(g, alpha=(1.0 - beta1))
        exp_avg_sq_t = exp_avg_sq.mul(beta2).addcmul(g, g, value=(1.0 - beta2))
        t_eff = step + 1
        bc1 = 1.0 - (beta1 ** t_eff)
        bc2 = 1.0 - (beta2 ** t_eff)
        step_per_lr = (bc1 / max(bc2, 1e-16) ** 0.5)
        denom = exp_avg_sq_t.sqrt().add(eps)
        adam_comp = -(step_per_lr) * (exp_avg_t / denom)
        wd_comp = -weight_decay * p.detach()
        v = (adam_comp + wd_comp).detach().to("cpu", torch.float32)
        vec_named[name] = v
        adam_norm += (adam_comp.double() ** 2).sum()
        wd_norm += (wd_comp.double() ** 2).sum()

    # Clear grads
    model.zero_grad(set_to_none=True)

    stats = {
        "method": "adamw_manual",
        "vec_norm": float(torch.sqrt(sum((v.double() ** 2).sum() for v in vec_named.values())).item()),
        "adam_comp_norm": float(adam_norm.sqrt().item()),
        "wd_comp_norm": float(wd_norm.sqrt().item()),
    }
    return vec_named, stats


class _NullLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass
