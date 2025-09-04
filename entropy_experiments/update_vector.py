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
from .delta_entropy_is import DeltaEntropyIS


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

    Returns name->cpu fp32 tensor for the direction: dir = -(m_hat / (sqrt(v_hat)+eps) + wd * p).
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
        hp = hparams.get(pid, None)
        if hp is None:
            # Fallback hyperparams
            betas = (0.9, 0.999)
            eps = 1e-8
            weight_decay = 0.0
            step = 0
        else:
            betas = hp["betas"]
            eps = hp["eps"]
            weight_decay = hp["weight_decay"]
            step = hp["step"]

        beta1, beta2 = float(betas[0]), float(betas[1])
        st = optimizer.state.get(p, {})
        m = st.get("exp_avg", torch.zeros_like(p))
        v = st.get("exp_avg_sq", torch.zeros_like(p))

        # Update first/second moments with current grad
        g = p.grad
        m_t = beta1 * m + (1.0 - beta1) * g
        v_t = beta2 * v + (1.0 - beta2) * (g * g)
        # Bias corrections use step+1 for the update being formed
        t_eff = step + 1
        bc1 = 1.0 - (beta1 ** t_eff)
        bc2 = 1.0 - (beta2 ** t_eff)
        m_hat = m_t / max(bc1, 1e-16)
        v_hat = v_t / max(bc2, 1e-16)
        denom = v_hat.sqrt() + eps
        adam_term = m_hat / denom
        wd_term = weight_decay * p.detach()
        direction = -(adam_term + wd_term)
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

    total_loss_val = 0.0
    num_microbatches = 0

    for start_b in range(0, B, importance_mb_size):
        end_b = min(start_b + importance_mb_size, B)
        B_mb = end_b - start_b

        mb_seqs = sequences[start_b:end_b].to(device)
        mb_masks = attention_masks[start_b:end_b].to(device)
        mb_adv = advantages[start_b:end_b]  # [B_mb, G]
        mb_Lmax = max_lengths[start_b:end_b]  # list/1D
        mb_prompt = prompt_lens[start_b:end_b]

        # Flat forward
        flat_seqs = mb_seqs.view(-1, Lmax)
        flat_masks = mb_masks.view(-1, Lmax)
        logits = model(flat_seqs, attention_mask=flat_masks).logits
        logits = (logits / max(temp, 1e-8)).float()
        logp_all = F.log_softmax(logits, dim=-1)
        targets = flat_seqs[:, 1:].unsqueeze(-1)
        new_logp = logp_all[:, :-1].gather(-1, targets).squeeze(-1)  # [B_mb*G, L-1]
        new_logp = torch.nan_to_num(new_logp, neginf=-80.0, posinf=0.0).clamp(min=-80.0, max=0.0)
        new_logp = new_logp.view(B_mb, G, -1)

        mb_loss_terms = []
        for b in range(B_mb):
            prompt_len = int(mb_prompt[b])
            L_max_b = max(int(mb_Lmax[b]), 1)
            gen_start = prompt_len - 1
            lp_gen = new_logp[b, :, gen_start:]              # (G, Tg)
            mask_gen = mb_masks[b, :, prompt_len:].float()   # (G, Tg)
            if lp_gen.numel() == 0 or mask_gen.numel() == 0:
                continue
            Tg = min(lp_gen.shape[1], mask_gen.shape[1])
            lp_gen = lp_gen[:, :Tg]
            mask_gen = mask_gen[:, :Tg]

            # Weighted token loss (naive path): − (A · mask · logp)
            adv_exp = mb_adv[b].unsqueeze(1).expand(-1, Tg)    # (G, Tg)
            weighted_logp = adv_exp * mask_gen * lp_gen
            loss_b = -weighted_logp.sum() / (G * L_max_b + 1e-8)
            mb_loss_terms.append(loss_b)

        mb_loss = (
            torch.stack(mb_loss_terms).mean() if mb_loss_terms else torch.tensor(0.0, device=device)
        )
        scale = (B_mb / B) if B > 0 else 1.0
        (mb_loss * scale).backward()
        total_loss_val += float(mb_loss.item()) * B_mb
        num_microbatches += 1

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


class _NullLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass
