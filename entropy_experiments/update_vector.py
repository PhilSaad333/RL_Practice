"""
Update vector builders (Option A: AdamW math from grads; Option B: single optimizer step Δθ/lr).

The update vector is name-keyed and normalized by learning rate: update_vector = Δθ / lr_used.
All outputs are CPU fp32 tensors for numerical stability and portability.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Tuple

import torch
import gc
import torch.nn.functional as F

from .param_registry import (
    get_trainable_named,
    get_optimizer_named_params,
    to_cpu_fp32_named,
)





def compute_update_vector(*, model, optimizer, U_batch, config, logger=None):
    """Preferred entry point: AdamW-from-grads update vector (per-unit-LR)."""
    return compute_update_vector_adamw(
        model=model, optimizer=optimizer, U_batch=U_batch, config=config, logger=logger
    )



def compute_update_vector_adamw(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    U_batch: Dict[str, Any],
    config: Dict[str, Any],
    logger=None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Build update_vector using AdamW math from current grads and optimizer state.

    Matches _rl_update slicing/masking/normalization (minus PPO/KL):
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
    
    if torch.cuda.is_available():
        print(f"  [AdamW] Before forward pass: GPU alloc={torch.cuda.memory_allocated(0)/1024**3:.2f}GB")
    
    try:
        total_loss_val = rl_loss(
            U_batch,
            model,
            temp=temp,
            mb_size=importance_mb_size,
            amp_dtype=_resolve_amp_dtype(config),
            use_amp=bool(config.get("memory_config", {}).get("amp", False)),
            backward_per_microbatch=True,
        )
        num_microbatches = (B + importance_mb_size - 1) // importance_mb_size
    finally:
        model.train(was_training)

    # Optional grad clipping
    max_norm = float(config.get("max_grad_norm", 0.0))
    if max_norm and max_norm > 0.0:
        # Clip over exactly the params the optimizer will step
        params_to_clip = []
        for g in optimizer.param_groups:
            params_to_clip.extend([p for p in g.get("params", []) if p is not None])
        if params_to_clip:
            torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm)

    # Build direction using AdamW math (per unit lr)
    dir_named = _adamw_direction_from_grads(model, optimizer, only_optimizer_params=True)

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



def _adamw_direction_from_grads(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    only_optimizer_params: bool = True,
) -> Dict[str, torch.Tensor]:
    """Compute per-parameter AdamW direction (per-unit-LR) from grads/state.

    Mirrors PyTorch 2.5.x AdamW math excluding LR scaling:
      step_size_per_lr = 1 / (1 - beta1^t)
      denom            = sqrt(v_t) / sqrt(1 - beta2^t) + eps
      dir_per_lr       = - [ step_size_per_lr * (exp_avg_t / denom) + weight_decay * p ]

    If amsgrad=True, denom uses max_exp_avg_sq. Operates over optimizer params by default.
    """
    hparams = _infer_group_hparams(optimizer)
    trainable = get_optimizer_named_params(model, optimizer) if only_optimizer_params else get_trainable_named(model)
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
        amsgrad = bool(hp.get("amsgrad", False))

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
        # torch.optim.AdamW bias correction:
        # step_size_per_lr = 1 / bc1; denom = sqrt(v_t)/sqrt(bc2) + eps
        step_factor = 1.0 / max(bc1, 1e-16)
        if amsgrad:
            max_v = st.get("max_exp_avg_sq", exp_avg_sq)
            v_eff = torch.maximum(max_v, exp_avg_sq_t)
            denom = v_eff.sqrt().div(max(bc2, 1e-16) ** 0.5).add(eps)
        else:
            denom = exp_avg_sq_t.sqrt().div(max(bc2, 1e-16) ** 0.5).add(eps)
        adam_dir = -step_factor * (exp_avg_t / denom)
        wd_dir = -weight_decay * p.detach()
        direction = (adam_dir + wd_dir)
        dir_named[name] = direction.detach().to("cpu", torch.float32)
    return dir_named


def _resolve_amp_dtype(config: Dict[str, Any]) -> torch.dtype:
    name = str(config.get("memory_config", {}).get("dtype", "bfloat16")).lower()
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(name, torch.bfloat16)

def rl_loss(
    U_batch: Dict[str, Any],
    model: torch.nn.Module,
    *,
    temp: float,
    mb_size: int,
    amp_dtype: torch.dtype | None = None,
    use_amp: bool = False,
    backward_per_microbatch: bool = True,
) -> torch.Tensor | float:
    """Compute RL-aligned naive loss over U with exact slicing/masking/normalization.

    Loss per prompt b: - sum_{g,t in gen} A_{b,g} * logp_{b,g,t} / (G * L_max_b)
    Returns a scalar tensor on the model device. Does not zero or step optimizer.
    """
    device = next(model.parameters()).device
    sequences = U_batch['sequences']          # [B_U, G, L]
    prompt_lens = U_batch['prompt_lens']      # [B_U]
    attention_masks = U_batch['attention_masks']  # [B_U, G, L]
    advantages = U_batch['advantages'].to(device)        # [B_U, G]
    max_lengths = U_batch['max_lengths']      # [B_U]
    B_U, G, Lmax = sequences.shape

    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    total_loss_val: float = 0.0
    for start_b in range(0, B_U, mb_size):
        end_b = min(start_b + mb_size, B_U)
        B_mb = end_b - start_b

        mb_seqs = sequences[start_b:end_b].to(device)
        mb_masks = attention_masks[start_b:end_b].to(device)
        mb_adv = advantages[start_b:end_b]
        mb_Lmax = max_lengths[start_b:end_b]
        mb_prompt = prompt_lens[start_b:end_b]

        flat_seqs = mb_seqs.view(-1, Lmax)
        flat_masks = mb_masks.view(-1, Lmax)

        with torch.amp.autocast("cuda", dtype=amp_dtype or torch.bfloat16, enabled=use_amp):
            logits = model(flat_seqs, attention_mask=flat_masks).logits
        logits = (logits / max(float(temp), 1e-8)).float()
        logp_all = torch.nn.functional.log_softmax(logits, dim=-1)
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

            adv_exp = mb_adv[b].unsqueeze(1).expand(-1, Tg)    # (G, Tg)
            weighted_logp = adv_exp * mask_gen * lp_gen
            loss_b = -weighted_logp.sum() / (G * L_max_b + 1e-8)
            mb_loss_terms.append(loss_b)

        mb_loss = (
            torch.stack(mb_loss_terms).mean() if mb_loss_terms else torch.tensor(0.0, device=device)
        )
        if backward_per_microbatch:
            (mb_loss * (B_mb / max(B_U, 1))).backward()
            total_loss_val += float(mb_loss.item()) * B_mb
        else:
            total_loss += (mb_loss * (B_mb / max(B_U, 1))).to(total_loss.dtype)

    if backward_per_microbatch:
        return (total_loss_val / max(B_U, 1))
    return total_loss





# -----------------------------------------------------------------------------
#                                FOR DEBUG
#   The routines below are not required for the primary flow. They are useful
#   for parity checks, ablations, and investigations only.
# -----------------------------------------------------------------------------








def _infer_group_hparams(optimizer: torch.optim.Optimizer) -> Dict[int, Dict[str, Any]]:
    """Map param_id to group hyperparams and per-param step.

    Returns a dict mapping id(param) -> {
        'betas', 'eps', 'weight_decay', 'lr', 'amsgrad', 'step'
    } using the optimizer's param_groups and state.
    """
    param_hparams: Dict[int, Dict[str, Any]] = {}
    for group in optimizer.param_groups:
        betas = group.get("betas", (0.9, 0.999))
        eps = float(group.get("eps", 1e-8))
        weight_decay = float(group.get("weight_decay", 0.0))
        lr = float(group.get("lr", 1.0))
        amsgrad = bool(group.get("amsgrad", False))
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
                "amsgrad": amsgrad,
                "step": step,
            }
    return param_hparams




def compute_update_vector_step(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    U_batch: Dict[str, Any],
    config: Dict[str, Any],
    logger=None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Option B: Build update_vector by performing a single real RL step and normalizing Δθ by lr.

    Uses DeltaEntropyIS._snapshot_model_optimizer and _rl_update to ensure parity with existing code.
    """
    de = DeltaEntropyIS(model=model, config=config, logger=logger or _NullLogger())
    
    # Don't move U_batch to device here - let _rl_update handle it with microbatching!
    # This was causing OOM by moving ALL data to GPU at once instead of streaming
    
    # Ensure model parameters require gradients (may have been disabled)
    for p in model.parameters():
        if hasattr(p, 'requires_grad'):
            p.requires_grad_(True)
    
    # Record before params first (before snapshot which detaches)
    # IMPORTANT: Clone to CPU to avoid doubling GPU memory usage!
    trainable = get_optimizer_named_params(model, optimizer)
    before = {name: p.detach().clone().cpu() for name, p in trainable.items()}
    
    # Now snapshot for restoration later
    cpu_snaps, opt_state_snapshot = de._snapshot_model_optimizer(model, optimizer, snapshot_device="cpu")
    
    if torch.cuda.is_available():
        print(f"  [Step] After snapshot: GPU alloc={torch.cuda.memory_allocated(0)/1024**3:.2f}GB")

    # Execute one RL-aligned step on U
    rl_grad_accum = int(config.get("computation_options", {}).get("rl_grad_accum", 1))
    imp_mb = int(config.get("true_delta_h", {}).get("microbatch_size", 1))
    print(f"  [Step] Calling _rl_update with rl_grad_accum={rl_grad_accum}, imp_mb={imp_mb}")
    de._rl_update(U_batch, optimizer, rl_grad_accum, imp_mb)
    
    if torch.cuda.is_available():
        print(f"  [Step] After RL update: GPU alloc={torch.cuda.memory_allocated(0)/1024**3:.2f}GB")

    # Build per-parameter LR map from optimizer groups
    lr_map: Dict[int, float] = {}
    lr_groups: Dict[float, int] = {}
    for gi, group in enumerate(optimizer.param_groups):
        lr_g = float(group.get("lr", 1.0))
        lr_groups[lr_g] = lr_groups.get(lr_g, 0) + 1
        for p in group.get("params", []):
            if p is not None:
                lr_map[id(p)] = lr_g
    delta_over_lr: Dict[str, torch.Tensor] = {}
    for name, p in trainable.items():
        after = p.detach().cpu()  # Move to CPU for comparison
        dtheta = (after - before[name])  # before is already on CPU
        lr_used = lr_map.get(id(p), 1.0)
        vec = (dtheta / max(lr_used, 1e-38)).to("cpu", torch.float32)
        delta_over_lr[name] = vec

    # Restore snapshot
    de._restore_model_optimizer(model, optimizer, cpu_snaps, opt_state_snapshot)

    vec_norm = torch.sqrt(sum((v.to(torch.float64) ** 2).sum() for v in delta_over_lr.values())).item()
    stats = {
        "method": "single_step",
        "lr_groups": sorted(list(lr_groups.keys())),
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

    Matches _rl_update slicing/masking/normalization (minus PPO/KL):
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
    
    if torch.cuda.is_available():
        print(f"  [AdamW] Before forward pass: GPU alloc={torch.cuda.memory_allocated(0)/1024**3:.2f}GB")
    
    try:
        total_loss_val = rl_loss(
            U_batch,
            model,
            temp=temp,
            mb_size=importance_mb_size,
            amp_dtype=_resolve_amp_dtype(config),
            use_amp=bool(config.get("memory_config", {}).get("amp", False)),
            backward_per_microbatch=True,
        )
        num_microbatches = (B + importance_mb_size - 1) // importance_mb_size
    finally:
        model.train(was_training)

    # Optional grad clipping
    max_norm = float(config.get("max_grad_norm", 0.0))
    if max_norm and max_norm > 0.0:
        # Clip over exactly the params the optimizer will step
        params_to_clip = []
        for g in optimizer.param_groups:
            params_to_clip.extend([p for p in g.get("params", []) if p is not None])
        if params_to_clip:
            torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm)

    # Build direction using AdamW math (per unit lr)
    dir_named = _adamw_direction_from_grads(model, optimizer, only_optimizer_params=True)

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
        _ = rl_loss(
            U_batch,
            model,
            temp=temp,
            mb_size=importance_mb_size,
            amp_dtype=_resolve_amp_dtype(config),
            use_amp=bool(config.get("memory_config", {}).get("amp", False)),
            backward_per_microbatch=True,
        )
    finally:
        model.train(was_training)

    # Optional grad clipping
    max_norm = float(config.get("max_grad_norm", 0.0))
    if max_norm and max_norm > 0.0:
        params_to_clip = []
        for g in optimizer.param_groups:
            params_to_clip.extend([p for p in g.get("params", []) if p is not None])
        if params_to_clip:
            torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm)

    # Manual AdamW direction (per unit lr)
    hparams = _infer_group_hparams(optimizer)
    trainable = get_optimizer_named_params(model, optimizer)
    vec_named: Dict[str, torch.Tensor] = {}
    adam_norm = torch.zeros((), dtype=torch.float64)
    wd_norm = torch.zeros((), dtype=torch.float64)
    for name, p in trainable.items():
        pid = id(p)
        hp = hparams.get(pid, {})  # betas, eps, weight_decay, step, amsgrad
        beta1, beta2 = map(float, hp.get("betas", (0.9, 0.999)))
        eps = float(hp.get("eps", 1e-8))
        weight_decay = float(hp.get("weight_decay", 0.0))
        step = int(hp.get("step", 0))
        amsgrad = bool(hp.get("amsgrad", False))

        st = optimizer.state.get(p, None)
        g = p.grad if p.grad is not None else torch.zeros_like(p)  # already allocated by backward

        # --- avoid zeros_like(p) for missing state ---
        if st is not None and "exp_avg" in st and "exp_avg_sq" in st:
            exp_avg = st["exp_avg"]
            exp_avg_sq = st["exp_avg_sq"]
            exp_avg_t = exp_avg.mul(beta1).add(g, alpha=(1.0 - beta1))
            exp_avg_sq_t = exp_avg_sq.mul(beta2).addcmul(g, g, value=(1.0 - beta2))
        else:
            # First-step math without allocating exp_avg/exp_avg_sq:
            #   exp_avg_t    = (1 - beta1) * g
            #   exp_avg_sq_t = (1 - beta2) * (g*g)
            exp_avg_t = g.mul(1.0 - beta1)
            exp_avg_sq_t = g.mul(g).mul(1.0 - beta2)

        t_eff = step + 1
        bc1 = 1.0 - (beta1 ** t_eff)
        bc2 = 1.0 - (beta2 ** t_eff)
        step_per_lr = 1.0 / max(bc1, 1e-16)

        if amsgrad and st is not None and "max_exp_avg_sq" in st:
            v_eff = torch.maximum(st["max_exp_avg_sq"], exp_avg_sq_t)
            denom = v_eff.sqrt().div(max(bc2, 1e-16) ** 0.5).add(eps)
        else:
            denom = exp_avg_sq_t.sqrt().div(max(bc2, 1e-16) ** 0.5).add(eps)

        adam_comp = -(step_per_lr) * (exp_avg_t / denom)
        wd_comp = -weight_decay * p.detach()

        v = (adam_comp + wd_comp).detach().to("cpu", torch.float32)
        vec_named[name] = v

        # accumulate component norms before freeing temporaries
        adam_norm += (adam_comp.detach().to("cpu", torch.float64) ** 2).sum()
        wd_norm += (wd_comp.detach().to("cpu", torch.float64) ** 2).sum()

        # reduce residency of temporaries
        del exp_avg_t, exp_avg_sq_t, adam_comp, wd_comp, denom

    # Clear grads
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

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
