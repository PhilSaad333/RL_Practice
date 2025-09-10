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

from .utils.param_registry import (
    get_trainable_named,
    get_optimizer_named_params,
    to_cpu_fp32_named,
)

from entropy_experiments.utils.precision_utils import force_grads_fp32, str_to_dtype





def _resolve_amp_dtype_from_cfg(prec_section: dict, default: str = "bfloat16") -> torch.dtype:
    """
    Map a user-facing dtype string in `precision.update_vector.amp_dtype`
    to a torch.dtype. Accepts synonyms and falls back to bf16.
    """
    name = str(prec_section.get("amp_dtype", default)).lower()
    mapping = {
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        "float16": torch.float16,   "fp16": torch.float16,
        "float32": torch.float32,   "fp32": torch.float32,
    }
    return mapping.get(name, torch.bfloat16)



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
    """
    Build update_vector using AdamW math from current grads and optimizer state.
    Output is name→tensor (CPU, fp32), normalized per-unit LR (Δθ / lr).

    Precision policy:
      - Default: pure fp32 forward+backward (no autocast).
      - If precision.update_vector.use_amp=True, uses autocast with dtype=precision.update_vector.amp_dtype
        ONLY for the forward; grads still accumulate into fp32 params. You can additionally force
        fp32 grad storage via precision.update_vector.grads_fp32.
    """
    device = next(model.parameters()).device
    B = int(U_batch["sequences"].shape[0])

    # Precision knobs (opt-in AMP, default OFF)
    uv_prec = (config.get("precision", {}) or {}).get("update_vector", {}) or {}
    use_amp: bool = bool(uv_prec.get("use_amp", False))
    amp_dtype: torch.dtype = _resolve_amp_dtype_from_cfg(uv_prec)

    # Zero grads
    model.zero_grad(set_to_none=True)

    # Ensure train mode for grad path
    was_training = model.training
    model.train()

    if torch.cuda.is_available():
        print(f"  [AdamW] Before forward pass: GPU alloc={torch.cuda.memory_allocated(0)/1024**3:.2f}GB")

    # Loss / backward accumulates across microbatches
    try:
        total_loss_val = rl_loss(
            U_batch,
            model,
            temp=float(config.get("generation", {}).get("temperature", 1.0)),
            mb_size=int(config.get("true_delta_h", {}).get("microbatch_size", 1)),
            amp_dtype=amp_dtype,
            use_amp=use_amp,
            backward_per_microbatch=True,
        )
        num_microbatches = (B + int(config.get("true_delta_h", {}).get("microbatch_size", 1)) - 1) // int(
            config.get("true_delta_h", {}).get("microbatch_size", 1)
        )
    finally:
        model.train(was_training)

    # (Optional) force grads to fp32 storage if anything leaked in mixed modes
    if bool(uv_prec.get("grads_fp32", True)):
        force_grads_fp32(model)

    # Optional grad clipping
    max_norm = float(config.get("max_grad_norm", 0.0))
    if max_norm > 0.0:
        params_to_clip = []
        for g in optimizer.param_groups:
            params_to_clip.extend([p for p in g.get("params", []) if p is not None])
        if params_to_clip:
            torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm)

    # Build per-unit-LR direction using AdamW math over optimizer params (LoRA trainables)
    dir_named = _adamw_direction_from_grads(model, optimizer, only_optimizer_params=True)

    # Clear grads
    model.zero_grad(set_to_none=True)

    vnorm64 = torch.sqrt(sum((v.to(torch.float64) ** 2).sum() for v in dir_named.values()))
    vec_norm = float(vnorm64.item()) if vnorm64.numel() else 0.0

    stats = {
        "method": "adamw_from_grads",
        "vec_norm": vec_norm,
        "num_params": len(dir_named),
        "avg_mb_loss": (total_loss_val / max(B, 1)),
        "num_microbatches": num_microbatches,
        "amp": {"enabled": use_amp, "dtype": str(amp_dtype).split(".")[-1]},
    }
    if logger:
        logger.info(f"[update-vector] built over {len(dir_named)} params; ||v||₂ ≈ {vec_norm:.3e}; AMP={use_amp}")
    return dir_named, stats



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


def _resolve_amp_dtype_from_cfg(cfg: Dict[str, Any]) -> torch.dtype:
    """
    Map config string to torch dtype for autocast when precision.update_vector.use_amp=True.
    Accepts: {"amp_dtype": "bfloat16"|"bf16"|"float16"|"fp16"|"float32"|"fp32"} (case-insensitive).
    """
    name = str(cfg.get("amp_dtype", "bfloat16")).lower()
    return {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }.get(name, torch.bfloat16)


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
    """
    Compute RL-aligned naive loss over U with exact slicing/masking/normalization.

    Precision policy:
      - If use_amp=False (default): run in pure fp32.
      - If use_amp=True: wrap *forward* in autocast(dtype=amp_dtype); logits are then cast to fp32
        before log_softmax. Gradients still accumulate into param dtype (fp32 if you've forced the model).
    """
    device = next(model.parameters()).device
    sequences = U_batch["sequences"]          # [B_U, G, L]
    prompt_lens = U_batch["prompt_lens"]      # [B_U]
    attention_masks = U_batch["attention_masks"]  # [B_U, G, L]
    advantages = U_batch["advantages"].to(device)        # [B_U, G]
    max_lengths = U_batch["max_lengths"]      # [B_U]
    B_U, G, Lmax = sequences.shape

    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    total_loss_val: float = 0.0

    for start_b in range(0, B_U, mb_size):
        end_b = min(start_b + mb_size, B_U)
        B_mb = end_b - start_b

        mb_seqs = sequences[start_b:end_b].to(device, non_blocking=True)
        mb_masks = attention_masks[start_b:end_b].to(device, non_blocking=True)
        mb_adv = advantages[start_b:end_b]                       # [B_mb, G]
        mb_Lmax = max_lengths[start_b:end_b]                     # [B_mb]
        mb_prompt = prompt_lens[start_b:end_b]                   # [B_mb]

        flat_seqs = mb_seqs.view(-1, Lmax)
        flat_masks = mb_masks.view(-1, Lmax)

        # Forward in desired precision
        if use_amp:
            with torch.amp.autocast("cuda", dtype=amp_dtype or torch.bfloat16):
                out = model(flat_seqs, attention_mask=flat_masks)
                logits = out.logits if hasattr(out, "logits") else out[0]
        else:
            # Explicitly disable autocast to guarantee pure fp32 math
            with torch.amp.autocast("cuda", enabled=False):
                out = model(flat_seqs, attention_mask=flat_masks)
                logits = out.logits if hasattr(out, "logits") else out[0]

        # Normalize temperature and upcast to fp32 for stable log_softmax
        logits = (logits / max(float(temp), 1e-8)).to(torch.float32)

        logp_all = torch.nn.functional.log_softmax(logits, dim=-1)
        targets = flat_seqs[:, 1:].unsqueeze(-1)
        new_logp = logp_all[:, :-1].gather(-1, targets).squeeze(-1)  # [B_mb*G, L-1]

        # sanitize numerics (consistent with your downstream expectations)
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

        mb_loss = torch.stack(mb_loss_terms).mean() if mb_loss_terms else torch.tensor(0.0, device=device)
        if backward_per_microbatch:
            (mb_loss * (B_mb / max(B_U, 1))).backward()
            total_loss_val += float(mb_loss.item()) * B_mb
        else:
            total_loss += (mb_loss * (B_mb / max(B_U, 1))).to(total_loss.dtype)

    return (total_loss_val / max(B_U, 1)) if backward_per_microbatch else total_loss





