"""
Update vector builders (Option A: AdamW math from grads; Option B: single optimizer step Δθ/lr).

The update vector is name-keyed and normalized by learning rate: update_vector = Δθ / lr_used.
All outputs are CPU fp32 tensors for numerical stability and portability.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import math
import torch

from .utils.param_registry import (
    get_trainable_named,
    get_optimizer_named_params,
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


def _get_trainable_items(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> List[Tuple[str, torch.nn.Parameter]]:
    """Return named parameters restricted to the optimizer's trainable set."""

    named = get_optimizer_named_params(model, optimizer)
    return list(named.items())


def _init_named_buffer(
    trainable_items: List[Tuple[str, torch.nn.Parameter]],
    *,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, torch.Tensor]:
    """Allocate zero buffers matching the LoRA trainables (CPU float32)."""

    buffer: Dict[str, torch.Tensor] = {}
    for name, param in trainable_items:
        buffer[name] = torch.zeros_like(param.detach(), dtype=torch.float32, device=device)
    return buffer


def _collect_sequence_gradients(
    *,
    model: torch.nn.Module,
    U_batch: Dict[str, Any],
    trainable_items: List[Tuple[str, torch.nn.Parameter]],
    temp: float,
    mb_size: int,
    amp_dtype: torch.dtype,
    use_amp: bool,
    aggregate_holder: Optional[Dict[str, torch.Tensor]] = None,
    callback: Optional[Callable[[int, Dict[str, torch.Tensor], Dict[str, Any]], None]] = None,
    sequence_records: Optional[List[Any]] = None,
) -> Tuple[float, int]:
    """Iterate over the U batch and accumulate sequence-wise gradients.

    Returns
    -------
    total_loss_val : float
        Sum of per-sequence losses across the batch (pre-normalisation as in RL loss).
    num_microbatches : int
        Number of microbatch slices processed.
    """

    sequences = U_batch["sequences"]
    attention_masks = U_batch["attention_masks"]
    prompt_lens = U_batch["prompt_lens"]
    advantages = U_batch["advantages"]
    max_lengths = U_batch.get("max_lengths")

    if not isinstance(prompt_lens, (list, tuple)):
        prompt_lens = list(prompt_lens)
    if max_lengths is None:
        max_lengths = [max(int(pl), 1) for pl in prompt_lens]

    device = next(model.parameters()).device
    advantages = advantages.to(device)

    B_U, G, Lmax = sequences.shape
    params = [param for _, param in trainable_items]

    total_sequences = B_U * G
    seq_counter = 0
    total_loss_val = 0.0
    num_microbatches = 0

    for start in range(0, B_U, mb_size):
        end = min(start + mb_size, B_U)
        B_mb = end - start
        num_microbatches += 1

        mb_seqs = sequences[start:end].to(device, non_blocking=True)
        mb_masks = attention_masks[start:end].to(device, non_blocking=True)
        mb_adv = advantages[start:end]

        flat_seqs = mb_seqs.view(-1, Lmax)
        flat_masks = mb_masks.view(-1, Lmax)

        if use_amp:
            autocast_ctx = torch.amp.autocast("cuda", dtype=amp_dtype or torch.bfloat16)
        else:
            autocast_ctx = torch.amp.autocast("cuda", enabled=False)

        with autocast_ctx:
            out = model(flat_seqs, attention_mask=flat_masks)
            logits = out.logits if hasattr(out, "logits") else out[0]

        logits = (logits / max(float(temp), 1e-8)).to(torch.float32)
        logp_all = torch.nn.functional.log_softmax(logits, dim=-1)
        targets = flat_seqs[:, 1:].unsqueeze(-1)
        new_logp = logp_all[:, :-1].gather(-1, targets).squeeze(-1)
        new_logp = torch.nan_to_num(new_logp, neginf=-80.0, posinf=0.0).clamp(min=-80.0, max=0.0)
        new_logp = new_logp.view(B_mb, G, -1)

        for local_b in range(B_mb):
            prompt_index = start + local_b
            prompt_len = int(prompt_lens[prompt_index])
            L_max_b = max(int(max_lengths[prompt_index]), 1)
            gen_start = max(prompt_len - 1, 0)

            lp_gen = new_logp[local_b, :, gen_start:]
            mask_gen = mb_masks[local_b, :, prompt_len:].float()
            Tg = min(lp_gen.shape[1], mask_gen.shape[1]) if lp_gen.ndim > 1 else 0
            if Tg == 0:
                continue
            lp_gen = lp_gen[:, :Tg]
            mask_gen = mask_gen[:, :Tg]

            denom = (G * L_max_b + 1e-8)

            for g in range(G):
                adv_scalar = mb_adv[local_b, g]
                lp_seq = lp_gen[g]
                mask_seq = mask_gen[g]
                seq_loss = - (adv_scalar * mask_seq * lp_seq).sum() / denom

                retain = seq_counter < (total_sequences - 1)
                grads = torch.autograd.grad(
                    seq_loss,
                    params,
                    retain_graph=retain,
                    allow_unused=True,
                )

                grad_named: Dict[str, torch.Tensor] = {}
                grad_norm_sq = 0.0
                for (name, param), grad in zip(trainable_items, grads):
                    if grad is None:
                        grad_cpu = torch.zeros_like(param.detach(), dtype=torch.float32, device="cpu")
                    else:
                        grad_cpu = grad.detach().to(torch.float32).cpu()
                    grad_named[name] = grad_cpu
                    if aggregate_holder is not None:
                        aggregate_holder[name] += grad_cpu
                    grad_norm_sq += float((grad_cpu.to(torch.float64) ** 2).sum().item())

                sequence_meta: Dict[str, Any] = {
                    "prompt_index": prompt_index,
                    "completion_index": g,
                    "advantage": float(mb_adv[local_b, g].item()),
                    "prompt_length": prompt_len,
                    "gen_length": int(Tg),
                    "loss": float(seq_loss.item()),
                    "grad_norm": math.sqrt(grad_norm_sq),
                }

                if sequence_records is not None and seq_counter < len(sequence_records):
                    record = sequence_records[seq_counter]
                    sequence_meta.setdefault("sequence_id", getattr(record, "sequence_id", None))
                    sequence_meta.setdefault("global_prompt_id", getattr(record, "global_prompt_id", None))

                if callback is not None:
                    callback(seq_counter, grad_named, sequence_meta)

                total_loss_val += float(seq_loss.item())
                seq_counter += 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return total_loss_val, num_microbatches


def _adamw_direction_from_named_grads(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    grad_named: Dict[str, torch.Tensor],
    *,
    include_components: bool = False,
) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, Dict[str, Any]]]]:
    """Compute AdamW direction given explicit named gradients (CPU tensors)."""

    direction: Dict[str, torch.Tensor] = {}
    components: Optional[Dict[str, Dict[str, Any]]] = {} if include_components else None

    trainable_items = _get_trainable_items(model, optimizer)

    for name, param in trainable_items:
        grad = grad_named.get(name)
        if grad is None:
            grad_cpu = torch.zeros_like(param.detach(), dtype=torch.float32, device="cpu")
        else:
            grad_cpu = grad.to(torch.float32).cpu()

        state = optimizer.state.get(param, {})
        betas = state.get("betas", None)
        if betas is None:
            betas = optimizer.defaults.get("betas", (0.9, 0.999))
        beta1, beta2 = map(float, betas)
        eps = float(state.get("eps", optimizer.defaults.get("eps", 1e-8)))
        weight_decay = float(state.get("weight_decay", optimizer.defaults.get("weight_decay", 0.0)))
        step = int(state.get("step", 0))
        amsgrad = bool(state.get("amsgrad", optimizer.defaults.get("amsgrad", False)))

        exp_avg = state.get("exp_avg")
        if exp_avg is None:
            exp_avg_cpu = torch.zeros_like(param.detach(), dtype=torch.float32, device="cpu")
        else:
            exp_avg_cpu = exp_avg.detach().to(torch.float32).cpu()

        exp_avg_sq = state.get("exp_avg_sq")
        if exp_avg_sq is None:
            exp_avg_sq_cpu = torch.zeros_like(param.detach(), dtype=torch.float32, device="cpu")
        else:
            exp_avg_sq_cpu = exp_avg_sq.detach().to(torch.float32).cpu()

        t_eff = step + 1
        bc1 = 1.0 - (beta1 ** t_eff)
        bc2 = 1.0 - (beta2 ** t_eff)
        step_factor = 1.0 / max(bc1, 1e-16)

        one_minus_beta1 = (1.0 - beta1)
        one_minus_beta2 = (1.0 - beta2)

        exp_avg_t = exp_avg_cpu.mul(beta1).add(grad_cpu, alpha=one_minus_beta1)
        exp_avg_sq_t = exp_avg_sq_cpu.mul(beta2).addcmul(grad_cpu, grad_cpu, value=one_minus_beta2)

        if amsgrad:
            max_exp_avg_sq = state.get("max_exp_avg_sq")
            if max_exp_avg_sq is None:
                max_exp_avg_sq_cpu = torch.zeros_like(exp_avg_sq_t)
            else:
                max_exp_avg_sq_cpu = max_exp_avg_sq.detach().to(torch.float32).cpu()
            v_eff = torch.maximum(max_exp_avg_sq_cpu, exp_avg_sq_t)
        else:
            v_eff = exp_avg_sq_t

        denom = v_eff.sqrt().div(math.sqrt(max(bc2, 1e-16))).add(eps)

        momentum_term = -step_factor * (beta1 * exp_avg_cpu / denom)
        grad_term = -step_factor * (one_minus_beta1 * grad_cpu / denom)
        weight_decay_term = -weight_decay * param.detach().to(torch.float32).cpu()

        direction[name] = (momentum_term + grad_term + weight_decay_term).to(torch.float32)

        if include_components and components is not None:
            components[name] = {
                "step_factor": step_factor,
                "one_minus_beta1": one_minus_beta1,
                "denom": denom,
                "momentum_term": momentum_term,
                "weight_decay_term": weight_decay_term,
            }

    return direction, components



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
    sequence_callback: Optional[Callable[[int, Dict[str, torch.Tensor], Dict[str, Any]], None]] = None,
    sequence_records: Optional[List[Any]] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Build update_vector using AdamW math from current grads and optimizer state.
    Output is name→tensor (CPU, fp32), normalized per-unit LR (Δθ / lr).

    Precision policy:
      - Default: pure fp32 forward+backward (no autocast).
      - If precision.update_vector.use_amp=True, uses autocast with dtype=precision.update_vector.amp_dtype
        ONLY for the forward; grads still accumulate into fp32 params. You can additionally force
        fp32 grad storage via precision.update_vector.grads_fp32.
    """
    sequences = U_batch["sequences"]
    B = int(sequences.shape[0])
    G = int(sequences.shape[1]) if sequences.ndim >= 2 else 1
    total_sequences = max(B * G, 1)

    uv_prec = (config.get("precision", {}) or {}).get("update_vector", {}) or {}
    use_amp: bool = bool(uv_prec.get("use_amp", False))
    amp_dtype: torch.dtype = _resolve_amp_dtype_from_cfg(uv_prec)
    mb_size = int(config.get("true_delta_h", {}).get("microbatch_size", 1))
    temp = float(config.get("generation", {}).get("temperature", 1.0))

    was_training = model.training
    model.train()
    model.zero_grad(set_to_none=True)

    trainable_items = _get_trainable_items(model, optimizer)
    aggregate_grads = _init_named_buffer(trainable_items)

    total_loss_val, num_microbatches = _collect_sequence_gradients(
        model=model,
        U_batch=U_batch,
        trainable_items=trainable_items,
        temp=temp,
        mb_size=mb_size,
        amp_dtype=amp_dtype,
        use_amp=use_amp,
        aggregate_holder=aggregate_grads,
        callback=None,
        sequence_records=None,
    )

    grad_scale = 1.0
    max_norm = float(config.get("max_grad_norm", 0.0))
    if max_norm > 0.0:
        total_norm_sq = 0.0
        for tensor in aggregate_grads.values():
            total_norm_sq += float((tensor.to(torch.float64) ** 2).sum().item())
        total_norm = math.sqrt(total_norm_sq)
        if total_norm > max_norm and total_norm > 0.0:
            grad_scale = max_norm / (total_norm + 1e-12)
            for tensor in aggregate_grads.values():
                tensor.mul_(grad_scale)
    else:
        total_norm = math.sqrt(
            sum(float((tensor.to(torch.float64) ** 2).sum().item()) for tensor in aggregate_grads.values())
        )

    dir_named, components = _adamw_direction_from_named_grads(
        model,
        optimizer,
        aggregate_grads,
        include_components=sequence_callback is not None,
    )

    baseline_named: Dict[str, torch.Tensor] = {}
    if components is not None:
        for name, comp in components.items():
            baseline_named[name] = (comp["momentum_term"] + comp["weight_decay_term"]).to(torch.float32)
    else:
        for name, tensor in dir_named.items():
            baseline_named[name] = torch.zeros_like(tensor)

    baseline_norm64 = torch.sqrt(sum((b.to(torch.float64) ** 2).sum() for b in baseline_named.values()))
    baseline_norm = float(baseline_norm64.item()) if baseline_norm64.numel() else 0.0

    vnorm64 = torch.sqrt(sum((v.to(torch.float64) ** 2).sum() for v in dir_named.values()))
    vec_norm = float(vnorm64.item()) if vnorm64.numel() else 0.0

    stats = {
        "method": "adamw_from_grads",
        "vec_norm": vec_norm,
        "baseline_norm": baseline_norm,
        "num_params": len(dir_named),
        "avg_mb_loss": (total_loss_val / max(B, 1)),
        "num_microbatches": num_microbatches,
        "amp": {"enabled": use_amp, "dtype": str(amp_dtype).split(".")[-1]},
        "per_sequence": {"count": total_sequences},
        "gradient_clip": {
            "max_norm": max_norm,
            "applied_scale": grad_scale,
            "preclip_norm": total_norm,
        },
    }

    if sequence_callback is not None and components is not None:
        baseline_shares: Dict[str, torch.Tensor] = {}
        for name in components:
            share = baseline_named[name] / float(total_sequences)
            baseline_shares[name] = share
            components[name]["baseline_share"] = share

        def _sequence_direction_callback(
            index: int,
            grad_named: Dict[str, torch.Tensor],
            meta: Dict[str, Any],
        ) -> None:
            dir_per_seq: Dict[str, torch.Tensor] = {}
            for name, grad in grad_named.items():
                if grad_scale != 1.0:
                    grad = grad * grad_scale
                comp = components[name]
                grad_term = -comp["step_factor"] * (
                    comp["one_minus_beta1"] * grad / comp["denom"]
                )
                dir_per_seq[name] = grad_term.to(torch.float32)
            sequence_callback(index, dir_per_seq, meta)

        _collect_sequence_gradients(
            model=model,
            U_batch=U_batch,
            trainable_items=trainable_items,
            temp=temp,
            mb_size=mb_size,
            amp_dtype=amp_dtype,
            use_amp=use_amp,
            aggregate_holder=None,
            callback=_sequence_direction_callback,
            sequence_records=sequence_records,
        )

    model.train(was_training)

    if bool(uv_prec.get("grads_fp32", True)):
        force_grads_fp32(model)

    if logger:
        logger.info(
            f"[update-vector] built over {len(dir_named)} params; ||v||₂ ≈ {vec_norm:.3e}; AMP={use_amp}"
        )

    return dir_named, baseline_named, stats



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
    """Backward-compatible helper using parameter grads already stored on tensors."""

    trainable = (
        get_optimizer_named_params(model, optimizer)
        if only_optimizer_params
        else get_trainable_named(model)
    )

    grad_named: Dict[str, torch.Tensor] = {}
    for name, param in trainable.items():
        if param.grad is None:
            grad_named[name] = torch.zeros_like(param.detach(), dtype=torch.float32, device="cpu")
        else:
            grad_named[name] = param.grad.detach().to(torch.float32).cpu()

    direction, _ = _adamw_direction_from_named_grads(
        model,
        optimizer,
        grad_named,
        include_components=False,
    )
    return direction


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
