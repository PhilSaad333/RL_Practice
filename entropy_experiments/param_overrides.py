# param_overrides.py
# Phase 0 utilities: build functional parameter/buffer dicts, with optional update vector.
# -----------------------------------------------------------------------------
# Usage:
#   from param_registry import get_trainable_named, get_named_buffers
#   params_dict, buffers_dict = build_functional_params_named(model, v_named, eta)
#   merged = merge_params_and_buffers(params_dict, buffers_dict)
#   out = torch.func.functional_call(model, merged, (inputs,), {'attention_mask': am})
#
# Notes:
# - v_named must be a dict[str, Tensor] keyed by *parameter names*.
# - Only parameters whose names appear in v_named are perturbed; all others (including frozen/base)
#   are passed through unchanged.
# - Buffers are provided by param_registry.get_named_buffers(model).
# - Unknown keys in v_named are validated against the registry and error by default (strict=True).
# - Set allow_frozen_updates=True to validate against *all* named parameters rather than only
#   the trainable set (useful if you intentionally perturb frozen/base weights for experiments).
# -----------------------------------------------------------------------------

from __future__ import annotations
from typing import Dict, Tuple, Optional, Iterable
import warnings
import torch
import torch.nn as nn

from .param_registry import (
    get_trainable_named,
    get_named_buffers,
)

# ------------------------------ helpers ------------------------------

def _unwrap_module(module: nn.Module) -> nn.Module:
    """
    Unwrap common wrappers (e.g., DDP .module). Extend if you use other wrappers.
    """
    m = module
    while hasattr(m, "module"):
        m = getattr(m, "module")
    if hasattr(m, "_orig_mod"):
        try:
            m = getattr(m, "_orig_mod")
        except Exception:
            pass
    return m

def _to_like(t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Move/cast tensor t to ref's device/dtype (for floating types only)."""
    if t.device != ref.device or t.dtype != ref.dtype:
        if t.is_floating_point():
            return t.to(device=ref.device, dtype=ref.dtype)
        else:
            return t.to(device=ref.device)
    return t

def _validate_update_keys(
    v_named: Dict[str, torch.Tensor],
    allowed_names: Iterable[str],
    *,
    strict: bool = True,
    context: str = "trainable",
) -> None:
    """Ensure all update keys exist among allowed_names."""
    allowed = set(allowed_names)
    unknown = [k for k in v_named.keys() if k not in allowed]
    if unknown:
        msg = (
            f"[param_overrides] update vector has keys not in {context} parameters: "
            f"{unknown[:5]}{'...' if len(unknown) > 5 else ''}"
        )
        if strict:
            raise KeyError(msg)
        else:
            warnings.warn(msg, RuntimeWarning)

# ------------------------------ main API ------------------------------

def build_functional_params_named(
    model: nn.Module,
    v_named: Optional[Dict[str, torch.Tensor]] = None,
    eta: float = 0.0,
    *,
    strict: bool = True,
    allow_frozen_updates: bool = False,
    detach_params: bool = True,
    detach_buffers: bool = True,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Build name-keyed dictionaries of parameters and buffers suitable for torch.func.functional_call.
    If v_named is provided, produce effective parameters θ_eff[name] = θ[name] + eta * v_named[name]
    for those names present in v_named. All other parameters (and all buffers) pass through unchanged.

    Args:
        model: nn.Module (possibly wrapped).
        v_named: dict name->Tensor of per-unit-LR updates (same shape as the corresponding params).
        eta: scalar learning-rate multiplier for v_named.
        strict: if True, error on unknown keys in v_named; if False, warn and ignore.
        allow_frozen_updates:
            - False (default): validate v_named against the **trainable** set (LoRA-safe).
            - True: validate against **all** named parameters (allows perturbing frozen/base).
        detach_params: if True, parameters are detached (recommended for functional_call use).
        detach_buffers: if True, buffers are detached.

    Returns:
        (params_dict, buffers_dict): both dict[str, Tensor] for functional_call.
    """
    m = _unwrap_module(model)

    # Registry: authoritative enumeration
    trainable_named = get_trainable_named(m)     # ordered mapping of trainables
    buffers_named   = get_named_buffers(m)       # ordered mapping of buffers

    # Validation set
    if v_named is not None:
        if allow_frozen_updates:
            # Validate against all parameters
            all_param_names = [name for name, _ in m.named_parameters()]
            _validate_update_keys(v_named, all_param_names, strict=strict, context="all")
        else:
            # Validate against trainables (LoRA adapters etc.)
            _validate_update_keys(v_named, trainable_named.keys(), strict=strict, context="trainable")

    # Build parameter dict: include *all* named parameters; add deltas where provided.
    params_out: Dict[str, torch.Tensor] = {}
    for name, p in m.named_parameters():
        base = p.detach() if detach_params else p
        if v_named is not None and name in v_named:
            v = v_named[name]
            if v is None:
                raise ValueError(f"[param_overrides] v_named['{name}'] is None.")
            if tuple(v.shape) != tuple(p.shape):
                raise ValueError(
                    f"[param_overrides] shape mismatch for '{name}': param {tuple(p.shape)} vs v {tuple(v.shape)}"
                )
            eff = base + (eta * _to_like(v, p))
            params_out[name] = eff
        else:
            params_out[name] = base

    # Buffers dict (pass-through; some backbones carry critical buffers)
    buffers_out: Dict[str, torch.Tensor] = {}
    for name, b in buffers_named.items():
        if isinstance(b, torch.Tensor) and detach_buffers:
            buffers_out[name] = b.detach()
        else:
            buffers_out[name] = b

    return params_out, buffers_out


def merge_params_and_buffers(
    params_dict: Dict[str, torch.Tensor],
    buffers_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Merge params and buffers to a single mapping for functional_call.
    (If any name collisions exist—which is unusual—parameters take precedence.)
    """
    merged = dict(params_dict)
    for k, t in buffers_dict.items():
        if k not in merged:
            merged[k] = t
    return merged


def build_merged_functional_state(
    model: nn.Module,
    v_named: Optional[Dict[str, torch.Tensor]] = None,
    eta: float = 0.0,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """
    Convenience: directly return a single merged dict for functional_call.
    kwargs are forwarded to build_functional_params_named (e.g., strict, allow_frozen_updates).
    """
    p, b = build_functional_params_named(model, v_named, eta, **kwargs)
    return merge_params_and_buffers(p, b)
