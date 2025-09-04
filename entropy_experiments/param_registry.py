"""
Parameter and buffer registry utilities.

Provides a single source of truth for named trainable parameters and buffers,
and helpers to convert to CPU/fp32 and compute simple contractions.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterable, Tuple

import torch


def get_trainable_named(model: torch.nn.Module) -> OrderedDict[str, torch.nn.Parameter]:
    """Return ordered mapping of trainable parameters (requires_grad=True).

    Keys match `model.named_parameters()` fully-qualified names.
    """
    params: OrderedDict[str, torch.nn.Parameter] = OrderedDict()
    for name, p in model.named_parameters():
        if p.requires_grad:
            params[name] = p
    return params


def get_named_buffers(model: torch.nn.Module) -> OrderedDict[str, torch.Tensor]:
    """Return ordered mapping of named buffers from the model."""
    bufs: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, b in model.named_buffers():
        bufs[name] = b
    return bufs


def to_cpu_fp32_named(named_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Detach, move to CPU, and cast to float32 (cloned)."""
    out: Dict[str, torch.Tensor] = {}
    for k, v in named_tensors.items():
        out[k] = v.detach().to("cpu", torch.float32).clone()
    return out


def dot_named(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Compute sum_k <a_k, b_k> over intersection of names.

    Assumes tensors are on CPU and same dtype; upcasts to float64 accumulation.
    """
    acc = torch.zeros((), dtype=torch.float64)
    for k, ta in a.items():
        tb = b.get(k)
        if tb is None:
            continue
        acc += (ta.to(torch.float64) * tb.to(torch.float64)).sum()
    return acc


def flatten_named(a: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten a name-keyed dict to a single 1-D CPU float32 tensor (for cosine checks)."""
    if not a:
        return torch.zeros(0, dtype=torch.float32)
    parts = [v.detach().to("cpu", torch.float32).reshape(-1) for _, v in a.items()]
    return torch.cat(parts, dim=0)

