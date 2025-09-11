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


def get_optimizer_named_params(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> OrderedDict[str, torch.nn.Parameter]:
    """Return ordered mapping of parameters that the optimizer updates.

    Filters model.named_parameters() by membership in optimizer.param_groups.
    """
    # Build set of parameter object IDs present in optimizer groups
    opt_param_ids = set()
    for group in optimizer.param_groups:
        for p in group.get("params", []):
            if p is not None:
                opt_param_ids.add(id(p))

    params: OrderedDict[str, torch.nn.Parameter] = OrderedDict()
    for name, p in model.named_parameters():
        if id(p) in opt_param_ids:
            params[name] = p
    return params


def to_cpu_fp32_named(named_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Detach, move to CPU, and cast to float32 (cloned)."""
    out: Dict[str, torch.Tensor] = {}
    for k, v in named_tensors.items():
        out[k] = v.detach().to("cpu", torch.float32).clone()
    return out


def dot_named_old(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> torch.Tensor:
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



def _pairwise_sum64(scalars: list[torch.Tensor]) -> torch.Tensor:
    """Pairwise (tree) reduction in float64 over a list of 0-D tensors on CPU."""
    if not scalars:
        return torch.zeros((), dtype=torch.float64)
    vals = [s.to("cpu", torch.float64) for s in scalars]
    while len(vals) > 1:
        nxt = []
        for i in range(0, len(vals), 2):
            if i + 1 < len(vals):
                nxt.append(vals[i] + vals[i + 1])
            else:
                nxt.append(vals[i])
        vals = nxt
    return vals[0]


def _dot_1d_pairwise64(x: torch.Tensor, y: torch.Tensor, block_elems: int = 2_000_000) -> torch.Tensor:
    """
    Float64 dot over 1-D CPU tensors using blockwise product + pairwise block reduction.
    Avoids huge temporaries and improves numeric stability.
    """
    x = x.detach().to("cpu", torch.float64).reshape(-1)
    y = y.detach().to("cpu", torch.float64).reshape(-1)
    n = x.numel()
    if n == 0:
        return torch.zeros((), dtype=torch.float64)
    partials: list[torch.Tensor] = []
    for i in range(0, n, block_elems):
        j = min(i + block_elems, n)
        # Each block sum is already float64, but we still reduce pairwise across blocks
        partials.append((x[i:j] * y[i:j]).sum())
    return _pairwise_sum64(partials)


def dot_named(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor], *, block_elems: int = 2_000_000) -> torch.Tensor:
    """
    Numerically-stable ∑_k ⟨a_k, b_k⟩ over the intersection of names.
    - Upcasts to float64 on CPU.
    - Uses blockwise inner sums and pairwise reduction across blocks and across names.
    """
    per_name: list[torch.Tensor] = []
    for name, ta in a.items():
        tb = b.get(name)
        if tb is None:
            continue
        if ta.shape != tb.shape:
            raise ValueError(f"Shape mismatch for {name}: {tuple(ta.shape)} vs {tuple(tb.shape)}")
        per_name.append(_dot_1d_pairwise64(ta, tb, block_elems=block_elems))
    return _pairwise_sum64(per_name)
