"""
Variance utilities for microbatch shard contributions.

Functions:
- gather_named_grads: collect grads for intersecting params (CPU/fp32 snapshot).
- update_shard_contrib: update cumulative dot and append microbatch contribution.
- compute_variance_info: compute SE across microbatches and optional jackknife SE.
"""

from __future__ import annotations

from typing import Dict, List

import math
import torch

from entropy_experiments.utils.param_registry import dot_named


def gather_named_grads(name_to_param: "Dict[str, torch.nn.Parameter]") -> Dict[str, torch.Tensor]:
    grads_named: Dict[str, torch.Tensor] = {}
    for n, p in name_to_param.items():
        if p.grad is None:
            grads_named[n] = torch.zeros_like(p.detach()).to("cpu", torch.float32)
        else:
            grads_named[n] = p.grad.detach().to("cpu", torch.float32).clone()
    return grads_named


def update_shard_contrib(
    name_to_param: "Dict[str, torch.nn.Parameter]",
    v_named_cpu: Dict[str, torch.Tensor],
    contribs: List[float],
    last_dot_val: float,
) -> float:
    grads_named_mb: Dict[str, torch.Tensor] = {}
    for n, p in name_to_param.items():
        if p.grad is None:
            grads_named_mb[n] = torch.zeros_like(p.detach()).to("cpu", torch.float32)
        else:
            grads_named_mb[n] = p.grad.detach().to("cpu", torch.float32)
    dot_now = float(dot_named(grads_named_mb, v_named_cpu).item())
    contribs.append(dot_now - last_dot_val)
    return dot_now


def compute_variance_info(
    contribs: List[float],
    *,
    debug: bool = False,
    use_jackknife: bool = True,
) -> Dict[str, object]:
    if not contribs:
        return {"num_shards": 0, "se_shard": 0.0}
    M = len(contribs)
    mean_c = sum(contribs) / M
    var_c = sum((x - mean_c) ** 2 for x in contribs) / max(M - 1, 1)
    se_shard = (var_c ** 0.5) / (M ** 0.5)
    out: Dict[str, object] = {"num_shards": M, "se_shard": se_shard}
    if use_jackknife and M > 1:
        S = sum(contribs)
        jack = [(S - contribs[i]) / (M - 1) for i in range(M)]
        jack_mean = sum(jack) / M
        jack_var = (M - 1) * sum((m - jack_mean) ** 2 for m in jack) / M
        out["se_jackknife"] = jack_var ** 0.5
        if debug:
            out["contribs"] = contribs
    return out

