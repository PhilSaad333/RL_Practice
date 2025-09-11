"""
JVP utilities for forward-mode directional derivatives with torch.func.

Provides:
- snapshot_base_functional_state(model): merged params+buffers mapping at base θ (fp32 params).
- intersect_jvp_primals_tangents(model, base_map, v_named): ordered names and aligned primals/tangents tuples.
- make_seq_outputs_closure(...): builds a per-sequence closure f(params_tuple) -> (logp_vec, H_sum).

These are factored out from DeltaEntropyApprox to reduce bloat and improve reuse.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch.func import functional_call

from entropy_experiments.utils.param_overrides import (
    build_functional_params_named,
    merge_params_and_buffers,
)


@torch.no_grad()
def snapshot_base_functional_state(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Return merged params+buffers mapping at base θ (detached, fp32 params)."""
    p_dict, b_dict = build_functional_params_named(
        model,
        v_named=None,
        eta=0.0,
        strict=True,
        allow_frozen_updates=False,
        detach_params=True,
        detach_buffers=True,
        force_param_dtype=torch.float32,
        force_buffer_dtype=None,
    )
    return merge_params_and_buffers(p_dict, b_dict)


def intersect_jvp_primals_tangents(
    model: torch.nn.Module,
    base_map: Dict[str, torch.Tensor],
    v_named: Dict[str, torch.Tensor],
) -> Tuple[List[str], Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
    """
    From base_map and v_named, build the ordered intersecting names and aligned
    primals/tangents tuples on the correct device/dtype.
    """
    device = next(model.parameters()).device
    names: List[str] = []
    primals: List[torch.Tensor] = []
    tangents: List[torch.Tensor] = []
    for n, p in model.named_parameters():
        if (not p.requires_grad) or (n not in v_named):
            continue
        names.append(n)
        primals.append(base_map[n].to(device=device))
        tangents.append(v_named[n].to(device=device, dtype=base_map[n].dtype))
    if len(names) == 0:
        raise ValueError("[jvp_utils] No intersecting trainables for JVP.")
    return names, tuple(primals), tuple(tangents)


def make_seq_outputs_closure(
    *,
    model: torch.nn.Module,
    base_map: Dict[str, torch.Tensor],
    names: List[str],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_len: int,
    T: int,
):
    """
    Build a closure f(params_tuple) -> (logp_vec[T], H_sum[scalar]) for use with torch.func.jvp.
    - Merges replacements under names into base_map, runs functional_call without autocast or cache.
    - Computes log-probs on realized tokens and full-softmax entropy sum in fp32.
    """
    mdl = model

    def _f(params_tuple):
        params_map = dict(base_map)
        for n, t in zip(names, params_tuple):
            params_map[n] = t
        out = functional_call(
            mdl, params_map,
            (input_ids.unsqueeze(0),),
            {"attention_mask": attention_mask.unsqueeze(0), "use_cache": False}
        )
        logits = out.logits
        start = max(int(prompt_len) - 1, 0)
        end = start + int(T)
        logits_slice = logits[:, start:end, :]
        targets = input_ids[prompt_len: prompt_len + T]
        logp_full = torch.log_softmax(logits_slice.to(torch.float32), dim=-1)
        logp_vec = logp_full.gather(-1, targets.view(1, T, 1)).squeeze(-1).squeeze(0)
        p = torch.exp(logp_full)
        H_t = -(p * logp_full).sum(dim=-1).squeeze(0)
        H_sum = H_t.sum()
        return logp_vec, H_sum

    return _f

