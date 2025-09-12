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

        # Some HF models register a forward hook (make_inputs_require_grads) when
        # gradient checkpointing is enabled. That hook calls requires_grad_ inside
        # the transformed function, which is incompatible with functorch.jvp.
        # Temporarily disable gradient checkpointing / input-requires-grad hook.
        restore_ckpt = False
        restore_input_req = False
        try:
            if getattr(mdl, "is_gradient_checkpointing", False):
                try:
                    mdl.gradient_checkpointing_disable()
                    restore_ckpt = True
                except Exception:
                    pass
            if hasattr(mdl, "disable_input_require_grads"):
                try:
                    mdl.disable_input_require_grads()
                    restore_input_req = True
                except Exception:
                    pass

            out = functional_call(
                mdl, params_map,
                (input_ids.unsqueeze(0),),
                {"attention_mask": attention_mask.unsqueeze(0), "use_cache": False}
            )
        finally:
            # Restore model hooks/state if we disabled them
            if restore_input_req and hasattr(mdl, "enable_input_require_grads"):
                try:
                    mdl.enable_input_require_grads()
                except Exception:
                    pass
            if restore_ckpt:
                try:
                    mdl.gradient_checkpointing_enable()
                except Exception:
                    pass
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


def make_mb_outputs_closure(
    *,
    model: torch.nn.Module,
    base_map: Dict[str, torch.Tensor],
    names: List[str],
    input_ids_mb: torch.Tensor,          # [B, L]
    attention_mask_mb: torch.Tensor,     # [B, L]
    prompt_lens: List[int],              # [B]
    gen_lens: List[int],                 # [B] (T per sequence)
):
    """
    Build a microbatch closure F_mb(params_tuple) -> (logp_cat[T_mb], H_sum[scalar]) where:
      - logp_cat concatenates per-token log p on realized tokens across all sequences in the
        microbatch (skips sequences with T_b == 0), in order b=0..B-1.
      - H_sum is the sum of full-softmax entropies across all tokens in the microbatch.

    Returns a closure suitable for use with torch.func.jvp.
    """
    mdl = model

    def _F(params_tuple):
        # Merge params
        params_map = dict(base_map)
        for n, t in zip(names, params_tuple):
            params_map[n] = t

        # Disable gradient checkpoint hooks that may call requires_grad_()
        restore_ckpt = False
        restore_input_req = False
        try:
            if getattr(mdl, "is_gradient_checkpointing", False):
                try:
                    mdl.gradient_checkpointing_disable()
                    restore_ckpt = True
                except Exception:
                    pass
            if hasattr(mdl, "disable_input_require_grads"):
                try:
                    mdl.disable_input_require_grads()
                    restore_input_req = True
                except Exception:
                    pass

            logp_list = []
            H_sum_total = torch.zeros((), dtype=torch.float32, device=input_ids_mb.device)

            B = int(input_ids_mb.shape[0])
            for b in range(B):
                T = int(gen_lens[b])
                if T <= 0:
                    continue
                input_ids = input_ids_mb[b]
                attention_mask = attention_mask_mb[b]
                out = functional_call(
                    mdl, params_map,
                    (input_ids.unsqueeze(0),),
                    {"attention_mask": attention_mask.unsqueeze(0), "use_cache": False}
                )
                logits = out.logits  # [1, L, V]
                start = max(int(prompt_lens[b]) - 1, 0)
                end = start + T
                logits_slice = logits[:, start:end, :]    # [1, T, V]
                targets = input_ids[prompt_lens[b]: prompt_lens[b] + T]
                logp_full = torch.log_softmax(logits_slice.to(torch.float32), dim=-1)
                tok_logp = logp_full.gather(-1, targets.view(1, T, 1)).squeeze(-1).squeeze(0)  # [T]
                logp_list.append(tok_logp)

                p = torch.exp(logp_full)
                H_t = -(p * logp_full).sum(dim=-1).squeeze(0)  # [T]
                H_sum_total = H_sum_total + H_t.sum()

            logp_cat = torch.cat(logp_list, dim=0) if len(logp_list) > 0 else torch.zeros(0, device=input_ids_mb.device)
            return logp_cat, H_sum_total
        finally:
            if restore_input_req and hasattr(mdl, "enable_input_require_grads"):
                try:
                    mdl.enable_input_require_grads()
                except Exception:
                    pass
            if restore_ckpt:
                try:
                    mdl.gradient_checkpointing_enable()
                except Exception:
                    pass

    return _F
