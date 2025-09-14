"""
JVP utilities for forward-mode directional derivatives with torch.func.

Provides:
- snapshot_base_functional_state(model): merged params+buffers mapping at base θ (fp32 params).
- intersect_jvp_primals_tangents(model, base_map, v_named): ordered names and aligned primals/tangents tuples.
- make_seq_outputs_closure(...): builds a per-sequence closure f(params_tuple) -> (logp_vec, H_sum).

These are factored out from DeltaEntropyApprox to reduce bloat and improve reuse.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import torch
from torch.func import functional_call

from entropy_experiments.utils.param_overrides import (
    build_functional_params_named,
    merge_params_and_buffers,
)


def _rb_entropy_full(a: torch.Tensor) -> torch.Tensor:
    """
    Full-softmax RB entropy per step: H = logsumexp(a) - sum softmax(a) * a
    a: [*, V] in fp32
    returns: H: [*]
    """
    Z = torch.logsumexp(a, dim=-1)               # [*]
    p = torch.softmax(a, dim=-1)                 # [*, V]
    return Z - (p * a).sum(dim=-1)               # [*]

def _rb_entropy_top_p(a: torch.Tensor, top_p: float, *, detach_mask: bool = True) -> torch.Tensor:
    """
    Top-p truncated RB entropy per step. If detach_mask=True, the keep mask is
    computed under no_grad to avoid differentiating through sorting/thresholds.
    a: [*, V] in fp32
    returns: H: [*]
    """
    if top_p >= 1.0:
        return _rb_entropy_full(a)
    # compute probabilities for ranking only
    with torch.no_grad() if detach_mask else torch.enable_grad():
        p = torch.softmax(a, dim=-1)             # [*, V]
        p_sorted, idx_sorted = p.sort(dim=-1, descending=True)   # [*, V]
        csum = p_sorted.cumsum(dim=-1)                           # [*, V]
        keep_sorted = (csum - p_sorted) <= top_p                 # minimal set with prev csum <= top_p
        keep_sorted[..., 0] = True                               # ensure ≥1 token
        keep = torch.zeros_like(p, dtype=torch.bool)
        keep.scatter_(dim=-1, index=idx_sorted, src=keep_sorted)
    a_masked = a.masked_fill(~keep, float('-inf'))
    Z_S = torch.logsumexp(a_masked, dim=-1)                      # [*]
    q = torch.softmax(a_masked, dim=-1)                          # [*, V]
    return Z_S - (q * a).sum(dim=-1)                             # [*]



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
    temperature: float = 1.0,
    top_p: float = 1.0,
    truncate_grad: bool = True,
):
    """
    Build a closure f(params_tuple) -> (logp_vec[T], H_sum[scalar]) for use with torch.func.jvp.
    - Computes token log-probs on realized tokens in fp32.
    - Computes RB entropy sum using the same formula as the "true" path:
        * full-softmax if top_p >= 1
        * top-p truncation (masking & renormalization) otherwise.
      If truncate_grad is True (default), the truncation mask is treated as constant.
    """
    mdl = model
    t_eps = max(float(temperature), 1e-8)

    def _f(params_tuple):
        params_map = dict(base_map)
        for n, t in zip(names, params_tuple):
            params_map[n] = t

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
            if restore_input_req and hasattr(mdl, "enable_input_require_grads"):
                try: mdl.enable_input_require_grads()
                except Exception: pass
            if restore_ckpt:
                try: mdl.gradient_checkpointing_enable()
                except Exception: pass

        logits = out.logits                         # [1, L, V]
        start = max(int(prompt_len) - 1, 0)
        end = start + int(T)
        logits_slice = logits[:, start:end, :]      # [1, T, V]
        a = (logits_slice.to(torch.float32)) / t_eps

        # token log-prob on realized tokens (full-softmax, same as before)
        logp_full = torch.log_softmax(a, dim=-1)    # [1, T, V]
        targets = input_ids[prompt_len: prompt_len + T]
        logp_vec = logp_full.gather(-1, targets.view(1, T, 1)).squeeze(-1).squeeze(0)  # [T]

        # RB entropy (uses same formula as delta_entropy_true)
        if top_p >= 1.0:
            H_t = _rb_entropy_full(a.squeeze(0))                    # [T]
        else:
            H_t = _rb_entropy_top_p(a.squeeze(0), top_p, detach_mask=truncate_grad)  # [T]
        H_sum = H_t.sum()                                           # scalar

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
    gen_lens: List[int],                 # [B]
    temperature: float = 1.0,
    top_p: float = 1.0,
    truncate_grad: bool = True,
):
    """
    F_mb(params_tuple) -> (logp_cat[T_mb], H_sum[scalar]) with RB entropy.
    - logp_cat concatenates realized-token log-probs across sequences (full-softmax).
    - H_sum sums RB entropies token-wise; for top_p<1 uses truncated RB with an
      optional detached mask (default).
    """
    mdl = model
    t_eps = max(float(temperature), 1e-8)

    def _F(params_tuple):
        params_map = dict(base_map)
        for n, t in zip(names, params_tuple):
            params_map[n] = t

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
                logits = out.logits                      # [1, L, V]
                start = max(int(prompt_lens[b]) - 1, 0)
                end = start + T
                logits_slice = logits[:, start:end, :]   # [1, T, V]
                a = (logits_slice.to(torch.float32)) / t_eps

                # realized-token log-probs (full-softmax, unchanged)
                logp_full = torch.log_softmax(a, dim=-1) # [1, T, V]
                targets = input_ids[prompt_lens[b]: prompt_lens[b] + T]
                tok_logp = logp_full.gather(-1, targets.view(1, T, 1)).squeeze(-1).squeeze(0)
                logp_list.append(tok_logp)

                # RB entropy contribution
                if top_p >= 1.0:
                    H_t = _rb_entropy_full(a.squeeze(0))                    # [T]
                else:
                    H_t = _rb_entropy_top_p(a.squeeze(0), top_p, detach_mask=truncate_grad)  # [T]
                H_sum_total = H_sum_total + H_t.sum()

            logp_cat = torch.cat(logp_list, dim=0) if len(logp_list) > 0 else torch.zeros(0, device=input_ids_mb.device)
            return logp_cat, H_sum_total
        finally:
            if restore_input_req and hasattr(mdl, "enable_input_require_grads"):
                try: mdl.enable_input_require_grads()
                except Exception: pass
            if restore_ckpt:
                try: mdl.gradient_checkpointing_enable()
                except Exception: pass

    return _F
