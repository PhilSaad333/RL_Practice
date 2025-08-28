
#!/usr/bin/env python3
"""
Alpha‑Trick Gradient Probe for LoRA/QLoRA (single‑GPU, non‑DDP)
---------------------------------------------------------------

Goal
    Reproduce and diagnose per‑sample gradient issues (e.g., None/zero grads) when
    combining PEFT (LoRA/QLoRA) with PyTorch autograd, using the "alpha trick":
        L_vec = [L_1, ..., L_B]  (per‑sequence losses)
        L_alpha = sum_i alpha_i * L_i
    and verifying that
        ∂L_alpha/∂θ  ==  sum_i alpha_i * ∂L_i/∂θ
    via both .backward() and torch.autograd.grad (vector‑Jacobian product).

What it does
    1) Loads base + PEFT adapter (QLoRA‑ready).
    2) Synthesizes a tiny batch of token sequences (no tokenizer needed).
    3) Runs a forward pass to compute:
        • per‑token log‑probs,
        • per‑sequence log‑prob S_i,
        • per‑sequence NLL L_i = −S_i (masked).
    4) Runs several gradient experiments over LoRA trainables:
        A) backward on L_alpha (alpha=ones, random, basis vectors),
        B) autograd.grad on L_vec with grad_outputs=alpha,
       and compares norms and max differences parameter‑wise.
    5) Prints detailed tables of which LoRA params have non‑None/ non‑zero grads.

Notes
    • This script intentionally avoids DDP; run it on a single GPU for clarity.
    • Gradient checkpointing is disabled by default; it can interfere with input
      grad plumbing if not set up exactly right. You may pass --use-checkpointing
      to opt in.
    • Requires: transformers, peft, bitsandbytes installed in your environment.
"""

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------
# Utilities
# ----------------------

def hr(title: str = "", ch: str = "=", width: int = 80):
    print("\n" + ch * width)
    if title:
        print(title)
        print(ch * width)

def fmt(x: float) -> str:
    ax = abs(x)
    if ax == 0: return "0"
    if ax < 1e-4 or ax >= 1e4: return f"{x:.3e}"
    return f"{x:.6f}"

def collect_lora_named_params(model) -> List[Tuple[str, nn.Parameter]]:
    m = model.module if hasattr(model, "module") else model
    named = []
    for n, p in m.named_parameters():
        n_low = n.lower()
        if ("lora_a" in n_low) or ("lora_b" in n_low) or ("lm_head" in n_low and p.requires_grad):
            if p.requires_grad:
                named.append((n, p))
    return named

def zero_grads(params: Sequence[nn.Parameter]):
    for p in params:
        if p.grad is not None:
            p.grad = None

def flat_grads(params: Sequence[nn.Parameter]) -> torch.Tensor:
    chunks = []
    for p in params:
        if p.grad is None:
            chunks.append(torch.zeros(p.numel(), dtype=torch.float32, device="cpu"))
        else:
            chunks.append(p.grad.detach().float().reshape(-1).cpu())
    return torch.cat(chunks) if chunks else torch.zeros(0)

def autograd_grad_vector(L_vec: torch.Tensor,
                         params: Sequence[nn.Parameter],
                         alpha: torch.Tensor,
                         retain_graph: bool = True,
                         allow_unused: bool = True) -> List[torch.Tensor]:
    """
    Compute sum_i alpha_i * ∂L_i/∂θ via vector‑Jacobian product.
    Returns a list of per‑param tensors (or None when unused and allow_unused=True).
    """
    g = torch.autograd.grad(
        outputs=L_vec, inputs=list(params),
        grad_outputs=alpha, retain_graph=retain_graph,
        allow_unused=allow_unused, create_graph=False
    )
    return list(g)

def list_to_flat(grads: List[torch.Tensor], params: Sequence[nn.Parameter]) -> torch.Tensor:
    out = []
    for gi, p in zip(grads, params):
        if gi is None:
            out.append(torch.zeros(p.numel(), dtype=torch.float32, device="cpu"))
        else:
            out.append(gi.detach().float().reshape(-1).cpu())
    return torch.cat(out) if out else torch.zeros(0)

# ----------------------
# Model loading (QLoRA‑ready)
# ----------------------

def load_peft_model(base_id: str, adapter_path: str, device_map: str = "auto", use_checkpointing: bool = False):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel, prepare_model_for_kbit_training

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_id, device_map=device_map, torch_dtype=torch.float16
    )
    base = prepare_model_for_kbit_training(base)
    if use_checkpointing and hasattr(base, "gradient_checkpointing_enable"):
        base.gradient_checkpointing_enable()
        if hasattr(base.config, "use_cache"):
            base.config.use_cache = False

    peft_model = PeftModel.from_pretrained(base, adapter_path, is_trainable=True)
    # required for proper gradient plumbing under k-bit + checkpointing
    if hasattr(peft_model, "enable_input_require_grads"):
        peft_model.enable_input_require_grads()

    peft_model.train(False)   # disable dropout for determinism
    return peft_model

# ----------------------
# Synthetic batch + forward w/ logprobs
# ----------------------

@torch.no_grad()
def make_dummy_batch(vocab_size: int, B: int, T: int, device: torch.device):
    # avoid special tokens at 0..9 for safety; keep labels aligned for next‑token
    x = torch.randint(low=10, high=vocab_size - 1, size=(B, T), device=device, dtype=torch.long)
    attn = torch.ones((B, T), device=device, dtype=torch.long)
    y = x.clone()
    y[:, :-1] = x[:, 1:]
    y[:, -1] = -100  # ignore last token
    return x, attn, y

def forward_with_logprobs(model, x, attn, y):
    """
    Returns:
        logp_tok:  [B, T]  log‑probs for each token position (masked at ignore_index)
        S_seq:     [B]     sequence log‑prob sums over valid positions
        L_vec:     [B]     per‑sequence NLL (−S_seq)
    """
    out = model(input_ids=x, attention_mask=attn)
    logits = out.logits  # [B, T, V]
    lsm = F.log_softmax(logits, dim=-1)
    # gather per‑token logp
    # for ignore_index, substitute a valid index (e.g., 0) then zero it out with mask
    gather_y = y.masked_fill(y < 0, 0)
    logp_tok = lsm.gather(-1, gather_y.unsqueeze(-1)).squeeze(-1)  # [B, T]
    mask = (y != -100).to(logp_tok.dtype)
    logp_tok = logp_tok * mask
    S_seq = logp_tok.sum(dim=1)                    # [B]
    L_vec = -S_seq                                 # NLL per sequence
    return logp_tok, S_seq, L_vec

# ----------------------
# Experiments
# ----------------------

def run_experiments(model, B: int, T: int, device: torch.device):
    vocab = int(getattr(model.config, "vocab_size", 32000))
    x, attn, y = make_dummy_batch(vocab, B, T, device)
    # we need grads through parameters, not through inputs here
    for t in (x, attn, y):
        t.requires_grad_(False)

    logp_tok, S_seq, L_vec = forward_with_logprobs(model, x, attn, y)
    hr("Forward summary")
    print(f"B={B}  T={T}  vocab={vocab}")
    print(f"S_seq (first 4): {', '.join(fmt(float(v)) for v in S_seq[:4].tolist())}")

    # choose parameter set = LoRA (and trainable lm_head if any)
    lora_named = collect_lora_named_params(model)
    params = [p for _, p in lora_named]
    names  = [n for n, _ in lora_named]
    print(f"LoRA(+lm_head) tensors: {len(params)}")

    if len(params) == 0:
        print("No trainable LoRA parameters found. Are adapters loaded with is_trainable=True?")
        return

    # common alphas to test
    ones = torch.ones(B, dtype=torch.float32, device=device)
    rnd  = torch.randn(B, dtype=torch.float32, device=device)
    basis = [torch.zeros(B, dtype=torch.float32, device=device) for _ in range(min(B, 3))]
    for i, e in enumerate(basis):
        e[i] = 1.0

    # Helper: compare backward vs autograd for a given alpha
    def compare_for_alpha(alpha: torch.Tensor, label: str):
        # Method 1: backward on (alpha * L_vec).sum()
        zero_grads(params)
        L_alpha = (alpha * L_vec).sum()
        L_alpha.backward(retain_graph=True)
        g_back = flat_grads(params)

        # Method 2: autograd.grad on L_vec with grad_outputs=alpha
        g_list = autograd_grad_vector(L_vec, params, alpha, retain_graph=True, allow_unused=True)
        # Count Nones
        none_ct = sum(int(gi is None) for gi in g_list)
        g_auto = list_to_flat(g_list, params)

        # Diagnostics
        diff = (g_back - g_auto).abs()
        l2_back = float(torch.linalg.vector_norm(g_back).item())
        l2_auto = float(torch.linalg.vector_norm(g_auto).item())
        l2_diff = float(torch.linalg.vector_norm(diff).item())
        lin_rel = l2_diff / max(1e-12, max(l2_back, l2_auto))

        print(f"[{label}] ||g_back||={fmt(l2_back)}  ||g_auto||={fmt(l2_auto)}  ||diff||={fmt(l2_diff)}  rel={fmt(lin_rel)}  none_in_auto={none_ct}")

        # Top‑k param‑wise norms
        k = min(10, len(params))
        per_param = []
        off = 0
        for (n, p), gi in zip(lora_named, g_list):
            sz = p.numel()
            gi_norm = 0.0 if gi is None else float(gi.detach().float().reshape(-1).norm().item())
            per_param.append((n, gi_norm))
            off += sz
        per_param.sort(key=lambda t: t[1], reverse=True)
        print("   Top‑10 by ||grad|| (autograd):")
        for j in range(k):
            n, v = per_param[j]
            print(f"     {j+1:>2}. {n:60s}  {fmt(v)}")

    hr("Experiment A: alpha = ones")
    compare_for_alpha(ones, "alpha=ones")

    hr("Experiment B: alpha = random N(0,1)")
    compare_for_alpha(rnd, "alpha=random")

    for i, e in enumerate(basis):
        hr(f"Experiment C{i+1}: alpha = e_{i}")
        compare_for_alpha(e, f"alpha=e_{i}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, required=True, help="HF id / local path (e.g., Qwen/Qwen2.5-1.5B)")
    ap.add_argument("--adapter", type=str, required=True, help="Path to PEFT adapter dir (e.g., .../training_state/step_X/model)")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seqlen", type=int, default=32)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--use-checkpointing", action="store_true", help="Enable gradient checkpointing on the base model")
    args = ap.parse_args()

    torch.set_grad_enabled(True)

    device = torch.device(args.device)
    model = load_peft_model(args.base, args.adapter, device_map=args.device, use_checkpointing=args.use_checkpointing)
    model.to(device)

    with torch.autograd.set_detect_anomaly(False):
        run_experiments(model, B=args.batch, T=args.seqlen, device=device)

if __name__ == "__main__":
    main()
