#!/usr/bin/env python3
"""
variance_grad_debug.py
----------------------
Purpose
    Reproduce and debug the conditional-variance (alpha-trick) failure:
    "S does not backprop to LoRA params in aligned batch computation."

What it does
    • Loads base + PEFT adapter (QLoRA-ready; 4-bit by default).
    • Builds a left-padded, aligned multi-prompt batch (like your variance path).
    • Computes S over the generated region and probes gradient flow to LoRA params.
    • Compares two variants:
        (A) Correct: adapter activated BEFORE forward (expected to give LoRA grads)
        (B) Wrong:   adapter DISABLED DURING forward (expected to kill LoRA grads)
    • Prints non-None gradient counts and norms; also a LoRA on/off delta sanity check.

Usage
    python variance_grad_debug.py \
      --base Qwen/Qwen2.5-1.5B \
      --adapter /path/to/training_state/step_XX/model \
      --batch 6 --G 4 --prompt-lens 48 --gen-lens 64

Notes
    - Single GPU, no DDP. Uses synthetic tokens (no tokenizer).
    - Ensure bitsandbytes, transformers, peft are installed.
"""

import argparse
from typing import List, Tuple
import torch
import torch.nn.functional as F

# ---------------- Utilities ----------------

def hr(title=""):
    print("\n" + "="*80)
    if title:
        print(title)
        print("="*80)

def load_peft_model(base_id: str, adapter_path: str, device_map: str = "auto"):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel, prepare_model_for_kbit_training

    # 4-bit config for QLoRA-style loading
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_id,
        device_map=device_map,
        torch_dtype=torch.float16,
        quantization_config=bnb
    )
    base = prepare_model_for_kbit_training(base)

    peft = PeftModel.from_pretrained(base, adapter_path, is_trainable=True)
    if hasattr(peft, "enable_input_require_grads"):
        peft.enable_input_require_grads()
    peft.eval()  # deterministic; autograd remains enabled
    return peft

def collect_lora_params(peft_model):
    named = list(peft_model.named_parameters())
    lora = [(n,p) for (n,p) in named
            if p.requires_grad and ("lora_a" in n.lower()
                                    or "lora_b" in n.lower()
                                    or "lm_head" in n)]
    return lora

def make_aligned_batch(B: int, G: int, prompt_len: int, gen_len: int, vocab_size: int, device: str):
    """
    Build aligned batch akin to _align_prompt_batch_with_padding():
      sequences:       [B, G, total_len] left-padded so generation starts together
      attention_masks: [B, G, total_len] 1 for token, 0 for pad
      max_prompt_len:  int (same for all after alignment)
    """
    total_len = prompt_len + gen_len
    pad_id = 0
    seqs = []
    masks = []
    for b in range(B):
        this_prompt = max(1, prompt_len - (b % 3))
        left_pad = prompt_len - this_prompt
        this_gen = max(1, gen_len - (b % 5))
        right_pad = gen_len - this_gen

        per_prompt = []
        per_mask = []
        for g in range(G):
            ptoks = torch.randint(low=10, high=vocab_size-1, size=(this_prompt,), device=device, dtype=torch.long)
            gtoks = torch.randint(low=10, high=vocab_size-1, size=(this_gen,), device=device, dtype=torch.long)
            left  = torch.full((left_pad,),  pad_id, device=device, dtype=torch.long) if left_pad  > 0 else torch.empty(0, dtype=torch.long, device=device)
            right = torch.full((right_pad,), pad_id, device=device, dtype=torch.long) if right_pad > 0 else torch.empty(0, dtype=torch.long, device=device)
            seq = torch.cat([left, ptoks, gtoks, right], dim=0)  # [total_len]
            per_prompt.append(seq)
            per_mask.append((seq != pad_id).long())

        seqs.append(torch.stack(per_prompt, dim=0))   # [G, total_len]
        masks.append(torch.stack(per_mask,  dim=0))   # [G, total_len]

    sequences = torch.stack(seqs, dim=0)         # [B, G, total_len]
    attention_masks = torch.stack(masks, dim=0)  # [B, G, total_len]
    max_prompt_len = prompt_len
    return sequences, attention_masks, max_prompt_len

def compute_S_from_sequences(peft_model, sequences, attention_masks, max_prompt_len,
                             *, activate_adapter_before: bool, disable_adapter_during: bool):
    """
    Compute sequence log-prob sum over generated region only, aligned like your code.
    - If activate_adapter_before=True, we call set_adapter('default') BEFORE forward.
    - If disable_adapter_during=True, we temporarily disable the adapter DURING forward.
    """
    B, G, total_len = sequences.shape
    flat_x = sequences.view(B*G, total_len)
    flat_m = attention_masks.view(B*G, total_len)

    if activate_adapter_before and hasattr(peft_model, "set_adapter"):
        peft_model.set_adapter("default")

    ctx = peft_model.disable_adapter() if disable_adapter_during and hasattr(peft_model, "disable_adapter") else torch.enable_grad()
    # ensure autograd context with a uniform interface
    if not disable_adapter_during:
        # no-op context that supports __enter__/__exit__
        class _Dummy:
            def __enter__(self): return None
            def __exit__(self, exc_type, exc, tb): return False
        ctx = _Dummy()

    with ctx:
        out = peft_model(input_ids=flat_x, attention_mask=flat_m)
        logits = out.logits.float()  # [B*G, T, V]
        lsm = torch.log_softmax(logits, dim=-1)
        targets = flat_x[:, 1:].unsqueeze(-1)
        tok_lp = lsm[:, :-1].gather(2, targets).squeeze(-1)  # [B*G, T-1]
        tok_lp = tok_lp.view(B, G, total_len-1)

        gen_start = int(max_prompt_len) - 1
        gen_lp = tok_lp[:, :, gen_start:]                     # [B, G, Lgen]
        gen_tokens = sequences[:, :, max_prompt_len:]
        gen_mask = (gen_tokens != 0).float()
        L = min(gen_lp.shape[2], gen_mask.shape[2])
        S = (gen_lp[:, :, :L] * gen_mask[:, :, :L]).sum(dim=2)  # [B, G]
        return S

def count_non_none_autograd(S, lora_params):
    # Probe gradient wrt LoRA params using one scalar probe from S
    probe = S[0].sum()
    g = torch.autograd.grad(probe, [p for _,p in lora_params], allow_unused=True, retain_graph=False)
    nnc = sum(int(gi is not None) for gi in g)
    # simple norms for visibility
    nz_norms = sorted([float(gi.detach().float().norm().item()) for gi in g if gi is not None], reverse=True)
    return nnc, nz_norms[:10]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--G", type=int, default=4)
    ap.add_argument("--prompt-lens", type=int, default=48)
    ap.add_argument("--gen-lens", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.set_grad_enabled(True)

    peft = load_peft_model(args.base, args.adapter, device_map=args.device)
    peft.to(device)

    lora = collect_lora_params(peft)
    print(f"LoRA(+lm_head) tensors: {len(lora)}")

    # sanity: LoRA on/off delta
    with torch.no_grad():
        vocab = int(getattr(peft.config, "vocab_size", 32000))
        x = torch.randint(10, vocab-1, (1, 16), device=device, dtype=torch.long)
        m = torch.ones_like(x, device=device)
        peft.set_adapter("default")
        logits_on = peft(x, attention_mask=m).logits
        with peft.disable_adapter():
            logits_off = peft(x, attention_mask=m).logits
        delta = (logits_on - logits_off).abs().max().item()
    print(f"[sanity] LoRA on/off delta: {delta:.3e}")

    vocab = int(getattr(peft.config, "vocab_size", 32000))
    seqs, masks, max_prompt_len = make_aligned_batch(
        args.batch, args.G, args.prompt_lens, args.gen_lens, vocab, device=str(device)
    )

    hr("Variant A (CORRECT): adapter activated BEFORE forward")
    S_a = compute_S_from_sequences(peft, seqs, masks, max_prompt_len,
                                   activate_adapter_before=True,
                                   disable_adapter_during=False)
    nnc_a, norms_a = count_non_none_autograd(S_a, lora)
    print(f"S_a.requires_grad={S_a.requires_grad}, non-None LoRA grads={nnc_a}/{len(lora)}")
    print("Top-10 ||grad|| (Variant A):", ", ".join(f"{v:.3e}" for v in norms_a) if norms_a else "(none)")

    hr("Variant B (WRONG): adapter DISABLED DURING forward (simulated failure mode)")
    S_b = compute_S_from_sequences(peft, seqs, masks, max_prompt_len,
                                   activate_adapter_before=False,
                                   disable_adapter_during=True)
    nnc_b, norms_b = count_non_none_autograd(S_b, lora)
    print(f"S_b.requires_grad={S_b.requires_grad}, non-None LoRA grads={nnc_b}/{len(lora)}")
    print("Top-10 ||grad|| (Variant B):", ", ".join(f"{v:.3e}" for v in norms_b) if norms_b else "(none)")

    hr("Observation")
    print("* Variant A should produce non-None gradients for many LoRA params.")
    print("* Variant B should largely kill LoRA gradients—mimicking your error.")
    print("* If Variant A works here but fails in your repo, move set_adapter('default')")
    print("  *before* the forward that creates logits in _compute_aligned_batch_logprobs,")
    print("  and call the exact PEFT module whose parameters you pass to autograd.grad.")

if __name__ == "__main__":
    main()
