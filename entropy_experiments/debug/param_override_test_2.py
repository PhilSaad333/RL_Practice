#!/usr/bin/env python3
"""
Fixed-Sequence Param-Override Sanity Probe (no SequenceProcessor)
=================================================================

What this does
--------------
1) Initializes OfflineEntropyProbe from your config and loads the LoRA checkpoint.
2) Samples a small U-batch and computes the update vector v_named (per-unit-LR).
3) Uses param_overrides.build_functional_params_named(...) to form θ' = θ + η·v.
4) Runs torch.func.functional_call directly on a fixed tokenized sentence
   (no SequenceProcessor), with use_cache=False for determinism.
5) Reports Δlogits norms across an η-grid and a linearity sanity ratio.

Why this matters
----------------
If this script shows clean linear scaling of Δlogits with η, your param-override
path and functional_call mapping are sound. Any nonlinearity then likely lives
in SequenceProcessor post-processing or precision divergence between paths.

Usage
-----
$ python fixed_seq_param_override_sanity.py \
    --config entropy_experiments/configs/colab_config.yaml \
    --eta_main 1e-5 --eta_tiny 1e-10 \
    --text "this is a test sequence. the logits should change continuously!"

Dependencies in your repo
-------------------------
- entropy_experiments/offline_entropy_probe.py
- entropy_experiments/update_vector.py             (compute_update_vector)
- entropy_experiments/utils/param_overrides.py     (build_functional_params_named, merge)
- entropy_experiments/utils/precision_utils.py     (optional: apply_global_precision)
"""

import os
import sys
import gc
import math
import argparse
from typing import Dict, Tuple, Any

import torch
import numpy as np
import yaml
from transformers import AutoTokenizer

# --- CLI ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True,
                   help="Path to YAML config (e.g., entropy_experiments/configs/colab_config.yaml)")
    p.add_argument("--text", type=str, default="this is a test sequence. the logits should change continuously!",
                   help="Fixed text to tokenize and probe.")
    p.add_argument("--tokenizer_id", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                   help="Tokenizer ID to use (must match model's vocab).")
    p.add_argument("--eta_main", type=float, default=1e-5, help="Main η used in linearity ratio.")
    p.add_argument("--eta_tiny", type=float, default=1e-10, help="Tiny η used in linearity ratio.")
    p.add_argument("--eta_grid", type=float, nargs="*", default=[0.0, 1e-10, 1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5],
                   help="Grid of η values to sweep.")
    p.add_argument("--u_batch", type=int, default=8, help="U batch size for update-vector estimation.")
    p.add_argument("--u_G", type=int, default=4, help="Generations per prompt for U.")
    p.add_argument("--print_topk", type=int, default=5, help="How many token positions to preview by max Δ∞.")
    p.add_argument("--dtype", type=str, default="float32", choices=["float32","float64"],
                   help="Compute dtype for the forward path and overrides.")
    return p.parse_args()

# --- Utilities ----------------------------------------------------------------

def setup_pythonpath(project_root: str):
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    os.environ["PYTHONPATH"] = project_root + (":" + os.environ.get("PYTHONPATH","") if os.environ.get("PYTHONPATH") else "")

def unwrap(module: torch.nn.Module) -> torch.nn.Module:
    m = module
    while hasattr(m, "module"):
        m = m.module
    if hasattr(m, "_orig_mod"):
        try:
            m = m._orig_mod
        except Exception:
            pass
    return m

def get_model_device(module: torch.nn.Module) -> torch.device:
    return next(unwrap(module).parameters()).device

def report_mem(tag: str):
    if not torch.cuda.is_available():
        print(f"[{tag}] CPU-only.")
        return
    torch.cuda.synchronize()
    gb = 1024**3
    alloc = torch.cuda.memory_allocated() / gb
    reserv = torch.cuda.memory_reserved() / gb
    print(f"[{tag}] GPU mem: alloc={alloc:.2f} GB, reserved={reserv:.2f} GB")

def logits_under_params(mdl, input_ids, attention_mask, *, mapping: Dict[str, torch.Tensor] | None):
    """
    Eval forward pass with optional name→tensor mapping via functional_call.
    Always disables KV cache to avoid hidden state carryover.
    Returns logits [1, T, V] (on device).
    """
    mdl = unwrap(mdl)
    was_training = mdl.training
    mdl.eval()
    try:
        if mapping is None:
            with torch.no_grad():
                out = mdl(input_ids, attention_mask=attention_mask, use_cache=False)
        else:
            out = torch.func.functional_call(
                mdl, mapping, (input_ids,), {"attention_mask": attention_mask, "use_cache": False}
            )
        return out.logits if hasattr(out, "logits") else out[0]
    finally:
        if was_training:
            mdl.train()

def tokenwise_norms(delta_logits_cpu: torch.Tensor):
    """delta_logits_cpu: [1, T, V] -> per-token (∞-norm, ℓ2)"""
    d = delta_logits_cpu.squeeze(0)         # [T, V]
    linf = d.abs().max(dim=-1).values       # [T]
    l2   = d.pow(2).sum(dim=-1).sqrt()      # [T]
    return linf, l2

def safe_ratio(a: float, b: float) -> float:
    return float(a / b) if (b != 0.0 and math.isfinite(b)) else float("nan")

# --- Main ---------------------------------------------------------------------

def main():
    args = parse_args()

    # Project root = CWD; adjust if launching elsewhere
    project_root = os.getcwd()
    setup_pythonpath(project_root)

    # Torch/global precision: keep it explicit and consistent
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True  # ok for f32; set False if you want stricter match
    if args.dtype == "float64":
        torch.set_default_dtype(torch.float64)
    else:
        torch.set_default_dtype(torch.float32)

    # Load config
    cfg_path = os.path.join(project_root, args.config)
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    print(f"✓ Loaded config: {args.config}")

    # Apply optional global precision profile if you keep one in your repo
    try:
        from entropy_experiments.utils.precision_utils import apply_global_precision
        apply_global_precision(
            allow_tf32=True,
            matmul_precision="high",
        )
    except Exception:
        pass  # optional

    # Import project bits late (PYTHONPATH is ready)
    from entropy_experiments.offline_entropy_probe import OfflineEntropyProbe
    from entropy_experiments.update_vector import compute_update_vector
    from entropy_experiments.utils.param_overrides import (
        build_functional_params_named,
        merge_params_and_buffers,
    )

    # Build the probe & load checkpoint
    probe = OfflineEntropyProbe(cfg)
    ckpt_cfg = cfg.get("checkpoint", {})
    adapter_path   = ckpt_cfg.get("checkpoint_path", "")
    optimizer_path = ckpt_cfg.get("optimizer_path", "")
    if not adapter_path:
        raise ValueError("Config missing checkpoint.checkpoint_path")

    print("Loading LoRA checkpoint …")
    probe.load_checkpoint(adapter_path, optimizer_path if optimizer_path else None)
    model = probe.model
    device = get_model_device(model)
    print(f"✓ Model ready on {device}: {type(unwrap(model)).__name__}")

    # --- Phase A: compute update vector from a tiny U-batch -------------------
    print("Sampling a small U-batch and computing update vector …")
    probe._ensure_sequence_processor()
    dataset = cfg["batch_config"]["dataset_name"]
    _, U_split = probe._get_splits()

    U_sequences, U_logprobs, _ = probe._sequence_processor.generate_with_logprobs(
        prompts=None, G=args.u_G, dataset_name=dataset, split=U_split,
        num_prompts=args.u_batch, compute_rb=True
    )
    U_batch = probe._pack_U_from_sequences(U_sequences, U_logprobs.rewards)

    # Build or reuse optimizer
    if getattr(probe, "optimizer", None) is not None:
        optimizer = probe.optimizer
    else:
        trainable = [p for _, p in model.named_parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=args.eta_main)

    v_named, stats = compute_update_vector(
        model=model, optimizer=optimizer, U_batch=U_batch, config=probe.config, logger=probe.logger
    )
    v_norm = float(torch.sqrt(sum((v.double()**2).sum() for v in v_named.values())).item())
    print(f"✓ Update vector: {len(v_named)} tensors, ||v||₂ ≈ {v_norm:.3e}")

    # --- Phase B: fixed-sequence logits under θ and θ+ηv ---------------------
    print("\nFixed-sequence functional_call probe (no SequenceProcessor) …")
    tok = AutoTokenizer.from_pretrained(args.tokenizer_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    enc = tok(args.text, return_tensors="pt", add_special_tokens=True, padding=False)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # Guard: vocab mismatch
    mdl_unwrapped = unwrap(model)
    vocab_param = None
    vocab_size  = None
    for n, p in mdl_unwrapped.named_parameters():
        if n.endswith("model.embed_tokens.weight"):
            vocab_param, vocab_size = n, p.shape[0]
            break
    if vocab_size is None:
        for n, p in mdl_unwrapped.named_parameters():
            if n.endswith("lm_head.weight"):
                vocab_param, vocab_size = n, p.shape[0]
                break
    if vocab_size is not None:
        max_id = int(input_ids.max().item())
        if max_id >= vocab_size:
            raise RuntimeError(
                f"Token id {max_id} exceeds model vocab size {vocab_size} (param: {vocab_param}). "
                f"Tokenizer/model mismatch."
            )

    # Baseline logits at θ (direct call; no mapping)
    with torch.no_grad():
        logits0 = logits_under_params(model, input_ids, attention_mask, mapping=None)
    logits0_cpu = logits0.detach().cpu()    # [1, T, V]
    T = logits0_cpu.shape[1]
    tokens = tok.convert_ids_to_tokens(input_ids.squeeze(0).tolist())
    print(f"Sequence length T={T}, vocab={logits0_cpu.shape[-1]}")

    # Sweep over η
    records = []
    for eta in args.eta_grid:
        # Build θ' = θ + η v (params only), plus buffers passthrough
        params_dict, buffers_dict = build_functional_params_named(
            model, v_named, eta,
            detach_params=True, detach_buffers=True,
            force_param_dtype=torch.float32 if args.dtype == "float32" else torch.float64,
            force_buffer_dtype=torch.float32 if args.dtype == "float32" else torch.float64,
        )
        mapping = merge_params_and_buffers(params_dict, buffers_dict)

        logits_eta = logits_under_params(model, input_ids, attention_mask, mapping=mapping)
        d_logits = (logits_eta.detach().cpu() - logits0_cpu)  # [1,T,V]
        linf_all = float(d_logits.abs().max().item())
        l2_all   = float(d_logits.pow(2).sum().sqrt().item())

        # per-token norms (∞, 2)
        linf_tok, l2_tok = tokenwise_norms(d_logits)
        idx_top = torch.topk(linf_tok, k=min(args.print_topk, T)).indices.tolist()
        preview = ", ".join([f"(t={i}, tok='{tokens[i]}', Δ∞={linf_tok[i].item():.2e})" for i in idx_top])

        print(f"η={eta: .1e}  ||Δlogits||∞={linf_all:.3e}  ||Δlogits||₂={l2_all:.3e}")
        print(f"   top Δ∞ tokens: {preview}")

        records.append({
            "eta": eta, "linf_all": linf_all, "l2_all": l2_all,
            "linf_tok": linf_tok, "l2_tok": l2_tok
        })

        # cleanup
        del params_dict, buffers_dict, mapping, logits_eta, d_logits
        torch.cuda.empty_cache()

    # Linearity check: compare two smallest non-zero η
    nz = [r for r in records if r["eta"] > 0]
    if len(nz) >= 2:
        a, b = nz[0], nz[1]      # e.g., 1e-10 and 1e-8
        r_inf = safe_ratio(a["linf_all"], b["linf_all"])
        r_l2  = safe_ratio(a["l2_all"], b["l2_all"])
        print(f"\nLinearity sanity (η={a['eta']:.1e} vs η={b['eta']:.1e}): "
              f"Δ∞ ratio≈{r_inf:.3f}, Δ₂ ratio≈{r_l2:.3f}, expected≈{a['eta']/b['eta']:.3f}")

    # Targeted ratio: tiny vs main
    eta_to_rec = {r["eta"]: r for r in records}
    if args.eta_tiny in eta_to_rec and args.eta_main in eta_to_rec:
        ra = eta_to_rec[args.eta_tiny]; rb = eta_to_rec[args.eta_main]
        r_inf = safe_ratio(ra["linf_all"], rb["linf_all"])
        r_l2  = safe_ratio(ra["l2_all"],  rb["l2_all"])
        print(f"\nTargeted ratio η_tiny/η_main={args.eta_tiny/args.eta_main:.3e}: "
              f"Δ∞ ratio≈{r_inf:.3e}, Δ₂ ratio≈{r_l2:.3e}")

    report_mem("after fixed-seq probe")
    del logits0, logits0_cpu
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
