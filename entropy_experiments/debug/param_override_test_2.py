#!/usr/bin/env python3
"""
Fixed-Sequence Param-Override Sanity Probe
==========================================

- Loads your model via OfflineEntropyProbe and LoRA checkpoint
- Builds an update vector v_named from a tiny U-batch
- Runs two functional_call probes on a fixed tokenized sentence:
  A) PARAMS-ONLY mapping (expected: Δlogits ∝ η in small-η regime)
  B) PARAMS+BUFFERS mapping (demonstrates instability if buffers are injected)

This script is self-locating and does not depend on the current working dir.
"""

import os, sys, gc, math, argparse
from pathlib import Path
from typing import Dict, Any, Tuple
import torch
import yaml
import numpy as np
from transformers import AutoTokenizer

# --- BEGIN: self-locating import bootstrap -----------------------------------
def _infer_project_root_from_argv():
    # If --config .../entropy_experiments/configs/... is provided, repo root is parent of "entropy_experiments"
    for i, a in enumerate(sys.argv):
        if a == "--config" and i + 1 < len(sys.argv):
            cfg_path = Path(sys.argv[i + 1]).resolve()
            parts = list(cfg_path.parts)
            if "entropy_experiments" in parts:
                idx = parts.index("entropy_experiments")
                return Path(*parts[:idx])
            return cfg_path.parent.parent
    return Path.cwd()

PROJECT_ROOT = _infer_project_root_from_argv()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ["PYTHONPATH"] = (
    f"{PROJECT_ROOT}:{os.environ.get('PYTHONPATH','')}"
    if os.environ.get("PYTHONPATH") else str(PROJECT_ROOT)
)

pkg_dir = PROJECT_ROOT / "entropy_experiments"
if pkg_dir.is_dir():
    init_file = pkg_dir / "__init__.py"
    if not init_file.exists():
        try: init_file.touch()
        except Exception: pass
# --- END: self-locating import bootstrap -------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True,
                   help="Path to YAML config (e.g. /content/RL_Practice/entropy_experiments/configs/colab_config.yaml)")
    p.add_argument("--text", default="this is a test sequence. the logits should change continuously!",
                   help="Fixed text to tokenize and probe.")
    p.add_argument("--tokenizer_id", default="Qwen/Qwen2.5-1.5B-Instruct",
                   help="Tokenizer ID; must be vocab-compatible with the model.")
    p.add_argument("--eta_grid", type=float, nargs="*", default=[0.0, 1e-10, 1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5])
    p.add_argument("--eta_main", type=float, default=1e-5)
    p.add_argument("--eta_tiny", type=float, default=1e-10)
    p.add_argument("--u_batch", type=int, default=8)
    p.add_argument("--u_G", type=int, default=4)
    p.add_argument("--print_topk", type=int, default=5)
    p.add_argument("--fp64", action="store_true", help="Use float64 for the probe (safer, slower).")
    return p.parse_args()


# --- small helpers ------------------------------------------------------------
def unwrap(mod: torch.nn.Module) -> torch.nn.Module:
    m = mod
    while hasattr(m, "module"): m = m.module
    if hasattr(m, "_orig_mod"):
        try: m = m._orig_mod
        except Exception: pass
    return m

def device_of(mod: torch.nn.Module) -> torch.device:
    return next(unwrap(mod).parameters()).device

def report_mem(tag: str):
    if not torch.cuda.is_available():
        print(f"[{tag}] CPU-only.")
        return
    torch.cuda.synchronize()
    gb = 1024**3
    print(f"[{tag}] alloc={torch.cuda.memory_allocated()/gb:.2f} GB, "
          f"reserved={torch.cuda.memory_reserved()/gb:.2f} GB")

def logits_under_mapping(mdl, input_ids, attention_mask, mapping: Dict[str, torch.Tensor] | None):
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
    d = delta_logits_cpu.squeeze(0)         # [T, V]
    linf = d.abs().max(dim=-1).values
    l2   = d.pow(2).sum(dim=-1).sqrt()
    return linf, l2

def safe_ratio(a, b):
    return float(a / b) if (b != 0.0 and math.isfinite(a) and math.isfinite(b)) else float("nan")


def main():
    args = parse_args()

    # Precision defaults for the probe
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    if args.fp64:
        torch.set_default_dtype(torch.float64)
    else:
        torch.set_default_dtype(torch.float32)

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    print(f"✓ Loaded config: {args.config}")

    # Optional global precision from your utils (ignored if missing)
    try:
        from entropy_experiments.utils.precision_utils import apply_global_precision
        apply_global_precision(allow_tf32=True, matmul_precision="high")
    except Exception:
        pass

    # Project imports
    from entropy_experiments.offline_entropy_probe import OfflineEntropyProbe
    from entropy_experiments.update_vector import compute_update_vector
    from entropy_experiments.utils.param_overrides import (
        build_functional_params_named,
        merge_params_and_buffers,   # used only in the buffers probe
    )

    # Build probe & load checkpoint
    probe = OfflineEntropyProbe(cfg)
    ckpt_cfg = cfg.get("checkpoint", {})
    adapter_path   = ckpt_cfg.get("checkpoint_path", "")
    optimizer_path = ckpt_cfg.get("optimizer_path", "")
    if not adapter_path:
        raise ValueError("Config missing checkpoint.checkpoint_path")
    print("Loading LoRA checkpoint …")
    probe.load_checkpoint(adapter_path, optimizer_path if optimizer_path else None)
    model = probe.model
    dev = device_of(model)
    print(f"✓ Model on {dev}: {type(unwrap(model)).__name__}")

    # --- Compute update vector v_named from a tiny U-batch --------------------
    probe._ensure_sequence_processor()
    dataset = cfg["batch_config"]["dataset_name"]
    _, U_split = probe._get_splits()
    U_sequences, U_logprobs, _ = probe._sequence_processor.generate_with_logprobs(
        prompts=None, G=args.u_G, dataset_name=dataset, split=U_split,
        num_prompts=args.u_batch, compute_rb=True
    )
    U_batch = probe._pack_U_from_sequences(U_sequences, U_logprobs.rewards)

    # Optimizer (only to get state if needed by your compute_update_vector)
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

    # --- Fixed sequence & baseline logits ------------------------------------
    tok = AutoTokenizer.from_pretrained(args.tokenizer_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    enc = tok(args.text, return_tensors="pt", add_special_tokens=True, padding=False)
    input_ids = enc["input_ids"].to(dev); attention_mask = enc["attention_mask"].to(dev)

    # Vocab guard
    mdl_unwrapped = unwrap(model)
    vocab_size = None
    for n, p in mdl_unwrapped.named_parameters():
        if n.endswith("model.embed_tokens.weight"):
            vocab_size = p.shape[0]; break
    if vocab_size is None:
        for n, p in mdl_unwrapped.named_parameters():
            if n.endswith("lm_head.weight"): vocab_size = p.shape[0]; break
    if vocab_size is not None and int(input_ids.max()) >= vocab_size:
        raise RuntimeError("Tokenizer/model vocab mismatch (token id exceeds vocab size).")

    with torch.no_grad():
        logits0 = logits_under_mapping(model, input_ids, attention_mask, mapping=None)
    logits0_cpu = logits0.detach().cpu()
    T = logits0_cpu.shape[1]
    toks = tok.convert_ids_to_tokens(input_ids.squeeze(0).tolist())
    print(f"\nFixed-sequence functional_call probe …")
    print(f"Sequence length T={T}, vocab={logits0_cpu.shape[-1]}")

    # --- PROBE A: PARAMS-ONLY mapping (expected: good scaling) ---------------
    print("\n[PROBE A] PARAMS-ONLY mapping (recommended)")
    records_A = []
    for eta in args.eta_grid:
        # Build θ' with PARAMS ONLY; force params to float32/64 for continuity
        params_dict, _ = build_functional_params_named(
            model, v_named, eta,
            detach_params=True, detach_buffers=True,
            force_param_dtype=(torch.float64 if args.fp64 else torch.float32),
            force_buffer_dtype=None,   # ← do not cast buffers; not used
        )
        mapping = params_dict  # ← PARAMS ONLY
        logits_eta = logits_under_mapping(model, input_ids, attention_mask, mapping=mapping)
        d = (logits_eta.detach().cpu() - logits0_cpu)
        linf = float(d.abs().max().item()); l2 = float(d.pow(2).sum().sqrt().item())
        records_A.append((eta, linf, l2))
        # per-token preview
        linf_tok, _ = tokenwise_norms(d)
        idx_top = torch.topk(linf_tok, k=min(args.print_topk, T)).indices.tolist()
        preview = ", ".join([f"(t={i}, tok='{toks[i]}', Δ∞={linf_tok[i].item():.2e})" for i in idx_top])
        print(f"η={eta: .1e}  ||Δlogits||∞={linf:.3e}  ||Δlogits||₂={l2:.3e}")
        print("   top Δ∞ tokens:", preview)
        del params_dict, mapping, logits_eta, d
        torch.cuda.empty_cache()

    # Linearity: compare tiniest two nonzero etas if present
    nzA = [(e, L, N) for (e, L, N) in records_A if e > 0]
    if len(nzA) >= 2:
        a, b = nzA[0], nzA[1]
        print(f"Linearity sanity (A): (η={a[0]:.1e} vs {b[0]:.1e}) "
              f"Δ∞ ratio≈{safe_ratio(a[1], b[1]):.3f}, Δ₂ ratio≈{safe_ratio(a[2], b[2]):.3f}, "
              f"expected≈{a[0]/b[0]:.3f}")

    # --- PROBE B: PARAMS+BUFFERS mapping (to reproduce instability) ----------
    print("\n[PROBE B] PARAMS+BUFFERS mapping (diagnostic; often unstable)")
    records_B = []
    for eta in args.eta_grid:
        params_dict, buffers_dict = build_functional_params_named(
            model, v_named, eta,
            detach_params=True, detach_buffers=True,
            force_param_dtype=(torch.float64 if args.fp64 else torch.float32),
            force_buffer_dtype=(torch.float64 if args.fp64 else torch.float32),
        )
        # Merge for mapping; PARAMETERS TAKE PRECEDENCE on name collisions
        mapping = dict(buffers_dict); mapping.update(params_dict)
        logits_eta = logits_under_mapping(model, input_ids, attention_mask, mapping=mapping)
        d = (logits_eta.detach().cpu() - logits0_cpu)
        linf = float(d.abs().max().item()); l2 = float(d.pow(2).sum().sqrt().item())
        records_B.append((eta, linf, l2))
        linf_tok, _ = tokenwise_norms(d)
        idx_top = torch.topk(linf_tok, k=min(args.print_topk, T)).indices.tolist()
        preview = ", ".join([f"(t={i}, tok='{toks[i]}', Δ∞={linf_tok[i].item():.2e})" for i in idx_top])
        print(f"η={eta: .1e}  ||Δlogits||∞={linf:.3e}  ||Δlogits||₂={l2:.3e}")
        print("   top Δ∞ tokens:", preview)
        del params_dict, buffers_dict, mapping, logits_eta, d
        torch.cuda.empty_cache()

    nzB = [(e, L, N) for (e, L, N) in records_B if e > 0]
    if len(nzB) >= 2:
        a, b = nzB[0], nzB[1]
        print(f"Linearity sanity (B): (η={a[0]:.1e} vs {b[0]:.1e}) "
              f"Δ∞ ratio≈{safe_ratio(a[1], b[1]):.3f}, Δ₂ ratio≈{safe_ratio(a[2], b[2]):.3f}, "
              f"expected≈{a[0]/b[0]:.3f}")

    report_mem("after probes")
    del logits0, logits0_cpu
    gc.collect(); torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
