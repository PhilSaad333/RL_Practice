#!/usr/bin/env python3
"""
Fixed-Sequence Param-Override Sanity Probe
==========================================

- Loads your model via OfflineEntropyProbe and LoRA checkpoint
- Builds an update vector v_named from a tiny U-batch
- Runs two functional_call probes on a fixed tokenized sentence:
  A) PARAMS-ONLY mapping (expected: Δlogits ∝ η in small-η regime)
  B) PARAMS+BUFFERS mapping (diagnostic; can be unstable)

This script is self-locating and does not depend on the working directory.
"""

import os, sys, gc, math, argparse
from pathlib import Path
from typing import Dict, Any
import torch
import yaml
import numpy as np
from transformers import AutoTokenizer


# --- BEGIN: self-locating import bootstrap -----------------------------------
def _infer_project_root_from_argv():
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
                   help="Path to YAML config (e.g. .../entropy_experiments/configs/colab_config.yaml)")
    p.add_argument("--text", default="this is a test sequence. the logits should change continuously!",
                   help="Fixed text to tokenize and probe.")
    p.add_argument("--tokenizer_id", default="Qwen/Qwen2.5-1.5B-Instruct",
                   help="Tokenizer ID; must be vocab-compatible with the model.")
    p.add_argument("--eta_grid", type=float, nargs="*",
                   default=[0.0, 1e-10, 1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5])
    p.add_argument("--eta_main", type=float, default=1e-5)
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

def tokenwise_norms(delta_logits_cpu: torch.Tensor):
    d = delta_logits_cpu.squeeze(0)         # [T, V]
    linf = d.abs().max(dim=-1).values
    l2   = d.pow(2).sum(dim=-1).sqrt()
    return linf, l2

def safe_ratio(a, b):
    return float(a / b) if (b != 0.0 and math.isfinite(a) and math.isfinite(b)) else float("nan")


# --- unified functional_call forward, autocast disabled ---
@torch.no_grad()
def _fc_logits_noautocast(mdl, input_ids, attention_mask, mapping: dict[str, torch.Tensor]):
    m = unwrap(mdl)
    was = m.training
    m.eval()
    try:
        # same precision path for EVERY call (baseline + all etas)
        with torch.autocast(device_type="cuda", enabled=False):
            out = torch.func.functional_call(
                m, mapping, (input_ids,),
                {'attention_mask': attention_mask, 'use_cache': False}
            )
        return out.logits if hasattr(out, "logits") else out[0]
    finally:
        if was: m.train()




# --- FP32/FP64 param-only mapping builders -----------------------------------
from entropy_experiments.utils.param_overrides import build_functional_params_named

def _params_only_fp(m: torch.nn.Module, v_named, eta: float, dtype=torch.float32):
    # Build mapping keyed for THIS module, parameters only
    params, _ = build_functional_params_named(
        m, v_named, eta,
        detach_params=True, detach_buffers=True,
        force_param_dtype=dtype,
        force_buffer_dtype=None
    )
    return params

def _params_plus_buffers_fp(m: torch.nn.Module, v_named, eta: float, dtype=torch.float32):
    params, bufs = build_functional_params_named(
        m, v_named, eta,
        detach_params=True, detach_buffers=True,
        force_param_dtype=dtype,
        force_buffer_dtype=dtype
    )
    merged = dict(bufs); merged.update(params)  # params take precedence
    return merged



@torch.no_grad()
def forward_under_mapping_noautocast(mdl, input_ids, attention_mask, mapping: Dict[str, torch.Tensor]):
    """functional_call forward with autocast disabled and use_cache=False."""
    m = unwrap(mdl)
    was_training = m.training
    m.eval()
    try:
        with torch.autocast(device_type="cuda", enabled=False):
            out = torch.func.functional_call(
                m, mapping, (input_ids,),
                {'attention_mask': attention_mask, 'use_cache': False}
            )
        return out.logits if hasattr(out, "logits") else out[0]
    finally:
        if was_training: m.train()


def main():
    args = parse_args()

    # Precision defaults for the probe
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_default_dtype(torch.float64 if args.fp64 else torch.float32)

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    print(f"✓ Loaded config: {args.config}")

    # Optional global precision
    try:
        from entropy_experiments.utils.precision_utils import apply_global_precision
        apply_global_precision(allow_tf32=True, matmul_precision="high")
    except Exception:
        pass

    # Project imports
    from entropy_experiments.offline_entropy_probe import OfflineEntropyProbe
    from entropy_experiments.update_vector import compute_update_vector

    # Build probe & load checkpoint
    probe = OfflineEntropyProbe(cfg)
    ckpt_cfg = cfg.get("checkpoint", {})
    adapter_path   = ckpt_cfg.get("checkpoint_path", "")
    optimizer_path = ckpt_cfg.get("optimizer_path", "")
    if not adapter_path:
        raise ValueError("Config missing checkpoint.checkpoint_path")
    print("Loading LoRA checkpoint …")
    probe.load_checkpoint(adapter_path, optimizer_path if optimizer_path else None)
    probe._ensure_sequence_processor()
    sp = probe._sequence_processor
    mdl_target = sp._mdl_target

    dev = device_of(mdl_target)

    fp_dtype = torch.float64 if args.fp64 else torch.float32

    print("fo dtype:", sp._fo_cfg.get("dtype"))
    print("tf dtype:", sp._tf_cfg.get("dtype"))
    print("example LoRA A dtype:",
        next(p for n,p in sp._mdl_target.named_parameters() if "lora_A" in n).dtype)





    # --- Compute update vector v_named from a tiny U-batch --------------------
    dataset = cfg["batch_config"]["dataset_name"]
    _, U_split = probe._get_splits()

    U_sequences, U_logprobs, _ = sp.generate_with_logprobs(
        prompts=None, G=args.u_G, dataset_name=dataset, split=U_split,
        num_prompts=args.u_batch, compute_rb=False
    )

    U_batch = probe._pack_U_from_sequences(U_sequences, U_logprobs.rewards)

    # Optimizer (only to get state if needed by compute_update_vector)
    optimizer = getattr(probe, "optimizer", None)
    if optimizer is None:
        trainable = [p for _, p in mdl_target.named_parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=args.eta_main)

    v_named, stats = compute_update_vector(
        model=mdl_target, optimizer=optimizer, U_batch=U_batch, config=probe.config, logger=probe.logger
    )
    v_norm = float(torch.sqrt(sum((v.double()**2).sum() for v in v_named.values())).item())
    print(f"✓ Update vector: {len(v_named)} tensors, ||v||₂ ≈ {v_norm:.3e}")





    #tok = AutoTokenizer.from_pretrained(args.tokenizer_id, use_fast=True)
    #if tok.pad_token is None:
    #    tok.pad_token = tok.eos_token

    # 1) Build a tiny E-batch of sequences.
    #    If you already created sequences with your generator, re-use them.
    #    Otherwise, use the SequenceProcessor to generate quickly.
    #processor_cfg = GenerationConfig()
    #processor = SequenceProcessor(model, tok, logger=None, config={"precision": {
    #    "allow_tf32": False,            # strict math
    #    "matmul_precision": "high",
    #    "deterministic_probe": True,
    #    "func_override": {"autocast": False, "cast_params": True, "dtype": "float32"},
    #    "tf_nograd":    {"autocast": False, "cast_logits_fp32": True},
    #}})


    # force identical precision profiles for this probe
    cfg.setdefault('precision', {})
    cfg['precision'].setdefault('tf_nograd', {}).update(
        {'autocast': False, 'dtype': 'float32', 'cast_logits_fp32': True})
    cfg['precision'].setdefault('func_override', {}).update(
        {'autocast': False, 'dtype': 'float32', 'cast_params': False})

    # Do NOT replace SequenceProcessor.config (a GenerationConfig) with a dict.
    # Instead, update precision profiles in-place so attribute access (e.g., gen_batch_size)
    # continues to work.
    prec = cfg['precision']
    try:
        sp._fo_cfg.update(prec.get('func_override', {}))
        sp._tf_cfg.update(prec.get('tf_nograd', {}))
        # Optional global precision knobs
        from entropy_experiments.utils.precision_utils import apply_global_precision
        apply_global_precision(
            allow_tf32=bool(prec.get('allow_tf32', True)),
            matmul_precision=prec.get('matmul_precision', 'high')
        )
    except Exception:
        pass



    # Generate E batch for entropy evaluation  
    print("Generating E batch for entropy evaluation...")


    E_sequences, _E_logprobs, _E_diag = sp.generate_with_replacement_sampling(
        total_sequences=16,
        dataset_name=dataset,
        split='test',
        G=1,  # E batch uses G=1 (single generation per prompt)
        compute_rb=True,
    )
    E_batch = probe._pack_E_from_sequences(E_sequences)

    print("fo dtype:", sp._fo_dtype, "tf dtype:", sp._tf_dtype)
    dbg = sp.teacher_force_debug_probe(E_sequences, b_idx=0, g_idx=0,
                                    params_override=sp._build_params_override(None, 0.0))
    print("[dbg] logits dtype:", dbg["topk_vals"].dtype)



    # --- SP LOGITS CONTINUITY (full vocab) ---

    b, g = 0, 0
    seq = E_sequences.sequences[b, g]
    pl  = int(E_sequences.prompt_lens[b])
    Tg  = min(int(E_sequences.gen_lens[b][g]), 64)
    L   = int(seq.size(0)); Luse = min(pl + Tg, L)
    ids = seq[:Luse].unsqueeze(0).to(dev)
    msk = E_sequences.attention_masks[b, g, :Luse].unsqueeze(0).to(dev)

    def map_for_eta(e):
        # Use SP's builder so keyspace/dtypes match exactly
        return sp._build_params_override(v_named=v_named, eta=float(e))

    # Baseline (fc-vs-fc)
    logits0 = sp._fc_logits_noautocast(ids, msk, map_for_eta(0.0))[0]  # [L,V]
    gen_slice = slice(pl-1, pl-1 + Tg)  # next-token slice for T steps
    lo0 = logits0[gen_slice]  # [T,V]

    for eta in [1e-8, 3e-8, 1e-7, 3e-7, 1e-6]:
        loe = sp._fc_logits_noautocast(ids, msk, map_for_eta(eta))[0][gen_slice]
        d = (loe - lo0)
        linf = float(d.abs().max().item())
        l2   = float(d.pow(2).sum().sqrt().item())
        print(f"[SP full-V] eta={eta:>8g}  ||Δlogits||∞={linf:.3e}  ||Δlogits||₂={l2:.3e}")







    # --- SP NAIVE ENTROPY CONTINUITY on the same (b,g) ---
    import torch.nn.functional as F

    def H_naive_for_eta(e):
        lo = sp._fc_logits_noautocast(ids, msk, map_for_eta(e))[0][gen_slice]  # [T,V]
        logp = F.log_softmax(lo, dim=-1)
        p = logp.exp()
        H = (-(p * logp).sum(dim=-1)).sum()   # sum over T
        return float(H.item())

    H0 = H_naive_for_eta(0.0)
    for eta in [1e-8, 3e-8, 1e-7, 3e-7, 1e-6]:
        dH = H_naive_for_eta(eta) - H0
        print(f"[SP naive H] eta={eta:>8g}  ΔH={dH:.6e}")










    # If you don't already have sequences, create a quick E-batch:
    #prompts = ["Compute: 37+58 = </think>", "Factor: 84 = </think>"]  # examples
    #E_sequences = processor.generate_batched(prompts, G=1)
    # Otherwise, re-use your existing generated 'sequences' object:
    #E_sequences = sequences  # re-use if available

    # 2) (Optional) Zero-check: direct vs functional_call(η=0)


    # After constructing 'processor' and building a minimal 1-sequence batch 'E_sequences'
    b0, g0 = 0, 0
    seq = E_sequences.sequences[b0, g0]
    pl  = int(E_sequences.prompt_lens[b0])
    Tg  = int(E_sequences.gen_lens[b0][g0])
    L   = int(seq.size(0))
    assert Tg > 0 and L > pl
    ids  = seq[:pl+Tg].unsqueeze(0).to(next(mdl_target.parameters()).device)
    mask = E_sequences.attention_masks[b0, g0, :pl+Tg].unsqueeze(0).to(ids.device)

    # Build η=0 mapping (params-only, baseline dtype preserved by patch (B))
    m0 = sp._build_params_override(v_named=None, eta=0.0)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", enabled=False):
            out_dir = mdl_target(ids, attention_mask=mask, use_cache=False)
    logits_dir = out_dir.logits if hasattr(out_dir, "logits") else out_dir[0]
    logits_fc  = sp._fc_logits_noautocast(ids, mask, m0)

    print("[sanity] direct finite:", bool(torch.isfinite(logits_dir).all().item()),
        " fc finite:", bool(torch.isfinite(logits_fc).all().item()))
    print("[sanity] max|direct - fc(η=0)| =",
        float((logits_dir - logits_fc).abs().max().item()))







    # pick one sequence (b,g) and build the same left-pad-aware slice as SP
    b, g = 0, 0
    seq = E_sequences.sequences[b, g]                        # [L_tot]
    pl  = int(E_sequences.prompt_lens[b])
    Tg  = int(E_sequences.gen_lens[b][g])
    Ltot= int(seq.size(0))
    assert Tg > 0 and Ltot > pl, "Chosen (b,g) has no generated tokens."

    Luse = min(pl + min(Tg, 64), Ltot)                       # cap T for speed/debug
    ids  = seq[:Luse].unsqueeze(0).to(mdl_target.device)            # [1, Luse]
    msk  = E_sequences.attention_masks[b, g, :Luse].unsqueeze(0).to(mdl_target.device)

    # convenience: build the params-only mapping using SP's builder (keyspace/dtype match)
    def map_for_eta(eta: float):
        return sp._build_params_override(
            v_named=v_named,  # use cached zero-map when eta==0
            eta=float(eta)
        )

    # sanity: what dtype are we in?
    dbg = sp.teacher_force_debug_probe(E_sequences, b_idx=b, g_idx=g,
                                    params_override=map_for_eta(0.0))
    print("[dbg] logits dtype (from SP debug probe):", dbg["topk_vals"].dtype)

    # baseline via functional_call (η=0), next-token slice
    logits0 = sp._fc_logits_noautocast(ids, msk, map_for_eta(0.0))[0]         # [Luse, V]
    gen_slice = slice(pl-1, pl-1 + min(Tg, 64))                                # [T,V]
    lo0 = logits0[gen_slice]

    for eta in [1e-8, 3e-8, 1e-7, 3e-7, 1e-6]:
        loe = sp._fc_logits_noautocast(ids, msk, map_for_eta(eta))[0][gen_slice]
        d = loe - lo0
        linf = float(d.abs().amax().item())
        l2   = float(d.pow(2).sum().sqrt().item())
        print(f"[SP full-V] eta={eta:>8g}  ||Δlogits||∞={linf:.3e}  ||Δlogits||₂={l2:.3e}")







    # 3) Prepare the params-only mapping function.
    #    We’ll use the *same* update_vector you computed from the U-batch.

    def mapping_for_eta(eta: float) -> dict[str, torch.Tensor]:
        # Use the processor's own builder to ensure keyspace alignment and upcasting.
        return sp._build_params_override(v_named=v_named, eta=float(eta))


    eta_grid = [0.0, 1e-8, 3e-8, 1e-7, 3e-7, 1e-6]

    # 2) RB-entropy continuity (all through the *same* functional_call path)
    H_vals = []
    for eta in eta_grid:
        params_override = mapping_for_eta(eta)   # NOTE: even for eta=0
        logprob_res, _ = sp.teacher_force_logprobs(
            E_sequences,
            with_grad=False,
            tf_batch_size=sp.config.tf_batch_size,
            compute_rb=True,
            params_override=params_override,
            buffers_override=None,
        )
        # Aggregate RB entropy over all sequences/tokens in E
        H_sum = 0.0
        for b in range(len(logprob_res.rb_entropies)):
            for g in range(len(logprob_res.rb_entropies[b])):
                rb = logprob_res.rb_entropies[b][g]
                if rb is not None and len(rb) > 0:
                    H_sum += float(rb.sum())
        H_vals.append(H_sum)

    import numpy as np
    H_vals = np.array(H_vals, dtype=np.float64)
    H0, dH = H_vals[0], H_vals - H_vals[0]
    print("\n[SP continuity — RB entropy]")
    for eta, val, diff in zip(eta_grid, H_vals, dH):
        print(f"eta={eta:>8g}  H={val:.6f}  ΔH={diff:.6e}")

    print("\n[small-η ratios]")
    for i in range(2, len(eta_grid)):
        r_eta = (eta_grid[i] - eta_grid[0]) / (eta_grid[1] - eta_grid[0]) if eta_grid[1] != 0 else np.nan
        r_dH  = (dH[i]) / (dH[1] if dH[1] != 0 else np.nan)
        print(f"η/η_small ≈ {r_eta:>9.3g}  |  ΔH/ΔH_small ≈ {r_dH:>9.3g}")

    # 3) Optional: logits-level probe on one (b,g), short T (matches SP path exactly)
    b_idx, g_idx = 0, 0
    def get_gen_logits_for_eta(eta, max_T=64):
        pack = sp.teacher_force_debug_probe(
            E_sequences, b_idx=b_idx, g_idx=g_idx,
            params_override=mapping_for_eta(eta),  # η=0 included
            max_T=max_T, topk=1
        )
        # returns gen logits only indirectly; reconstruct from topk? Instead compute Δ using on-token logprobs/logits
        return pack  # compact per-token views in pack

    base = get_gen_logits_for_eta(0.0)
    for eta in [1e-8, 3e-8, 1e-7]:
        probe = get_gen_logits_for_eta(eta)
        # Use the “logit at realized token” channel which is stable and 1-D
        d_logit = (probe["logit_on_tok"] - base["logit_on_tok"]).abs()
        d_logp  = (probe["logprob_on_tok"] - base["logprob_on_tok"]).abs()
        linf_L  = float(d_logit.max() if d_logit.numel() else 0.0)
        linf_P  = float(d_logp.max()  if d_logp.numel()  else 0.0)
        print(f"[logits probe] eta={eta:>8g}  ||Δlogit_on_tok||∞={linf_L:.3e}  ||Δlogp_on_tok||∞={linf_P:.3e}")






    report_mem("after probes")
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
