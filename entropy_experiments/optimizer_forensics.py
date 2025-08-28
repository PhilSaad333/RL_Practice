
#!/usr/bin/env python3
"""
Optimizer & LoRA State Forensics
--------------------------------

Purpose
    Thoroughly inspect a saved Adam(W) optimizer state and its relationship to a
    PEFT/LoRA model checkpoint. Detect whether "zero exp_avg_sq" arises from:
      (1) Early checkpointing (no optimizer steps yet),
      (2) Parameter-set mismatch when loading,
      (3) Genuine lack of Adam updates to LoRA params, or
      (4) Accidental training over a different param set than expected.

What it does
    • Reads optimizer.pt and training_info.json from a training_state/<step_X> dir
    • Summarizes group structure and state counts; computes non‑zero statistics
    • (Optional) Loads the base + PEFT model, enumerates:
        - all parameters (by tensor),
        - LoRA-trainable parameters only (lora_A/B, and optionally lm_head if trainable)
    • Reconstructs two optimizers (all-params and LoRA-only), attempts to load
      the saved optimizer state into each, and reports detailed coverage:
        - how many tensors have a state entry,
        - how many have nonzero exp_avg / exp_avg_sq norms,
        - per‑module summaries for LoRA A/B pairs,
        - scalar param totals (numel) for context (e.g., “~90M trainables”).
    • (Optional) Runs a single synthetic “smoke” update to verify moments update.

Usage
    python optimizer_forensics.py /path/to/training_state/step_40 \
        --base Qwen/Qwen2.5-1.5B \
        --adapter /path/to/training_state/step_40/model \
        --topk 20 \
        --smoke-update

Notes
    - Internet access may be required the first time you load the base model if
      it is not locally cached. If you cannot or do not wish to load the model,
      omit --base/--adapter and the script will still perform file‑level checks.
    - The “730 parameters” style counts reported here refer to *parameter tensors*
      (nn.Parameter objects), not scalar counts. Scalar counts (numel) are also printed.
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

# -------------------------
# Utility printing helpers
# -------------------------

def hr(title: str = "", ch: str = "=", width: int = 70) -> None:
    print("\n" + ch * width)
    if title:
        print(title)
        print(ch * width)

def fmt_int(n: int) -> str:
    return f"{n:,}"

def fmt_float(x: float) -> str:
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf"
    # scientific for very small/large
    ax = abs(x)
    if (ax != 0 and (ax < 1e-3 or ax >= 1e4)):
        return f"{x:.3e}"
    return f"{x:.6f}"

def human_numel(n: int) -> str:
    # human-friendly count of scalars
    if n >= 10**9: return f"{n/10**9:.2f}B"
    if n >= 10**6: return f"{n/10**6:.2f}M"
    if n >= 10**3: return f"{n/10**3:.2f}K"
    return str(n)

# -------------------------
# Check A: file-level optimizer structure
# -------------------------

def load_optimizer_state_dict(opt_path: Path) -> Optional[Dict]:
    if not opt_path.exists():
        print(f"❌ No optimizer.pt found at {opt_path}")
        return None
    try:
        sd = torch.load(opt_path, map_location="cpu", weights_only=False)
        if not isinstance(sd, dict) or "state" not in sd or "param_groups" not in sd:
            print("❌ optimizer.pt does not look like an optimizer state_dict")
            return None
        return sd
    except Exception as e:
        print(f"❌ Failed to load optimizer state_dict: {e}")
        return None

def summarize_optimizer_state_dict(opt_sd: Dict, topk: int = 10) -> Dict:
    hr("CHECK A: Saved Optimizer Structure")
    pg_sizes = [len(g.get("params", [])) for g in opt_sd["param_groups"]]
    print(f"Saved param_groups: {len(pg_sizes)}")
    print(f"Group sizes (by tensors): {pg_sizes}")
    print(f"Total tensors with any state entry: {fmt_int(len(opt_sd['state']))}")
    # Infer whether exp_avg / exp_avg_sq present & nonzero
    nz_avg, nz_var, nz_step = 0, 0, 0
    norms_list = []
    for st in opt_sd["state"].values():
        has_avg = "exp_avg" in st and isinstance(st["exp_avg"], torch.Tensor)
        has_var = "exp_avg_sq" in st and isinstance(st["exp_avg_sq"], torch.Tensor)
        has_step = "step" in st
        if has_avg:
            if float(st["exp_avg"].norm().item()) > 0:
                nz_avg += 1
        if has_var:
            nrm = float(st["exp_avg_sq"].norm().item())
            if nrm > 0:
                nz_var += 1
            norms_list.append(nrm)
        if has_step:
            # allow both int and tensor step
            step_val = int(st["step"]) if not isinstance(st["step"], torch.Tensor) else int(st["step"].item())
            if step_val > 0:
                nz_step += 1

    print(f"Tensors with nonzero exp_avg:     {fmt_int(nz_avg)}/{fmt_int(len(opt_sd['state']))}")
    print(f"Tensors with nonzero exp_avg_sq:  {fmt_int(nz_var)}/{fmt_int(len(opt_sd['state']))}")
    print(f"Tensors with step>0:              {fmt_int(nz_step)}/{fmt_int(len(opt_sd['state']))}")

    if norms_list:
        import numpy as np
        arr = np.array(norms_list, dtype=np.float64)
        desc = {
            "count": int(arr.size),
            "min": float(arr.min(initial=0.0)),
            "p25": float(np.quantile(arr, 0.25)),
            "p50": float(np.quantile(arr, 0.50)),
            "p75": float(np.quantile(arr, 0.75)),
            "max": float(arr.max(initial=0.0)),
            "mean": float(arr.mean() if arr.size else 0.0),
        }
        print("exp_avg_sq norm distribution:",
              f"min={fmt_float(desc['min'])}  p50={fmt_float(desc['p50'])}  "
              f"p75={fmt_float(desc['p75'])}  max={fmt_float(desc['max'])}")
        # topk
        idxs = arr.argsort()[::-1][:topk]
        if idxs.size > 0:
            print(f"Top-{topk} exp_avg_sq norms:",
                  ", ".join(fmt_float(float(arr[i])) for i in idxs))
    else:
        print("No exp_avg_sq entries present in saved state.")

    return {
        "group_sizes": pg_sizes,
        "nz_exp_avg": nz_avg,
        "nz_exp_avg_sq": nz_var,
        "nz_step": nz_step,
        "state_count": len(opt_sd["state"]),
    }

# -------------------------
# Check C: training_info.json
# -------------------------

def load_training_info(info_path: Path) -> Optional[Dict]:
    if not info_path.exists():
        print(f"⚠️  training_info.json not found at {info_path}")
        return None
    try:
        with open(info_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  Failed to read training_info.json: {e}")
        return None

def summarize_training_info(info: Dict) -> None:
    hr("CHECK C: Training Progress Snapshot")
    step = info.get("step", None)
    global_step = info.get("global_step", None)
    print(f"Recorded 'step':        {step}")
    print(f"Recorded 'global_step': {global_step}")
    train_cfg = info.get("training_config", {})
    world = info.get("distributed_info", {}).get("world_size", 1)
    buf = train_cfg.get("buffer_size", None)
    mb = train_cfg.get("microbatch_size", None)
    print(f"buffer_size: {buf} | microbatch_size: {mb} | world_size: {world}")
    if isinstance(buf, int) and isinstance(mb, int) and buf > 0 and mb > 0:
        try:
            accum = max(1, buf // (max(1, world) * mb))
            print(f"Estimated grad_accum_steps per update: {accum}")
        except Exception:
            pass
    # Print any runner-provided counters
    for k in ["optimizer_updates", "samples_seen", "tokens_seen"]:
        if k in info:
            print(f"{k}: {info[k]}")

# -------------------------
# Optional: model loading & parameter enumeration
# -------------------------

def load_peft_model(base: str, adapter: Path, device: str = "auto"):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel, prepare_model_for_kbit_training
    # 4-bit by default for memory
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base, device_map=device, torch_dtype=torch.float16
    )
    base_model = prepare_model_for_kbit_training(base_model)
    peft_model = PeftModel.from_pretrained(base_model, str(adapter), is_trainable=True)
    peft_model.enable_input_require_grads()
    return peft_model

def enumerate_params_with_names(model) -> List[Tuple[str, torch.nn.Parameter]]:
    # use the unwrapped module if present
    mod = model.module if hasattr(model, "module") else model
    return [(n, p) for (n, p) in mod.named_parameters()]

def split_lora_params(named_params: List[Tuple[str, torch.nn.Parameter]], include_lm_head_if_trainable: bool = True):
    lora, non_lora = [], []
    for name, p in named_params:
        lname = name.lower()
        if ("lora_a" in lname) or ("lora_b" in lname):
            lora.append((name, p))
        elif include_lm_head_if_trainable and p.requires_grad and ("lm_head" in lname):
            # some PEFT setups also train lm_head
            lora.append((name, p))
        else:
            non_lora.append((name, p))
    return lora, non_lora

def count_numel(params: Sequence[torch.nn.Parameter]) -> int:
    return sum(int(p.numel()) for p in params)

# -------------------------
# Check B: Try loading state into reconstructed optimizers
# -------------------------

def build_adamw(param_list: Sequence[torch.nn.Parameter], lr: float = 1e-4, wd: float = 0.01):
    from torch.optim import AdamW
    return AdamW(list(param_list), lr=lr, weight_decay=wd)

def try_load(opt, sd: Dict) -> Tuple[bool, Optional[str]]:
    try:
        opt.load_state_dict(sd)
        return True, None
    except Exception as e:
        return False, str(e)

def analyze_optimizer_state_coverage(opt, named_params: List[Tuple[str, torch.nn.Parameter]], topk: int = 20):
    """
    For the given optimizer (post-load), compute coverage stats: which tensors
    have state, how many have nonzero exp_avg/exp_avg_sq, and show top‑k norms.
    """
    state = opt.state  # maps param -> state dict
    N = len(named_params)
    present, nz_avg, nz_var, nz_step = 0, 0, 0, 0
    rows = []
    for name, p in named_params:
        st = state.get(p, None)
        if st is None:
            rows.append((name, False, 0.0, 0.0, 0))
            continue
        present += 1
        avg_nrm = float(st["exp_avg"].norm().item()) if "exp_avg" in st else 0.0
        var_nrm = float(st["exp_avg_sq"].norm().item()) if "exp_avg_sq" in st else 0.0
        step_val = int(st["step"]) if "step" in st and not isinstance(st["step"], torch.Tensor) else int(st["step"].item()) if "step" in st else 0
        if avg_nrm > 0: nz_avg += 1
        if var_nrm > 0: nz_var += 1
        if step_val > 0: nz_step += 1
        rows.append((name, True, avg_nrm, var_nrm, step_val))

    hr("State coverage (post‑load)")
    print(f"Tensors present in optimizer.state: {present}/{N}")
    print(f"Nonzero exp_avg:                   {nz_avg}/{present}")
    print(f"Nonzero exp_avg_sq:                {nz_var}/{present}")
    print(f"step > 0:                          {nz_step}/{present}")

    # Top‑k by exp_avg_sq
    rows_sorted = sorted(rows, key=lambda r: r[3], reverse=True)
    print(f"\nTop‑{topk} tensors by exp_avg_sq norm:")
    for i, (name, pres, avg_nrm, var_nrm, step_val) in enumerate(rows_sorted[:topk], 1):
        print(f"{i:>2}. {name:60s} | present={pres} | exp_avg_sq={fmt_float(var_nrm)} | step={step_val}")

    return {
        "present": present, "nz_avg": nz_avg, "nz_var": nz_var, "nz_step": nz_step,
        "rows": rows_sorted,
    }

# -------------------------
# Optional: Smoke update
# -------------------------

def run_smoke_update(model, opt, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """
    Single synthetic token‑level CE step to verify that moments begin to move.
    Keeps sequence length small to be cheap; purely diagnostic.
    """
    model = model.to(device)
    model.train()
    torch.manual_seed(0)
    with torch.random.fork_rng(devices=[device] if device != "cpu" else []):
        B, T, V = 2, 16, getattr(model.config, "vocab_size", 32000)
        x = torch.randint(low=10, high=V-1, size=(B, T), device=device, dtype=torch.long)
        y = x.clone()
        y[:, :-1] = x[:, 1:]  # next‑token target
        y[:, -1] = 0
        out = model(input_ids=x, labels=y)
        loss = out.loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    return float(loss.detach().cpu().item())

# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=str, help="Path to training_state/step_X directory")
    ap.add_argument("--base", type=str, default=None, help="HF id or local path of base model (e.g., Qwen/Qwen2.5-1.5B)")
    ap.add_argument("--adapter", type=str, default=None, help="Path to PEFT adapter dir (defaults to <checkpoint>/model)")
    ap.add_argument("--device", type=str, default="auto", help="device_map for model loading (auto / cuda / cpu)")
    ap.add_argument("--topk", type=int, default=20, help="Top‑K tensors to display by exp_avg_sq norm")
    ap.add_argument("--smoke-update", action="store_true", help="Run one synthetic update to verify Adam moments move")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    opt_path = ckpt / "optimizer.pt"
    info_path = ckpt / "training_info.json"

    hr("Optimizer & LoRA State Forensics")

    # A. File‑level optimizer structure
    opt_sd = load_optimizer_state_dict(opt_path)
    if opt_sd is None:
        return
    A = summarize_optimizer_state_dict(opt_sd, topk=args.topk)

    # C. Training info snapshot
    info = load_training_info(info_path)
    if info:
        summarize_training_info(info)

    # If no model requested, stop here
    if not args.base:
        hr("Model loading skipped (no --base provided).")
        print("Provide --base and --adapter to perform parameter‑level analysis.")
        return

    # B. Model‑level parameter analysis
    adapter_path = Path(args.adapter) if args.adapter else (ckpt / "model")
    if not adapter_path.exists():
        hr("Warning: adapter path not found")
        print(f"Expected adapter at: {adapter_path}")
        print("Parameter‑level analysis will likely fail.")
    try:
        hr("Loading PEFT model")
        model = load_peft_model(args.base, adapter_path, device=args.device)
        print("✅ PEFT model loaded.")
    except Exception as e:
        print(f"❌ Model load failed: {e}")
        return

    named = enumerate_params_with_names(model)
    all_params = [p for _, p in named]
    lora_named, non_lora_named = split_lora_params(named, include_lm_head_if_trainable=True)
    lora_params = [p for _, p in lora_named]

    # Report tensor and scalar counts
    hr("Parameter inventory")
    print(f"Total parameter tensors:       {fmt_int(len(all_params))}")
    print(f"  · LoRA(+lm_head) tensors:    {fmt_int(len(lora_params))}")
    print(f"  · Non‑LoRA tensors:          {fmt_int(len(non_lora_named))}")
    print(f"Total scalar params (numel):   {human_numel(count_numel(all_params))}")
    print(f"Trainable LoRA scalar params:  {human_numel(count_numel(lora_params))}")

    # Build two optimizers
    from torch.optim import AdamW
    opt_all = AdamW(all_params, lr=1e-4, weight_decay=0.01)
    opt_lora = AdamW(lora_params, lr=1e-4, weight_decay=0.01)

    saved_group_sizes = [len(g.get("params", [])) for g in opt_sd["param_groups"]]
    all_group_sizes = [len(g.get("params", [])) for g in opt_all.state_dict()["param_groups"]]
    lora_group_sizes = [len(g.get("params", [])) for g in opt_lora.state_dict()["param_groups"]]

    hr("Param‑group size comparison")
    print(f"Saved group sizes:    {saved_group_sizes}")
    print(f"ALL‑params group sz:  {all_group_sizes}  -> match={all_group_sizes == saved_group_sizes}")
    print(f"LoRA‑only group sz:   {lora_group_sizes} -> match={lora_group_sizes == saved_group_sizes}")

    # Attempt loads
    ok_all, err_all = try_load(opt_all, opt_sd)
    ok_lora, err_lora = try_load(opt_lora, opt_sd)
    print("\nLoad results:")
    print(f"ALL‑params optimizer:  {'OK' if ok_all else 'FAIL'}{'' if ok_all else ('  ' + str(err_all))}")
    print(f"LoRA‑only optimizer:   {'OK' if ok_lora else 'FAIL'}{'' if ok_lora else ('  ' + str(err_lora))}")

    # Coverage reports
    hr("Coverage on ALL‑params optimizer")
    _ = analyze_optimizer_state_coverage(opt_all, named, topk=args.topk)

    hr("Coverage on LoRA‑only optimizer")
    _ = analyze_optimizer_state_coverage(opt_lora, lora_named, topk=args.topk)

    # Optional smoke step
    if args.smoke_update:
        hr("Synthetic smoke update (ALL‑params optimizer)")
        try:
            loss = run_smoke_update(model, opt_all)
            print(f"Smoke loss: {fmt_float(loss)}")
            # Re‑summarize after one step
            _ = analyze_optimizer_state_coverage(opt_all, lora_named, topk=min(args.topk, 10))
        except Exception as e:
            print(f"⚠️  Smoke update failed: {e}")

    hr("Summary & Guidance")
    print("* If file‑level A shows zero 'state' entries: this was saved before any optimizer step.")
    print("* If A shows many 'state' entries but LoRA coverage is near‑zero on BOTH optimizers:")
    print("    - Training likely never updated LoRA tensors (check which params were in the optimizer).")
    print("* If ALL‑params matches saved group sizes (and loads) but LoRA‑only does not:")
    print("    - Training optimizer was built over ALL params; ensure probe uses the same policy.")
    print("* If most LoRA tensors lack state but a few have nonzero moments:")
    print("    - Confirm accumulation -> step schedule actually calls optimizer.step() regularly;")
    print("      inspect 'training_info.json' and runner logs for accumulation boundaries.")
    print("* Consider persisting a 'param‑name list per param‑group' alongside checkpoints")
    print("  so that test harnesses can reconstruct identical optimizers by name rather than position.")
    print("\nDone.")


if __name__ == "__main__":
    main()
