"""
Model and optimizer loaders for offline probes.

Supports LoRA and QLoRA via a single entrypoint and loads Adam optimizer
state with parameter-ID remapping to current model parameter IDs.
"""

from typing import Optional, Dict, Any
from pathlib import Path
import torch




# --- runtime-precision helpers (local to loader) ---
def _force_model_fp32_runtime(m: torch.nn.Module) -> torch.nn.Module:
    """Upcast all parameters and buffers to fp32 and mark config dtype accordingly.
    No-op for int/quantized tensors. Safe for LoRA; do not call on 4-bit base weights."""
    # Cast parameters
    for p in m.parameters(recurse=True):
        if p.is_floating_point() and p.dtype is not torch.float32:
            p.data = p.data.float()
    # Cast buffers (e.g., layernorm stats, rotary caches, logits processors)
    for mod in m.modules():
        for name, buf in list(mod._buffers.items()):
            if buf is not None and buf.is_floating_point() and buf.dtype is not torch.float32:
                mod._buffers[name] = buf.float()
    # Best-effort config annotation
    if hasattr(m, "config") and getattr(m.config, "torch_dtype", None) != torch.float32:
        try:
            m.config.torch_dtype = torch.float32
        except Exception:
            pass
    return m




def load_peft_for_probe(
    base_id: str,
    adapter_path: str,
    *,
    use_qlora: bool = False,
    dtype: str = "fp32",         # "fp32" strongly recommended for your experiments
    device_map: str = "cuda",
    use_checkpointing: bool = False,
    force_fp32_runtime: bool = True,   # <— new: upcast after PEFT attach
):
    """Load a PEFT (LoRA/QLoRA) model ready for probe computations.

    Sets attn_implementation="eager" for VJP compatibility and disables
    gradient checkpointing unless explicitly requested.
    """
    from transformers import AutoModelForCausalLM
    from peft import PeftModel

    torch_dtype = {
        "fp32": torch.float32, "float32": torch.float32,
        "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
        "fp16": torch.float16,  "float16": torch.float16,
        "fp64": torch.float64,  "float64": torch.float64,  # rarely used at load
    }[dtype]

    if not use_qlora:
        # LoRA-simple (no quantization)
        base = AutoModelForCausalLM.from_pretrained(
            base_id,
            device_map=device_map,
            torch_dtype=torch_dtype,              # caller can still pass bf16/fp16 if desired
            attn_implementation="eager",          # required for stable VJP/functional_call
            trust_remote_code=True,
        )
        peft = PeftModel.from_pretrained(base, adapter_path, is_trainable=True)
        if hasattr(peft, "enable_input_require_grads"):
            peft.enable_input_require_grads()

        # Harden runtime precision as fp32 unless overridden explicitly
        if force_fp32_runtime:
            peft = _force_model_fp32_runtime(peft)

        return peft

    # QLoRA path
    from transformers import BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_id,
        device_map=device_map,
        torch_dtype=torch_dtype,
        quantization_config=bnb_cfg,
        attn_implementation="eager",
        trust_remote_code=True,
    )
    base = prepare_model_for_kbit_training(base)
    if use_checkpointing and hasattr(base, "gradient_checkpointing_enable"):
        base.gradient_checkpointing_enable()
        if hasattr(base.config, "use_cache"):
            base.config.use_cache = False
    peft = PeftModel.from_pretrained(base, adapter_path, is_trainable=True)
    if hasattr(peft, "enable_input_require_grads"):
        peft.enable_input_require_grads()
        
    # Do NOT upcast quantized weights. If you need full-fp32 runtime, load non-quantized LoRA instead.
    if force_fp32_runtime and any(getattr(p, "is_quantized", False) for p in peft.parameters()):
        # no-op but leave a breadcrumb via attribute for downstream logs
        setattr(peft, "_probe_quantized_runtime", True)
    return peft


def _remap_optimizer_state_ids(saved_state_dict: Dict[str, Any], optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
    """Map saved param IDs to current model param IDs by position.

    Handles the common case where parameter object IDs differ across runs.
    """
    # Current params in order
    current_params = []
    for group in optimizer.param_groups:
        current_params.extend(group["params"])  # list[Tensor]

    saved_state = saved_state_dict.get("state", {})
    saved_param_groups = saved_state_dict.get("param_groups", [])

    # Flatten saved param IDs in group order
    saved_param_ids = []
    for group in saved_param_groups:
        saved_param_ids.extend(group.get("params", []))

    # Build position mapping
    id_mapping = {}
    count = min(len(saved_param_ids), len(current_params))
    for i in range(count):
        id_mapping[saved_param_ids[i]] = id(current_params[i])

    # Remap state
    remapped_state = {}
    for old_id, st in saved_state.items():
        if old_id in id_mapping:
            remapped_state[id_mapping[old_id]] = st

    # Remap param_groups structure to current shape: put all params into first group
    # and keep hyperparams from the first non-empty saved group
    merged_cfg = {}
    for group in saved_param_groups:
        if group.get("params"):
            merged_cfg = {k: v for k, v in group.items() if k != "params"}
            break

    all_new_ids = [id(p) for p in current_params]
    remapped_param_groups = []
    for i, cur in enumerate(optimizer.param_groups):
        if i == 0:
            g = merged_cfg.copy()
            g["params"] = all_new_ids
            remapped_param_groups.append(g)
        else:
            g = {k: v for k, v in cur.items() if k != "params"}
            g["params"] = []
            remapped_param_groups.append(g)

    return {"state": remapped_state, "param_groups": remapped_param_groups}


def load_adam_optimizer_from_path(
    model: torch.nn.Module,
    optimizer_path: str | Path,
    *,
    lr: Optional[float] = None,
    weight_decay: float = 0.0,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> torch.optim.Optimizer:
    """Create AdamW and load state from optimizer.pt, remapping param IDs.

    If state cannot be loaded, returns a fresh AdamW with provided hparams.
    """
    from torch.optim import AdamW

    optimizer_path = Path(optimizer_path)
    if not optimizer_path.exists():
        raise FileNotFoundError(f"Optimizer state not found: {optimizer_path}")

    # Instantiate AdamW over LoRA trainables only (stable order via named_parameters)
    peft_mod = model.module if hasattr(model, "module") else model
    name2param = dict(peft_mod.named_parameters())
    trainable_items = [(n, p) for n, p in name2param.items() if p.requires_grad]
    trainable_params = [p for _, p in trainable_items]
    if not trainable_params:
        raise RuntimeError("No trainable (LoRA) parameters found; cannot build optimizer for probe.")

    # Prefer name-aware remap when optimizer_param_names.json is available
    names_json = optimizer_path.with_name("optimizer_param_names.json")
    try:
        state_dict = torch.load(str(optimizer_path), map_location="cpu")
        if names_json.exists():
            import json as _json
            with open(names_json, "r") as f:
                ordered_names_groups = _json.load(f)  # list[list[str]]

            # Flatten saved (name, old_id) pairs in group order
            saved_param_groups = state_dict.get("param_groups", [])
            saved_pairs: list[tuple[str, int]] = []
            for gi, group in enumerate(saved_param_groups):
                names_g = ordered_names_groups[gi] if gi < len(ordered_names_groups) else []
                ids_g = group.get("params", [])
                m = min(len(names_g), len(ids_g))
                for k in range(m):
                    saved_pairs.append((names_g[k], ids_g[k]))

            trainable_names = {n for n, _ in trainable_items}
            # Keep only trainables, preserving saved order
            ordered_trainable_names = [n for (n, _) in saved_pairs if n in trainable_names]
            ordered_trainable_params = [name2param[n] for n in ordered_trainable_names]

            # Build name->old_id mapping for trainables
            name_to_old_id = {n: old_id for (n, old_id) in saved_pairs if n in trainable_names}

            # Create optimizer with ordered trainables
            opt = AdamW(ordered_trainable_params, lr=(lr or 1e-4), weight_decay=weight_decay, betas=betas, eps=eps)

            # Remap state by name
            old_state = state_dict.get("state", {})
            remapped_state: dict[int, dict] = {}
            for n in ordered_trainable_names:
                old_id = name_to_old_id.get(n)
                if old_id is None:
                    continue
                st = old_state.get(old_id, None)
                if st is None:
                    continue
                remapped_state[id(name2param[n])] = st

            # Keep hyperparams from first non-empty saved group
            merged_cfg = {}
            for g in saved_param_groups:
                if g.get("params"):
                    merged_cfg = {k: v for k, v in g.items() if k != "params"}
                    break

            all_new_ids = [id(p) for p in ordered_trainable_params]
            remapped_param_groups = []
            # Current optimizer may have one group; mirror that
            g0 = merged_cfg.copy()
            g0["params"] = all_new_ids
            remapped_param_groups.append(g0)

            remapped = {"state": remapped_state, "param_groups": remapped_param_groups}
            opt.load_state_dict(remapped)
        else:
            # Fallback: positional remap after building optimizer over trainables
            opt = AdamW(trainable_params, lr=(lr or 1e-4), weight_decay=weight_decay, betas=betas, eps=eps)
            remapped = _remap_optimizer_state_ids(state_dict, opt)
            opt.load_state_dict(remapped)

        # Coverage summary: fraction of params with exp_avg_sq and step present
        have_v = 0
        have_step = 0
        total = 0
        for group in opt.param_groups:
            for p in group["params"]:
                total += 1
                st = opt.state.get(p, {})
                if isinstance(st.get("exp_avg_sq", None), torch.Tensor):
                    have_v += 1
                s = st.get("step", None)
                if isinstance(s, int) or isinstance(s, torch.Tensor):
                    have_step += 1
        coverage_v = have_v / max(total, 1)
        coverage_step = have_step / max(total, 1)
        print(f"[optimizer-state] coverage: v(exp_avg_sq) {have_v}/{total} = {coverage_v:.1%}, step {have_step}/{total} = {coverage_step:.1%}")

        return opt
    except Exception:
        # Fall back to a fresh optimizer if anything fails
        return AdamW(trainable_params, lr=(lr or 1e-4), weight_decay=weight_decay, betas=betas, eps=eps)
