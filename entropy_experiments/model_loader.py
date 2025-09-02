"""
Model and optimizer loaders for offline probes.

Supports LoRA and QLoRA via a single entrypoint and loads Adam optimizer
state with parameter-ID remapping to current model parameter IDs.
"""

from typing import Optional, Dict, Any
from pathlib import Path
import torch


def load_peft_for_probe(
    base_id: str,
    adapter_path: str,
    *,
    use_qlora: bool = False,
    dtype: str = "bf16",         # "bf16" or "fp16"
    device_map: str = "cuda",
    use_checkpointing: bool = False,
):
    """Load a PEFT (LoRA/QLoRA) model ready for probe computations.

    Sets attn_implementation="eager" for VJP compatibility and disables
    gradient checkpointing unless explicitly requested.
    """
    from transformers import AutoModelForCausalLM
    from peft import PeftModel

    torch_dtype = {
        "bf16": torch.bfloat16, 
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16, 
        "float16": torch.float16
    }[dtype]

    if not use_qlora:
        # LoRA-simple (no quantization)
        base = AutoModelForCausalLM.from_pretrained(
            base_id,
            device_map=device_map,
            torch_dtype=torch_dtype,
            attn_implementation="eager",
            trust_remote_code=True,
        )
        if hasattr(base, "gradient_checkpointing_disable"):
            base.gradient_checkpointing_disable()
        if hasattr(base.config, "use_cache"):
            base.config.use_cache = True

        peft = PeftModel.from_pretrained(base, adapter_path, is_trainable=True)
        if hasattr(peft, "enable_input_require_grads"):
            peft.enable_input_require_grads()
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

    # Instantiate AdamW with provided hparams (lr may be overridden later)
    opt = AdamW(model.parameters(), lr=(lr or 1e-4), weight_decay=weight_decay, betas=betas, eps=eps)

    try:
        state_dict = torch.load(str(optimizer_path), map_location="cpu")
        remapped = _remap_optimizer_state_ids(state_dict, opt)
        opt.load_state_dict(remapped)
        return opt
    except Exception:
        return opt  # Fall back to fresh optimizer