# models/__init__.py
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig



MODEL_REGISTRY = {
    "phi2":        "microsoft/phi-2",
    "phi1_5":      "microsoft/phi-1_5",
    "tinyllama":   "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "mistral7b":   "mistralai/Mistral-7B-Instruct-v0.3",
    "mixtral8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "gemma2b":     "google/gemma-2b-it",
    "orcamath":    "microsoft/orca-math-7b",          # eval-only
    "codellama7b": "codellama/CodeLlama-7b-hf",
    "qwen2":       "Qwen/Qwen2-0.5B",
    "qwen2_5_15":     "Qwen/Qwen2.5-1.5B",
    "qwen2_5_05":     "Qwen/Qwen2.5-0.5B",
}

def _is_local(path_like: str) -> bool:
    return Path(path_like).expanduser().exists()

def load_model(
    name_or_path: str,
    *,
    quantized: bool = False,
    **hf_kwargs
):
    """
    Load either a hub model (if `name_or_path` is in MODEL_REGISTRY or a valid hub id)
    or a *local* checkpoint directory.
    """
    # 1) Resolve remote vs local
    if name_or_path in MODEL_REGISTRY:
        model_id = MODEL_REGISTRY[name_or_path]
    elif _is_local(name_or_path):
        model_id = name_or_path                   # -> local folder on Drive
    else:
        # treat as raw HF hub id the user typed
        model_id = name_or_path

    # 2) Tokenizer
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    # 3) Quantisation toggle (works for both local + hub)
    if quantized:
        hf_kwargs.setdefault("device_map", "auto")
        hf_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype="bfloat16"
        )

    model = AutoModelForCausalLM.from_pretrained(model_id, **hf_kwargs)
    return model, tok


