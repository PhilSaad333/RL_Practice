# models/__init__.py
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_REGISTRY = {
    "phi2":        "microsoft/phi-2",
    "phi1_5":      "microsoft/phi-1_5",
    "tinyllama":   "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "mistral7b":   "mistralai/Mistral-7B-Instruct-v0.3",
    "mixtral8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "gemma2b":     "google/gemma-2b-it",
    "orcamath":    "microsoft/orca-math-7b",          # eval-only
    "codellama7b": "codellama/CodeLlama-7b-hf"
}

def load_model(name: str, quantized: bool = False, **hf_kwargs):
    """Return (model, tokenizer).  Set quantized=True for 4-bit QLoRA debug."""
    model_id = MODEL_REGISTRY[name]
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if quantized:
        from transformers import BitsAndBytesConfig
        hf_kwargs.setdefault("device_map", "auto")
        hf_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype="bfloat16"
        )
    model = AutoModelForCausalLM.from_pretrained(model_id, **hf_kwargs)
    return model, tok

