# models/__init__.py
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_REGISTRY = {
    "phi2":  "microsoft/phi-2",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "llama3-8b": "meta-llama/Meta-Llama-3-8B"
}
def load_model(name, **kw):
    tok = AutoTokenizer.from_pretrained(MODEL_REGISTRY[name], use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_REGISTRY[name], **kw)
    return model, tok
