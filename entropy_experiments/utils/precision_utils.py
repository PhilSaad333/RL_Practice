from __future__ import annotations
import contextlib
from contextlib import ExitStack
from typing import Iterator, Optional
import torch

def apply_global_precision(allow_tf32: bool = True, matmul_precision: str = "high") -> None:
    # Call once at startup (after parsing config)
    try:
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
    except Exception:
        pass
    if hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision(matmul_precision)
        except Exception:
            pass

@contextlib.contextmanager
def forward_precision_ctx(*, autocast: bool, dtype: torch.dtype) -> Iterator[None]:
    if autocast:
        with torch.autocast(device_type="cuda", dtype=dtype):
            yield
    else:
        # Ensure no stale autocast context is active
        with torch.cuda.amp.autocast(False):
            yield

def str_to_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32, "fp32": torch.float32,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        "float16": torch.float16, "fp16": torch.float16,
        "float64": torch.float64, "fp64": torch.float64,
    }.get(name.lower(), torch.float32)

def maybe_cast_logits_fp32(x):
    return x.float() if x.is_floating_point() and x.dtype != torch.float32 else x

def force_grads_fp32(module: torch.nn.Module) -> None:
    for p in module.parameters():
        if p.grad is not None and p.grad.dtype != torch.float32:
            p.grad = p.grad.float()
