from __future__ import annotations

import os
from typing import Any


def _stock(x: Any):
    import torch

    half = x.shape[-1] // 2
    return torch.nn.functional.silu(x[..., :half]) * x[..., half:]


def _load_extension():
    try:
        from . import _C

        return _C
    except ImportError:
        return None


def fused_available(x: Any) -> bool:
    return (
        os.getenv("VLLM_USE_FUSED_SILU_MUL", "0") == "1"
        and getattr(x, "is_cuda", False)
        and getattr(x, "is_contiguous", lambda: False)()
        and getattr(x, "ndim", 0) >= 2
        and x.shape[-1] % 2 == 0
        and str(x.dtype) in {"torch.float16", "torch.bfloat16", "torch.float32"}
        and _load_extension() is not None
    )


def silu_and_mul(x: Any):
    """Drop-in inference activation for vLLM's packed [gate | up] tensor."""
    if os.getenv("VLLM_USE_FUSED_SILU_MUL", "0") != "1":
        return _stock(x)
    if not fused_available(x):
        return _stock(x)
    original_shape = tuple(x.shape)
    if x.ndim != 2:
        x2 = x.reshape(-1, x.shape[-1])
        y2 = _load_extension().silu_mul(x2)
        return y2.reshape(*original_shape[:-1], original_shape[-1] // 2)
    return _load_extension().silu_mul(x)
