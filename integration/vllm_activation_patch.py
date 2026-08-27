"""Small V3 adapter used inside a fixed vLLM checkout/container.

Run with VLLM_USE_FUSED_SILU_MUL=1. The patch deliberately keeps stock
behavior as the fallback and does not touch the scheduler or KV cache.
"""

from importlib import import_module

from fused_swiglu.activation import silu_and_mul


def patch_vllm_activation(module_name: str = "vllm.model_executor.layers.activation"):
    module = import_module(module_name)
    cls = getattr(module, "SiluAndMul")
    original = cls.forward

    def forward(self, x):
        return silu_and_mul(x)

    cls.forward = forward
    return original


def install():
    return patch_vllm_activation()

