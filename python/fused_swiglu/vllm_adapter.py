from __future__ import annotations

from .activation import silu_and_mul, fused_available


def patch_vllm_activation(module):
    """Patch the v0.10.2 CUDA dispatch while preserving stock fallback."""
    original = module.SiluAndMul.forward_cuda

    def forward_cuda(self, x):
        if fused_available(x):
            return silu_and_mul(x)
        return original(self, x)

    module.SiluAndMul.forward_cuda = forward_cuda
    return original

