import os

import pytest


def test_fallback_matches_stock():
    torch = pytest.importorskip("torch")
    from fused_swiglu.activation import silu_and_mul

    os.environ["VLLM_USE_FUSED_SILU_MUL"] = "0"
    x = torch.randn(3, 14)
    expected = torch.nn.functional.silu(x[:, :7]) * x[:, 7:]
    assert torch.equal(silu_and_mul(x), expected)

