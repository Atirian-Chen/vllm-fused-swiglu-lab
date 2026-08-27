import os


if os.getenv("VLLM_USE_FUSED_SILU_MUL") == "1":
    try:
        import vllm.model_executor.layers.activation as _activation
        from fused_swiglu.vllm_adapter import patch_vllm_activation

        patch_vllm_activation(_activation)
    except Exception as _exc:
        print(f"[fused-swiglu] patch skipped: {_exc}")
