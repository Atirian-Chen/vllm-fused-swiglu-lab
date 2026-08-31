from vllm_fused_swiglu_lab.service_benchmark import _make_prompt


def test_make_prompt_scales_with_target_tokens():
    short = _make_prompt(0, 16)
    long = _make_prompt(0, 768)
    assert len(long) > len(short) * 20
    assert short.startswith("Request 0:")
