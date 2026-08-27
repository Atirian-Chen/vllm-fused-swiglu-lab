#pragma once

#include <cstdint>
#include <cuda_runtime.h>

void launch_fused_silu_mul_fp32(const float* x, float* y, int64_t tokens,
                                int64_t intermediate, cudaStream_t stream);
void launch_fused_silu_mul_fp16(const void* x, void* y, int64_t tokens,
                                int64_t intermediate, cudaStream_t stream);
void launch_fused_silu_mul_bf16(const void* x, void* y, int64_t tokens,
                                int64_t intermediate, cudaStream_t stream);

