#include "fused_swiglu.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {

__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }

__global__ void fused_fp32_kernel(const float* x, float* y, int64_t n, int64_t i) {
  int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  int64_t row = idx / i, col = idx % i;
  y[idx] = silu(x[row * (2 * i) + col]) * x[row * (2 * i) + i + col];
}

__global__ void fused_fp16_kernel(const __half* x, __half* y, int64_t n, int64_t i) {
  int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  int64_t row = idx / i, col = idx % i;
  y[idx] = __float2half_rn(silu(__half2float(x[row * (2 * i) + col])) *
                           __half2float(x[row * (2 * i) + i + col]));
}

__global__ void fused_bf16_kernel(const __nv_bfloat16* x, __nv_bfloat16* y,
                                  int64_t n, int64_t i) {
  int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  int64_t row = idx / i, col = idx % i;
  y[idx] = __float2bfloat16(silu(__bfloat162float(x[row * (2 * i) + col])) *
                            __bfloat162float(x[row * (2 * i) + i + col]));
}

}  // namespace

void launch_fused_silu_mul_fp32(const float* x, float* y, int64_t tokens,
                                int64_t intermediate, cudaStream_t stream) {
  const int64_t n = tokens * intermediate;
  fused_fp32_kernel<<<static_cast<int>((n + 255) / 256), 256, 0, stream>>>(
      x, y, n, intermediate);
}

void launch_fused_silu_mul_fp16(const void* x, void* y, int64_t tokens,
                                int64_t intermediate, cudaStream_t stream) {
  const int64_t n = tokens * intermediate;
  fused_fp16_kernel<<<static_cast<int>((n + 255) / 256), 256, 0, stream>>>(
      static_cast<const __half*>(x), static_cast<__half*>(y), n, intermediate);
}

void launch_fused_silu_mul_bf16(const void* x, void* y, int64_t tokens,
                                int64_t intermediate, cudaStream_t stream) {
  const int64_t n = tokens * intermediate;
  fused_bf16_kernel<<<static_cast<int>((n + 255) / 256), 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(x), static_cast<__nv_bfloat16*>(y), n,
      intermediate);
}
