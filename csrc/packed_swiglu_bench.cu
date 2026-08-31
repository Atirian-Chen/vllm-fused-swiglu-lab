#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(expr)                                                         \
  do {                                                                           \
    cudaError_t status = (expr);                                                 \
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status)); \
  } while (0)

struct Config {
  int64_t tokens = 128;
  int64_t intermediate = 8960;
  int warmup = 200;
  int iters = 200;
  std::string dtype = "fp16";
};

__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }

__global__ void stock_silu_fp32(const float* x, float* tmp, int64_t n, int64_t i) {
  int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) tmp[idx] = silu(x[(idx / i) * (2 * i) + (idx % i)]);
}
__global__ void stock_mul_fp32(const float* x, const float* tmp, float* y, int64_t n, int64_t i) {
  int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) y[idx] = tmp[idx] * x[(idx / i) * (2 * i) + i + (idx % i)];
}
__global__ void fused_fp32(const float* x, float* y, int64_t n, int64_t i) {
  int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    int64_t row = idx / i, col = idx % i;
    y[idx] = silu(x[row * (2 * i) + col]) * x[row * (2 * i) + i + col];
  }
}

__global__ void stock_silu_fp16(const __half* x, __half* tmp, int64_t n, int64_t i) {
  int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) tmp[idx] = __float2half_rn(silu(__half2float(x[(idx / i) * (2 * i) + (idx % i)])));
}
__global__ void stock_mul_fp16(const __half* x, const __half* tmp, __half* y, int64_t n, int64_t i) {
  int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) y[idx] = __hmul(tmp[idx], x[(idx / i) * (2 * i) + i + (idx % i)]);
}
__global__ void fused_fp16(const __half* x, __half* y, int64_t n, int64_t i) {
  int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    int64_t row = idx / i, col = idx % i;
    y[idx] = __float2half_rn(silu(__half2float(x[row * (2 * i) + col])) *
                             __half2float(x[row * (2 * i) + i + col]));
  }
}

__global__ void stock_silu_bf16(const __nv_bfloat16* x, __nv_bfloat16* tmp,
                                int64_t n, int64_t i) {
  int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    tmp[idx] = __float2bfloat16_rn(
        silu(__bfloat162float(x[(idx / i) * (2 * i) + (idx % i)])));
  }
}
__global__ void stock_mul_bf16(const __nv_bfloat16* x,
                               const __nv_bfloat16* tmp,
                               __nv_bfloat16* y, int64_t n, int64_t i) {
  int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    y[idx] = __float2bfloat16_rn(
        __bfloat162float(tmp[idx]) *
        __bfloat162float(x[(idx / i) * (2 * i) + i + (idx % i)]));
  }
}
__global__ void fused_bf16(const __nv_bfloat16* x, __nv_bfloat16* y,
                           int64_t n, int64_t i) {
  int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    int64_t row = idx / i, col = idx % i;
    y[idx] = __float2bfloat16_rn(
        silu(__bfloat162float(x[row * (2 * i) + col])) *
        __bfloat162float(x[row * (2 * i) + i + col]));
  }
}

float percentile(std::vector<float> values, float p) {
  std::sort(values.begin(), values.end());
  return values[static_cast<size_t>(std::ceil(values.size() * p)) - 1];
}

template <typename Fn>
float bench(Fn&& launch, int warmup, int iters) {
  for (int k = 0; k < warmup; ++k) launch();
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> samples;
  samples.reserve(iters);
  for (int k = 0; k < iters; ++k) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start)); launch(); CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    samples.push_back(ms); CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
  }
  return percentile(std::move(samples), 0.50f);
}

Config parse(int argc, char** argv) {
  Config c;
  for (int a = 1; a < argc; ++a) {
    std::string key = argv[a];
    if (a + 1 >= argc) throw std::invalid_argument("value missing for " + key);
    std::string value = argv[++a];
    if (key == "--tokens") c.tokens = std::stoll(value);
    else if (key == "--intermediate") c.intermediate = std::stoll(value);
    else if (key == "--warmup") c.warmup = std::stoi(value);
    else if (key == "--iters") c.iters = std::stoi(value);
    else if (key == "--dtype") c.dtype = value;
    else throw std::invalid_argument("unknown argument " + key);
  }
  if (c.tokens <= 0 || c.intermediate <= 0 || c.iters <= 0 ||
      (c.dtype != "fp16" && c.dtype != "bf16" && c.dtype != "fp32"))
    throw std::invalid_argument("invalid config");
  return c;
}

int main(int argc, char** argv) {
  try {
    Config c = parse(argc, argv);
    const int64_t n = c.tokens * c.intermediate;
    const int threads = 256;
    const int blocks = int((n + threads - 1) / threads);
    std::mt19937 rng(20260828); std::uniform_real_distribution<float> dist(-3, 3);
    cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    float stock_ms = 0, fused_ms = 0, max_abs = 0;
    if (c.dtype == "fp32") {
      std::vector<float> host(2 * n); for (float& v : host) v = dist(rng);
      float *x, *tmp, *stock, *fused; CUDA_CHECK(cudaMalloc(&x, 2*n*sizeof(float)));
      CUDA_CHECK(cudaMalloc(&tmp, n*sizeof(float))); CUDA_CHECK(cudaMalloc(&stock, n*sizeof(float))); CUDA_CHECK(cudaMalloc(&fused, n*sizeof(float)));
      CUDA_CHECK(cudaMemcpy(x, host.data(), 2*n*sizeof(float), cudaMemcpyHostToDevice));
      auto baseline = [&] { stock_silu_fp32<<<blocks,threads>>>(x,tmp,n,c.intermediate); stock_mul_fp32<<<blocks,threads>>>(x,tmp,stock,n,c.intermediate); };
      auto fused_launch = [&] { fused_fp32<<<blocks,threads>>>(x,fused,n,c.intermediate); };
      stock_ms = bench(baseline,c.warmup,c.iters); fused_ms = bench(fused_launch,c.warmup,c.iters); baseline(); fused_launch(); CUDA_CHECK(cudaDeviceSynchronize());
      std::vector<float> a(n), b(n); CUDA_CHECK(cudaMemcpy(a.data(),stock,n*sizeof(float),cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(b.data(),fused,n*sizeof(float),cudaMemcpyDeviceToHost));
      for (int64_t k=0;k<n;++k) max_abs=std::max(max_abs,std::abs(a[k]-b[k])); cudaFree(x); cudaFree(tmp); cudaFree(stock); cudaFree(fused);
    } else if (c.dtype == "fp16") {
      std::vector<__half> host(2 * n); for (__half& v : host) v = __float2half_rn(dist(rng));
      __half *x, *tmp, *stock, *fused; CUDA_CHECK(cudaMalloc(&x, 2*n*sizeof(__half)));
      CUDA_CHECK(cudaMalloc(&tmp, n*sizeof(__half))); CUDA_CHECK(cudaMalloc(&stock, n*sizeof(__half))); CUDA_CHECK(cudaMalloc(&fused, n*sizeof(__half)));
      CUDA_CHECK(cudaMemcpy(x, host.data(), 2*n*sizeof(__half), cudaMemcpyHostToDevice));
      auto baseline = [&] { stock_silu_fp16<<<blocks,threads>>>(x,tmp,n,c.intermediate); stock_mul_fp16<<<blocks,threads>>>(x,tmp,stock,n,c.intermediate); };
      auto fused_launch = [&] { fused_fp16<<<blocks,threads>>>(x,fused,n,c.intermediate); };
      stock_ms = bench(baseline,c.warmup,c.iters); fused_ms = bench(fused_launch,c.warmup,c.iters); baseline(); fused_launch(); CUDA_CHECK(cudaDeviceSynchronize());
      std::vector<__half> a(n), b(n); CUDA_CHECK(cudaMemcpy(a.data(),stock,n*sizeof(__half),cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(b.data(),fused,n*sizeof(__half),cudaMemcpyDeviceToHost));
      for (int64_t k=0;k<n;++k) max_abs=std::max(max_abs,std::abs(__half2float(a[k])-__half2float(b[k]))); cudaFree(x); cudaFree(tmp); cudaFree(stock); cudaFree(fused);
    } else {
      std::vector<__nv_bfloat16> host(2 * n);
      for (__nv_bfloat16& v : host) v = __float2bfloat16_rn(dist(rng));
      __nv_bfloat16 *x, *tmp, *stock, *fused;
      CUDA_CHECK(cudaMalloc(&x, 2*n*sizeof(__nv_bfloat16)));
      CUDA_CHECK(cudaMalloc(&tmp, n*sizeof(__nv_bfloat16)));
      CUDA_CHECK(cudaMalloc(&stock, n*sizeof(__nv_bfloat16)));
      CUDA_CHECK(cudaMalloc(&fused, n*sizeof(__nv_bfloat16)));
      CUDA_CHECK(cudaMemcpy(x, host.data(), 2*n*sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
      auto baseline = [&] { stock_silu_bf16<<<blocks,threads>>>(x,tmp,n,c.intermediate); stock_mul_bf16<<<blocks,threads>>>(x,tmp,stock,n,c.intermediate); };
      auto fused_launch = [&] { fused_bf16<<<blocks,threads>>>(x,fused,n,c.intermediate); };
      stock_ms = bench(baseline,c.warmup,c.iters); fused_ms = bench(fused_launch,c.warmup,c.iters); baseline(); fused_launch(); CUDA_CHECK(cudaDeviceSynchronize());
      std::vector<__nv_bfloat16> a(n), b(n);
      CUDA_CHECK(cudaMemcpy(a.data(),stock,n*sizeof(__nv_bfloat16),cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(b.data(),fused,n*sizeof(__nv_bfloat16),cudaMemcpyDeviceToHost));
      for (int64_t k=0;k<n;++k) max_abs=std::max(max_abs,std::abs(__bfloat162float(a[k])-__bfloat162float(b[k])));
      cudaFree(x); cudaFree(tmp); cudaFree(stock); cudaFree(fused);
    }
    std::cout << std::fixed << std::setprecision(6)
              << "{\n  \"device\": \"" << prop.name << "\",\n  \"tokens\": " << c.tokens
              << ",\n  \"intermediate\": " << c.intermediate << ",\n  \"dtype\": \"" << c.dtype
              << "\",\n  \"layout\": \"packed_gate_up\",\n  \"stock_two_kernel_p50_ms\": " << stock_ms
              << ",\n  \"fused_one_kernel_p50_ms\": " << fused_ms << ",\n  \"speedup_p50\": " << stock_ms/fused_ms
              << ",\n  \"max_abs_fused_vs_stock\": " << max_abs << "\n}\n";
  } catch (const std::exception& e) { std::cerr << "error: " << e.what() << '\n'; return 1; }
}
