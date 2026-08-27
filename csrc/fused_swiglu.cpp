#include "fused_swiglu.h"

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace {

void check_input(const at::Tensor& x) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(x.dim() == 2, "x must be [tokens, 2*intermediate]");
  TORCH_CHECK(x.size(1) > 0 && x.size(1) % 2 == 0, "last dimension must be positive and even");
  TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf ||
              x.scalar_type() == at::kBFloat16,
              "supported dtypes are float32, float16 and bfloat16");
}

}  // namespace

at::Tensor silu_mul_cuda(const at::Tensor& x) {
  check_input(x);
  const int64_t tokens = x.size(0);
  const int64_t intermediate = x.size(1) / 2;
  auto y = at::empty({tokens, intermediate}, x.options());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  if (x.scalar_type() == at::kFloat) {
    launch_fused_silu_mul_fp32(x.data_ptr<float>(), y.data_ptr<float>(), tokens,
                               intermediate, stream.stream());
  } else if (x.scalar_type() == at::kHalf) {
    launch_fused_silu_mul_fp16(x.data_ptr(), y.data_ptr(), tokens, intermediate,
                               stream.stream());
  } else {
    launch_fused_silu_mul_bf16(x.data_ptr(), y.data_ptr(), tokens, intermediate,
                               stream.stream());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("silu_mul", &silu_mul_cuda, "Packed SiLU(gate) * up (CUDA)");
}

