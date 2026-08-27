# vLLM + CUDA Fused SwiGLU Lab

一个最小可运行的 V0→V4 实验项目：把 packed SwiGLU CUDA 微基准，逐步接入固定版本 vLLM 的 Qwen2 MLP activation，并对比 kernel、模型层和服务层数据。

## 当前实现

- **V0**：CMake CUDA benchmark，stock 双 kernel vs fused 单 kernel。
- **V1**：输入布局对齐为 `[tokens, 2 * intermediate] = [gate | up]`，输出 `[tokens, intermediate]`；支持 FP16/FP32。
- **V2**：PyTorch CUDA extension，Python wrapper、当前 stream、fallback 和正确性测试；kernel 同时包含 BF16 launcher。
- **V3**：固定 vLLM tag 的 adapter，`VLLM_USE_FUSED_SILU_MUL=1` 时替换 `SiluAndMul`。
- **V4**：kernel shape matrix；服务层用 SSE benchmark 对 stock/fused 镜像做相同参数 A/B。

## 快速开始：V0/V1

在带 CUDA Toolkit 的 Windows shell：

```powershell
cd projects/vllm-fused-swiglu-lab
powershell -ExecutionPolicy Bypass -File .\scripts\build_bench.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\run_v0_v1.ps1 -Tokens 128 -Intermediate 8960 -Dtype fp16
```

输出包括 `stock_two_kernel_p50_ms`、`fused_one_kernel_p50_ms`、`speedup_p50` 和 `max_abs_fused_vs_stock`。

## V2：构建 extension

V2 需要 Linux/CUDA 容器中的 PyTorch 开发头文件：

```bash
BUILD_FUSED_OP=1 pip install --no-build-isolation .
python - <<'PY'
import torch
from fused_swiglu import silu_and_mul
x = torch.randn(4, 2 * 8960, device='cuda', dtype=torch.float16)
print(silu_and_mul(x).shape)
PY

# 已安装 fused 镜像中可复测三种 dtype 的误差
python scripts/check_v2.py --output results/v2_correctness.json
```

Windows 主机 Python 没有 torch 时，V2/V3 不会构建；V0/V1 仍可用本机 `nvcc`。

## V3：vLLM 接入

在固定 `vllm/vllm-openai:v0.10.2` 的 Linux/CUDA 镜像中构建：

```bash
docker build -f Dockerfile.fused -t vllm-fused-swiglu:v0.1 .
docker run --gpus all -p 8000:8000 -e VLLM_USE_FUSED_SILU_MUL=1 \
  vllm-fused-swiglu:v0.1 --model Qwen/Qwen2.5-1.5B-Instruct
```

`integration/vllm_activation_patch.py` 是最小 adapter。实际 vLLM 版本若使用不同 custom-op registry，应把同一 `silu_and_mul` 调用放入该版本的 `SiluAndMul` 实现；不要把 patch 文件本身误认为已经改写了 vLLM 镜像。

## V4：服务 A/B

可以手动启动 stock/fused 服务，也可以直接运行脚本（脚本会轮流停止其中一个服务，避免 8GB 显存上的资源争用）：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_service_ab.ps1 `
  -Model Qwen/Qwen2.5-1.5B-Instruct -Concurrency 4 -Requests 24 -OutputTokens 48
```

脚本会保持模型、GPU、`max-model-len`、`max-num-seqs` 和客户端参数一致，并生成 `service_stock.json`、`service_fused.json` 和 `service_ab_summary.json`。手动运行 benchmark 时：

```powershell
python -m vllm_fused_swiglu_lab.service_benchmark `
  --base-url http://127.0.0.1:8000 `
  --model Qwen/Qwen2.5-1.5B-Instruct `
  --implementation fused --concurrency 8 --requests 60 `
  --output results/service_fused.json
```

对照结果至少记录：TTFT P50、延迟 P95、输出 token 数和成功率。该脚本只负责最小服务数据采集，不修改 scheduler、KV cache 或 attention backend。

## 重要边界

- 当前项目是新建的实验实现，不等同于已有 `kernel-benchmark-lab` 或 `vllm-serving-lab` 已完成端到端接入。
- `8960` 是默认可修改的实验维度；真正接入 vLLM 前必须读取目标模型的 `intermediate_size`。
- CUDA kernel 的局部 speedup 不能直接外推为服务 speedup。
- 训练/反向路径未实现；训练模式应回退 stock。
- vLLM Scheduler、Continuous Batching、Paged KV Cache 没有被修改。
