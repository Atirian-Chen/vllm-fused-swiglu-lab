# V0–V4 实测记录

测试设备：NVIDIA GeForce RTX 4070 Laptop GPU（8GB）。项目优化的是 packed SwiGLU 的 `SiLU(gate) * up`，输入 `[T, 2I] = [gate | up]`，输出 `[T, I]`。

## V0/V1：CUDA 微基准

FP16、`T=128`、`I=8960`、warm-up/iterations 均为 200：

| 实现 | P50 |
| --- | ---: |
| stock 双 kernel | 0.029696 ms |
| fused 单 kernel | 0.018432 ms |
| speedup | 1.611x |

融合结果相对 stock 的最大绝对误差为 `0.007812`（FP16 舍入范围内）。完整 shape 矩阵见 [`kernel_matrix.json`](kernel_matrix.json)。

## V2：PyTorch CUDA extension

`fused_swiglu._C.silu_mul` 支持 FP32、FP16、BF16，使用当前 CUDA stream，并在不满足条件时回退到 PyTorch 实现。容器内测试 shape 为 `[3, 17920] -> [3, 8960]`，固定随机种子为 0，参考实现为同 dtype 的 PyTorch `silu(gate) * up`：

- FP32 最大误差 `9.54e-7`；
- FP16 最大误差 `0.0078125`；
- BF16 最大误差 `0.03125`。

FP16/BF16 的误差来自 fused kernel 在 float 中计算 SiLU 和乘法后再转换回输入 dtype，属于对应低精度格式的量化误差；原始数据见 [`v2_correctness.json`](v2_correctness.json)。

## V3：vLLM v0.10.2 接入

`VLLM_USE_FUSED_SILU_MUL=1` 时，镜像中的 `SiluAndMul.forward_cuda` 来源为 `fused_swiglu.vllm_adapter`；不满足 CUDA/连续布局/支持 dtype 条件时走原始实现。没有修改 scheduler、continuous batching、Paged KV Cache 或 attention backend。

## V4：服务层 A/B

模型为 `Qwen/Qwen2.5-1.5B-Instruct`，`max-model-len=1024`、`max-num-seqs=8`、并发 4、24 个请求、每个最多生成 48 token。两个容器轮流单独占用 GPU，并各发送一次预热请求。

| 指标 | stock | fused | fused - stock |
| --- | ---: | ---: | ---: |
| TTFT P50 | 40.15 ms | 43.20 ms | +3.05 ms |
| latency P95 | 1233.09 ms | 1347.75 ms | +114.66 ms |
| output throughput | 185.07 tok/s | 173.43 tok/s | -11.64 tok/s |
| success rate | 100% | 100% | — |

这次短测没有观察到服务级收益，说明单个 activation kernel 的局部加速不能直接等价为端到端收益；服务结果还会受编译缓存、调度、采样和测量顺序影响。原始请求级数据在 [`service_stock.json`](service_stock.json) 和 [`service_fused.json`](service_fused.json)，汇总在 [`service_ab_summary.json`](service_ab_summary.json)。

## 扩展 shape sweep（2026-08-31）

为回答“是否所有 shape 都比原版好”，补跑了完整的 `T` 扫描：

- `T ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096}`；
- `I=8960`，对应 Qwen2.5-1.5B 的 MLP intermediate size；
- FP16、BF16、FP32 各 13 个 shape，共 39 组；
- 每组 warm-up 200 次、计时 200 次，统计 CUDA event P50。

本轮结果见 [`kernel_matrix_v2.json`](kernel_matrix_v2.json)：

| dtype | speedup 最小值 | speedup 中位数 | speedup 最大值 | fused 是否全部更快 |
| --- | ---: | ---: | ---: | --- |
| FP16 | 1.350x（T=128） | 1.791x | 2.453x（T=512） | 是（13/13） |
| BF16 | 1.253x（T=1） | 1.701x | 2.355x（T=512） | 是（13/13） |
| FP32 | 1.167x（T=2） | 1.798x | 3.635x（T=256） | 是（13/13） |

因此，在“项目内两 kernel baseline”这个比较对象下，本轮 39/39 个 shape 都是 fused 更快；但收益不稳定，最低只有 1.17x，短 kernel 的 P50 也会受时钟和系统噪声影响。这个结果不能外推成生产 vLLM 的 39/39，因为 vLLM 原生实现可能已经使用融合 activation，必须用 profiler 确认实际 kernel。

## 为什么 V4 没有直接覆盖多种矩阵

服务层的 `T` 不是用户可以直接指定的单一矩阵维度：prefill 时它近似是一批请求的 prompt token 总数，decode 时通常接近当前活跃序列数，并且会随 continuous batching 动态变化。原 V4 只使用并发 4、短 prompt、固定 48 output token 的单个负载，主要得到的是一个混合的 prefill/decode 测试，不能代表不同 `T`。

已新增可复现的服务矩阵脚本 [`run_service_matrix.ps1`](../scripts/run_service_matrix.ps1)，默认覆盖：

| case | prompt 目标 token | output token | 并发 | 关注阶段 |
| --- | ---: | ---: | ---: | --- |
| decode_t1 | 16 | 96 | 1 | 单请求 decode |
| decode_t4 | 16 | 96 | 4 | 低并发 decode |
| decode_t8 | 16 | 96 | 8 | 高并发 decode |
| prefill_small | 64 | 8 | 1 | 小 prefill |
| prefill_medium | 256 | 8 | 4 | 中等 prefill |
| prefill_large | 768 | 8 | 4 | 大 prefill |
| mixed | 256 | 64 | 4 | prefill + decode |

脚本会以交叉顺序运行 stock/fused，每个 case 先发 warm-up 请求，并记录实际 prompt token、TTFT P50/P95、端到端 latency P50/P95、TPOT P50、输入/输出吞吐和成功率。当前机器在 2026-08-31 运行该脚本前 Docker Desktop 后端因遗留 Windows socket（`dockerInference` / `dockerEthernetVfkit` / `engine.sock`）无法启动，所以新的服务矩阵尚未产生数据；不能用旧的单一 V4 结果填充这些 case。Docker/WSL 重启后可直接运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_service_matrix.ps1 -Rounds 2
```
