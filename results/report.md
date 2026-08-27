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
