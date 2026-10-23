# 17 Kernel 与双平台

| 直觉 | vLLM 的做法 | DeepInfer 的做法 |
| --- | --- | --- |
| 框架层的设计只决定「做什么」，真正的时间花在 kernel 里。两种提速方式：低精度（FP8/FP4 少读字节、算力翻倍）和融合（把几个小 kernel 合成一个，中间结果不落显存）。 | 可插拔后端：FlashAttention、FlashInfer、FlashMLA、Triton、CUTLASS、DeepGEMM 等；torch.compile 自动做部分融合。 | DeepSeek 自家的 kernel 库 + 约 30 个 TileLang 自定义 kernel，每个都写了 CUDA 和昇腾两版。 |

| 库 | 在 DeepInfer 里干什么 | 典型调用点 |
| --- | --- | --- |
| FlashMLA | 稀疏 MLA 注意力（prefill / decode） | [`attention_backend.py`](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/layers/attention_backend.py#L1) |
| DeepGEMM | FP8/FP4 GEMM、Mega MoE、mHC 融合、indexer 打分 | [`hc.py:270`](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/models/v4_modules/hc.py#L270) [`mlp.py:96`](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/models/v4_modules/mlp.py#L96) |
| DeepEP | EP dispatch/combine、CP 的 bucket buffer | [`userbuffer.py:63`](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/parallel_utils/userbuffer.py#L63) |
| TileKernels | FP8/FP4 量化与反量化、RoPE、RMSNorm、Engram 哈希与门控 | [`cache_common.py:9`](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/layers/tilelang_ops/cache_common.py#L9) [`deepseek.py:177`](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/models/deepseek.py#L177) |
| DeepSelect | 采样器里的大词表 top-k（昇腾上暂时回退到 `torch.topk`） | [`sampler_utils.py:4`](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/layers/sampler_utils.py#L4) |
| TileLang 自定义 | KV 读写、采样、置信度、ZOS 修正、CP 索引、SO mask… | [`tilelang_ops/ops.py:15`](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/layers/tilelang_ops/ops.py#L15) |

平台由 `DEEPINFER_ACCELERATOR` 决定（`deep_cluster/hardware/device.py`）。`PlatformGPU` 用 `tilelang_ops/cuda/` + NCCL，`PlatformNPU` 用 `tilelang_ops/asc/` + HCCL；同名 kernel 两套实现，缺了就抛 `KernelUnavailableError`。很多 kernel 文件末尾直接带 pytest，用 Python 参考实现对拍：想读懂一个 kernel，先读同文件里的 `*_ref` 函数最快。[`platform.py:55`](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/platform.py#L55)

低精度格式汇总（来自部署文档，与 `cache_common.py` 的字节数吻合）：主 KV 为 per-16 e4m3 scale 的 FP4；indexer K 与路由专家权重为 MXFP4（per-32 ue8m0）；SWA KV 为 MXFP8；其余线性层 FP8 e4m3、32×32 block scale。

---

[← 上一篇](./16-parallelism.md) · [交互版本节](https://middle-chunjie.github.io/deepinfer-visual-guide/interactive/#hw) · [下一篇 →](./18-practice.md)
