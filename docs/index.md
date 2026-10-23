---
layout: home
hero:
  name: DeepInfer 可视化导读
  text: 图解 DeepSeek 推理框架
  tagline: 先讲清「大模型推理慢在哪」和「vLLM 怎么做」，再图解 DeepInfer 每个子系统的设计直觉与源码。非官方学习资料。
  actions:
    - theme: brand
      text: 开始阅读
      link: /chapters/01-positioning
    - theme: alt
      text: 打开交互版
      link: https://middle-chunjie.github.io/deepinfer-visual-guide/interactive/
    - theme: alt
      text: GitHub
      link: https://github.com/middle-chunjie/deepinfer-visual-guide
features:
  - title: 零基础可读
    details: 第二部分从 prefill / decode、显存带宽瓶颈、KV cache、PagedAttention 讲起，不要求事先读过推理框架。
  - title: 逐项对照 vLLM
    details: 第三部分每节都按「直觉 → vLLM 的做法 → DeepInfer 的做法」展开，说明专用框架为什么这样取舍。
  - title: 25 张机制图 + 交互版
    details: 文字版配静态 SVG 图；交互版里可以拖动显存带宽计算器、逐层查看模型、试变长验证和 CP 切分。
  - title: 每个结论都有出处
    details: 源码结论链接到 DeepInfer 固定提交 9bd9cf2 的具体行；估算和推断单独标注。
---

## 怎么读

| 你的情况 | 建议路线 |
| --- | --- |
| 没读过推理框架，想知道 DeepInfer 为什么快 | 从 [04 推理到底在算什么](./chapters/04-inference-basics.md) 开始，读完第二部分再进第三部分 |
| 熟悉 vLLM / SGLang，想看 DeepInfer 的差异 | 先看 [07 的总对比表](./chapters/07-vllm.md)，再挑第三部分感兴趣的章节 |
| 准备读源码 | [08 一个请求的旅程](./chapters/08-request-journey.md) + [18 推荐阅读顺序](./chapters/18-practice.md) |

## 目录

**第一部分 · 全局**：[01 定位与开源边界](./chapters/01-positioning.md) · [02 仓库地图](./chapters/02-repo-map.md) · [03 进程模型与启动](./chapters/03-processes.md)

**第二部分 · 推理基础**：[04 推理到底在算什么](./chapters/04-inference-basics.md) · [05 瓶颈：显存带宽](./chapters/05-memory-bound.md) · [06 KV cache 与 PagedAttention](./chapters/06-kv-cache.md) · [07 vLLM 是怎么设计的](./chapters/07-vllm.md)

**第三部分 · 拆解 DeepInfer**：[08 一个请求的旅程](./chapters/08-request-journey.md) · [09 调度器](./chapters/09-scheduler.md) · [10 CPU 开销：异步调度与 CUDA Graph](./chapters/10-cpu-overhead.md) · [11 V4.1-Flash 模型结构](./chapters/11-model.md) · [12 注意力与 KV 存储](./chapters/12-attention-kv.md) · [13 前缀缓存、YoCo 与有界重放](./chapters/13-prefix-yoco.md) · [14 DSpark 投机解码](./chapters/14-dspark.md) · [15 采样、确定性、结构化输出](./chapters/15-sampling.md) · [16 并行：DP / EP / CP / EPLB](./chapters/16-parallelism.md) · [17 Kernel 与双平台](./chapters/17-kernels.md) · [18 动手与自测](./chapters/18-practice.md)

## 和其他社区资料的关系

这份导读偏「建立直觉」。想继续深入，推荐搭配：

- [cluster1900/deepinfer-reader](https://github.com/cluster1900/deepinfer-reader)：逐行对照源码的中英文书，并从零实现教学引擎 `minidi`。适合读完本导读后深挖。
- [algebra-MCX/mini-DeepInfer](https://github.com/algebra-MCX/mini-DeepInfer)、[hoppespy/mini-DeepInfer](https://github.com/hoppespy/mini-DeepInfer)：用几千行 PyTorch 重写的可运行精简版。
- [geekchow/deepseek-v41-flash-inference-guide](https://github.com/geekchow/deepseek-v41-flash-inference-guide)：侧重整套部署系统（deepseek-recipe、KV 存储、负载均衡）。
- 官方：[DeepInfer 仓库](https://github.com/deepseek-ai/DeepInfer) 与其中的 `doc/inference_deployment_qa.md`。

## 准确性说明

本导读基于静态源码阅读，没有在 SM100 / 昇腾硬件上运行验证。V4.1 超参数取自 HuggingFace 上的 `DeepSeek-V4.1-Flash-1017/config.json`，生产数字引自官方部署文档，vLLM 的描述来自其公开文档（版本间可能有差异）。标「估算」「推断」的地方没有实测或直接出处。发现错误欢迎提 [Issue](https://github.com/middle-chunjie/deepinfer-visual-guide/issues)。
