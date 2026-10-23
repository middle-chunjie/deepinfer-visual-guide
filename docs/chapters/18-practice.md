# 18 动手与自测

## 跑起来

```text
export DEEPINFER_ACCELERATOR=nvidia      # 或 ascend；运行时也要保留
./build.sh                                # 装依赖 → 编 vendor 子模块 → 编 _C
export WORLD_SIZE=1 RANK=0 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python examples/offline_inference.py --ep 8           # dp=8，每卡一个引擎
python examples/offline_inference.py --ep 2 --cp 8    # 试 zigzag CP
DEEPINFER_ENABLE_MTP_VARLEN=1 python examples/benchmark.py --decode_num_requests 64
```

要求 Linux、GCC ≥ 13、Python 3.12、CUDA 13（或 CANN 9.2）、SM100 GPU。没有这类硬件也能读：C++ 调度器和大部分 TileLang 参考实现是纯逻辑，可以照着测试读。如果你想先在普通 GPU 上建立直觉，可以用 vLLM 跑一个小模型，打开 `--enable-prefix-caching`、`--speculative-config` 等选项观察吞吐变化，再回来读 DeepInfer。

| 环境变量 | 代码默认 | 作用 |
| --- | --- | --- |
| `DEEPINFER_USE_NEXTN_DECODE` | 5 | 草稿块长；0 关闭投机 |
| `DEEPINFER_ENABLE_MTP_VARLEN` | 0 | 置信度驱动的变长验证 |
| `DEEPINFER_ZERO_OVERHEAD_SCHEDULING` | 按引擎类型 | ZOS；NORMAL/COMPLETION 默认开 |
| `DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE` | off | off / mask / compact |
| `DEEPINFER_BATCH_INVARIANT` | 0 | 批次不变（离线示例在 NVIDIA 上默认设 1） |
| `DEEPINFER_PREFILL_ATTEMPT_INTERVAL` / `_ANY_` | 70 / 100 | DP 投票点间隔 |
| `DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS` | 0 | EPLB 冗余专家下限 |

源码：[`deepinfer/environ.py（完整列表）`](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/environ.py#L1)

## 推荐阅读顺序

- [ ] **本教程第 04–07 节 + vLLM 论文《PagedAttention》**：先把基线建立起来。
- [ ] **examples/offline_inference.py → entrypoints/llm.py**：看清 DP 进程怎么起、generate 的循环怎么写。约 300 行。
- [ ] **engine/llm_engine.py 的 step / _process_model_outputs**：引擎主循环和所有「虚函数」扩展点。
- [ ] **worker/worker.py 全文**：四段式 execute_model_step 与 ZOS 修正，283 行，最值得精读。
- [ ] **csrc/engine/scheduler.hpp 的 _schedule / _schedule_prefill / _schedule_decode / process_outputs**：投票、chunk、抢占、乐观推进。
- [ ] **hugging_face/config.py → models/deepseek.py → decoder_layer.py**：先看配置怎么变成层，再看 forward。
- [ ] **v4_modules/attn.py、index_attn.py、hc.py**：最硬的一块，配合第 11–12 节的图读。
- [ ] **mtp/dflash.py → layers/sampler.py → csrc/engine/mtp_varlen.hpp**：投机解码从草稿到验证到调度的闭环。
- [ ] **worker/cache_engine.py + tilelang_ops/store_kv_cache.py**：KV 布局与字节格式。
- [ ] **parallel_utils/ + eplb/ + tilelang_ops/prepare_cp_mirror.py**：并行与负载均衡。
- [ ] **doc/inference_deployment_qa.md**：最后读，把代码对回生产部署参数。

## 自测题

<details>
<summary>1. 为什么 decode 阶段把 batch 从 1 加到 64，每步耗时几乎不变？</summary>

decode 每步要把全部权重从显存读一遍，这个时间和 batch 大小无关；而 64 个 token 的计算量远没到 GPU 算力上限。所以耗时由「读权重」决定，batch 变大只是让同一次读取服务更多 token。可以用第 05 节的计算器（交互版）或表格验证。

</details>

<details>
<summary>2. PagedAttention 解决了什么问题？DeepInfer 的块大小和 vLLM 有什么不同？</summary>

解决按最大长度连续预留导致的显存浪费和碎片，并顺带支持前缀共享与抢占。vLLM 默认 16 token 一块，DeepInfer 是 128 token 一块（代码里写死，见 `model_metadata_utils.py:134`；文档里 KV 落盘、SWA 存储也都以 128 token 为粒度）。

</details>

<details>
<summary>3. 为什么 NORMAL 引擎不像 vLLM 那样把 prefill 和 decode 混在一个 batch 里？</summary>

DP rank 之间在 MoE 层做 all-to-all，同一步所有 rank 的计算量要接近，否则都等最慢的；纯 decode 步形状稳定，可以整图捕获 CUDA Graph；线上本来就是 PD 分离部署。所以每步通过 DP all-reduce 投票选一种（`scheduler.hpp:275`）。

</details>

<details>
<summary>4. ZOS 下，第 N+1 步调度时 CPU 不知道第 N 步接受了几个草稿，KV 长度怎么算对？</summary>

CPU 先按上限准备，`prepare_zos_inputs` 给出两步 batch 的 remap，然后 GPU 上的 `compute_seqlens_k_zos` 用上一步的 `num_accepted_tokens_gpu` 修正 seqlen_k，变长验证时再用置信度修正 seqlen_q 并异步拷回 CPU 给采样用。

</details>

<details>
<summary>5. 置信度头估错了会让输出分布变吗？</summary>

不会。它只决定验证多少个草稿；接受与否由 `min(1, p/q)` 拒绝采样决定，输出分布与不投机一致。估错只影响吞吐。

</details>

<details>
<summary>6. 为什么 DeepSeek 的模型不用张量并行做注意力？</summary>

MLA 只有 1 个 KV 头（512 维 latent），TP 无法按头切分，每张卡都要存一整份 KV，显存浪费 TP 倍；每层还多两次 all-reduce。DP 注意力让每卡只存自己请求的 KV，通信只留在 MoE 层。

</details>

<details>
<summary>7. layer 27 的压缩 KV 从哪来？它的 SWA KV 呢？</summary>

压缩 KV 用 layer 20 的（`kv_source_layer_ids` 中不晚于 27 的最大值）；indexer 模块用 layer 24 的；SWA KV 是 layer 27 自己算的。可以在第 11 节的逐层配置表里核对。

</details>

<details>
<summary>8. 开 cp=8 能让单卡存下更长的上下文吗？</summary>

从代码看不能。K/V 在 CP 组内 all-gather 后每个 rank 都写完整序列的 KV，CP 摊的是 prefill 计算和激活显存。

</details>

<details>
<summary>9. 为什么 YoCo 截断保留 140 个 token，而不是 128？</summary>

decode 时最多会回退 nextn+1 个未接受的位置，SWA 窗口需要覆盖 128 + 最大回溯（默认 6），所以要求 KEEP_TOKENS ≥ 窗口 + `DEEPINFER_MAX_BACKTRACKING_TOKENS`，README 取 140。

</details>

## 代码与文档不一致处

- `scheduler.hpp:557` 注释说抢占输出最短的请求，实现先抢占输出最长的（第 09 节）。
- `hugging_face/config.py:159` 的 docstring 说 layer 20「属于 prefill 侧」，而 `v4_modules/yoco.py:19` 把 layer 20 算作第一个 consumer；以后者为准。
- README 写 balanced CP 是特性，部署文档写生产暂不用 CP：前者是能力，后者是生产配置。
- 冗余专家默认值：引擎 0，示例与 README 32。

## 参考资料

- DeepInfer 仓库与 `doc/inference_deployment_qa.md`：github.com/deepseek-ai/DeepInfer
- vLLM V1 架构博客（2025-01）：vllm.ai/blog/2025-01-27-v1-alpha-release
- vLLM 架构概览、CUDA Graph 设计、专家并行部署文档：docs.vllm.ai/en/latest/design/、…/serving/expert_parallel_deployment
- vLLM Model Runner V2（2026-03）：vllm.ai/blog/2026-03-24-mrv2
- 论文：Kwon 等《Efficient Memory Management for LLM Serving with PagedAttention》（SOSP 2023）；Yu 等《Orca》（OSDI 2022）；Leviathan 等《Fast Inference from Transformers via Speculative Decoding》（ICML 2023）

本教程为静态源码阅读结果，没有在 GPU 上运行验证。V4.1 超参数取自 HuggingFace 上的 `DeepSeek-V4.1-Flash-1017/config.json`；生产数字引自 `doc/inference_deployment_qa.md`；vLLM 的描述来自上述公开文档，版本间可能有差异。标「估算」「推断」的地方是从代码或配置推出、没有实测或直接说明的结论。第 05 节计算器与第 16 节专家权重估算只用于理解量级。

---

[← 上一篇](./17-kernels.md) · [交互版本节](https://middle-chunjie.github.io/deepinfer-visual-guide/interactive/#practice) · [回到目录 →](../index.md)
