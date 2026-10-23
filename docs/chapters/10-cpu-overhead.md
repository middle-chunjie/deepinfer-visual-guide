# 10 CPU 开销：异步调度与 CUDA Graph

| 直觉 | vLLM 的做法 | DeepInfer 的做法 |
| --- | --- | --- |
| GPU 一步 decode 只要几毫秒。如果 CPU 每步要花 1–2 ms 调度、准备输入、逐个发射几千个 kernel、再处理输出，GPU 有相当一部分时间在等 CPU。这是第 05 节的杠杆 ⑤。 | CUDA Graph（默认整图 + 分段图）消掉 kernel 发射开销；`--async-scheduling` 让 CPU 提前一步调度；Model Runner V2 把输入准备也搬到 GPU。 | 调度与输入准备用 C++；decode 整图捕获；ZOS（零开销调度）把第 N 步的后处理藏到第 N+1 步 forward 后面，长度修正在 GPU 上做，整条路径禁止 CPU-GPU 同步。 |

## CUDA Graph：把几千次发射变成一次

一个 40 层 MoE 模型的一步 decode 由几千个 kernel 组成，每个 kernel 从 CPU 发射要几微秒，加起来可能和 GPU 计算时间相当。CUDA Graph 先把整串 kernel「录」下来，之后每步只需一次重放。限制是录下来的形状固定，所以要按 batch 大小分桶，各录一张图。

#### `vLLM` 为什么要分段图

混合 batch 里有长短不一的 prefill，注意力的形状每步都在变，很多注意力后端不能整图捕获。于是 vLLM 用 torch.compile 在注意力处把模型「切开」，注意力以外的部分录成图，注意力 eager 执行。纯 decode batch 再额外用整图（`FULL_AND_PIECEWISE`）。

#### `DeepInfer` 只录 decode，整图

因为一步只做一种（第 09 节），decode 步的形状只取决于总 token 数，注意力也能一起录。按 `build_graph_batch_sizes` 分桶：1、2、4，然后每 8 一档；实际 token 数向上取到最近的桶，用 padding 序列补齐。prefill 计算量大、发射开销占比小，直接 eager。[`cuda_graph_runner.py:41`](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/worker/cuda_graph_runner.py#L41)

## 异步调度：让 CPU 和 GPU 同时干活

串行流程是 `schedule(N) → forward(N) → sample(N) → 同步 → post_process(N)`，GPU 在后处理和下一步调度期间空闲。ZOS 把第 N 步的后处理推迟到第 N+1 步 forward 发射之后做，CPU 工作被藏进 GPU 计算里。NORMAL/COMPLETION 引擎默认开启。[`arg_utils.py:48`](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/engine/arg_utils.py#L48)

![串行模式时间线](../assets/figures/10-1.svg)

![ZOS 异步模式时间线](../assets/figures/10-2.svg)

图例：青绿＝CPU：调度 / 发射 / 后处理；深蓝＝GPU：forward / sample；浅红＝GPU 空闲

<small>示意时长（非实测）：调度 1.0、发射 forward 0.4、发射 sample 0.3、后处理 1.0、GPU forward 3.0、sample 0.8，单位任意。代码结构见 `Worker.execute_model_step` 的四段：forward 当前步 → 后处理上一步 → sample 当前步 → 视情况后处理当前步。[`worker.py:89`](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/worker/worker.py#L89)</small>

## 难点：CPU 看到的状态晚一步

没有投机解码时，异步调度相对简单：每个请求每步固定多 1 个 token，下一步的长度可以提前算好。开了投机解码就麻烦了：调度第 N+1 步时，第 N 步还没验证完，CPU 不知道每个请求接受了几个草稿，也就不知道 KV 长度、下一步从哪个位置开始。vLLM 社区也提到这个组合在 V1 里很难做干净。DeepInfer 的做法：

- CPU 按「乐观假设」准备好 `seqlen_q/seqlen_k`，再由 C++ 的 `prepare_zos_inputs` 算出本步和上一步 batch 的对应关系 `remap`。[`zos.cpp`](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/csrc/engine/zos.cpp#L1)
- 在 GPU 上用上一步真实的接受数和置信度修正长度：`compute_seqlens_k_zos`、`compute_seqlens_q_by_confidence_rates`（两个 TileLang kernel）。[`worker.py:191`](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/worker/worker.py#L191)
- 上一步刚采出来的 token 直接在 GPU 上从 `prev_sampled_token_ids` 取，不回 CPU。[`zos_status.py:24`](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/worker/zos_status.py#L24)

整条路径上的 `execute_main_model`/`execute_sample` 都包在 `forbid_cuda_sync()` 里：一旦有人写了隐式同步（比如 `.item()`），直接报错。`tests/test_cuda_sync_guards.py` 专门测这个。

---

[← 上一篇](./09-scheduler.md) · [交互版本节](https://middle-chunjie.github.io/deepinfer-visual-guide/interactive/#zos) · [下一篇 →](./11-model.md)
