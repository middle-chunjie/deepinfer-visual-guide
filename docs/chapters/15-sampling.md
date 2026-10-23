# 15 采样、确定性、结构化输出

| 直觉 | vLLM 的做法 | DeepInfer 的做法 |
| --- | --- | --- |
| 模型每步输出词表上 12.9 万个分数（logits），采样就是按温度、top-p、top-k 把它变成概率再抽一个 token。看似简单，但有两个工程难题：同样的输入为什么结果会变？怎么强制输出合法 JSON？ | 采样器参数更全（含 min-p、各种 penalty、logits processor）；结构化输出用 xgrammar / guidance 生成 token bitmask。 | 参数精简到线上需要的；无状态随机数 + batch-invariant 模式实现逐位可复现；结构化输出同样用 xgrammar，但异步编译并与投机解码逐草稿配合。 |

## 为什么同样的输入会得到不同的输出

浮点加法不满足结合律：(a+b)+c 和 a+(b+c) 的结果可能差最后一位。GPU kernel 为了快，会根据 batch 大小选不同的切分方式和归约顺序。于是同一个请求，和不同的其他请求拼在一起时，算出的 logits 有微小差异，采样时偶尔就会选到不同的 token，之后整段输出都不一样。2025 年 Thinking Machines 的文章让这个问题广为人知。

#### batch-invariant 推理

`DEEPINFER_BATCH_INVARIANT=1` 把 DeepGEMM、注意力、通信切到「结果与 batch 组成无关」的算法（仅 NVIDIA），代价是一点性能。离线示例在 NVIDIA 上默认打开。[`runtime_config.py:63`](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/worker/runtime_config.py#L63)

#### 确定性采样

随机数不来自全局 RNG，而是 `SplitMix64(seed + f(3·pos + case))`：case 0/1/2 分别是普通采样、残差重采样、接受判定。同一序列同一位置永远拿到同一个随机数，与 batch 里还有谁无关。要复现必须显式传 `seq_seed`（默认随机），并关掉变长验证。

## 采样器参数

- 支持：temperature、top-p、top-k、top_logprobs、token 黑名单、`first_n_token_no_eos`。**没有** min-p 和 repetition/frequency/presence penalty。[`sampling_params.py`](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/sampling_params.py#L49)
- 温度 0 被钳到 1e-5，没有单独的 argmax 分支。
- top-p/top-k 用直方图找阈值，不做全词表排序；`DEEPINFER_SAMPLER_FORCE_TOPK=4096` 时只在前 4096 个里采（用 DeepSelect 做大词表 top-k）。

## 结构化输出的流水

1. `SamplingParams(so_args={"schema_type": "json"|"regex"|"ebnf", "schema": ...})`。
2. 请求先进线程池用 xgrammar 编译成语法自动机（超时 20 s），编译完才入调度队列，不阻塞引擎主循环。[`so_matcher_builder.py:58`](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/structure_output/so_matcher_builder.py#L58)
3. 每步在 CPU 上推进自动机、填「哪些 token 合法」的 bitmask，与 GPU forward 重叠；TileLang kernel 把非法位置的 logits 置为 −∞。
4. 和投机解码配合：逐个把草稿喂给自动机，为每个验证位置生成对应的 mask，遇到第一个非法草稿就停，最后按实际接受数回滚（`max_rollback_tokens = nextn + 2`）。非法草稿在被 mask 的目标分布里概率为 0，自然会被拒绝。[`sampler.py:361`](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/layers/sampler.py#L361)

---

[← 上一篇](./14-dspark.md) · [交互版本节](https://middle-chunjie.github.io/deepinfer-visual-guide/interactive/#sample) · [下一篇 →](./16-parallelism.md)
