# DeepInfer 可视化导读

> 非官方学习资料，与 DeepSeek 无隶属关系。

[DeepInfer](https://github.com/deepseek-ai/DeepInfer) 是 DeepSeek 开源的生产级推理框架，用来服务 DeepSeek-V4.1-Flash。这份导读面向**没读过推理框架**的读者：先讲清「大模型推理慢在哪」和「vLLM 是怎么做的」，再逐个拆解 DeepInfer 的子系统，每节按「直觉 → vLLM 的做法 → DeepInfer 的做法 → 源码」展开。

- 📖 **网页版**：<https://middle-chunjie.github.io/deepinfer-visual-guide/>
- 🧪 **交互版**（单个 HTML，可下载离线看）：<https://middle-chunjie.github.io/deepinfer-visual-guide/interactive/>
- 📝 **Markdown 原文**：[`docs/chapters/`](docs/chapters/)，在 GitHub 上可以直接读

源码结论对照 DeepInfer 提交 [`9bd9cf2`](https://github.com/deepseek-ai/DeepInfer/tree/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9)（2026-10-21），每处都链接到具体行。

## 目录

**第一部分 · 全局**

1. [定位与开源边界](docs/chapters/01-positioning.md)：完整服务是 E/P/D 分离的集群，仓库开源的是哪一块
2. [仓库地图](docs/chapters/02-repo-map.md)：从入口到 kernel 的分层
3. [进程模型与启动](docs/chapters/03-processes.md)：DP rank、CP follower、初始化顺序

**第二部分 · 推理基础**

4. [推理到底在算什么](docs/chapters/04-inference-basics.md)：prefill / decode、术语表
5. [瓶颈：显存带宽](docs/chapters/05-memory-bound.md)：一个公式和五条加速杠杆（全篇的核心直觉）
6. [KV cache 与 PagedAttention](docs/chapters/06-kv-cache.md)
7. [vLLM 是怎么设计的](docs/chapters/07-vllm.md)：连续批、V1 进程结构、关键设计及理由、与 DeepInfer 的总对比

**第三部分 · 拆解 DeepInfer**

8. [一个请求的旅程](docs/chapters/08-request-journey.md)：10 步调用链，每步对照 vLLM
9. [调度器](docs/chapters/09-scheduler.md)：为什么不混批、DP 组投票、抢占、乐观推进
10. [CPU 开销：异步调度与 CUDA Graph](docs/chapters/10-cpu-overhead.md)：ZOS 与整图捕获
11. [V4.1-Flash 模型结构](docs/chapters/11-model.md)：mHC、43 个注意力模块逐层配置、MoE、Engram
12. [注意力与 KV 存储](docs/chapters/12-attention-kv.md)：每 token KV 从 320 KB 到约 0.9 KB
13. [前缀缓存、YoCo 与有界重放](docs/chapters/13-prefix-yoco.md)
14. [DSpark 投机解码](docs/chapters/14-dspark.md)：并行草稿、置信度变长验证、拒绝采样
15. [采样、确定性、结构化输出](docs/chapters/15-sampling.md)
16. [并行：DP / EP / CP / EPLB](docs/chapters/16-parallelism.md)：为什么不用 TP、为什么 decode 要 64 卡以上
17. [Kernel 与双平台](docs/chapters/17-kernels.md)
18. [动手与自测](docs/chapters/18-practice.md)：运行命令、环境变量、推荐阅读顺序、9 道自测题

## 三种形式怎么选

| 形式 | 适合 | 位置 |
| --- | --- | --- |
| 网页版 | 系统阅读、站内搜索 | VitePress 构建，GitHub Pages 托管 |
| Markdown | 在 GitHub 上直接读、提 PR 勘误 | `docs/chapters/*.md`，配图在 `docs/assets/figures/` |
| 交互版 | 拖参数建立直觉（显存带宽计算器、逐层浏览、变长验证、CP 切分、ZOS 时间线） | `docs/public/interactive/index.html`，单文件，无外部依赖（字体除外） |

## 本地预览

```bash
npm ci
npm run dev        # http://localhost:5173/deepinfer-visual-guide/
npm run build      # 输出到 docs/.vitepress/dist
```

校验源码引用（文件存在、行号不越界）：

```bash
git clone https://github.com/deepseek-ai/DeepInfer /tmp/DeepInfer
git -C /tmp/DeepInfer checkout 9bd9cf243fc53ef9afbf8f04d391d989117d6ab9
python3 scripts/check_refs.py /tmp/DeepInfer
```

`scripts/convert_interactive.py` 是从交互版生成 Markdown 初稿和 SVG 配图的工具，Markdown 之后经过人工修订，不要直接整体重新生成覆盖。

## 和其他社区资料的关系

这份导读偏「建立直觉」，深度上推荐继续读：

- [cluster1900/deepinfer-reader](https://github.com/cluster1900/deepinfer-reader)：逐行对照源码的中英文书，并从零实现教学引擎 `minidi`
- [algebra-MCX/mini-DeepInfer](https://github.com/algebra-MCX/mini-DeepInfer)、[hoppespy/mini-DeepInfer](https://github.com/hoppespy/mini-DeepInfer)：可运行的精简重写
- [geekchow/deepseek-v41-flash-inference-guide](https://github.com/geekchow/deepseek-v41-flash-inference-guide)：整套部署系统（deepseek-recipe、KV 存储、负载均衡）

## 准确性与勘误

- 基于静态源码阅读，**没有**在 SM100 / 昇腾硬件上运行验证。
- V4.1 超参数取自 HuggingFace `DeepSeek-V4.1-Flash-1017/config.json`；生产数字引自官方 `doc/inference_deployment_qa.md`；vLLM 的描述来自其公开文档，版本之间可能有差异。
- 标「估算」「推断」的地方（例如 V4.1 每 token KV 约 0.9 KB、EP8/EP64 每卡专家权重）是从配置和代码推算的，没有实测。
- 发现错误或过时内容，欢迎提 Issue 或 PR。

## 许可

- 正文与配图：[CC BY 4.0](LICENSE)
- 脚本（`scripts/`）：[MIT](LICENSE-CODE)
- 文中引用的 DeepInfer 源码片段版权归 DeepSeek-AI，遵循 Apache-2.0
