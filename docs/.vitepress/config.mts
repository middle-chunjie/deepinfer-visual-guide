import { defineConfig } from 'vitepress'

const INTERACTIVE = 'https://middle-chunjie.github.io/deepinfer-visual-guide/interactive/'

const chapters = (items: [string, string][]) =>
  items.map(([file, text]) => ({ text, link: `/chapters/${file}` }))

export default defineConfig({
  lang: 'zh-CN',
  title: 'DeepInfer 可视化导读',
  description: '从推理基础与 vLLM 对照出发，图解 DeepSeek 开源推理框架 DeepInfer 的设计与源码（非官方）。',
  base: '/deepinfer-visual-guide/',
  cleanUrls: true,
  lastUpdated: true,
  head: [['meta', { name: 'theme-color', content: '#2F49D8' }]],
  markdown: { lineNumbers: false },
  themeConfig: {
    nav: [
      { text: '开始阅读', link: '/chapters/01-positioning' },
      { text: '交互版', link: INTERACTIVE },
      { text: 'DeepInfer 源码', link: 'https://github.com/deepseek-ai/DeepInfer/tree/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9' },
    ],
    sidebar: [
      {
        text: '第一部分 · 全局',
        items: chapters([
          ['01-positioning', '01 定位与开源边界'],
          ['02-repo-map', '02 仓库地图'],
          ['03-processes', '03 进程模型与启动'],
        ]),
      },
      {
        text: '第二部分 · 推理基础',
        items: chapters([
          ['04-inference-basics', '04 推理到底在算什么'],
          ['05-memory-bound', '05 瓶颈：显存带宽'],
          ['06-kv-cache', '06 KV cache 与 PagedAttention'],
          ['07-vllm', '07 vLLM 是怎么设计的'],
        ]),
      },
      {
        text: '第三部分 · 拆解 DeepInfer',
        items: chapters([
          ['08-request-journey', '08 一个请求的旅程'],
          ['09-scheduler', '09 调度器'],
          ['10-cpu-overhead', '10 CPU 开销：异步调度与 CUDA Graph'],
          ['11-model', '11 V4.1-Flash 模型结构'],
          ['12-attention-kv', '12 注意力与 KV 存储'],
          ['13-prefix-yoco', '13 前缀缓存、YoCo 与有界重放'],
          ['14-dspark', '14 DSpark 投机解码'],
          ['15-sampling', '15 采样、确定性、结构化输出'],
          ['16-parallelism', '16 并行：DP / EP / CP / EPLB'],
          ['17-kernels', '17 Kernel 与双平台'],
          ['18-practice', '18 动手与自测'],
        ]),
      },
      {
        text: '其他形式',
        items: [{ text: '交互版（单页，可离线）', link: INTERACTIVE }],
      },
    ],
    outline: { level: [2, 3], label: '本页目录' },
    socialLinks: [{ icon: 'github', link: 'https://github.com/middle-chunjie/deepinfer-visual-guide' }],
    editLink: {
      pattern: 'https://github.com/middle-chunjie/deepinfer-visual-guide/edit/main/docs/:path',
      text: '在 GitHub 上修改此页',
    },
    footer: {
      message: '非官方学习资料，与 DeepSeek 无隶属关系。正文 CC BY 4.0，脚本 MIT。',
      copyright: 'DeepInfer 源码版权归 DeepSeek-AI（Apache-2.0）',
    },
    docFooter: { prev: '上一篇', next: '下一篇' },
    lastUpdated: { text: '最后更新' },
    returnToTopLabel: '回到顶部',
    sidebarMenuLabel: '目录',
    darkModeSwitchLabel: '深色模式',
    search: {
      provider: 'local',
      options: {
        miniSearch: {
          options: {
            // 中文按「单字 + 相邻两字」切分，英文与代码标识符按词切分；函数会被序列化到浏览器端，必须自包含
            tokenize: (text: string) => {
              const out: string[] = []
              const re = /[㐀-鿿豈-﫿]+|[A-Za-z0-9_.\-]+/g
              let m: RegExpExecArray | null
              while ((m = re.exec(text))) {
                const w = m[0]
                if (/[㐀-鿿豈-﫿]/.test(w)) {
                  for (let i = 0; i < w.length; i++) {
                    out.push(w[i])
                    if (i + 1 < w.length) out.push(w.slice(i, i + 2))
                  }
                } else {
                  out.push(w)
                }
              }
              return out
            },
          },
          searchOptions: { combineWith: 'AND', fuzzy: 0.1, prefix: true, boost: { title: 4, text: 2, titles: 1 } },
        },
        translations: {
          button: { buttonText: '搜索', buttonAriaLabel: '搜索' },
          modal: {
            noResultsText: '没有找到结果',
            resetButtonTitle: '清除',
            footer: { selectText: '选择', navigateText: '切换', closeText: '关闭' },
          },
        },
      },
    },
  },
})
