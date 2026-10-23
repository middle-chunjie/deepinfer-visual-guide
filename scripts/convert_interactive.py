"""把交互版页面（渲染后的 DOM）拆成 Markdown 章节 + 独立 SVG 配图。

用法：
    # 1. 用无头 Chrome 渲染交互页，得到包含 JS 生成图形的 DOM
    chrome --headless=new --dump-dom file://$PWD/docs/public/interactive/index.html > /tmp/dom.html
    # 2. 生成 docs/chapters/*.md 与 docs/assets/figures/*.svg
    python3 scripts/convert_interactive.py /tmp/dom.html

交互页是唯一的「源」时用它重新生成；手工改过 Markdown 之后不要再整体覆盖。
依赖：beautifulsoup4。
"""

from __future__ import annotations

import json
import math
import re
import string
import subprocess
import sys
from pathlib import Path

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

ROOT = Path(__file__).resolve().parent.parent
CHAP_DIR = ROOT / "docs" / "chapters"
FIG_DIR = ROOT / "docs" / "assets" / "figures"
INTERACTIVE = ROOT / "docs" / "public" / "interactive" / "index.html"
REPO = "https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/"
SITE_INTERACTIVE = "https://middle-chunjie.github.io/deepinfer-visual-guide/interactive/"

LIGHT = {
    "--bg": "#EEF1F4", "--surface": "#FFFFFF", "--sunk": "#E3E8ED", "--ink": "#111A23", "--muted": "#56636F",
    "--line": "#CDD5DD", "--accent": "#2F49D8", "--accent-soft": "#DCE2FB", "--vllm": "#C4521B", "--vllm-soft": "#F8E3D7",
    "--swa": "#0B8577", "--c2": "#3D72CF", "--c1": "#7E46CF", "--mtp": "#C47A0E", "--bad": "#B8402A", "--good": "#2E8A47",
    "--gpu": "#2F49D8", "--cpu": "#0B8577", "--idle": "#B8402A",
}
SVG_STYLE = (
    "<style>text{font-family:'Noto Sans SC','PingFang SC','Microsoft YaHei','IBM Plex Sans',sans-serif}"
    "text:not([fill]){fill:#111A23}.m{font-family:'JetBrains Mono',Menlo,Consolas,monospace}</style>"
)

# 章节 id -> 文件名（保持与交互版锚点一致）
CHAPTERS = [
    ("pos", "01-positioning"), ("map", "02-repo-map"), ("boot", "03-processes"),
    ("basics", "04-inference-basics"), ("roof", "05-memory-bound"), ("kvbasic", "06-kv-cache"), ("vllm", "07-vllm"),
    ("journey", "08-request-journey"), ("sched", "09-scheduler"), ("zos", "10-cpu-overhead"), ("model", "11-model"),
    ("attn", "12-attention-kv"), ("yoco", "13-prefix-yoco"), ("spec", "14-dspark"), ("sample", "15-sampling"),
    ("par", "16-parallelism"), ("hw", "17-kernels"), ("practice", "18-practice"),
]


def resolve_vars(s: str) -> str:
    return re.sub(r"var\((--[a-z0-9-]+)\)", lambda m: LIGHT[m.group(1)], s)


def standalone_svg(raw: str) -> str:
    """把页面内嵌 SVG 变成可独立显示的文件：解析颜色变量、白底、补字体。"""
    m = re.match(r"<svg([^>]*)>(.*)</svg>\s*$", raw, re.S)
    attrs, body = m.group(1), m.group(2)
    vb = re.search(r'viewBox="([^"]+)"', attrs).group(1)
    _, _, w, h = (float(x) for x in vb.split())
    label = re.search(r'aria-label="([^"]*)"', attrs)
    title = f"<title>{label.group(1)}</title>" if label else ""
    body = resolve_vars(body)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vb}" width="{w:g}" height="{h:g}" color="#111A23">'
        f'{title}{SVG_STYLE}<rect x="0" y="0" width="{w:g}" height="{h:g}" rx="10" fill="#FFFFFF"/>{body}</svg>\n'
    )


def raw_svg(dom: str, *, label: str | None = None, svg_id: str | None = None) -> str:
    key = f'aria-label="{label.replace("&", "&amp;")}"' if label else f'id="{svg_id}"'
    start = dom.index(key)
    start = dom.rindex("<svg", 0, start)
    end = dom.index("</svg>", start) + len("</svg>")
    return dom[start:end]


# ---------------- inline / block conversion ----------------

def ref_url(r: str) -> str:
    i = r.rindex(":")
    return f"{REPO}{r[:i]}#L{r[i + 1:]}"


def esc_text(s: str) -> str:
    return s.replace("<", "&lt;").replace(">", "&gt;").replace("*", "\\*").replace("{{", "{ {")


def inline(node) -> str:
    out = []
    for c in node.children:
        if isinstance(c, Comment):
            continue
        if isinstance(c, NavigableString):
            out.append(esc_text(re.sub(r"\s+", " ", str(c))))
            continue
        if not isinstance(c, Tag):
            continue
        name = c.name
        if name == "code":
            t = c.get_text()
            out.append(f"``{t}``" if "`" in t else f"`{t}`")
        elif name in ("b", "strong"):
            out.append(bold(inline(c).strip()))
        elif name in ("i", "em"):
            t = inline(c).strip()
            if t:
                out.append(f"*{t}*")
        elif name == "a":
            if "ref" in (c.get("class") or []):
                r = c.get("data-ref") or ""
                out.append(f"[`{c.get_text()}`]({ref_url(r)})")
            else:
                href = c.get("href", "")
                if href.startswith("#"):
                    out.append(inline(c))
                else:
                    out.append(f"[{inline(c)}]({href})")
        elif name == "sup":
            out.append(f"<sup>{c.get_text()}</sup>")
        elif name == "br":
            out.append("<br>")
        elif name == "span" and "tag" in (c.get("class") or []):
            out.append(f"`{c.get_text()}`")
        elif name == "span" and "progress" in (c.get("class") or []):
            continue
        else:
            out.append(inline(c))
    return "".join(out)


PUNCT = set("，。、；：？！「」『』（）《》—…·【】") | set(string.punctuation)


def bold(t: str) -> str:
    """CJK 标点紧邻 ** 时 CommonMark 不认加粗，这种情况改用 <strong>。"""
    if not t:
        return ""
    if t[0] in PUNCT or t[-1] in PUNCT:
        return f"<strong>{t}</strong>"
    return f"**{t}**"


COLOR_NAMES = {
    "--swa": "青绿", "--c2": "蓝", "--c1": "紫", "--mtp": "橙", "--accent": "深蓝", "--gpu": "深蓝",
    "--cpu": "青绿", "--idle": "浅红", "--vllm": "橙红", "--bad": "红", "--good": "绿",
}


def legend_md(leg: Tag) -> str:
    items = []
    for sp in leg.find_all("span", recursive=False):
        i = sp.find("i")
        color = ""
        if i is not None:
            st = i.get("style", "")
            m = re.search(r"background:var\((--[a-z0-9-]+)\)", st)
            if m:
                color = COLOR_NAMES.get(m.group(1), "")
            elif "dashed" in st:
                color = "虚线框"
            elif "solid" in st:
                color = "实线框"
            i.extract()
        text = clean(inline(sp))
        if text:
            items.append(f"{color}＝{text}" if color else text)
    return "图例：" + "；".join(items)


def clean(s: str) -> str:
    s = re.sub(r"[ \t]+", " ", s)
    return re.sub(r" *\n *", "\n", s).strip()


def table_md(tbl: Tag) -> str:
    rows = tbl.find_all("tr")
    lines = []
    for i, tr in enumerate(rows):
        cells = [clean(inline(td)).replace("|", "\\|").replace("\n", " ") for td in tr.find_all(["th", "td"])]
        lines.append("| " + " | ".join(cells) + " |")
        if i == 0:
            lines.append("|" + "|".join(" --- " for _ in cells) + "|")
    return "\n".join(lines)


def list_md(lst: Tag, ordered: bool) -> str:
    out = []
    for i, li in enumerate(lst.find_all("li", recursive=False), 1):
        bullet = f"{i}." if ordered else "-"
        out.append(f"{bullet} {clean(inline(li))}")
    return "\n".join(out)


def _wrap(a: Tag) -> Tag:
    w = BeautifulSoup("<span></span>", "html.parser").span
    w.append(a.__copy__())
    return w


class Ctx:
    def __init__(self, dom: str, chap: str):
        self.dom = dom
        self.chap = chap
        self.nfig = 0

    def save_svg(self, raw: str, alt: str) -> str:
        self.nfig += 1
        name = f"{self.chap}-{self.nfig}.svg"
        (FIG_DIR / name).write_text(standalone_svg(raw), encoding="utf-8")
        return f"![{alt}](../assets/figures/{name})"


def figure_md(fig: Tag, ctx: Ctx) -> str:
    parts = []
    for svg in fig.find_all("svg"):
        label = svg.get("aria-label") or ""
        sid = svg.get("id")
        raw = raw_svg(ctx.dom, svg_id=sid) if sid else raw_svg(ctx.dom, label=label)
        parts.append(ctx.save_svg(raw, label or sid))
    leg = fig.find(class_="legend")
    if leg:
        parts.append(legend_md(leg))
    cap = fig.find("figcaption")
    if cap:
        parts.append(f"<small>{clean(inline(cap))}</small>")
    return "\n\n".join(parts)


def callout_md(div: Tag) -> str:
    cls = div.get("class") or []
    kind = "TIP" if "key" in cls else "WARNING" if "warn" in cls else "NOTE"
    body = clean(inline(div))
    return f"> [!{kind}]\n> " + body.replace("\n", "\n> ")


def card_md(card: Tag) -> str:
    out = []
    h4 = card.find("h4")
    if h4:
        out.append(f"#### {clean(inline(h4))}")
    for el in card.children:
        if not isinstance(el, Tag) or el.name == "h4":
            continue
        if el.name == "p":
            out.append(clean(inline(el)))
        elif el.name == "ul":
            out.append(list_md(el, False))
        elif el.name == "ol":
            out.append(list_md(el, True))
        elif "refs" in (el.get("class") or []):
            out.append("源码：" + " · ".join(clean(inline(_wrap(a))) for a in el.find_all("a")))
    if not h4:
        out.append(clean(inline(card)))
    return "\n\n".join(o for o in out if o)


def trio_md(div: Tag) -> str:
    cards = div.find_all(class_="card", recursive=False)
    head = "| 直觉 | vLLM 的做法 | DeepInfer 的做法 |\n| --- | --- | --- |"
    cells = []
    for c in cards:
        lab = c.find(class_="lab")
        if lab:
            lab.extract()
        cells.append(clean(inline(c)).replace("|", "\\|").replace("\n", " "))
    return head + "\n| " + " | ".join(cells) + " |"


def gloss_md(dl: Tag) -> str:
    rows = ["| 术语 | 含义 |", "| --- | --- |"]
    for dt in dl.find_all("dt"):
        dd = dt.find_next_sibling("dd")
        rows.append(f"| `{dt.get_text()}` | {clean(inline(dd))} |")
    return "\n".join(rows)


def details_md(d: Tag) -> str:
    s = d.find("summary")
    q = clean(inline(s))
    s.extract()
    body = "\n\n".join(clean(inline(p)) for p in d.find_all("p"))
    return f"<details>\n<summary>{q}</summary>\n\n{body}\n\n</details>"


def checklist_md(ul: Tag) -> str:
    out = []
    for li in ul.find_all("li"):
        lab = li.find("label")
        b = lab.find("b")
        title = clean(inline(b))
        b.extract()
        desc = clean(inline(lab)).strip().removeprefix("<br>").strip()
        out.append(f"- [ ] **{title}**：{desc}")
    return "\n".join(out)


SPECIAL = {}  # 交互部件 id -> callable(figure, ctx) -> markdown


def block(el: Tag, ctx: Ctx) -> str | None:
    cls = el.get("class") or []
    if el.name == "div" and "chap-head" in cls:
        return None
    if el.name == "p" and "why" in cls:
        return f"> {clean(inline(el))}"
    if el.name == "p":
        return clean(inline(el))
    if el.name == "h3":
        return f"## {clean(inline(el))}"
    if el.name == "ul" and "check" in cls:
        return checklist_md(el)
    if el.name == "ul":
        return list_md(el, False)
    if el.name == "ol":
        return list_md(el, True)
    if el.name == "pre":
        return "```text\n" + el.get_text().rstrip() + "\n```"
    if el.name == "dl":
        return gloss_md(el)
    if el.name == "details":
        return details_md(el)
    if el.name == "figure":
        first = el.find(id=True)
        hook = SPECIAL.get(first.get("id") if first else None)
        if hook:
            return hook(el, ctx)
        return figure_md(el, ctx)
    if el.name == "div" and "trio" in cls:
        return trio_md(el)
    if el.name == "div" and "grid2" in cls:
        return "\n\n".join(card_md(c) for c in el.find_all(class_="card", recursive=False))
    if el.name == "div" and "tbl" in cls:
        return table_md(el.find("table"))
    if el.name == "div" and "callout" in cls:
        return callout_md(el)
    if el.name == "div" and "refs" in cls:
        return "源码：" + " · ".join(clean(inline(a.parent if False else _wrap(a))) for a in el.find_all("a"))
    raise ValueError(f"unhandled block {el.name} {cls} in {ctx.chap}")


# ---------------- 交互部件的静态替代 ----------------

def node_eval_array(js_src: str, var: str) -> list:
    start = js_src.index(f"var {var}=[")
    end = js_src.index("];", start) + 1
    code = js_src[start:end] + f";process.stdout.write(JSON.stringify({var}))"
    return json.loads(subprocess.run(["node", "-e", code], capture_output=True, text=True, check=True).stdout)


def journey_md(fig: Tag, ctx: Ctx) -> str:
    steps = node_eval_array(ctx.dom, "J")
    rows = ["| # | 环节 | 做了什么 | 源码 | vLLM 里对应 |", "| --- | --- | --- | --- | --- |"]
    for i, s in enumerate(steps, 1):
        d = clean(inline(BeautifulSoup(s["d"], "html.parser")))
        v = clean(inline(BeautifulSoup(s["v"], "html.parser")))
        refs = "<br>".join(f"[`{r}`]({ref_url(r)})" for r in s["r"])
        name = f"`{s['n']}`" + ("（C++）" if s["cpp"] else "")
        d, v = d.replace("|", "\\|"), v.replace("|", "\\|")
        rows.append(f"| {i} | {name}<br>**{s['t']}** | {d} | {refs} | {v} |")
    link = f"[▶ 在交互版中逐步点击浏览]({SITE_INTERACTIVE}#journey)"
    return link + "\n\n" + "\n".join(rows)


BS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
KVF = {"gqa": (327680, 0, "GQA（Llama-3-70B 式）"), "mla": (70272, 0, "MLA（DeepSeek-V3 式）"), "v41": (890, 3.0e6, "V4.1-Flash 式")}


def roof(B, ctx_len, kvf, k):
    b, s, _ = KVF[kvf]
    kvb = B * (ctx_len * b + s)
    tw, tk, tc = 70e9 / 8e12, kvb / 8e12, 2 * 70e9 * B * k / 4.5e15
    t = max(tw + tk, tc)
    return t, B * (1 if k == 1 else 3.2) / t, 70e9 + kvb <= 180e9, (tw + tk) >= tc


def roof_md(fig: Tag, ctx: Ctx) -> str:
    out = [f"[▶ 在交互版中拖动参数]({SITE_INTERACTIVE}#roof)"]
    out.append("下图是交互版默认参数（8 个并发、32K 上下文、GQA、不开投机）时的耗时分解和吞吐曲线：")
    out.append(ctx.save_svg(raw_svg(ctx.dom, svg_id="rBars"), "一步耗时的分解"))
    out.append(ctx.save_svg(raw_svg(ctx.dom, svg_id="rCurve"), "吞吐随并发请求数的变化"))
    out.append("同一个玩具模型在 32K 上下文下的几组对照（单卡吞吐 token/s；✗ 表示 KV 装不下）：")
    rows = ["| 并发请求数 | " + " | ".join(v[2] for v in KVF.values()) + " | V4.1 式 + 投机解码 |", "| --- | --- | --- | --- | --- |"]
    for B in [1, 8, 16, 64, 256]:
        cells = []
        for key in KVF:
            t, tps, fit, _ = roof(B, 32768, key, 1)
            cells.append(f"{tps:,.0f}（{t * 1e3:.1f} ms）" if fit else "✗")
        t, tps, fit, _ = roof(B, 32768, "v41", 6)
        cells.append(f"{tps:,.0f}（{t * 1e3:.1f} ms）" if fit else "✗")
        rows.append(f"| {B} | " + " | ".join(cells) + " |")
    out.append("\n".join(rows))
    cap = fig.find("figcaption")
    out.append(f"<small>{clean(inline(cap))}</small>")
    return "\n\n".join(out)


def layers_md(fig: Tag, ctx: Ctx) -> str:
    ratios = [0, 0] + [2] * 18 + [1] * 20
    kv_src, idx_src = [2, 8, 14, 20], [2, 8, 14, 20, 24, 28, 32, 36]

    def src(arr, l):
        r = None
        for s in arr:
            if s <= l:
                r = s
        return r

    rows = ["| layer | 注意力 | 压缩 | YoCo 角色 | 压缩 KV 来自 | indexer 来自 | 其他 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for i, r in enumerate(ratios):
        att = "纯 SWA" if r == 0 else "SWA + 稀疏 top-512"
        comp = "—" if r == 0 else ("C2（2→1）" if r == 2 else "C1")
        role = "prefill 侧" if i < 20 else ("第一个 consumer" if i == 20 else "consumer")
        ks = src(kv_src, i) if r else None
        is_ = src(idx_src, i) if r else None
        kvc = "—" if ks is None else ("**本层**" if ks == i else f"layer {ks}")
        idc = "—" if is_ is None else ("**本层**" if is_ == i else f"layer {is_}")
        other = []
        if i in (1, 14):
            other.append("Engram")
        if i in (37, 38, 39):
            other.append("特征送 DSpark")
        if i == 20:
            other.append("候选块源")
        rows.append(f"| {i} | {att} | {comp} | {role} | {kvc} | {idc} | {'、'.join(other)} |")
    rows.append("| 40–42 | 纯 SWA（DSpark 草稿层） | — | — | — | — | 128 专家 top-3 小 MoE |")
    head = f"[▶ 在交互版中逐层点击查看]({SITE_INTERACTIVE}#model)"
    leg = fig.find(class_="legend")
    return head + "\n\n<details>\n<summary>展开 43 个注意力模块的逐层配置表</summary>\n\n" + "\n".join(rows) + "\n\n</details>"


def varlen_md(fig: Tag, ctx: Ctx) -> str:
    conf = [[0.95, 0.9, 0.85, 0.7, 0.6], [0.9, 0.6, 0.5, 0.4, 0.3], [0.98, 0.95, 0.9, 0.88, 0.8], [0.7, 0.5, 0.4, 0.3, 0.2]]
    graph = [1, 2, 4, 8, 16, 24]
    q = []
    for s, row in enumerate(conf):
        a = 1.0
        for j, c in enumerate(row):
            a *= c
            q.append((a, s, j))
    q.sort(key=lambda x: (-x[0], x[1]))
    exp, bsz, best, best_b = 4.0, 4, 0.0, 4
    for e in [None] + q:
        if e:
            exp += e[0]
            bsz += 1
        g = next((x for x in graph if x >= bsz), None)
        if g is None:
            continue
        th = exp / (6 + 0.4 * g + 1)
        if th > best:
            best, best_b = th, bsz
    picked = {(s, j) for _, s, j in q[: best_b - 4]}
    rows = ["| 请求 | 草稿 1 | 草稿 2 | 草稿 3 | 草稿 4 | 草稿 5 | 本步验证 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for s, row in enumerate(conf):
        a, cells = 1.0, []
        for j, c in enumerate(row):
            a *= c
            t = f"{a:.2f}（conf {c:.2f}）"
            cells.append(f"**{t}** ✓" if (s, j) in picked else t)
        n = 1 + sum(1 for j in range(5) if (s, j) in picked)
        rows.append(f"| {s} | " + " | ".join(cells) + f" | {n} 个 token |")
    out = [f"[▶ 在交互版中换一组置信度试试]({SITE_INTERACTIVE}#spec)",
           f"下表是交互版的默认例子：4 个请求、每个 5 个草稿。格子里是累乘接受率（括号里是置信度头原始输出），✓ 表示被选中验证。按示意耗时曲线，最优是本步共验证 **{best_b}** 个 token（固定验证需要 24 个）。",
           "\n".join(rows),
           ctx.save_svg(raw_svg(ctx.dom, svg_id="vchart"), "每个 total_q 对应的估算吞吐")]
    cap = fig.find("figcaption")
    out.append(f"<small>{clean(inline(cap))}</small>")
    return "\n\n".join(out)


def zos_svg(mode: str) -> str:
    D = dict(S=1.0, L=0.4, Ls=0.3, P=1.0, F=3.0, Sa=0.8)
    cpu = gpu = 0.0
    C, G, prev_s = [], [], None
    for k in range(6):
        s0 = cpu
        C.append((s0, s0 + D["S"], f"调度{k}"))
        lf = s0 + D["S"] + D["L"]
        C.append((s0 + D["S"], lf, ""))
        f0 = max(lf, gpu)
        f1 = f0 + D["F"]
        G.append((f0, f1, f"forward {k}", 0))
        if mode == "serial":
            ls = lf + D["Ls"]
            C.append((lf, ls, ""))
            s1 = max(f1, ls)
            s2 = s1 + D["Sa"]
            G.append((s1, s2, "", 1))
            gpu = s2
            p0 = max(ls, s2)
            C.append((p0, p0 + D["P"], f"后处理{k}"))
            cpu = p0 + D["P"]
        else:
            c = lf
            if prev_s is not None:
                p0 = max(c, prev_s)
                C.append((p0, p0 + D["P"], f"后处理{k - 1}"))
                c = p0 + D["P"]
            ls = c + D["Ls"]
            C.append((c, ls, ""))
            s1 = max(f1, ls)
            s2 = s1 + D["Sa"]
            G.append((s1, s2, "", 1))
            gpu = prev_s = s2
            cpu = ls
    T, W, x0 = 20, 860, 60
    sc = (W - x0 - 10) / T
    h = ['<text x="8" y="44" font-size="12">CPU</text><text x="8" y="104" font-size="12">GPU</text>']
    h.append(f'<line x1="{x0}" y1="130" x2="{W - 10}" y2="130" stroke="#111A23" stroke-opacity=".3"/>')
    for t in range(0, T + 1, 2):
        h.append(f'<text x="{x0 + t * sc:.1f}" y="146" font-size="10" text-anchor="middle" class="m" opacity=".7">{t}</text>')
    busy = last = 0.0
    for a, b, l, s in sorted(G):
        if a >= T:
            continue
        b = min(b, T)
        if a > last + 1e-9:
            h.append(f'<rect x="{x0 + last * sc:.1f}" y="84" width="{(a - last) * sc:.1f}" height="30" fill="#B8402A" opacity=".28"/>')
        busy += b - a
        last = max(last, b)
        h.append(f'<rect x="{x0 + a * sc:.1f}" y="84" width="{max(1, (b - a) * sc - 1):.1f}" height="30" rx="3" fill="#2F49D8" opacity="{.65 if s else 1}"/>')
        if not s and (b - a) * sc > 60:
            h.append(f'<text x="{x0 + (a + b) / 2 * sc:.1f}" y="103" font-size="10.5" text-anchor="middle" fill="#fff">{l}</text>')
    for a, b, l in C:
        if a >= T:
            continue
        b = min(b, T)
        h.append(f'<rect x="{x0 + a * sc:.1f}" y="24" width="{max(1, (b - a) * sc - 1):.1f}" height="30" rx="3" fill="#0B8577" opacity="{1 if l else .55}"/>')
        if l and (b - a) * sc > 40:
            h.append(f'<text x="{x0 + (a + b) / 2 * sc:.1f}" y="43" font-size="10" text-anchor="middle" fill="#fff">{l}</text>')
    title = "串行" if mode == "serial" else "ZOS 异步"
    h.append(f'<text x="{W - 10}" y="14" font-size="11" text-anchor="end">{title}：GPU 忙碌占比 ≈ {busy / T * 100:.0f}%</text>')
    return f'<svg viewBox="0 0 {W} 150" aria-label="{title}模式下 CPU 与 GPU 时间线">' + "".join(h) + "</svg>"


def zos_md(fig: Tag, ctx: Ctx) -> str:
    out = [ctx.save_svg(zos_svg("serial"), "串行模式时间线"), ctx.save_svg(zos_svg("zos"), "ZOS 异步模式时间线")]
    out.append(legend_md(fig.find(class_="legend")))
    out.append(f"<small>{clean(inline(fig.find('figcaption')))}</small>")
    return "\n\n".join(out)


def cp_svg(cp: int) -> tuple[str, list[float], list[float]]:
    L = 64
    ps = math.ceil(L / (2 * cp))
    owner = [0] * L
    for r in range(cp):
        for blk in (r, 2 * cp - 1 - r):
            for j in range(ps):
                t = blk * ps + j
                if t < L:
                    owner[t] = r
    naive, zig = [0] * cp, [0] * cp
    for t in range(L):
        zig[owner[t]] += t + 1
        naive[min(cp - 1, t // (L // cp))] += t + 1
    pal = ["#3D72CF", "#0B8577", "#C47A0E", "#7E46CF", "#B8402A", "#2E8A47", "#2F49D8", "#56636F"]
    W, x0, cw = 860, 20, (860 - 40) / L
    h = [f'<text x="{x0}" y="18" font-size="11.5">64 个 token 的序列，颜色 = 归属的 CP rank（cp={cp}，切成 {2 * cp} 块）</text>']
    for t in range(L):
        h.append(f'<rect x="{x0 + t * cw:.1f}" y="28" width="{cw - 1.5:.1f}" height="22" fill="{pal[owner[t]]}"/>')
    mx = max(naive)
    for col, (name, arr) in enumerate([("连续切分", naive), ("zigzag 首尾配对", zig)]):
        bx = x0 + col * 420
        imb = max(arr) / (sum(arr) / cp)
        h.append(f'<text x="{bx}" y="80" font-size="12" font-weight="600">{name}  <tspan class="m" font-weight="400" font-size="11">最慢/平均 = {imb:.2f}</tspan></text>')
        for r, v in enumerate(arr):
            y = 92 + r * 22
            h.append(f'<text x="{bx}" y="{y + 12}" font-size="10.5" class="m">rank{r}</text>')
            h.append(f'<rect x="{bx + 50}" y="{y}" width="{v / mx * 330:.1f}" height="14" rx="3" fill="{pal[r]}"/>')
    H = 92 + cp * 22 + 10
    return f'<svg viewBox="0 0 {W} {H}" aria-label="CP={cp} 时连续切分与 zigzag 切分的负载对比">' + "".join(h) + "</svg>", naive, zig


def cp_md(fig: Tag, ctx: Ctx) -> str:
    svg, _, _ = cp_svg(4)
    out = [f"[▶ 在交互版中切换 cp=2/4/8]({SITE_INTERACTIVE}#par)", ctx.save_svg(svg, "CP=4 的切分与负载")]
    rows = ["| cp | 连续切分 最慢/平均 | zigzag 最慢/平均 |", "| --- | --- | --- |"]
    for cp in (2, 4, 8):
        _, n, z = cp_svg(cp)
        rows.append(f"| {cp} | {max(n) / (sum(n) / cp):.2f} | {max(z) / (sum(z) / cp):.2f} |")
    out.append("\n".join(rows))
    out.append(f"<small>{clean(inline(fig.find('figcaption')))}</small>")
    return "\n\n".join(out)


SPECIAL.update({  # 以 figure 内第一个带 id 的元素区分交互部件
    "jflow": journey_md,
    "rB": roof_md,
    "strip": layers_md,
    "vRand": varlen_md,
    "zSerial": zos_md,
    "cpviz": cp_md,
})


def main(dom_path: str) -> None:
    dom = Path(dom_path).read_text(encoding="utf-8")
    soup = BeautifulSoup(dom, "html.parser")
    CHAP_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for old in FIG_DIR.glob("*.svg"):
        old.unlink()
    for idx, (sid, fname) in enumerate(CHAPTERS, 1):
        sec = soup.find("section", id=sid)
        ctx = Ctx(dom, f"{idx:02d}")
        title = clean(inline(sec.find("h2")))
        parts = [f"# {idx:02d} {title}"]
        for el in sec.children:
            if isinstance(el, Tag):
                md = block(el, ctx)
                if md:
                    parts.append(md)
        prev_link = f"[← 上一篇](./{CHAPTERS[idx - 2][1]}.md)" if idx > 1 else "[← 目录](../index.md)"
        next_link = f"[下一篇 →](./{CHAPTERS[idx][1]}.md)" if idx < len(CHAPTERS) else "[回到目录 →](../index.md)"
        parts.append(f"---\n\n{prev_link} · [交互版本节]({SITE_INTERACTIVE}#{sid}) · {next_link}")
        (CHAP_DIR / f"{fname}.md").write_text("\n\n".join(parts) + "\n", encoding="utf-8")
        print(f"{fname}.md  figures={ctx.nfig}")


if __name__ == "__main__":
    main(sys.argv[1])
