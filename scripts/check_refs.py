"""检查文档里所有指向 DeepInfer 源码的「文件:行号」链接在固定提交上仍然有效。

用法：
    git clone https://github.com/deepseek-ai/DeepInfer /tmp/DeepInfer
    git -C /tmp/DeepInfer checkout 9bd9cf243fc53ef9afbf8f04d391d989117d6ab9
    python3 scripts/check_refs.py /tmp/DeepInfer

检查两类引用：Markdown / HTML 里的 GitHub blob 链接，以及交互版里的 data-ref 属性。
只校验文件存在且行号不超过文件行数；行号是否指向「正确的那一行」仍需人工复核。
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PINNED = "9bd9cf243fc53ef9afbf8f04d391d989117d6ab9"
BLOB = re.compile(r"github\.com/deepseek-ai/DeepInfer/blob/([0-9a-f]{40})/([^#)\s\"']+)#L(\d+)")
DATA_REF = re.compile(r'data-ref="([^":]+):(\d+)"')
JS_REF = re.compile(r'"((?:deepinfer|csrc|deep_cluster|examples|tests)/[^":]+):(\d+)"')


def collect() -> list[tuple[str, str, int, str]]:
    refs = []
    files = list((ROOT / "docs").rglob("*.md")) + [ROOT / "README.md", ROOT / "docs/public/interactive/index.html"]
    for f in files:
        text = f.read_text(encoding="utf-8")
        where = str(f.relative_to(ROOT))
        for sha, path, line in BLOB.findall(text):
            refs.append((sha, path, int(line), where))
        if f.suffix == ".html":
            for path, line in DATA_REF.findall(text) + JS_REF.findall(text):
                refs.append((PINNED, path, int(line), where))
    return refs


def main(checkout: str) -> int:
    repo = Path(checkout)
    head = subprocess.run(["git", "-C", checkout, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if head != PINNED:
        print(f"checkout 在 {head}，应为 {PINNED}")
        return 1
    refs = collect()
    errors = 0
    line_counts: dict[str, int] = {}
    for sha, path, line, where in refs:
        if sha != PINNED:
            print(f"[sha] {where}: {path}#L{line} 使用了 {sha[:7]}，应为 {PINNED[:7]}")
            errors += 1
            continue
        f = repo / path
        if not f.is_file():
            print(f"[missing] {where}: {path}")
            errors += 1
            continue
        n = line_counts.setdefault(path, len(f.read_text(encoding="utf-8", errors="replace").splitlines()))
        if line > n:
            print(f"[line] {where}: {path}#L{line} 超出文件长度 {n}")
            errors += 1
    print(f"检查了 {len(refs)} 处引用（{len({(p, l) for _, p, l, _ in refs})} 个不同位置），错误 {errors} 处")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
