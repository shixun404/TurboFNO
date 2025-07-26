#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cutlass_pretty_to_json.py
========================
把 NVCC / CUTLASS 的 “pretty‑function” 输出转换成层次化 JSON：
  * <…>  ⇒  { "outer": [...]}      （outer 为模板 / 类型名）
  * […]  ⇒  [ … ]                  （保持数组语义）
"""

import json
import re
import sys
from typing import List

# --------------------------------------------------------------------------- #
# 1. 工具函数
# --------------------------------------------------------------------------- #
def tokenize_brackets(s: str, open_sym: str, close_sym: str) -> List[str]:
    """把 *最外层* open_sym … close_sym 里的逗号分段，返回子串列表"""
    tokens, current, depth = [], "", 0
    for c in s:
        if c == open_sym:
            if depth:
                current += c
            depth += 1
        elif c == close_sym:
            depth -= 1
            if depth:
                current += c
            else:
                tokens.append(current.strip())
                current = ""
        elif c == ',' and depth == 1:
            tokens.append(current.strip())
            current = ""
        else:
            current += c
    if current.strip():
        tokens.append(current.strip())
    return tokens


def find_matching(text: str, open_sym: str, close_sym: str, start: int) -> int:
    """给定 open_sym 的索引，返回其匹配 close_sym 的索引"""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == open_sym:
            depth += 1
        elif text[i] == close_sym:
            depth -= 1
            if depth == 0:
                return i
    raise ValueError(f"No matching {close_sym!r} for {open_sym!r} at pos {start}")


# --------------------------------------------------------------------------- #
# 2. 递归解析
# --------------------------------------------------------------------------- #
def parse_nested(s: str):
    """把单个类型/模板串解析成嵌套 JSON 结构"""
    s = s.strip()

    # 先查 '<'，再查 '['，保证优先处理模板 <> 形式
    for open_sym, close_sym in (('<', '>'), ('[', ']')):
        if open_sym in s:
            start = s.index(open_sym)
            end   = find_matching(s, open_sym, close_sym, start)
            outer = s[:start].strip()
            inner = s[start + 1 : end]

            subtokens = tokenize_brackets(open_sym + inner + close_sym,
                                          open_sym, close_sym)
            return {
                outer: [parse_nested(token) for token in subtokens]
            }

    # 纯标识符 / 数字常量
    return s


# --------------------------------------------------------------------------- #
# 3. 提取 [with …] 块
# --------------------------------------------------------------------------- #
def extract_with_block(text: str) -> str | None:
    """截出 '[with …]' 并返回 'T = …' 后面的主体，忽略后续赋值串"""
    m = re.search(r'\[with\s+T\s*=\s*(.*?)\](?!\s*\w)', text, re.S)
    return m.group(1).strip() if m else None


# --------------------------------------------------------------------------- #
# 4. 主程序
# --------------------------------------------------------------------------- #
def main() -> None:
    text = sys.stdin.read() or """
    /* 这里放你的 pretty-function 字符串，或用管道/重定向输入 */
    """

    with_block = extract_with_block(text)
    if not with_block:
        sys.exit("❌  没找到 '[with …]' 块，请检查输入。")

    structured = parse_nested(with_block)
    print(json.dumps(structured, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
