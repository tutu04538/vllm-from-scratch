"""结构化输出的小工具（对应 vLLM `v1/structured_output/utils.py` 的本关子集 + `apply_grammar_bitmask`）。

三块内容：

1. **grammar 规格的规范化**（`grammar_is_likely_lark` / `convert_lark_to_ebnf` / `choice_as_grammar`）：
   上游这几个函数是纯字符串处理，与后端无关，本仓库逐字保留（含那两个正则与 `clean_line`）。
   为什么要转 Lark：xgrammar 只吃 EBNF 与 JSON schema（068 §3.4"不重造 JSON parser"——
   所以我们也不写自己的语法解析器，只做上游那套字符串改写）。
2. **正则编译超时**（`compile_regex_with_timeout`）：`(a+)+b` 这类嵌套量词会把 DFA 状态空间
   炸掉，编译线程卡住整个推理；上游用线程池 + 超时把它变成可报的错。
3. **把 bitmask 应用到 logits**（`apply_grammar_bitmask`）：上游同名函数在 `worker/gpu_model_runner`
   里被调用（`xgr.apply_token_bitmask_inplace`）。掩码里 bit=0 的位置表示"不允许"，内核把它写成
   `-inf` —— 于是**采样分布上**就不可能出现非法 token（068 §1 的小例子：候选在冒号前提出字母，
   要在验证分布上屏蔽，而不是生成之后删字母）。

**与上游的一处签名差异**（刻意，写在 docs/step68_alignment.md）：上游用它自己的 `InputBatch`
自己算"请求 → logits 行号"的映射；本仓库的非投机路径只为**要采样的行**算 logits（57 关以来的
教学差异），所以行号映射只有 Runner 知道，这里改成接收 Runner 算好的 `logit_index_of_req`。
数值与掩码顺序都按上游那套（每请求 `1 + K` 行：K 个候选位 + 1 个 bonus 位）。
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError


def compile_regex_with_timeout(fn, pattern: str):
    """带超时地跑一次"正则 → grammar"编译（上游同名函数）。

    超时值取上游同一个环境变量名 `VLLM_REGEX_COMPILATION_TIMEOUT_S`（默认 5 秒）；
    `<= 0` 表示不设超时（上游同款）。
    """
    timeout = int(os.getenv("VLLM_REGEX_COMPILATION_TIMEOUT_S", "5"))
    if timeout <= 0:
        return fn(pattern)

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(fn, pattern)
    try:
        result = future.result(timeout=timeout)
    except TimeoutError:
        future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise ValueError(
            f"正则编译超过 {timeout} 秒仍未完成：这个 pattern 可能构造出了指数级的状态空间"
            f"（例如嵌套量词 `(a+)+b`）。Pattern: {pattern[:200]}") from None
    else:
        executor.shutdown(wait=False)
        return result


def grammar_is_likely_lark(grammar_str: str) -> bool:
    """粗判一份 grammar 是不是 Lark 写法：有任何一行出现 `::=` 就认为它不是（上游同款）。

    上游的判据只看 EBNF 的 `::=` 记号；注释用 `#` 或 `//`。
    """
    if not grammar_str or not isinstance(grammar_str, str):
        return False

    for line in grammar_str.split("\n"):
        line = re.sub(r"(#|//).*$", "", line).strip()
        if not line:
            continue
        if "::=" in line:
            return False

    return True


def convert_lark_to_ebnf(grammar_str: str) -> str:
    """把 Lark 写法的 grammar 转成 EBNF（上游同名函数，逐字保留其规则与报错）。

    例：`rule: 'hello'` → `root ::= rule` + `rule ::= "hello"`；单引号统一换成双引号。
    上游的注释里给了参照 `llama.cpp` 的 EBNF 语法；本仓库不改任何一条规则——
    自己"优化"这份转换只会让同一份 grammar 在不同引擎里接受不同的语言。
    """
    if not isinstance(grammar_str, str):
        raise ValueError(f"grammar 必须是字符串，收到 {type(grammar_str)}")
    if not grammar_str.strip():
        raise ValueError("grammar 字符串不能为空")

    defined_rules = set()
    referenced_rules = set()
    output_lines = []

    def clean_line(line: str) -> str:
        return re.sub(r"(#|//).*$", "", line).strip()

    def check_quotes(text: str, rule_name: str, line_num: int) -> None:
        if text.count("'") % 2 != 0 or text.count('"') % 2 != 0:
            raise ValueError(f"{rule_name} 第 {line_num} 行的引号不配对")

    def extract_references(text: str) -> set[str]:
        text = re.sub(r'"[^"]*"', "", text)
        text = re.sub(r"[+*?()|\[\]{}]", " ", text)
        return set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", text))

    lines = [clean_line(line) for line in grammar_str.split("\n")]
    first_rule = None

    for line_num, line in enumerate(lines, 1):
        if not line or line.startswith("|"):
            continue
        if ":" in line:
            try:
                name = line.split(":", 1)[0].strip().strip("?")
                defined_rules.add(name)
                if first_rule is None:
                    first_rule = name
                if name == "start":
                    first_rule = "start"
            except IndexError as e:
                raise ValueError(
                    f"第 {line_num} 行的规则格式不对，应该是 'rule_name: definition'") from e

    if not defined_rules:
        raise ValueError("grammar 里没有找到任何规则")

    output_lines.append(f"root ::= {first_rule}")

    current_rule = None
    current_definition: list[str] = []

    for line_num, line in enumerate(lines, 1):
        if not line:
            continue
        try:
            if ":" in line and not line.startswith("|"):
                if current_rule:
                    output_lines.append(
                        f"{current_rule} ::= {' | '.join(current_definition)}")
                name, definition = line.split(":", 1)
                current_rule = name.strip().strip("?")
                check_quotes(definition, f"规则 '{current_rule}'", line_num)
                definition = re.sub(r"'([^']*)'", r'"\1"', definition)
                referenced_rules.update(extract_references(definition))
                current_definition = [definition.strip()]
            elif line.startswith("|"):
                if not current_rule:
                    raise ValueError(f"第 {line_num} 行出现了 '|'，但它前面没有规则定义")
                alt_def = line[1:].strip()
                check_quotes(alt_def, f"规则 '{current_rule}' 的备选分支", line_num)
                alt_def = re.sub(r"'([^']*)'", r'"\1"', alt_def)
                referenced_rules.update(extract_references(alt_def))
                current_definition.append(alt_def)
        except ValueError as e:
            raise ValueError(f"第 {line_num} 行：{e}") from e

    if current_rule:
        output_lines.append(f"{current_rule} ::= {' | '.join(current_definition)}")

    undefined_rules = referenced_rules - defined_rules - {"root"}
    if undefined_rules:
        raise ValueError(f"引用了没有定义的规则：{', '.join(sorted(undefined_rules))}")

    return "\n".join(output_lines)


def choice_as_grammar(choice: list[str]) -> str:
    """把"只能输出这几个字符串之一"变成一份 EBNF（上游同名函数）。"""
    def escape_ebnf_string(s: str) -> str:
        return re.sub(r'(["\\])', r"\\\1", s)

    escaped_choices = (escape_ebnf_string(c) for c in choice)
    return "root ::= " + " | ".join(f'"{c}"' for c in escaped_choices)


def apply_grammar_bitmask(logits, grammar_output, scheduled_spec_decode_tokens: dict,
                          logit_index_of_req: dict[str, int]) -> None:
    """把语法掩码原地打到 `logits` 的对应行上（上游 `v1/structured_output/utils.py` 同名函数）。

    三个输入：
        grammar_output               掩码（紧凑排列：每个结构化请求 `1 + K` 行）
        scheduled_spec_decode_tokens 本轮每请求采用了几枚草稿（决定"1 + K"里的 K）
        logit_index_of_req           请求 ID → 它在本轮 `logits` 里的**第一行**行号（Runner 算）

    掩码行序与 logits 行序必须逐请求一致：第 i 份掩码打给"请求 r 的第 j 行"。行映射算错
    不会报错，只会把 A 的掩码打到 B 的行上（静默错），所以这里对总数做一次断言。

    **不返回新张量**：上游同样是原地改（`xgr.apply_token_bitmask_inplace`），这样"掩码之后的
    logits"就是采样器看到的那一份——`processed_logits` 模式的 logprobs 因此**天然包含**掩码，
    不需要在 logprobs 侧再补一次（068 §3.5 的"索引正确"就建立在这条同源关系上）。
    """
    import numpy as np
    import torch

    bitmask = torch.from_numpy(np.ascontiguousarray(grammar_output.grammar_bitmask))
    out_indices: list[int] = []
    total_masks = 0
    for req_id in grammar_output.structured_output_request_ids:
        num_spec_tokens = len(scheduled_spec_decode_tokens.get(req_id, ()))
        logit_index = logit_index_of_req.get(req_id)
        if logit_index is not None:
            out_indices.extend(logit_index + i for i in range(1 + num_spec_tokens))
        total_masks += 1 + num_spec_tokens

    if total_masks != len(out_indices):
        # 掩码是按"所有结构化请求"生成的；logits 里少了某一行说明行映射算错了。
        # 不报错的话，第 i 份掩码会打到别人那一行上（**不报错的静默错**）。
        raise RuntimeError(
            f"语法掩码有 {total_masks} 行，但只找到 {len(out_indices)} 个对应的 logits 行："
            f"请求 → logits 行的映射与 grammar_bitmask 的生成顺序不一致")

    bitmask = bitmask.to(logits.device)
    if len(out_indices) == logits.shape[0]:
        # 掩码行与 logits 行完全对齐时不必传 indices（上游同款优化）
        _apply_bitmask(logits, bitmask, None)
        return
    _apply_bitmask(logits, bitmask, out_indices)


def _apply_bitmask(logits, bitmask, indices) -> None:
    """调后端内核（顺带处理 dtype，上游同款）。

    CPU 上的 xgrammar 内核只吃 float32（上游 issue #31901 的绕法）：fp16/bf16 的 logits
    先转 fp32 打掩码、再拷回去。不改 dtype 会在 CPU 路径直接抛内核错误。
    """
    import torch
    import xgrammar as xgr

    if logits.dtype != torch.float32:
        logits_fp32 = logits.to(torch.float32)
        xgr.apply_token_bitmask_inplace(logits_fp32, bitmask, indices=indices)
        logits.copy_(logits_fp32.to(logits.dtype))
        return
    xgr.apply_token_bitmask_inplace(logits, bitmask, indices=indices)
