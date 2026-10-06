"""xgrammar 后端（对应 vLLM `v1/structured_output/backend_xgrammar.py`）。

本仓库**只接这一个后端**，理由见 `backend_types.py` 的模块说明（068 §3.4：选本机已有的
backend，不重造 JSON parser）。这一份是上游的核心逻辑，逐块对应：

    XgrammarBackend.__post_init__     建 TokenizerInfo（从 HF tokenizer）+ GrammarCompiler
    compile_grammar                   JSON / JSON_OBJECT / REGEX / GRAMMAR / STRUCTURAL_TAG
    allocate_token_bitmask            xgr.allocate_token_bitmask(max_num_seqs, vocab_size)
    XgrammarGrammar.accept_tokens     推进 FSM（有一个 token 不合法就 False）
    XgrammarGrammar.validate_tokens   试走 + 回滚（**不推进**）
    XgrammarGrammar.rollback          退 num_tokens 步（68 关的试走靠它成对收尾）
    XgrammarGrammar.fill_bitmask      把"当前位置允许哪些 token"写进第 idx 行
    validate_xgrammar_grammar         请求期校验（含 `choice` → EBNF 的改写）

**两处刻意的差异**（都写在 docs/step68_alignment.md）：

1. `compile_regex_with_timeout` 的调用点：上游正则与 JSON schema 都走超时包装；本仓库只有
   REGEX 走（JSON_OBJECT 的 schema 是常量，JSON 走的是 xgrammar 自己的编译）。
2. `max_rollback_tokens`：上游用 `num_speculative_tokens` 断言"回滚不会超过 K 步"，
   本仓库同款（不设它的话 xgrammar 默认只能回滚 0 步，68 关的试走会直接失败）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .backend_types import (
    StructuredOutputBackend,
    StructuredOutputGrammar,
    StructuredOutputOptions,
)
from .utils import (
    choice_as_grammar,
    compile_regex_with_timeout,
    convert_lark_to_ebnf,
    grammar_is_likely_lark,
)

#: xgrammar 支持的 string format（上游同名集合，逐字保留）
STRING_SUPPORTED_FORMATS = {
    "email", "date", "time", "date-time", "duration", "ipv4", "ipv6", "hostname",
    "uuid", "uri", "uri-reference", "uri-template", "json-pointer",
    "relative-json-pointer",
}


def has_xgrammar_unsupported_json_features(schema: dict) -> bool:
    """这份 JSON schema 有没有 xgrammar 不支持的关键字（上游同名函数，规则逐条保留）。

    不支持就**请求期报错**，而不是让 xgrammar 悄悄忽略这个关键字——"要求整数在 1~10 之间"
    被忽略的话，模型照样可能吐 999，而用户以为约束生效了。
    """
    def check_object(obj) -> bool:
        if not isinstance(obj, dict):
            return False

        if obj.get("type") in ("integer", "number") and ("multipleOf" in obj):
            return True
        if obj.get("type") == "array" and any(
                key in obj for key in ("uniqueItems", "contains", "minContains",
                                       "maxContains")):
            return True
        if (obj.get("type") == "string" and "format" in obj
                and obj["format"] not in STRING_SUPPORTED_FORMATS):
            return True
        if obj.get("type") == "object" and any(
                key in obj for key in ("patternProperties", "propertyNames")):
            return True

        for value in obj.values():
            if isinstance(value, dict):
                if check_object(value):
                    return True
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and check_object(item):
                        return True
        return False

    return check_object(schema)


def validate_xgrammar_grammar(sampling_params) -> None:
    """请求期校验 + 规范化（上游同名函数）。

    **它会改写 `structured_outputs`**：`choice=["a","b"]` 会被转成 EBNF 并写进 `grammar`
    （上游同样这么做，见其 L306-316）。所以 `StructuredOutputRequest.structured_output_key`
    必须在它**之后**才算（上游用 `cached_property` 保证这一点）。
    """
    import xgrammar as xgr

    if sampling_params.structured_outputs is None:
        return

    so_params = sampling_params.structured_outputs

    if so_params.regex:
        # NUL 字节在正则里没有意义，xgrammar 的原生转换也处理不了：在进原生代码之前挡住
        if "\x00" in so_params.regex:
            raise ValueError("structured_outputs.regex 里不能有 NUL 字符（'\\x00'）")
        try:
            compile_regex_with_timeout(xgr.Grammar.from_regex, so_params.regex)
        except Exception as err:
            raise ValueError(f"正则转 grammar 失败：{err}") from err

    if so_params.choice:
        choice_grammar = choice_as_grammar(so_params.choice)
        try:
            xgr.Grammar.from_ebnf(choice_grammar)
        except Exception as err:
            raise ValueError(f"choice 转 grammar 失败：{err}") from err
        so_params.choice = None
        so_params.grammar = choice_grammar
        return

    if so_params.json:
        if isinstance(so_params.json, str):
            try:
                schema = json.loads(so_params.json)
            except json.JSONDecodeError as e:
                raise ValueError("structured_outputs.json 不是合法 JSON") from e
        else:
            schema = so_params.json

        if has_xgrammar_unsupported_json_features(schema):
            raise ValueError("这份 JSON schema 含 xgrammar 不支持的关键字（见 "
                             "has_xgrammar_unsupported_json_features 的清单）")
        try:
            xgr.Grammar.from_json_schema(schema)
        except Exception as err:
            raise ValueError(f"JSON schema 转 grammar 失败：{err}") from err
        return

    if so_params.grammar:
        if grammar_is_likely_lark(so_params.grammar):
            try:
                so_params.grammar = convert_lark_to_ebnf(so_params.grammar)
            except ValueError as e:
                raise ValueError(f"Lark grammar 转 EBNF 失败：{e}") from e
        try:
            xgr.Grammar.from_ebnf(so_params.grammar)
        except Exception as e:
            raise ValueError(f"grammar 不合法：{e}") from e
        return

    if so_params.structural_tag:
        try:
            s_tag = json.loads(so_params.structural_tag)
            if "structures" in s_tag:
                tags = [
                    xgr.StructuralTagItem(begin=s["begin"],
                                          schema=json.dumps(s["schema"]), end=s["end"])
                    for s in s_tag["structures"]
                ]
                xgr.Grammar.from_structural_tag(tags, s_tag["triggers"])
            else:
                xgr.Grammar.from_structural_tag(so_params.structural_tag)
        except Exception as e:
            raise ValueError(f"structural_tag 不合法：{e}") from e


@dataclass
class XgrammarBackend(StructuredOutputBackend):
    def __post_init__(self) -> None:
        import xgrammar as xgr

        self.disable_any_whitespace = getattr(
            getattr(self.vllm_config, "structured_outputs_config", None),
            "disable_any_whitespace", False)

        # 词表从 tokenizer 那边取（上游同款：不用 `tokenizer.vocab_size`，它会把解码错误的
        # token 折叠成一个）。`vocab_size` 由调用方按 model config 传进来。
        tokenizer_info = xgr.TokenizerInfo.from_huggingface(
            self.tokenizer, vocab_size=self.vocab_size)
        self.compiler = xgr.GrammarCompiler(tokenizer_info, max_threads=8,
                                            cache_enabled=True)

        self.num_speculative_tokens = 0
        spec_config = getattr(self.vllm_config, "speculative_config", None)
        if spec_config is not None:
            self.num_speculative_tokens = spec_config.num_speculative_tokens

    def compile_grammar(self, request_type: StructuredOutputOptions, grammar_spec: str,
                        stop_token_ids: set[int] | None = None) -> StructuredOutputGrammar:
        import xgrammar as xgr

        if request_type == StructuredOutputOptions.JSON:
            ctx = self.compiler.compile_json_schema(
                grammar_spec, any_whitespace=not self.disable_any_whitespace)
        elif request_type == StructuredOutputOptions.JSON_OBJECT:
            ctx = self.compiler.compile_json_schema(
                '{"type": "object"}', any_whitespace=not self.disable_any_whitespace)
        elif request_type == StructuredOutputOptions.GRAMMAR:
            ctx = self.compiler.compile_grammar(grammar_spec)
        elif request_type == StructuredOutputOptions.REGEX:
            ctx = compile_regex_with_timeout(self.compiler.compile_regex, grammar_spec)
        elif request_type == StructuredOutputOptions.STRUCTURAL_TAG:
            s_tag = json.loads(grammar_spec)
            if "structures" in s_tag:
                tags = [
                    xgr.StructuralTagItem(begin=s["begin"],
                                          schema=json.dumps(s["schema"]), end=s["end"])
                    for s in s_tag["structures"]
                ]
                ctx = self.compiler.compile_structural_tag(tags, s_tag["triggers"])
            else:
                ctx = self.compiler.compile_structural_tag(grammar_spec)
        else:
            raise ValueError(
                f"{request_type!s} 不是 xgrammar 能编译的类型：请求期校验应该已经挡掉了它"
                f"（068 §3.6 要求缺能力就报错，不许静默换一种解释）")

        return XgrammarGrammar(
            matcher=xgr.GrammarMatcher(
                ctx,
                override_stop_tokens=list(stop_token_ids) if stop_token_ids else None,
                # 68 关：试走 K 步之后要能整体回滚。不设这个上限的话 xgrammar 默认
                # max_rollback_tokens=0，`rollback(K)` 会直接断言失败。
                max_rollback_tokens=self.num_speculative_tokens),
            vocab_size=self.vocab_size,
            ctx=ctx,
        )

    def allocate_token_bitmask(self, max_num_seqs: int):
        import xgrammar as xgr

        return xgr.allocate_token_bitmask(max_num_seqs, self.vocab_size)

    def destroy(self) -> None:
        del self.compiler


@dataclass
class XgrammarGrammar(StructuredOutputGrammar):
    vocab_size: int
    matcher: object = field(hash=False)
    ctx: object = field(hash=False)
    num_processed_tokens: int = field(default=0, repr=False, hash=False, init=False)
    _is_terminated: bool = field(default=False, repr=False, hash=False)

    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        """推进 FSM；有一个 token 不合法就返回 False（**已经推进的那几步不回退**，上游同款）。

        68 关的重要性质：这个方法**只有 Scheduler**在真正提交 token 之后才调（068 §2：
        "不要让 proposer 永久推进 grammar"）。bitmask 阶段用的是 validate/rollback 那对。
        """
        if self._is_terminated:
            return False
        for token in tokens:
            if not self.matcher.accept_token(token):
                return False
            self.num_processed_tokens += 1
        self._is_terminated = self.matcher.is_terminated()
        return True

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        """试走一串 token，返回被接受的前缀；无论接受多少都**回滚到原状态**。"""
        accepted_tokens = []
        for token in tokens:
            if self.matcher.accept_token(token):
                accepted_tokens.append(token)
            else:
                break
        if len(accepted_tokens) > 0:
            self.matcher.rollback(len(accepted_tokens))
        return accepted_tokens

    def rollback(self, num_tokens: int) -> None:
        self.matcher.rollback(num_tokens)
        self.num_processed_tokens -= num_tokens
        self._is_terminated = self.matcher.is_terminated()

    def fill_bitmask(self, bitmask, idx: int) -> None:
        self.matcher.fill_next_token_bitmask(bitmask, idx)

    def is_terminated(self) -> bool:
        return self._is_terminated

    def reset(self) -> None:
        self.num_processed_tokens = 0
        self.matcher.reset()
