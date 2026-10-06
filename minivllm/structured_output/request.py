"""请求级结构化输出状态（对应 vLLM `v1/structured_output/request.py`）。

`Request` 上挂一个它，就表示"这条请求的输出要受语法约束"：

    structured_output_key    (类型, 规格) —— 编译 grammar 的输入
    grammar                  编译出来的请求级 FSM（真正的状态在这里）

**与上游的差异**（写在 docs/step68_alignment.md）：

1. 上游的 `grammar` 可能是 `Future`（异步编译：`grammar_init` 丢进线程池，Scheduler 轮询
   `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR`）。本仓库**同步编译**并且**在提交请求时**就编译完
   （编译失败 → `add_request` 直接报错，请求根本没进引擎）。这是上游在
   `distributed_executor_backend == "external_launcher"` 下用的那条同步分支，
   异步编译属于"本项目尚未接入"（三态矩阵里明写）。
2. 思考模式（reasoning parser）整块未接入，所以没有 `reasoner` / `reasoning_ended` /
   `reasoning_end_token_index` 这些字段——`should_fill_bitmask()` 因此恒为 True（上游在
   没有 reasoner 时也是 True）。
"""

import dataclasses
import functools
import json

from .backend_types import StructuredOutputGrammar, StructuredOutputKey, StructuredOutputOptions


@dataclasses.dataclass
class StructuredOutputRequest:
    params: object
    grammar: StructuredOutputGrammar | None = None

    @staticmethod
    def from_sampling_params(sampling_params):
        """采样参数 → 请求级状态；没要求结构化输出就返回 None（上游同名静态方法）。"""
        if sampling_params is None:
            return None
        params = getattr(sampling_params, "structured_outputs", None)
        if not params or params.all_constraints_none():
            return None
        return StructuredOutputRequest(params=params)

    @functools.cached_property
    def structured_output_key(self) -> StructuredOutputKey:
        """`(类型, 规格)`。**必须在 `validate_xgrammar_grammar` 之后取**。

        上游用 `cached_property` 就是为这条：校验会把 `choice` 改写成 EBNF 并写进 `grammar`，
        提前取 key 会拿到 `CHOICE`，而 xgrammar 的 `compile_grammar` 没有 CHOICE 分支。
        """
        return get_structured_output_key(self.params)


def get_structured_output_key(params) -> StructuredOutputKey:
    """按 upstream 的优先级把参数折成一种约束（上游同名函数，顺序逐条保留）。"""
    if params.json is not None:
        json_str = params.json if isinstance(params.json, str) else json.dumps(params.json)
        return StructuredOutputOptions.JSON, json_str
    if params.json_object:
        return StructuredOutputOptions.JSON_OBJECT, ""
    if params.regex is not None:
        return StructuredOutputOptions.REGEX, params.regex
    if params.choice is not None:
        json_str = (params.choice if isinstance(params.choice, str)
                    else json.dumps(params.choice))
        return StructuredOutputOptions.CHOICE, json_str
    if params.grammar is not None:
        return StructuredOutputOptions.GRAMMAR, params.grammar
    if params.structural_tag is not None:
        return StructuredOutputOptions.STRUCTURAL_TAG, params.structural_tag
    raise ValueError("structured_outputs 里没有任何一种约束（None 与非 None 的判定错了）")
