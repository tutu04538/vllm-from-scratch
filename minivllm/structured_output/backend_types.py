"""结构化输出的两个抽象：请求级的 `StructuredOutputGrammar` 与引擎级的 `StructuredOutputBackend`。

对应 vLLM `v1/structured_output/backend_types.py`（逐字对齐：枚举、`StructuredOutputKey`、
两个 ABC 的方法名与语义）。

为什么要有抽象：**"谁能解码"是后端库的事**（xgrammar / guidance / outlines / lm-format-enforcer），
"什么时候推进状态、什么时候回滚、每行填哪份掩码"是引擎的事（068 §2）。分成两层之后，
投机验证要的"试走 + 回滚"只需要 `StructuredOutputGrammar` 这一个接口，与具体后端无关。

    引擎级 backend      建 grammar（编译 schema/regex/grammar）、分配 bitmask 缓冲
    请求级 grammar      接受 token（推进 FSM）、试走（validate_tokens，不推进）、回滚、填掩码

**本仓库只接 xgrammar**（068 §3.4："选择本机已有 backend，不重造 JSON parser"；本机装了
xgrammar 0.2.3 / outlines_core / lmformatenforcer，上游的 `auto` 默认也是 xgrammar）。
其它 backend 的字段照旧留在配置里（`StructuredOutputsConfig.backend`），但选择它们会在
**请求期**明确报错——不是静默用一个别的后端顶上（068 §3.6）。
"""

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass


class StructuredOutputOptions(enum.Enum):
    JSON = enum.auto()
    JSON_OBJECT = enum.auto()
    REGEX = enum.auto()
    GRAMMAR = enum.auto()
    CHOICE = enum.auto()
    STRUCTURAL_TAG = enum.auto()


#: `(类型, 规格字符串)`。上游同名类型别名。
StructuredOutputKey = tuple[StructuredOutputOptions, str]


class StructuredOutputGrammar(ABC):
    """请求级的后端状态（上游同名 ABC，方法逐个对应）。"""

    @abstractmethod
    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        """接受一串 token 并**推进** FSM；有一个不合法就返回 False（不推进到那一步之后）。

        只有"已经真正提交的 token"才该走这条路（Scheduler 在结果回来之后调，068 §2）。
        """

    @abstractmethod
    def validate_tokens(self, tokens: list[int]) -> list[int]:
        """试走一串 token 但**不推进** FSM；返回被接受的前缀。

        用途一（67 关起就有）：Scheduler 收下草稿时先裁掉不合语法的尾巴，别让它们进预算；
        用途二（068 §2）：本关的 bitmask 生成要对每个候选位置"假设前面都被接受"地试走一遍，
        算完必须撤销——**proposer 不能永久推进 grammar**。
        """

    @abstractmethod
    def rollback(self, num_tokens: int) -> None:
        """把 FSM 退回 `num_tokens` 步（试走之后必须成对调用；上游同款）。"""

    @abstractmethod
    def fill_bitmask(self, bitmask, batch_index: int) -> None:
        """把"当前位置允许哪些 token"写进 bitmask 的第 `batch_index` 行。"""

    @abstractmethod
    def is_terminated(self) -> bool:
        """FSM 是否已经走完（结束之后不再填掩码，也不会再接受 token）。"""

    @abstractmethod
    def reset(self) -> None:
        """回到初始状态。"""


@dataclass
class StructuredOutputBackend(ABC):
    """引擎级的后端（上游同名 dataclass）：编译 grammar + 分配掩码缓冲。"""

    vllm_config: object
    tokenizer: object
    vocab_size: int

    @abstractmethod
    def compile_grammar(self, request_type: StructuredOutputOptions, grammar_spec: str,
                        stop_token_ids: set[int] | None = None) -> StructuredOutputGrammar:
        """把规格编译成请求级 grammar。

        `stop_token_ids` 是这条请求的 EOS + 显式 stop token（上游 `all_stop_token_ids`）：
        编译期交给后端，让它在 FSM **结束之后**才把这些 token 放出来。
        """

    @abstractmethod
    def allocate_token_bitmask(self, max_num_seqs: int):
        """分配 `[max_num_seqs, ceil(vocab/32)]` 的 int32 掩码缓冲（后端决定布局）。"""

    @abstractmethod
    def destroy(self) -> None:
        """后端自己的清理（上游用它释放编译缓存）。"""
