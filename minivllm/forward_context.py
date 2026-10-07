"""一次 forward 的上下文 + CUDA Graph 的批次描述（对应 vLLM `vllm/forward_context.py`）。

69 关之前，这个上下文只有一样东西："层名 → AttentionMetadata"。这一关要往里加**两样**：

    cudagraph_runtime_mode   本轮是 eager(NONE) 还是重放图(FULL/PIECEWISE)
    batch_descriptor         本轮（补齐后）的批次形状 —— 就是图缓存的键

为什么必须放进上下文，而不是当参数传给模型：模型层的 `forward()` 签名是固定的
（`forward(input_ids, positions)`，198 §6），不认 Runner 的参数。而注意力层要决定
"这一轮能不能走图内那条固定形状的路径"、图包装器要决定"这一轮是捕获、重放还是直通"，
它们都只能从这里读。上游也是同一个做法（`vllm/forward_context.py`）。

**`BatchDescriptor` 是什么**：图形状必须完全固定，所以键里不能只写"这一轮有多少 token"，
还要写清"这些 token 摊在几条请求上、是不是每条请求一样长、批次里有几套 LoRA"。

    num_tokens          补齐后的**总行数**（padding 之后，不是真实行数）
    num_reqs            补齐后的请求数；PIECEWISE 图可以处理任意请求数，所以那里是 None
    uniform             所有请求的 query 长度是否相同（统一 decode：每请求 1+K 行）
    has_lora / num_active_loras
                        本仓库没有 LoRA，两个字段照抄只是为了键的形状与上游一致（恒 False/0）

`uniform` 这一条是投机解码与 CUDA Graph 的接缝：**只有"每请求恰好 1+K 行"的批，
形状才是固定的**，上游因此把它单独拎出来叫 uniform decode（`uniform_decode_query_len = 1 + K`）。
本仓库不做 LoRA（字段保留），也不实现 PIECEWISE（那需要算子拆分编译，配置期明确拒绝）。
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from .config import CUDAGraphMode


@dataclass(frozen=True)
class BatchDescriptor:
    """CUDA Graph 的键（对应上游 `vllm/forward_context.py:29-58` 的同名类）。"""

    num_tokens: int
    num_reqs: int | None = None
    """批里几条请求。PIECEWISE 图能处理任意请求数，那里是 None；FULL 图必须精确匹配。"""
    uniform: bool = False
    """是不是"所有请求 query 长度相同"的批（统一 decode：每请求 `1 + K` 行）。"""
    has_lora: bool = False
    num_active_loras: int = 0


@dataclass
class ForwardContext:
    """一次 forward 的只读上下文（只留本仓库有能力兑现的字段）。

    上游还有 `no_compile_layers` / `dp_metadata` / `ubatch_slices` 等字段，那些属于
    "多卡 + 编译期静态上下文 + 微批"，本仓库没有对应实现，所以**不留空壳字段**：
    留了却没有生产者，读的人会以为它是有效数据。
    """

    attn_metadata: dict = field(default_factory=dict)
    num_tokens: int = 0
    # 本轮允许的图运行模式（NONE = 纯 eager）。图包装器按它决定是否重放。
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE
    # 本轮（补齐后）的批次形状 = 图缓存键。eager 时可以是 None（上游默认值）。
    batch_descriptor: BatchDescriptor | None = None
    # 本轮每个输入 token 写到哪个物理槽位；padding 行是 `PADDING_SLOT_ID(-1)`。
    # 上游把它放进上下文是给"在图里自己算槽位的层"用（本仓库的 attention 后端直接读
    # metadata，所以这里只作为一致性来源保留）。
    slot_mapping: dict | None = None
    additional_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.cudagraph_runtime_mode.is_valid_runtime_mode():
            raise ValueError(
                f"forward 上下文里的 cudagraph_runtime_mode 只能是 NONE/PIECEWISE/FULL 三种"
                f"具体运行模式之一，收到 {self.cudagraph_runtime_mode}")


# 模块级全局：与 vLLM 一样用"设进去、finally 恢复"的方式，而不是 contextvars
_forward_context: ForwardContext | None = None


def get_forward_context() -> ForwardContext:
    if _forward_context is None:
        raise RuntimeError("当前不在 forward 上下文里：先 set_forward_context()（模型层不该"
                           "直接拿 metadata，必须经过这个上下文）")
    return _forward_context


def is_forward_context_available() -> bool:
    """不在上下文里时返回 False（而不是抛）。图包装器要用它判断"这次调用是不是推理路径"
    ——上游同款：没有上下文就直通，不做捕获也不做重放。"""
    return _forward_context is not None


def create_forward_context(attn_metadata: Any, num_tokens: int = 0,
                           cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
                           batch_descriptor: BatchDescriptor | None = None,
                           slot_mapping: dict | None = None,
                           additional_kwargs: dict | None = None) -> ForwardContext:
    """造一个上下文对象（上游 `create_forward_context`）。

    与上游的差异：上游签名里要 `vllm_config`，因为它要用它填"编译期静态上下文"（
    `static_forward_context` / DP metadata）。本仓库没有编译期静态层表，也不做 DP，
    所以不需要 config——这一条记在 `docs/step69_alignment.md` 的差异账上。

    上游还有一条便利规则：`runtime_mode != NONE` 且给了 `num_tokens` 时自动补一个
    `BatchDescriptor(num_tokens=...)`（包装器匹配不上时自然会直通，不会错）。这里照抄。
    """
    if cudagraph_runtime_mode != CUDAGraphMode.NONE and num_tokens is not None \
            and batch_descriptor is None:
        batch_descriptor = BatchDescriptor(num_tokens=num_tokens)
    return ForwardContext(
        attn_metadata=attn_metadata or {},
        num_tokens=num_tokens,
        cudagraph_runtime_mode=cudagraph_runtime_mode,
        batch_descriptor=batch_descriptor,
        slot_mapping=slot_mapping,
        additional_kwargs=additional_kwargs or {},
    )


@contextmanager
def override_forward_context(forward_context: ForwardContext | None):
    """临时覆盖当前上下文（上游 `override_forward_context`；测试与嵌套场景用）。"""
    global _forward_context
    previous = _forward_context
    _forward_context = forward_context
    try:
        yield
    finally:
        _forward_context = previous


@contextmanager
def set_forward_context(attn_metadata: dict, num_tokens: int = 0,
                        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
                        batch_descriptor: BatchDescriptor | None = None,
                        slot_mapping: dict | None = None,
                        additional_kwargs: dict | None = None):
    """设上下文，`finally` 里恢复（异常退出也要恢复：否则下一次 forward 会读到上一轮的
    metadata / 图模式，那是"图在错误的形状上重放"这类最难查的错）。

    与上游的差异：上游的第二个位置参数是 `vllm_config`（见 `create_forward_context` 的说明），
    本仓库不需要它，所以位置参数留给了 `num_tokens`——**位置变了，关键字不变**，
    调用点一律用关键字（本仓库所有调用点都是关键字调用）。
    """
    forward_context = create_forward_context(
        attn_metadata, num_tokens, cudagraph_runtime_mode, batch_descriptor,
        slot_mapping, additional_kwargs)
    with override_forward_context(forward_context):
        yield forward_context
