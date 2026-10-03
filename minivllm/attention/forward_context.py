"""一次 forward 的上下文（对应 vLLM `vllm/forward_context.py`，只留本关需要的部分）。

解决的问题：**模型层怎么知道自己这一层的 AttentionMetadata？**

模型不该拿 Runner 的参数（`forward()` 的签名里没有块表、没有 metadata）。做法是：Runner 在
forward 之前把"层名 → AttentionMetadata"放进一个临时的上下文，`Attention.forward()` 再按**自己
唯一的层名**取出来。

三条约定：

1. **只在一次 forward 内有效**：`set_forward_context()` 必须在 `finally` 里恢复（异常退出也要恢复，
   否则下一次 forward 会读到上一次的 metadata——那是最难查的一类错）。
2. **按层名索引**：`attn_metadata[layer_name]`；层名在构造时就固定（`Attention` 持有它）。
3. 上下文里**只有 metadata**：没有 Request、没有 Scheduler、没有 KV 池对象。
"""

from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class ForwardContext:
    """一次 forward 的只读上下文。"""

    attn_metadata: dict = field(default_factory=dict)
    num_tokens: int = 0


# 模块级全局：与 vLLM 一样用"设进去、finally 恢复"的方式，而不是 contextvars
_forward_context: ForwardContext | None = None


def get_forward_context() -> ForwardContext:
    if _forward_context is None:
        raise RuntimeError("当前不在 forward 上下文里：先 set_forward_context()（模型层不该"
                           "直接拿 metadata，必须经过这个上下文）")
    return _forward_context


@contextmanager
def set_forward_context(attn_metadata: dict, num_tokens: int):
    global _forward_context
    previous = _forward_context
    _forward_context = ForwardContext(attn_metadata=attn_metadata, num_tokens=num_tokens)
    try:
        yield _forward_context
    finally:
        _forward_context = previous          # 异常也要恢复
