"""`Attention`：模型里的"注意力边界"（对应 vLLM `vllm/attention/layer.py`）。

它持有的四样东西，正好是"一层注意力需要知道的全部"：

    layer_name    唯一的层名 —— 用来从 forward 上下文里取**自己这一层**的 metadata
    num_heads / head_size / scale   头参数（GQA 下还有 num_kv_heads）
    kv_cache      绑定的物理缓存（由 Runner 在初始化时灌进来；模型自己不知道块表）
    impl          后端实现（Torch / Triton / FlashAttention 可换）

`forward(q, k, v)` 只做一件事：**取本层 metadata → 交给后端**。它不认识 Request、不认识块表、
不认识采样。换后端时模型代码一个字不动。

**它是 `nn.Module`**（vLLM 的 `Attention` 也是）：没有参数，但要能被 `model.named_modules()`
看见——Runner 就是靠遍历模型找到所有 Attention 层、按 `layer_name` 绑 KV 缓存、并把同一份
metadata 挂到每个层名下的。做成普通 Python 对象的话它就"隐身"了（第一版就是这样，
`named_modules()` 里找不到它，KV 无处可绑）。
"""

import math

import torch
from torch import nn

from ..compilation.breakable_cudagraph import eager_break_during_capture
from .backends import TorchAttentionImpl
from .forward_context import get_forward_context

#: 当前模型加载时选用的 attention impl。默认 Torch；V2 Runner 在加载 target 前按
#: `vllm_config.attention_backend` 设置，加载完 target 后重置为 Torch，避免草稿层
#: 误用 target 的后端（草稿的 metadata/能力仍是 Torch 那套）。
_ATTENTION_IMPL_CLS = TorchAttentionImpl


def set_attention_impl_cls(cls) -> None:
    global _ATTENTION_IMPL_CLS
    _ATTENTION_IMPL_CLS = cls


def get_attention_impl_cls():
    return _ATTENTION_IMPL_CLS


class Attention(nn.Module):
    def __init__(self, num_heads: int, head_size: int, scale: float | None = None,
                 num_kv_heads: int | None = None, layer_name: str = "") -> None:
        super().__init__()
        self.layer_name = layer_name
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.scale = 1.0 / math.sqrt(head_size) if scale is None else scale
        self.impl = _ATTENTION_IMPL_CLS(
            num_heads, head_size, self.scale, self.num_kv_heads)
        # 由 Runner 在 KV 分配之后绑定：`[2, num_blocks, block_size, kv_heads, head_size]`
        self.kv_cache = None
        #: breakable PIECEWISE 的固定输出缓冲（runner 在 KV 分配后设置）。
        #: 图段之间靠**同一地址**传递 attention 输出；返回新张量会让下一段读旧内存。
        self._output_buffer: torch.Tensor | None = None

    def set_output_buffer(self, max_num_tokens: int, dtype: torch.dtype,
                          device: torch.device) -> None:
        self._output_buffer = torch.empty(
            max_num_tokens, self.num_heads * self.head_size,
            dtype=dtype, device=device)

    @eager_break_during_capture
    def forward(self, query, key, value):
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata.get(self.layer_name)
        if attn_metadata is None:
            raise KeyError(f"forward 上下文里没有层 {self.layer_name!r} 的 metadata；"
                           f"Runner 必须为每个 Attention 层都建一份（键是层名）")
        if self.kv_cache is None:
            raise RuntimeError(f"层 {self.layer_name!r} 的 kv_cache 还没绑定（Runner 初始化时绑定）")
        result = self.impl.forward(
            self, query, key, value, self.kv_cache, attn_metadata)
        if self._output_buffer is None:
            return result
        out = self._output_buffer[: result.shape[0]]
        out.copy_(result)
        return out
