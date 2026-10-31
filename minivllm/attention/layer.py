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

#: 当前模型加载时选用的 attention impl。默认 Torch；V2 Runner 在加载 target 与每个
#: speculator 草稿模型时按 `vllm_config.attention_backend` 临时设置，加载完恢复 Torch，
#: 避免全局状态污染。087 起草稿也有自己的 metadata builder，并跟随 target 的 backend。
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
        #: 80D：Q-only 层（Gemma4 assistant 这类）只算 Q，K/V 直接读另一个层的 cache。
        #: 置 True 后 `forward()` 走 `impl.forward_q_only()`，**不写** K/V。
        self.q_only = False
        #: Q-only 层的源层名（由 speculator 的映射建立，Runner 消费）。
        #: 非 None 表示"这一层的 KV 来自 target 的哪一层"；Runner 只绑 cache，不复制。
        self.kv_sharing_target_layer_name: str | None = None
        #: 80E.E2/81：本层属于哪个 KV group（Runner 分配时写）。Q-only 层取源层的 group。
        self.kv_cache_group_id: int | None = None
        #: 滑窗大小（None = 全注意力）。模型构造时写入；Runner 用它给 group 建 spec。
        self.per_layer_sliding_window: int | None = None
        #: breakable PIECEWISE 的固定输出缓冲（runner 在 KV 分配后设置）。
        #: 图段之间靠**同一地址**传递 attention 输出；返回新张量会让下一段读旧内存。
        self._output_buffer: torch.Tensor | None = None

    def set_output_buffer(self, max_num_tokens: int, dtype: torch.dtype,
                          device: torch.device) -> None:
        self._output_buffer = torch.empty(
            max_num_tokens, self.num_heads * self.head_size,
            dtype=dtype, device=device)

    def bind_shared_kv_cache(self, target: "Attention", cache: torch.Tensor,
                             *, draft_dtype: torch.dtype | None = None) -> None:
        """把本层绑到 target 源层的物理 KV cache 上（80D 的只读契约）。

        这里只做**共享关系与兼容性检查**，不分配、不复制、不释放：
        分配/写入/释放都由 target 拥有，本层只读。检查项是"读同一份 cache 所需的
        最小集合"：

            KV heads / head_size    K/V 的形状必须一致；
            dtype / 布局             cache 的形状 `[2, blocks, block_size, kv_heads, head]`
                                     必须连续；draft 的浮点 dtype 必须与 cache 一致；
            Q heads                 **不检查**：GQA 允许 draft 的 Q 头数与 target 不同。

        源层不能自己也是 Q-only：那会把"读别人的 KV"串成一条链，映射/生命周期都不再
        是单跳；冻结映射只选非共享源层，这里把这条约束在执行侧再挡一次。
        """
        if target.q_only:
            raise ValueError(
                f"Q-only 层 {self.layer_name!r} 的共享源 {target.layer_name!r} 自己也是 "
                f"Q-only：源层必须真正拥有 KV，不能是另一个共享消费者")
        if not target.layer_name:
            raise ValueError("Q-only 共享源层没有 layer_name，无法建立稳定映射")
        if target.kv_cache is not cache:
            raise ValueError(
                f"Q-only 层 {self.layer_name!r} 的源 {target.layer_name!r} 绑定的 cache "
                f"与传入的不是同一对象；Runner 必须传源层实际持有的那份 cache")
        if cache.ndim != 5 or cache.shape[0] != 2:
            raise ValueError(
                f"KV cache 布局应为 [2, num_blocks, block_size, kv_heads, head_size]，"
                f"收到 shape={tuple(cache.shape)}")
        # packed 布局的 cache 在 block 维上是 stride 视图，整体 `is_contiguous()` 为
        # False；但一个 block 内的 [block, kv_heads, head] 必须连续。旧布局与 packed
        # 布局都满足“页内连续”，只是页之间 stride 不同。
        page_elems = cache.shape[2] * cache.shape[3] * cache.shape[4]
        if (cache.stride(2) != cache.shape[3] * cache.shape[4]
                or cache.stride(3) != cache.shape[4]
                or cache.stride(4) != 1
                or cache.stride(1) < page_elems):
            raise ValueError(
                f"KV cache 的页内布局必须连续，且块 stride 不小于一页："
                f"shape={tuple(cache.shape)} stride={cache.stride()}")
        if target.num_kv_heads != self.num_kv_heads:
            raise ValueError(
                f"Q-only 共享要求 KV heads 一致：draft {self.layer_name!r}="
                f"{self.num_kv_heads}，target {target.layer_name!r}={target.num_kv_heads}")
        if target.head_size != self.head_size:
            raise ValueError(
                f"Q-only 共享要求 head_size 一致：draft {self.layer_name!r}="
                f"{self.head_size}，target {target.layer_name!r}={target.head_size}")
        if cache.shape[3] != self.num_kv_heads or cache.shape[4] != self.head_size:
            raise ValueError(
                f"KV cache 的 KV heads/head_size 与 Q-only 层不符：cache="
                f"{tuple(cache.shape[3:])}，draft={(self.num_kv_heads, self.head_size)}")
        if draft_dtype is not None and cache.dtype != draft_dtype:
            raise ValueError(
                f"Q-only 共享要求 dtype 一致：cache={cache.dtype}，draft={draft_dtype}")

        self.kv_cache = cache
        self.q_only = True
        self.kv_sharing_target_layer_name = target.layer_name

    @eager_break_during_capture
    def forward(self, query, key, value):
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata.get(self.layer_name)
        if attn_metadata is None:
            raise KeyError(f"forward 上下文里没有层 {self.layer_name!r} 的 metadata；"
                           f"Runner 必须为每个 Attention 层都建一份（键是层名）")
        if self.kv_cache is None:
            raise RuntimeError(f"层 {self.layer_name!r} 的 kv_cache 还没绑定（Runner 初始化时绑定）")
        if self.q_only:
            forward_q_only = getattr(self.impl, "forward_q_only", None)
            if forward_q_only is None:
                raise NotImplementedError(
                    f"后端 {type(self.impl).__name__} 不支持 Q-only 共享 KV 读取；"
                    f"不能退回普通 forward（那会写 K/V，破坏只读契约）")
            result = forward_q_only(self, query, self.kv_cache, attn_metadata)
        else:
            result = self.impl.forward(
                self, query, key, value, self.kv_cache, attn_metadata)
        if self._output_buffer is None:
            return result
        out = self._output_buffer[: result.shape[0]]
        if result.data_ptr() == out.data_ptr():
            # 086：TRITON_ATTN 直接把结果写进这个 buffer；再 copy_ 一次会
            # 变成自拷贝（至少浪费带宽，部分版本还会报 overlap）。
            return out
        out.copy_(result)
        return out
