"""旋转位置编码（对应 vLLM `layers/rotary_embedding.py` 的子集）。

做法与 vLLM 一致：**预计算一张 cos/sin 表**（长度 max_position，形状 [max_pos, rotary_dim]，
前后两半相同），forward 时按 positions 查表——比每次算三角函数便宜，也保证"同一位置永远同一个值"。

约定（与 HF Qwen3 一致，`rotate_half` 风格）：

    emb = [freqs, freqs]                     # 维度上拼接一次，rotary_dim 维
    q_rot = q * cos + rotate_half(q) * sin
    rotate_half(x) = cat(-x[..., d/2:], x[..., :d/2])

**positions 是绝对位置**（不是"本轮的相对序号"）：chunked prefill 的第二块、decode 的第一枚，
都要用它在整条请求里的真实位置。
"""

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(self, head_size: int, rotary_dim: int, max_position_embeddings: int,
                 base: float = 10000.0, *, proportional: bool = False) -> None:
        super().__init__()
        if rotary_dim % 2 != 0:
            raise ValueError(f"rotary_dim 必须是偶数，收到 {rotary_dim}")
        if rotary_dim > head_size:
            raise ValueError(
                f"rotary_dim={rotary_dim} 不能大于 head_size={head_size}")
        self.head_size = head_size
        # `proportional` 是 Gemma4 全注意力层的 RoPE：频率指数的分母用 head_size，
        # 未旋转的维度用 cos=1/sin=0 零填充；因为整张 cache 是 head_size 宽，下面把
        # 实际作用维度设成 head_size（非旋转维做的是恒等旋转）。
        # 普通路径保持原行为：cache 宽 = rotary_dim，只旋转前 rotary_dim 维。
        if proportional:
            rope_angles = rotary_dim // 2
            nope_angles = head_size // 2 - rope_angles
            if nope_angles < 0:
                raise ValueError(
                    f"proportional RoPE 要求 rotary_dim <= head_size，"
                    f"收到 rotary_dim={rotary_dim}, head_size={head_size}")
            inv_freq = 1.0 / (
                base ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float32)
                         / head_size))
            if nope_angles:
                inv_freq = torch.cat(
                    (inv_freq, torch.zeros(nope_angles, dtype=torch.float32)))
            apply_dim = head_size
        else:
            inv_freq = 1.0 / (
                base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32)
                         / rotary_dim))
            apply_dim = rotary_dim
        self.rotary_dim = apply_dim
        positions = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)          # [max_pos, apply_dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)           # [max_pos, apply_dim]
        # 注册成 buffer（跟着模型走设备/精度），但不参与训练
        self.register_buffer("cos_sin_cache",
                             torch.cat((emb.cos(), emb.sin()), dim=-1), persistent=False)

    def _rotate(self, positions: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        num_tokens = x.shape[0]
        x = x.view(num_tokens, -1, self.head_size)
        cos, sin = self.cos_sin_cache.split(self.rotary_dim, dim=-1)
        cos = cos[positions].unsqueeze(1)                 # [num_tokens, 1, rotary_dim]
        sin = sin[positions].unsqueeze(1)
        return _apply_rotary(x, cos, sin, self.rotary_dim).view(num_tokens, -1)

    def forward(self, positions: torch.Tensor, query: torch.Tensor,
                key: torch.Tensor | None = None):
        """入参与返回都是**扁平**布局 `[num_tokens, num_heads * head_size]`（vLLM 的约定：
        qkv 投影出来就是扁的，不在调用点手动 reshape）。这里按 head_size 折开、旋转、再折回去。

        `key=None` 是 Gemma4 Q-only 层的合法输入：那类层不产生 K，也不需要旋转 K；
        上游 `RotaryEmbedding.forward` 同样允许 `key=None`。
        """
        q = self._rotate(positions, query)
        if key is None:
            return q, None
        return q, self._rotate(positions, key)

    def apply_to_query(self, positions: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """只旋转一组 Q（Gemma4 Q-only attention 用）。"""
        return self._rotate(positions, query)

    def apply_to_key(self, positions: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """只旋转一组 K（76 关 DFlash 的上下文 KV 预计算用）。

        为什么需要单独一个入口：DFlash 的上下文行**不过模型**，它们的 K 是在
        `precompute_and_store_context_kv()` 里一次性算好、直接写进缓存的；没有 query 可传，
        所以不能走 `forward()`（它要求 q 也非空）。数学与 `forward()` 里的 K 那一路完全相同。
        """
        return self._rotate(positions, key)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                  rotary_dim: int) -> torch.Tensor:
    """只旋转前 `rotary_dim` 维；其余维原样保留（partial rotary 的通用写法）。"""
    if rotary_dim == x.shape[-1]:
        return x * cos + _rotate_half(x) * sin
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    x_rot = x_rot * cos + _rotate_half(x_rot) * sin
    return torch.cat((x_rot, x_pass), dim=-1)
