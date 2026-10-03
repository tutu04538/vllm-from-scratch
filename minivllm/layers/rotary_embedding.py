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
                 base: float = 10000.0) -> None:
        super().__init__()
        if rotary_dim % 2 != 0:
            raise ValueError(f"rotary_dim 必须是偶数，收到 {rotary_dim}")
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        # inv_freq[i] = base^(-2i/rotary_dim)
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32)
                                  / rotary_dim))
        positions = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)          # [max_pos, rotary_dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)           # [max_pos, rotary_dim]
        # 注册成 buffer（跟着模型走设备/精度），但不参与训练
        self.register_buffer("cos_sin_cache",
                             torch.cat((emb.cos(), emb.sin()), dim=-1), persistent=False)

    def forward(self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor):
        """入参与返回都是**扁平**布局 `[num_tokens, num_heads * head_size]`（vLLM 的约定：
        qkv 投影出来就是扁的，不在调用点手动 reshape）。这里按 head_size 折开、旋转、再折回去。
        """
        num_tokens = query.shape[0]
        q = query.view(num_tokens, -1, self.head_size)
        k = key.view(num_tokens, -1, self.head_size)
        cos, sin = self.cos_sin_cache.split(self.rotary_dim, dim=-1)
        cos = cos[positions].unsqueeze(1)                 # [num_tokens, 1, rotary_dim]
        sin = sin[positions].unsqueeze(1)
        return (_apply_rotary(q, cos, sin, self.rotary_dim).view(num_tokens, -1),
                _apply_rotary(k, cos, sin, self.rotary_dim).view(num_tokens, -1))


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
