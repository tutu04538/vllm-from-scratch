"""激活层（对应 vLLM `layers/activation.py` 的子集）：本关只用到 SiLU×mul 的合并形式。

`gate_up_proj` 一次算出 [gate | up]，`SiluAndMul` 把它折成 `silu(gate) * up` —— 这就是为什么
gate/up 要打包成一次 GEMM：中间结果从不落地成两个独立张量。
"""

import torch
import torch.nn.functional as F
from torch import nn


class SiluAndMul(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] % 2 != 0:
            raise ValueError(f"SiluAndMul 需要最后一维是偶数（gate|up 各占一半），收到 {tuple(x.shape)}")
        gate, up = x.chunk(2, dim=-1)
        return F.silu(gate) * up


class SwigluStepAndMul(nn.Module):
    """`silu(gate).clamp(max=limit) * up.clamp(-limit, limit)`（上游同名激活）。"""

    def __init__(self, limit: float = 7.0) -> None:
        super().__init__()
        if limit is None:
            raise ValueError("SwigluStepAndMul 需要 limit")
        self.limit = float(limit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = x.chunk(2, dim=-1)
        gate = F.silu(gate).clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        return gate * up
