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
