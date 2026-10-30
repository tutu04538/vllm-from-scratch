"""激活层（对应 vLLM `layers/activation.py` 的子集）。

`gate_up_proj` 一次算出 `[gate | up]`，`GeluAndMul` / `SiluAndMul` 把它折成
`act(gate) * up`——这就是为什么 gate/up 要打包成一次 GEMM：中间结果从不落地成两个
独立张量。CUDA 走 `layers/ops/activation.py` 的融合 kernel，CPU 走 Torch 参考。

Gemma4 默认 `gelu_pytorch_tanh`，对应 `GeluAndMul(approximate="tanh")`；不能和 exact
erf 版互换。`SwigluStepAndMul` 的 clamp 语义单独保留，不塞进这两个类里。
"""

import torch
import torch.nn.functional as F
from torch import nn

from .ops.activation import activation_and_mul


class GeluAndMul(nn.Module):
    """GeGLU：`gelu(gate) * up`，输入最后一维是 `[gate | up]`。"""

    def __init__(self, approximate: str = "none") -> None:
        super().__init__()
        if approximate not in ("none", "tanh"):
            raise ValueError(
                f"GeluAndMul 的 approximate 只能是 'none' 或 'tanh'，"
                f"收到 {approximate!r}")
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return activation_and_mul(
            x, gelu=True, approximate=self.approximate)

    def extra_repr(self) -> str:
        return f"approximate={self.approximate!r}"


class SiluAndMul(nn.Module):
    """SwiGLU：`silu(gate) * up`，输入最后一维是 `[gate | up]`。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return activation_and_mul(x, gelu=False)


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


_ACTIVATION_AND_MUL = {
    "gelu": lambda: GeluAndMul("none"),
    "gelu_pytorch_tanh": lambda: GeluAndMul("tanh"),
    "geglu": lambda: GeluAndMul("none"),
    "silu": lambda: SiluAndMul(),
    "swish": lambda: SiluAndMul(),
}


def get_act_and_mul_fn(name: str) -> nn.Module:
    """按 checkpoint 的 `hidden_activation` 选 `...AndMul`（对应上游同名工厂）。"""
    key = str(name).lower()
    try:
        return _ACTIVATION_AND_MUL[key]()
    except KeyError:
        raise ValueError(f"激活函数 {name!r} 未适配") from None


__all__ = [
    "GeluAndMul",
    "SiluAndMul",
    "SwigluStepAndMul",
    "get_act_and_mul_fn",
]
