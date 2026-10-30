"""RMSNorm：带“残差相加”的融合签名（对应 vLLM `layers/layernorm.py::RMSNorm`）。

vLLM 的约定是 `forward(x, residual=None) -> out | (out, residual)`：

- `residual is None`：只做归一化 → 返回 `out`
- 给了 `residual`：先 `x = x + residual`，再归一化 → 返回 `(norm(x), x)`

第二段才是关键：层与层之间**不把归一化后的张量当作残差传下去**，残差是“加过、还没归一化”的
那份。省掉一次显式的加法 kernel，也让模型结构（`layers[i].forward(pos, hidden, residual)`）
天然把“残差流”表达成一个独立参数。

数值上对齐 HF/上游 native：方差在 **float32** 里归约；权重乘法的 cast 次序在
`layers/ops/layernorm.py` 里逐项对照。CUDA 走本地 Triton 融合 kernel，CPU 走 Torch 参考。
`residual` 分支返回**新的** residual 张量，不原地改写调用方的输入（上游 native 同义；
上游 CUDA 的自定义 op 会原地写，差异记在 `docs/step85_alignment.md`）。
"""

import torch
from torch import nn
from torch.nn.parameter import Parameter

from .ops.layernorm import rms_norm


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6,
                 has_weight: bool = True) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.has_weight = has_weight
        if has_weight:
            self.weight = Parameter(torch.ones(hidden_size))
        else:
            # 无权重分支不保留全 1 向量：kernel/native 都走 HAS_WEIGHT=False，
            # 不会为了省一次判断而每层多存一份 hidden 大小的参数。
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        return rms_norm(
            x,
            self.weight if self.has_weight else None,
            self.variance_epsilon,
            residual,
            add_one=False,
        )

    def extra_repr(self) -> str:
        return (f"hidden_size={self.hidden_size}, eps={self.variance_epsilon}, "
                f"has_weight={self.has_weight}")


class GemmaRMSNorm(RMSNorm):
    """Gemma 系 RMSNorm：`x * (1 + w)`，权重零初始化（对应上游 `GemmaRMSNorm`）。

    与 `RMSNorm` 的 `x * w` 差一个 1。Step3p5 的 checkpoint（含 MTP 的 enorm/hnorm/
    shared_head.norm）按零中心存权重，用普通 `RMSNorm` 会把整层乘成 0——不会报错，只会
    让输出全错。这里把差异做成一个明确的类，不靠调用方记得传参数。

    上游还多一条 cast 差异：先 `(x * (1+w))` 在 weight 的 dtype（float32）里算，最后才回
    orig_dtype，而不是先回 orig_dtype 再乘权重。本类通过 `add_one=True` 保留这条次序。
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__(hidden_size, eps=eps, has_weight=True)
        with torch.no_grad():
            self.weight.zero_()

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        return rms_norm(
            x,
            self.weight,
            self.variance_epsilon,
            residual,
            add_one=True,
        )


__all__ = ["RMSNorm", "GemmaRMSNorm"]
