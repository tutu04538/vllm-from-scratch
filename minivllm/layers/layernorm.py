"""RMSNorm：带"残差相加"的融合签名（对应 vLLM `layers/layernorm.py::RMSNorm`）。

vLLM 的约定是 `forward(x, residual=None) -> out | (out, residual)`：

- `residual is None`：只做归一化 → 返回 `out`
- 给了 `residual`：先 `x = x + residual`，再归一化 → 返回 `(norm(x), x)`

第二段才是关键：层与层之间**不把归一化后的张量当作残差传下去**，残差是"加过、还没归一化"的
那份。省掉一次显式的加法 kernel，也让模型结构（`layers[i].forward(pos, hidden, residual)`）
天然把"残差流"表达成一个独立参数。

数值上对齐 HF：方差在 **float32** 里算（BF16 直接算方差会明显偏），最后再回原 dtype。
"""

import torch
from torch import nn
from torch.nn.parameter import Parameter


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.weight = Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        if residual is not None:
            x = x + residual
            residual = x
        input_dtype = x.dtype
        # 方差用 float32 算：bf16 下 pow(2).mean() 的累积误差足以让 logits 对不上参考实现
        x32 = x.to(torch.float32)
        variance = x32.pow(2).mean(-1, keepdim=True)
        out = x32 * torch.rsqrt(variance + self.variance_epsilon)
        out = out.to(input_dtype) * self.weight
        if residual is None:
            return out
        return out, residual
