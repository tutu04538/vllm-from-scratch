"""采样（对应 vLLM `v1/sample/` 的子集）。57B 只有最小贪心/温度采样，惩罚与 top_k/top_p 属 57D。"""

from .sampler import Sampler

__all__ = ["Sampler"]
