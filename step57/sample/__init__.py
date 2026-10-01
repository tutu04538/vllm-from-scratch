"""采样（对应 vLLM `v1/sample/`）。

    metadata.py                按行组织的采样参数（Sampler 不接 Request）
    ops/penalties.py           三种惩罚（会改变 argmax，必须在 greedy 之前）
    ops/topk_topp_sampler.py   top-k/top-p 筛选 + 指数竞赛抽样
    sampler.py                 顺序：约束 → 惩罚 → greedy/random 分流

57D 只做普通采样与停止；投机验证（57E）会复用同一套 metadata 与惩罚算子。
"""

from .metadata import SAMPLING_EPS, SamplingMetadata
from .ops.penalties import apply_all_penalties
from .ops.topk_topp_sampler import TopKTopPSampler, apply_top_k_top_p, random_sample
from .sampler import Sampler

__all__ = ["Sampler", "SamplingMetadata", "SAMPLING_EPS", "TopKTopPSampler",
           "apply_top_k_top_p", "random_sample", "apply_all_penalties"]
