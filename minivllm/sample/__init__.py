"""采样（对应 vLLM `v1/sample/`）。

    metadata.py                按行组织的采样参数（Sampler 不接 Request）
    ops/penalties.py           三种惩罚（会改变 argmax，必须在 greedy 之前）
    ops/topk_topp_sampler.py   top-k/top-p 筛选 + 指数竞赛抽样
    sampler.py                 顺序：惩罚 → 约束 → greedy/random 分流
    rejection_sampler.py       投机验证：Triton 批量拒绝采样（59）

普通采样与投机验证复用同一套 metadata 与惩罚/约束算子；拒绝采样对应上游
`vllm/v1/sample/rejection_sampler.py`，所以也放在本包。
"""

from .metadata import SAMPLING_EPS, SamplingMetadata
from .ops.penalties import apply_all_penalties
from .ops.topk_topp_sampler import TopKTopPSampler, apply_top_k_top_p, random_sample
from .rejection_sampler import (PLACEHOLDER_TOKEN_ID, RejectionSampler,
                                apply_sampling_constraints, expand_batch_to_tokens,
                                generate_uniform_probs, rejection_sample,
                                sample_recovered_tokens)
from .sampler import Sampler

__all__ = ["Sampler", "SamplingMetadata", "SAMPLING_EPS", "TopKTopPSampler",
           "apply_top_k_top_p", "random_sample", "apply_all_penalties",
           "RejectionSampler", "rejection_sample", "sample_recovered_tokens",
           "apply_sampling_constraints", "expand_batch_to_tokens",
           "generate_uniform_probs", "PLACEHOLDER_TOKEN_ID"]
