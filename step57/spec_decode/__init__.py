"""投机（对应 vLLM `v1/spec_decode/`）。

    metadata.py           一轮验证的索引：两个坐标系（forward 行 / 取完的 logits）
    rejection_sampler.py  验证草稿：greedy 比对 argmax、random 用 min(1, p/q)
    ngram_proposer.py     从历史里找重复片段当草稿（确定性，没有 q）
    draft_model.py        用小模型提议（SpecDecodeBaseProposer + DraftModelProposer）

**时序**（199 §4）：轮 t 验证时顺手提草稿 → `post_step` 取回 → 轮 t+1 才采用。
提议者**不改本轮计划**，也没有自己的调度器：它是"历史进、草稿出"的一段计算。

**KV**（199 §9）：draft 与 target 共用逻辑块表与 slot 编号（同一个 KV group），
但每个 Attention 层绑自己的物理 tensor；规格不兼容时**明确报错**，不给独立 pool 兜底。
"""

from .draft_model import DraftModelProposer, SpecDecodeBaseProposer
from .metadata import SpecDecodeMetadata
from .ngram_proposer import NgramProposer
from .rejection_sampler import PLACEHOLDER_TOKEN_ID, RejectionSampler, expand_batch_to_tokens

__all__ = ["SpecDecodeMetadata", "RejectionSampler", "PLACEHOLDER_TOKEN_ID",
           "expand_batch_to_tokens", "NgramProposer", "SpecDecodeBaseProposer",
           "DraftModelProposer"]
