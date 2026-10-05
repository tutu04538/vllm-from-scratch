"""投机（对应 vLLM `v1/spec_decode/`）。

    metadata.py           一轮验证的索引：两个坐标系（forward 行 / 取完的 logits）
    metrics.py            接受率统计（只统计已验证的候选）
    ngram_proposer.py     从历史里找重复片段当草稿（确定性，没有 q；CPU）
    ngram_proposer_gpu.py 同一套匹配的 GPU 版：显存常驻历史 + 增量写入 + 有效个数
    draft_model.py        用小模型提议（SpecDecodeBaseProposer + DraftModelProposer）
    extract_hidden_states.py  64 关：cache-only 特征提取（不猜 token，借 KV 缓存存特征）
    medusa.py                 66 关：Medusa 多头提议（N 个纯 MLP head 并行读同一份 target hidden）

**时序**（199 §4）：轮 t 验证时顺手提草稿 → `post_step` 取回 → 轮 t+1 才采用。
提议者**不改本轮计划**，也没有自己的调度器：它是"历史进、草稿出"的一段计算。

**KV**（199 §9）：draft 与 target 共用逻辑块表与 slot 编号（同一个 KV group），
但每个 Attention 层绑自己的物理 tensor；规格不兼容时**明确报错**，不给独立 pool 兜底。

**验证不在这里**（59 关）：拒绝采样对应上游 `vllm/v1/sample/rejection_sampler.py`，所以本项目的
生产实现放在 `minivllm/sample/rejection_sampler.py`（Torch 参考版在 `minivllm/testing/`）。
本包只管"怎么提草稿"和"草稿的索引/统计"。
"""

from .draft_model import DraftModelProposer, SpecDecodeBaseProposer
from .extract_hidden_states import ExtractHiddenStatesProposer
from .metadata import SpecDecodeMetadata
from .metrics import SpecDecodingStats
from .ngram_proposer import NgramProposer
from .ngram_proposer_gpu import NgramProposerGPU

__all__ = ["SpecDecodeMetadata", "SpecDecodingStats", "NgramProposer", "NgramProposerGPU",
           "SpecDecodeBaseProposer", "DraftModelProposer", "ExtractHiddenStatesProposer"]
