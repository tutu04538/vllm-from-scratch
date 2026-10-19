"""DSpark 的模型加载（对应 vLLM `v1/worker/gpu/spec_decode/dspark/utils.py`）。

上游在这一支做四件事：按 draft 配置重建一份（非因果 / KV dtype 覆盖）的 vllm_config、
用 draft 的量化配置覆盖、建模型、**与 target 共享 `embed_tokens` / `lm_head`**，外加一条
"不支持 PP"的硬检查。本仓库的对应关系：

    非因果 / 后端选择   本仓库只有一个注意力后端（`TorchAttentionBackend`），而"哪一层因果"
                        是**模型**按配置决定的（`get_draft_attn_causal()`）——没有
                        `attention_config.use_non_causal` 这一轴可设
    KV cache dtype     本仓库没有 `kv_cache_dtype` 配置轴（无量化 KV）
    quant config       本仓库没有量化轴
    共享 embed / lm_head  照抄：checkpoint 里**没有**这两份（或与 target 完全相同时）就把 target
                        的模块挂上去。判据与上游 `_should_share()` 同义，flag 由
                        `Qwen3DSparkForCausalLM.load_weights()` 在加载时记下
    PP 检查             照抄上游那条 `NotImplementedError`；本仓库没有分布式 PP 轴，所以
                        "PP 大小"从 `vllm_config.parallel_config` 读（缺省 1）
"""

import torch
import torch.nn as nn

from .....config import VllmConfig
from .....model_loader import get_model


def _should_share(draft: nn.Module, flag: str, draft_layer: nn.Module | None,
                  target_layer: nn.Module | None) -> bool:
    """checkpoint 没有自己的一份、或自己那份与 target 完全相同时共享（上游 `_should_share`）。

    `flag` 由模型的 `load_weights()` 写：`has_own_embed_tokens` / `has_own_lm_head`。
    "有自己的一份但与 target 逐元素相同"也共享——省一份显存，语义不变（上游同款）。
    """
    if not getattr(draft, flag, False) or draft_layer is None:
        return True
    if target_layer is None:
        return False
    return torch.equal(draft_layer.weight, target_layer.weight)


def _target_lm_head(target_model: nn.Module, target_language_model: nn.Module):
    """target 的 lm_head：`*ForConditionalGeneration` 走 `get_language_model()`，否则顶层。"""
    return (getattr(target_language_model, "lm_head", None)
            or getattr(target_model, "lm_head", None))


def _pipeline_parallel_size(vllm_config: VllmConfig) -> int:
    """PP 大小。本仓库没有 `ParallelConfig` 这一轴 → 缺省 1（有则按上游口径读）。"""
    parallel_config = getattr(vllm_config, "parallel_config", None)
    return int(getattr(parallel_config, "pipeline_parallel_size", 1) or 1)


def load_dspark_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config

    # 与上游同一条硬检查：DSpark 的串行 Markov 采样是**逐请求**的，跨 PP 段拿不到上一段刚
    # 采出的 token。放在建模**之前**报错（本仓库没有 PP 通路，早报错少一次白建模型；
    # 语义与上游相同）。
    if _pipeline_parallel_size(vllm_config) != 1:
        raise NotImplementedError("DSpark does not support pipeline parallelism.")

    draft_model = get_model(draft_model_config, vllm_config.device_config.device)

    target_language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    target_inner = target_language_model.model
    draft_inner = draft_model.model

    target_embed = getattr(target_inner, "embed_tokens", None)
    draft_embed = getattr(draft_inner, "embed_tokens", None)
    if target_embed is not None and _should_share(
            draft_model, "has_own_embed_tokens", draft_embed, target_embed):
        if draft_embed is not None:
            del draft_inner.embed_tokens
        draft_inner.embed_tokens = target_embed

    target_lm_head = _target_lm_head(target_model, target_language_model)
    draft_lm_head = getattr(draft_model, "lm_head", None)
    if target_lm_head is not None and _should_share(
            draft_model, "has_own_lm_head", draft_lm_head, target_lm_head):
        if draft_lm_head is not None:
            del draft_model.lm_head
        draft_model.lm_head = target_lm_head

    return draft_model
