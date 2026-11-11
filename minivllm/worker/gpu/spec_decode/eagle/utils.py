"""EAGLE draft 的加载（对应 vLLM `v1/worker/gpu/spec_decode/eagle/utils.py`）。

上游在这里做三件事：按 draft 配置建模型、**共享 target 的 embedding**、**共享 target 的
lm_head**。本仓库的对应关系：

- 建模型：`minivllm.model_loader.get_model(draft_model_config, device)`（与 V1 提议者同一条加载路径）。
- 共享 embedding 和 lm_head：对登记自有权重标志的 EAGLE3 使用冻结的 `_should_share`
  规则；其他已有模型保留原有加载路径。真实缩减词表 head 与 target 不同，不应共享。
"""

import torch
import torch.nn as nn

from .....config import VllmConfig
from .....model_loader import get_model


def _should_share(eagle: nn.Module, flag: str, draft, target) -> bool:
    """对齐冻结上游：无自有权重，或自有权重与目标逐值相同。"""
    if not getattr(eagle, flag, False) or draft is None:
        return True
    if target is None:
        return False
    weight = draft.weight
    if weight.is_cuda and torch.cuda.mem_get_info(weight.device)[0] < weight.numel() * 2:
        return torch.equal(weight.cpu(), target.weight.cpu())
    return torch.equal(weight, target.weight)


def load_eagle_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config

    eagle_model = get_model(draft_model_config, vllm_config.device_config.device)

    # 只有 EAGLE3 加载器登记了 own 标志；保留其他已有草稿的兼容路径。
    if not hasattr(eagle_model, "has_own_embed_tokens"):
        if hasattr(eagle_model, "share_embeddings"):
            eagle_model.share_embeddings(target_model)
        return eagle_model
    language_model = (target_model.get_language_model()
                      if hasattr(target_model, "get_language_model") else target_model)
    target_inner = language_model.model
    target_embed = getattr(target_inner, "embed_tokens", None)
    draft_embed = getattr(eagle_model.model, "embed_tokens", None)
    if target_embed is not None and _should_share(
            eagle_model, "has_own_embed_tokens", draft_embed, target_embed):
        eagle_model.model.embed_tokens = target_embed
    target_head = getattr(language_model, "lm_head", None) or getattr(target_model, "lm_head", None)
    draft_head = getattr(eagle_model, "lm_head", None)
    if target_head is not None and _should_share(
            eagle_model, "has_own_lm_head", draft_head, target_head):
        eagle_model.lm_head = target_head

    return eagle_model
