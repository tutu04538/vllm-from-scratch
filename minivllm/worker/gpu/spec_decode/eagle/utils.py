"""EAGLE draft 的加载（对应 vLLM `v1/worker/gpu/spec_decode/eagle/utils.py`）。

上游在这里做三件事：按 draft 配置建模型、**共享 target 的 embedding**、**共享 target 的
lm_head**。本仓库的对应关系：

- 建模型：`minivllm.model_loader.get_model(draft_model_config, device)`（与 V1 提议者同一条加载路径）。
- 共享 embedding：由 draft 模型自己的 `share_embeddings(target_model)` 决定（EAGLE3 的
  checkpoint 里**没有** `embed_tokens` 时才共享；见 `models/qwen3_eagle3.py`）。
  这条与上游 `_should_share(..., "has_own_embed_tokens", ...)` 的判据语义相同，
  只是本仓库把"检查点里有没有这一份"写进了模型的加载器（缺就共享、有就用自己的）。
- **不共享 lm_head**：本仓库的 EAGLE3 draft 有自己的 draft 词表 `lm_head` + `d2t` 映射
  （63 关收尾），与 target 的 lm_head 宽度不同，共享会直接错。上游只在
  `_should_share(eagle_model, "has_own_lm_head", ...)` 为假时共享——本仓库的 draft
  `has_own_lm_head` 恒为真，所以这一段等价于"不共享"。
"""

import torch.nn as nn

from .....config import VllmConfig
from .....model_loader import get_model


def load_eagle_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config

    eagle_model = get_model(draft_model_config, vllm_config.device_config.device)

    # 与 target 共享词表嵌入（仅当 draft 的 checkpoint 里缺这一份）。
    if hasattr(eagle_model, "share_embeddings"):
        eagle_model.share_embeddings(target_model)

    return eagle_model
