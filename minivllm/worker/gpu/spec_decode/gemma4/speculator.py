"""V2 Gemma4 assistant 提议者（80E.E2；对应冻结
`v1/worker/gpu/spec_decode/gemma4/speculator.py::Gemma4Speculator`）。

Gemma4 assistant 与 EAGLE/MTP 的输入循环相同（吃 target hidden、逐草稿步跑一次），
但有三条**不能靠父类默认值**的差异：

1. **位置不前进**：每步都从 target 最后一个位置预测（`advance_draft_positions=False`）。
   父类已经把这条做成属性；这里只覆盖它。位置若每步 +1，Q-only 层会去读 target 根本
   没写过的位置，得到"形状对、上下文错"的草稿。
2. **不写 KV**：所有 assistant 层是 Q-only，Runner 把它们的 cache 绑到 target 源层；
   这里不需要（也不允许）分配第二份 draft KV。
3. **宽度不同**：assistant hidden=256、backbone hidden=1536；`pre_projection` 吃
   `[embedding(1536) ‖ target hidden(1536)]`，`post_projection` 把 256 投回 1536 供下步反馈。
   embedding 必须用 target 的 backbone 宽 embedding 替换，而不是同形状共享。

本类只接线；模型数学在 `models/gemma4_mtp.py`，Q-only 读路径在 `attention/`。
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .....attention.layer import Attention
from .....model_loader import get_model
from .kv_sharing import setup_gemma4_kv_sharing
from ..autoregressive.speculator import AutoRegressiveSpeculator


class Gemma4Speculator(AutoRegressiveSpeculator):
    @property
    def advance_draft_positions(self) -> bool:
        return False

    @property
    def draft_attention_backend(self):
        """assistant 的 metadata 组跟随 target 选定的后端。

        086 已让 assistant 的 Q-only Impl 使用 target 的 TRITON_ATTN；087 把 metadata
        组也切过去，保证 `workspace_pool`、`update_draft_decode_metadata()` 与 Impl 是
        同一套契约。target 选 Torch 时仍返回 Torch，不强行升级。
        """
        from .....attention.backends import get_attention_backend

        return get_attention_backend(self.vllm_config.attention_backend)

    # -------- 模型加载与接线 --------

    def load_draft_model(self, target_model: nn.Module) -> nn.Module:
        """加载 assistant，并完成 target→draft 的 embedding / Q-only KV 接线。"""
        from .....models.gemma4_mtp import _text_config

        draft_model = get_model(
            self.speculative_config.draft_model_config, self.device)
        target_names = {
            name for name, module in target_model.named_modules()
            if isinstance(module, Attention)
        }
        target_text = _text_config(self.vllm_config.model_config.hf_config)
        draft_text = _text_config(
            self.speculative_config.draft_model_config.hf_config)
        setup_gemma4_kv_sharing(
            draft_model, target_names, draft_text, target_text)
        self._share_embeddings(draft_model, target_model)
        return draft_model

    def _share_embeddings(self, draft_model: nn.Module,
                          target_model: nn.Module) -> None:
        """把 target 的 backbone 宽 embedding 接到 assistant（对应冻结 `_share_embeddings`）。

        上游没有"维度必须相等"的检查（基础类的 `_maybe_share_embeddings` 有，但 Gemma4
        显式绕过它）；这里同样只检查**宽度必须是 assistant 的 backbone**，不要求等于
        assistant hidden=256。assistant 自己的 `lm_head` 保持 draft 宽，不能换成 target 的。
        """
        target_inner = getattr(target_model, "model", None)
        target_embed = getattr(target_inner, "embed_tokens", None)
        if target_embed is None:
            raise AttributeError(
                "Gemma4 target 没有 `model.embed_tokens`：无法共享 backbone 词嵌入")
        draft_inner = getattr(draft_model, "model", None)
        if draft_inner is None or not hasattr(draft_inner, "embed_tokens"):
            raise AttributeError("Gemma4 assistant 没有 `model.embed_tokens`")
        backbone = int(getattr(draft_model.model, "backbone_hidden_size", 0))
        if int(target_embed.embedding_dim) != backbone:
            raise ValueError(
                f"target embed_tokens 宽度 {target_embed.embedding_dim} 与 assistant "
                f"backbone_hidden_size={backbone} 不一致：拼 `[embedding ‖ hidden]` 会错位")
        del draft_inner.embed_tokens
        draft_inner.embed_tokens = target_embed

    # -------- 采样 --------

    def _greedy_sample_draft(self, hidden_states: torch.Tensor,
                             spec_step_idx: int = 0) -> torch.Tensor:
        """centroid 路径优先用 `get_top_tokens`（不物化完整词表）。

        对纯 greedy 来说，sparse 选簇后的 argmax 与完整 logits 的 argmax 相同；用
        `get_top_tokens` 可以避免在真实 262144 词表上白算一份 `[T, vocab]`。非
        ordered-embedding 检查点仍走父类的完整 logits。
        """
        masked = getattr(self.model, "masked_embedding", None)
        if masked is not None and self.model.use_ordered_embeddings:
            return self.model.get_top_tokens(hidden_states)
        return super()._greedy_sample_draft(
            hidden_states, spec_step_idx=spec_step_idx)


__all__ = ["Gemma4Speculator"]
