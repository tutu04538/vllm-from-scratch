"""`LogitsProcessor`（对应 vLLM `model_executor/layers/logits_processor.py` 的子集）。

它在 vLLM 里的职责是"hidden states → 词表 logits"这一步的**收尾**，一步步是：

    logits = lm_head(hidden_states)          # 词表 GEMM
    logits = logits[..., :org_vocab_size]    # 去掉词表 padding（TP 分片对齐用）
    logits = logits * scale                  # 有些模型的 logit 要缩放

本仓库普通原始词表在 TP 路径按冻结 CUDA 平台 all-gather 恢复完整 logits，
随后去掉 padding、soft cap、scale；不支持量化与 LoRA 词表分片。

**调用方**：
- 66 关的 Medusa（`models/medusa.py`）：每个 head 一个 `LogitsProcessor`，
  `truncated_vocab_size` 是"只在最常用的一小撮 token 上算草稿"的截断词表，切片就发生在这里。
- 78 关的 DFlash2（`models/qwen3_dflash2.py`）：`get_top_k_tokens()` 是候选选择器的第一步
  ——**不做整词表 logits、只取 top-k 的 (id, value)**，并按 checkpoint 的
  `output_multiplier` / `final_logit_softcapping` 收尾。
"""

import torch
from torch import nn


class LogitsProcessor(nn.Module):
    """词表 GEMM 的收尾（切片 + 缩放 + soft cap）。签名与上游一致，参数按本仓库用到的子集收窄。"""

    def __init__(self, vocab_size: int, org_vocab_size: int | None = None,
                 scale: float = 1.0, soft_cap: float | None = None) -> None:
        super().__init__()
        self.scale = scale
        self.vocab_size = vocab_size
        # 上游：`self.org_vocab_size = org_vocab_size or vocab_size`（`None` = 没有截断）
        self.org_vocab_size = org_vocab_size or vocab_size
        # 上游 `LogitsProcessor.soft_cap`（Gemma2 系用）。78 关的 DFlash2 候选打分按
        # checkpoint 的 `dflash_config.final_logit_softcapping` 打开它。
        self.soft_cap = soft_cap

    def forward(self, lm_head: nn.Module, hidden_states: torch.Tensor,
                embedding_bias: torch.Tensor | None = None) -> torch.Tensor:
        logits = lm_head(hidden_states)
        if embedding_bias is not None:
            logits = logits + embedding_bias
        # 冻结 CUDA 平台每个 rank 都获得完整词表 logits。
        if getattr(lm_head, "tp_size", 1) > 1:
            logits = lm_head.group.all_gather(logits, dim=-1)
        # 去掉词表 padding（上游 `logits[..., : self.org_vocab_size]`）
        logits = logits[..., : self.org_vocab_size]
        # 上游 forward 的顺序：先 soft cap、再 scale（与下面的 get_top_k_tokens 相反——
        # 那是上游自己的顺序，两个入口都照抄，不"顺手统一"）。
        if self.soft_cap is not None:
            logits = torch.tanh(logits / self.soft_cap) * self.soft_cap
        if self.scale != 1.0:
            logits = logits * self.scale
        return logits

    def get_top_k_tokens(self, lm_head: nn.Module, hidden_states: torch.Tensor, k: int,
                         embedding_bias: torch.Tensor | None = None
                         ) -> tuple[torch.Tensor, torch.Tensor]:
        """只取词表 top-k 的 `(ids, values)`（上游同名方法的 TP=1 等价物）。

        上游为什么要它：整份 `[B, V]` logits 的 all-gather 在 TP 下是 O(B·V) 通信，
        而候选选择器只需要 k 个候选；这条路径把通信量压到 O(B·2k·TP)。
        本仓库仅开放 TP=1 的候选选择：先按 `lm_head` 原 dtype 选 top-k，
        再 `float()`、乘 `scale`、过 soft cap —— 顺序与上游 `get_top_k_tokens` 逐行一致。

        物理权重可能补齐到 64 行，选择之前按真实词表大小排除补齐行。
        """
        if self.scale <= 0.0 and self.scale != 1.0:
            raise ValueError(
                "非正的 logit 缩放因子不支持 top-k 归约（上游同一句检查）："
                f"scale={self.scale}")
        if getattr(lm_head, "tp_size", 1) > 1:
            raise NotImplementedError("多 rank 局部 top-k 归约不属于 099")
        logits = lm_head(hidden_states)
        if embedding_bias is not None:
            logits = logits + embedding_bias
        logits = logits[..., :getattr(lm_head, "num_embeddings", logits.shape[-1])]
        values, ids = torch.topk(logits, k, dim=-1)
        values = values.float()
        if self.scale != 1.0:
            values = values * self.scale
        if self.soft_cap is not None:
            values = torch.tanh(values / self.soft_cap) * self.soft_cap
        return ids, values
