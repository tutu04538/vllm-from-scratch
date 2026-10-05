"""`LogitsProcessor`（对应 vLLM `model_executor/layers/logits_processor.py` 的子集）。

它在 vLLM 里的职责是"hidden states → 词表 logits"这一步的**收尾**，一步步是：

    logits = lm_head(hidden_states)          # 词表 GEMM
    logits = logits[..., :org_vocab_size]    # 去掉词表 padding（TP 分片对齐用）
    logits = logits * scale                  # 有些模型的 logit 要缩放

本仓库 **TP=1、不量化、无 LoRA、无 soft cap**，所以只保留这三步里对得上号的部分：
`org_vocab_size` 的切片与 `scale`。上游还会在这里做 TP 的 gather/all-gather
（`lm_head.tp_size > 1`）与 `head_dtype` 的处理，本仓库没有第二条 rank，不存在。

**第一个调用方是 66 关的 Medusa**（`models/medusa.py`）：它的每个 head 都配一个
`LogitsProcessor(vocab_size, truncated_vocab_size, logit_scale)`——`truncated_vocab_size`
是"只在最常用的一小撮 token 上算草稿"的截断词表，切片就发生在这里。
"""

import torch
from torch import nn


class LogitsProcessor(nn.Module):
    """词表 GEMM 的收尾（切片 + 缩放）。签名与上游一致，参数按本仓库的用到的子集收窄。"""

    def __init__(self, vocab_size: int, org_vocab_size: int | None = None,
                 scale: float = 1.0) -> None:
        super().__init__()
        self.scale = scale
        self.vocab_size = vocab_size
        # 上游：`self.org_vocab_size = org_vocab_size or vocab_size`（`None` = 没有截断）
        self.org_vocab_size = org_vocab_size or vocab_size

    def forward(self, lm_head: nn.Module, hidden_states: torch.Tensor,
                embedding_bias: torch.Tensor | None = None) -> torch.Tensor:
        logits = lm_head(hidden_states)
        if embedding_bias is not None:
            logits = logits + embedding_bias
        # 去掉词表 padding（上游 `logits[..., : self.org_vocab_size]`）
        logits = logits[..., : self.org_vocab_size]
        if self.scale != 1.0:
            logits = logits * self.scale
        return logits
