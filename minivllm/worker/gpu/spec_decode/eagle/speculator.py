"""EAGLE/EAGLE3 的 V2 提议者（对应 vLLM `v1/worker/gpu/spec_decode/eagle/speculator.py`）。

上游这个文件只有一件事：把 `load_draft_model` 指到 `load_eagle_model`。提议循环全在
`AutoRegressiveSpeculator` 里（EAGLE 与 MTP 共用）——本仓库照抄这个划分。
"""

import torch.nn as nn

from ..autoregressive.speculator import AutoRegressiveSpeculator
from .utils import load_eagle_model


class EagleSpeculator(AutoRegressiveSpeculator):
    def load_draft_model(
        self,
        target_model: nn.Module,
    ) -> nn.Module:
        return load_eagle_model(target_model, self.vllm_config)
