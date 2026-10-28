"""80G 教学对象：MTP 的 top-k 索引缓存与压紧契约。

真实 DeepSeek MTP 的 `compact_topk_indices()` 只做一件事：

    buffer[:num_slots] = buffer[slot_ids]

索引缓存归模型所有；proposer 只负责：
  - `set_skip_topk(False)`：step 0 自己算索引；
  - `compact_topk_indices(last_token_indices)`：把 step 0 的结果压到请求行；
  - `set_skip_topk(True)`：step 1+ 复用；
  - `set_skip_topk(False)`：结束恢复。

这里用固定的小 buffer 把语义钉住，不是稀疏 indexer 的数学实现。
"""

from __future__ import annotations

import torch


class TopKIndexBuffer:
    """每个 MTP 层一份 `[max_tokens, topk]` 的 int32 索引缓存。"""

    def __init__(self, max_tokens: int, topk: int, num_layers: int = 1) -> None:
        if max_tokens <= 0 or topk <= 0 or num_layers <= 0:
            raise ValueError("max_tokens/topk/num_layers 必须为正")
        self.max_tokens = max_tokens
        self.topk = topk
        self.buffers = [
            torch.zeros(max_tokens, topk, dtype=torch.int32)
            for _ in range(num_layers)
        ]
        self.skip_topk = False
        self.compute_calls = 0

    def set_skip_topk(self, skip: bool) -> None:
        self.skip_topk = bool(skip)

    def compute_indices(self, num_tokens: int,
                        layer: int = 0) -> torch.Tensor:
        """模拟 indexer 计算；`skip_topk=True` 时直接返回旧缓冲，不增加计数。"""
        if not 0 <= num_tokens <= self.max_tokens:
            raise ValueError(f"num_tokens={num_tokens} 超出 [0,{self.max_tokens}]")
        buffer = self.buffers[layer]
        if not self.skip_topk:
            self.compute_calls += 1
            # 只为测试提供可辨认的内容：第 t 行全是 t。
            buffer[:num_tokens] = torch.arange(
                num_tokens, dtype=torch.int32)[:, None]
        return buffer[:num_tokens]

    def compact_topk_indices(self, slot_ids: torch.Tensor) -> None:
        """对应 DeepSeek `compact_topk_indices`：先读 RHS，再写目标行。"""
        slot_ids = slot_ids.to(dtype=torch.int64)
        if slot_ids.ndim != 1:
            raise ValueError(f"slot_ids 必须是一维，收到 {tuple(slot_ids.shape)}")
        if slot_ids.numel() > self.max_tokens:
            raise ValueError("slot_ids 数量超过 max_tokens")
        if slot_ids.numel() and (int(slot_ids.min()) < 0
                                 or int(slot_ids.max()) >= self.max_tokens):
            raise IndexError("slot_ids 越界")
        num_slots = slot_ids.numel()
        source = [buffer[slot_ids].clone() for buffer in self.buffers]
        for buffer, rows in zip(self.buffers, source):
            buffer[:num_slots].copy_(rows)


class TeachingSparseMTPInner:
    """模拟 `model.model`：模型内部持有索引缓存并提供两个契约方法。"""

    def __init__(self, max_tokens: int = 16, topk: int = 4,
                 num_layers: int = 2) -> None:
        self.index_buffer = TopKIndexBuffer(max_tokens, topk, num_layers)

    def set_skip_topk(self, skip: bool) -> None:
        self.index_buffer.set_skip_topk(skip)

    def compact_topk_indices(self, slot_ids: torch.Tensor) -> None:
        self.index_buffer.compact_topk_indices(slot_ids)

    def compute_indices(self, num_tokens: int,
                        layer: int = 0) -> torch.Tensor:
        return self.index_buffer.compute_indices(num_tokens, layer)


class TeachingSparseMTPOuter:
    """模拟 `MTPSpeculator.model`：外层持有 `.model` 的模型包装。"""

    def __init__(self, inner: TeachingSparseMTPInner | None = None) -> None:
        self.model = inner or TeachingSparseMTPInner()
