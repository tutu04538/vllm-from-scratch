"""Attention 的元数据（对应 vLLM `v1/attention/backends/*.py` 里各后端的 AttentionMetadata）。

**字段是本关子集**，不冒充所有后端共用的统一签名。四个字段刚好覆盖"分页 KV + 变长 batch"：

    query_start_loc  [num_reqs + 1]  每个请求本轮的 query 在扁平张量里的起止（前缀和）
    seq_lens         [num_reqs]      每个请求**本轮之后**的上下文长度（含本轮 query）
    block_table      [num_reqs, max_blocks_per_req]  逻辑块号 → 物理块号
    slot_mapping     [num_tokens]    每个输入 token 的 KV 写到哪个物理槽位

`slot_mapping[i] = block_table[req, pos // block_size] * block_size + pos % block_size`，
`pos` 是该 token 的**绝对位置**——这条式子是"分页 KV 写在哪"的唯一答案，测试里有专门用例
（对齐 198 §4 的数字）。
"""

from dataclasses import dataclass

import torch


@dataclass
class AttentionMetadata:
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    block_size: int
    num_reqs: int


class AttentionMetadataBuilder:
    """从 InputBatch + 块表构造 metadata。**只做张量准备，不做模型数学**。"""

    def __init__(self, block_size: int) -> None:
        self.block_size = block_size

    def build(self, query_start_loc: torch.Tensor, seq_lens: torch.Tensor,
              block_table: torch.Tensor, slot_mapping: torch.Tensor,
              num_reqs: int) -> AttentionMetadata:
        return AttentionMetadata(
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            block_table=block_table,
            slot_mapping=slot_mapping,
            block_size=self.block_size,
            num_reqs=num_reqs,
        )
