"""Attention 的元数据（对应 vLLM `v1/attention/backends/*.py` 里各后端的 AttentionMetadata）。

**字段是本仓库子集**，不冒充所有后端共用的统一签名。五个字段刚好覆盖"分页 KV + 变长 batch"：

    query_start_loc  [num_reqs + 1]  每个请求本轮的 query 在扁平张量里的起止（前缀和）
    seq_lens         [num_reqs]      每个请求**本轮之后**的上下文长度（含本轮 query）
    block_table      [num_reqs, max_blocks_per_req]  逻辑块号 → 物理块号
    slot_mapping     [num_tokens]    每个输入 token 的 KV 写到哪个物理槽位
    block_size / num_reqs

`slot_mapping[i] = block_table[req, pos // block_size] * block_size + pos % block_size`，
`pos` 是该 token 的**绝对位置**——这条式子是"分页 KV 写在哪"的唯一答案，测试里有专门用例
（对齐 198 §4 的数字）。

**69 关新增 `uniform_query_len`**（对应上游各后端的 `max_query_len`）：统一 decode 批里
"每条请求恰好几行 query"。它必须是 Python int（不是张量）——图内路径靠它定形状，
从张量里 `.item()` 取值等于在图里插一次同步，而那正是图最忌讳的事。`None` = 非统一批，
走逐请求的通用路径（本仓库的通用路径就是 eager：非统一形状没有对应的图键）。
"""

from dataclasses import dataclass
from typing import ClassVar

import torch

from .backend import AttentionCGSupport


@dataclass
class AttentionMetadata:
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    block_size: int
    num_reqs: int
    #: 统一 decode（图内）路径的每请求 query 行数 `1 + K`；None = 通用（eager）路径。
    uniform_query_len: int | None = None


class AttentionMetadataBuilder:
    """从 InputBatch + 块表构造 metadata。**只做张量准备，不做模型数学**。"""

    #: 本后端的图能力档位。基类是 `NEVER`（上游同款默认），具体后端覆写它；
    #: 不要直接读这个字段，要用 `get_cudagraph_support()`（它给的是"对这份配置"的答案）。
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

    def __init__(self, block_size: int) -> None:
        self.block_size = block_size

    @classmethod
    def get_cudagraph_support(cls, vllm_config, kv_cache_spec=None) -> AttentionCGSupport:
        """这个后端在**这份引擎配置**下支持到哪一档（对应上游同名方法）。

        上游各后端还会看 `kv_cache_spec`（例如 MLA / 滑动窗口的实现差异）与 dtype。
        本仓库只有一个后端、一种分页 KV，能力只取决于 `_cudagraph_support` 这个类属性；
        `kv_cache_spec` 参数保留是为了调用点与上游同形（将来加后端时不用改调用方）。
        """
        return cls._cudagraph_support

    def build(self, query_start_loc: torch.Tensor, seq_lens: torch.Tensor,
              block_table: torch.Tensor, slot_mapping: torch.Tensor,
              num_reqs: int, uniform_query_len: int | None = None) -> AttentionMetadata:
        return AttentionMetadata(
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            block_table=block_table,
            slot_mapping=slot_mapping,
            block_size=self.block_size,
            num_reqs=num_reqs,
            uniform_query_len=uniform_query_len,
        )

    def build_for_cudagraph_capture(self, metadata: AttentionMetadata) -> AttentionMetadata:
        """捕获期专用（对应上游各后端的同名方法）。

        两件事，都有理由：

        1. **`seq_lens` 全部填 1**：图是"录一遍"而不是"真算一遍"，`seq_lens` 只影响 kernel 的
           网格/workspace 选择。上游的注释是"填 max_model_len 会让捕获慢到不可接受"；
           填 1 最快（注意力只扫一个位置）。**重放前 Runner 会把真实值写回这块缓冲**，
           所以捕获时填什么不影响结果——前提是缓冲是**同一块**（这正是静态工作区的意义）。
        2. `uniform_query_len` 必须已经设好：它定的是图里的形状，不能捕获完再变。
        """
        if metadata.uniform_query_len is None:
            raise ValueError(
                "捕获图必须走统一 decode 路径（uniform_query_len = 1+K）："
                "非统一形状没有对应的图键")
        metadata.seq_lens.fill_(1)
        return metadata
