"""教学用的 Attention 后端：逐请求 gather + Torch 数学（对应 vLLM `v1/attention/backends/torch_sdpa.py`）。

它在一次调用里做两件事，顺序不能反：

1. **写 KV**：把本轮的 key/value 按 `slot_mapping` 散射进分页缓存（这一步之后，本轮的 KV 就
   "在显存里"了）；
2. **算注意力**：对每条请求，从分页缓存里按块表 gather 出 key/value（长度 = 本轮上下文长度），
   和本轮的 query 做 causal attention。

**为什么必须显式构造 mask，不能 `is_causal=True`**：query 和 key 的长度不一样（decode 时
query 长 1、key 长 seq_len），而且 query 的绝对位置从 `seq_len - query_len` 开始。以"past=3、
本轮 2 个 token"为例：第一个 query（绝对位置 3）能看 key 0..3，第二个（位置 4）能看 0..4。
矩形 causal（只按 `key_idx <= query_idx`）会算错——**必须按绝对位置**。

这是**逐请求循环**的实现：清晰、慢，适合对照。真实生产路径是分块 kernel（第五十六关之前的
Triton attention 就是那条路），本关的验收不要求它。
"""

import math

import torch

from ..metadata import AttentionMetadata


class TorchAttentionImpl:
    """后端实现。签名与 198 §7 一致：`forward(layer, query, key, value, kv_cache, attn_metadata)`。"""

    def __init__(self, num_heads: int, head_size: int, scale: float,
                 num_kv_heads: int | None = None) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(f"GQA 要求 query 头数是 KV 头数的整数倍，收到 "
                             f"{self.num_heads} / {self.num_kv_heads}")
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.scale = scale
        if scale != 1.0 / math.sqrt(head_size):
            # 不是错误，但 Qwen 系默认就是这个；写出来免得后面有人误改
            pass

    # -------- 1) 写 KV --------

    def write_kv(self, key: torch.Tensor, value: torch.Tensor, kv_cache, attn_metadata):
        """按 `slot_mapping` 把本轮的 K/V 写进分页缓存。

        `kv_cache` 形状 `[2, num_blocks, block_size, num_kv_heads, head_size]`（K 在第 0 片）。
        `slot_mapping` 是**扁平的物理槽位**（blocks × block_size 摊平后的下标），所以写的时候
        把缓存看成 `[2, num_blocks * block_size, num_kv_heads, head_size]`。
        """
        num_blocks, block_size = kv_cache.shape[1], kv_cache.shape[2]
        flat = kv_cache.view(2, num_blocks * block_size, self.num_kv_heads, self.head_size)
        slots = attn_metadata.slot_mapping
        # 入参是扁平的 [num_tokens, kv_heads * head_size]（模型的 qkv 投影与 RoPE 都是扁平约定），
        # 折成 [num_tokens, kv_heads, head_size] 才能按槽位整行写入
        flat[0].index_copy_(0, slots, key.view(-1, self.num_kv_heads, self.head_size))
        flat[1].index_copy_(0, slots, value.view(-1, self.num_kv_heads, self.head_size))

    # -------- 2) 算注意力 --------

    def forward(self, layer, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                kv_cache, attn_metadata: AttentionMetadata):
        self.write_kv(key, value, kv_cache, attn_metadata)
        num_tokens = query.shape[0]
        out = torch.empty(num_tokens, self.num_heads * self.head_size,
                          dtype=query.dtype, device=query.device)
        start = 0
        for req_index in range(attn_metadata.num_reqs):
            query_len = int(attn_metadata.query_start_loc[req_index + 1]
                            - attn_metadata.query_start_loc[req_index])
            seq_len = int(attn_metadata.seq_lens[req_index])
            past_len = seq_len - query_len
            query_rows = query[start:start + query_len]
            keys, values = self._gather_kv(kv_cache, attn_metadata, req_index, seq_len)
            out[start:start + query_len] = self._attend(query_rows, keys, values,
                                                        past_len, query_len)
            start += query_len
        return out

    def _gather_kv(self, kv_cache, attn_metadata: AttentionMetadata, req_index: int,
                   seq_len: int):
        """按块表把这条请求**有效范围内**的 K/V 取出来，形状 [seq_len, kv_heads, head_size]。"""
        block_size = kv_cache.shape[2]
        num_blocks = (seq_len + block_size - 1) // block_size
        block_ids = attn_metadata.block_table[req_index, :num_blocks]
        num_blocks_total = kv_cache.shape[1]
        flat = kv_cache.view(2, num_blocks_total * block_size, self.num_kv_heads,
                             self.head_size)
        # [num_blocks, block_size, ...] → [num_blocks*block_size, ...] → 取前 seq_len 行
        keys = flat[0][(block_ids[:, None] * block_size
                        + torch.arange(block_size, device=block_ids.device)[None, :]
                        ).reshape(-1)][:seq_len]
        values = flat[1][(block_ids[:, None] * block_size
                          + torch.arange(block_size, device=block_ids.device)[None, :]
                          ).reshape(-1)][:seq_len]
        return keys, values

    def _attend(self, query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                past_len: int, query_len: int) -> torch.Tensor:
        """单请求的 causal attention：**按绝对位置**加 mask。

        query [q_len, heads, head_size]、keys/values [seq_len, kv_heads, head_size]。
        GQA 时把 KV 头复制到与 query 头数一致。
        """
        heads, head_size = self.num_heads, self.head_size
        q = query.view(query_len, heads, head_size).transpose(0, 1).float()      # [H, q, d]
        k = keys.view(-1, self.num_kv_heads, head_size).transpose(0, 1).float()   # [Hk, s, d]
        v = values.view(-1, self.num_kv_heads, head_size).transpose(0, 1).float()
        if self.num_queries_per_kv > 1:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=0)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=0)

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale               # [H, q, s]
        # 绝对位置 mask：第 i 个 query 的绝对位置是 past_len + i，只能看 key 0..past_len+i
        query_pos = torch.arange(past_len, past_len + query_len, device=q.device)
        key_pos = torch.arange(scores.shape[-1], device=q.device)
        mask = key_pos[None, :] > query_pos[:, None]
        scores = scores.masked_fill(mask, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v)                                            # [H, q, d]
        return out.transpose(0, 1).reshape(query_len, heads * head_size).to(query.dtype)
