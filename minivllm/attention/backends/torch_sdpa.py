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

### 69 关：同一个后端里的第二条路径（图内、固定形状）

CUDA Graph 要求形状固定 + 无 CPU 同步，而上面那条逐请求路径两样都违反
（`int(query_start_loc[i])` 是同步、请求数/query 长度都是动态量）。所以这个后端有两条路径，
由 forward 上下文里的 `cudagraph_runtime_mode` 决定：

    NONE                  → `_forward_generic`：逐请求循环（今天这条，eager 参考实现）
    FULL / PIECEWISE      → `_forward_padded`：统一 decode，一次批量算完

`_forward_padded` 成立的前提（也是"后端能支持图"的**准确表述**）：

    * 批里**每条请求恰好 `uniform_query_len = 1 + K` 行**（统一 decode，含投机验证批）；
    * 一条请求的 query 行在位置上连续（`query_pos = seq_len[req] - q_len + i`），
      所以 mask 是"按绝对位置"的规则矩阵，不需要逐请求循环；
    * `num_reqs`、`q_len`、块表宽度都是**捕获时就定死的 Python int**（形状），
      会变的只有 `seq_lens` 与 `slot_mapping` 的内容——它们每轮原地写回静态缓冲；
    * padding 行的槽位是 `PADDING_SLOT_ID(-1)`，写 KV 时**夹到 0 号块**（垃圾桶）。
      这就是上游把 0 号块留白的原因：夹取换掉了分支，代价是必须有一块"写了也没人读"的地方。

**能力边界（明确写出来）**：本仓库的图只覆盖"统一 decode"这一类批（`_forward_padded`）。
prefill、chunked prefill、混合批、以及"每条请求 K_i 不同"的投机批都**不能把注意力录进图**
（`_forward_generic` 里有 `int(张量)` 同步）。它们只能走 **PIECEWISE**：把注意力留在图外、
只把每层的"注意力前/后"两段（norm/gemm/MLP）录进图——这正是 `AttentionCGSupport.UNIFORM_BATCH`
这一档的含义，也是上游 `FULL + UNIFORM_BATCH → FULL_AND_PIECEWISE` 那条降级规则的由来。
"""

import math

import torch

from ..backend import AttentionCGSupport
from ..metadata import AttentionMetadata, AttentionMetadataBuilder


class TorchAttentionMetadataBuilder(AttentionMetadataBuilder):
    """本后端的元数据构造器（上游 `TorchAttentionMetadataBuilder` 的位置）。

    它比基类只多一件事：**声明这个后端的图能力**。上游把能力放在 backend 类上
    （`AttentionBackend.get_builder_cls().get_cudagraph_support(...)`），本仓库没有后端
    注册表，所以能力挂在这唯一一个 builder 上，由 Runner 在初始化图之前读一次。
    """

    #: 只有"每请求 query 长度相同"的批能把注意力录进图（见模块开头的能力边界）。
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH


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

        **哨兵槽位 `PADDING_SLOT_ID(-1)` 的处理分两条路**（58 关的"被拒尾部"与 69 关的
        padding 行都靠它）：

            eager：`slots >= 0` 掩掉它们（`index_copy_` 会把 -1 当成最后一个槽位，
                   真写进去就把池子最后一格覆盖了；上游 kernel 同样先判 slot >= 0）
            图内：**不能**做这个判断——`bool(valid.all())` 是一次 CPU 同步，图里不允许。
                   改成 `clamp_min(0)` 把 padding 行统统写进**0 号块**（垃圾桶）。
                   代价是 0 号块必须留白（`KVCacheManager` 在图模式下不把它分给任何请求），
                   否则 padding 会覆盖真实数据——这一条 64 关就记下了，69 关才真的兑现。
        """
        num_blocks, block_size = kv_cache.shape[1], kv_cache.shape[2]
        flat = kv_cache.view(2, num_blocks * block_size, self.num_kv_heads, self.head_size)
        slots = attn_metadata.slot_mapping
        keys = key.view(-1, self.num_kv_heads, self.head_size)
        values = value.view(-1, self.num_kv_heads, self.head_size)
        if self._graph_dispatched():
            # 图内路径：形状与索引全部是张量，不做任何数据相关的分支
            flat[0].index_copy_(0, slots.clamp_min(0), keys)
            flat[1].index_copy_(0, slots.clamp_min(0), values)
            return
        valid = slots >= 0
        if not bool(valid.all()):
            slots = slots[valid]
            keys = keys[valid]
            values = values[valid]
        if slots.numel():
            flat[0].index_copy_(0, slots, keys)
            flat[1].index_copy_(0, slots, values)

    @staticmethod
    def _graph_dispatched() -> bool:
        """这一轮的**注意力本身**是不是跑在捕获的图里（= 运行模式是 FULL）。

        两个用处，都只对"图内"成立：

            `write_kv` 的 `clamp_min(0)`（图里不能有数据相关分支）
            `forward()` 走 `_forward_padded`（固定形状的批量路径）

        **为什么 PIECEWISE 不算**：分段图把注意力**留在图外**（只把每层注意力前后的
        norm/gemm/MLP 录进图），所以这一轮的注意力就是普通 eager 代码——可以用 `slots >= 0`
        掩掉 padding 行（不必依赖 0 号垃圾桶），也可以走逐请求的通用路径。把 PIECEWISE 也算成
        "图内"会让一条 eager 路径去做只有图才需要的妥协（更强的假设、更差的数值路径）。
        """
        from ...config import CUDAGraphMode
        from ...forward_context import (get_forward_context,
                                        is_forward_context_available)

        if not is_forward_context_available():
            return False
        return get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.FULL

    # -------- 2) 算注意力 --------

    def forward(self, layer, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                kv_cache, attn_metadata: AttentionMetadata):
        self.write_kv(key, value, kv_cache, attn_metadata)
        if attn_metadata.uniform_query_len is not None and self._graph_dispatched():
            return self._forward_padded(query, kv_cache, attn_metadata)
        return self._forward_generic(query, kv_cache, attn_metadata)

    def _forward_generic(self, query: torch.Tensor, kv_cache,
                         attn_metadata: AttentionMetadata) -> torch.Tensor:
        """逐请求循环（eager 参考实现）。"""
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

    def _forward_padded(self, query: torch.Tensor, kv_cache,
                        attn_metadata: AttentionMetadata) -> torch.Tensor:
        """统一 decode 的批量路径（**图内可用**：形状固定、无 CPU 同步、无数据相关分支）。

        形状全部来自 Python int（`num_reqs`、`uniform_query_len`、块表宽度），只有张量的
        **内容**（`seq_lens`、`slot_mapping`）每轮在变，所以同一张图能重放任意一轮。

        行布局：批被补齐到 `B * q_len` 行，第 `b` 条请求的第 `i` 行在绝对位置
        `seq_lens[b] - q_len + i`（统一 decode 下这条式子就是"行在请求内的位置"）。

        mask 是两块合起来（都按**绝对位置**，不是矩形 causal）：

            key_pos >= seq_lens[b]          越界的 key 不该看（它属于下一轮的槽位）
            key_pos >  seq_lens[b] - q_len + i   未来的 key 不该看（causal）

        两条都命中的行（例如被补齐的假请求 `seq_lens=0`）会**整行被掩掉**。这里用
        `finfo.min`（有限大负数）而不是 `-inf`：全掩行的 softmax 在 `-inf` 下是 NaN，
        而 NaN 会写进 KV 缓存（padding 行）并在后续重放里扩散；用有限大负数时全掩行得到
        均匀分布（一堆垃圾的均值），**是垃圾但有限**。上游 kernel 用 `l==0 → out=0`
        达到同样目的（本仓库没有 kernel 内分支，就用这个等价写法）。

        **精度**：与 eager 路径一样把 K/V 升到 float32 再算。为什么不"省这一步直接在
        bf16/fp16 上乘"（那样显存能少一半）：两条路径的数值必须**足够接近**，否则 greedy
        取舍会在临界处翻掉——这不是理论担心，实测在真实 EAGLE3（bf16）上确实翻过一枚 token
        （eager 与图输出在第 7 个 token 分叉，见 `docs/step69_alignment.md` §5.2）。
        升到 fp32 之后两条路径只在"批量 matmul vs 逐请求 matmul"的规约顺序上有差别（~1e-6）。

        代价是 [B, P, Hk, D] 要多占一份 fp32；`P = 块表宽度 × block_size`（= 上下文上限），
        所以本后端的图路径显存是 `2 · B · P · Hk · D · 4` 字节量级。上游的 FlashAttention
        类 kernel 不物化这份 gather——这是教学后端的能力边界，写在文档里。
        """
        kv_heads, head_size = self.num_kv_heads, self.head_size
        num_blocks_total, block_size = kv_cache.shape[1], kv_cache.shape[2]
        flat = kv_cache.view(2, num_blocks_total * block_size, kv_heads, head_size)

        num_reqs = attn_metadata.num_reqs
        q_len = int(attn_metadata.uniform_query_len)
        num_positions = attn_metadata.block_table.shape[1] * block_size
        device = query.device

        # 每条请求的**全部**候选 key 槽位：[B, num_positions]
        # （越界位置也一并 gather：反正随后会被 mask 掉，而"先算后掩"才是固定形状的写法）
        key_pos = torch.arange(num_positions, device=device)
        block_slots = attn_metadata.block_table * block_size          # [B, nb]
        # slot = block_table[req, pos // bs] * bs + pos % bs（与 eager 路径同一条式子，
        # 只是这里对 [0, num_positions) 一次算完，形状固定）
        gather = block_slots[:, key_pos // block_size] + (key_pos % block_size)
        keys = flat[0][gather]                                        # [B, P, Hk, D]
        values = flat[1][gather]

        heads = self.num_heads
        q = query.view(num_reqs, q_len, heads, head_size).transpose(1, 2).float()
        k = keys.transpose(1, 2).float()                                     # [B, Hk, P, D]
        v = values.transpose(1, 2).float()
        if self.num_queries_per_kv > 1:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale           # [B,H,q,P]
        seq_lens = attn_metadata.seq_lens.view(num_reqs, 1, 1).to(torch.float32)
        query_pos = (seq_lens - q_len) + torch.arange(
            q_len, device=device, dtype=torch.float32).view(1, 1, q_len)
        positions = key_pos.to(torch.float32).view(1, 1, 1, num_positions)
        mask = (positions > query_pos.unsqueeze(-1)) | (positions >= seq_lens.unsqueeze(-1))
        scores = scores.masked_fill(mask, torch.finfo(torch.float32).min)
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v)                                  # [B, H, q, D]
        return out.transpose(1, 2).reshape(num_reqs * q_len,
                                           heads * head_size).to(query.dtype)

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


class TorchAttentionBackend:
    """后端契约（上游 `AttentionBackend` 的**只有调用方的**那一部分）。

    上游的后端还要回答名字、支持的 head size、是否用级联注意力、KV cache 形状等；本仓库
    只有一个后端、一种分页 KV，所以这里只保留真的被调用的三件事——其中前两件是 71 关新增的
    "能力协商"入口：
    """

    @staticmethod
    def get_builder_cls():
        return TorchAttentionMetadataBuilder

    @staticmethod
    def get_impl_cls():
        return TorchAttentionImpl
