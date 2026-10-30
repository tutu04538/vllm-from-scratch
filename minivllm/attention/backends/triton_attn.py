"""TRITON_ATTN：086 起的唯一 Triton 分页 attention 后端。

职责边界：

    * 这里只做 backend / metadata / 能力检查 / dispatch；
    * attention 数学在 `attention/ops/triton_unified_attention.py`；
    * KV 写入在 `attention/ops/triton_reshape_and_cache_flash.py`；
    * eager、FULL 图、Q-only 共享层全部走同一条 unified 路径，没有
      “eager 回 Torch”或“block/head 不满足就换旧 kernel”的 fallback。

本地没有上游的量化、ALiBi、sinks、cascade、MM prefix、RSWA、tensor
descriptor 分支；这些能力不在本后端的支持矩阵内。
"""

from __future__ import annotations

from typing import ClassVar

import torch

from ..backend import AttentionCGSupport
from ..metadata import AttentionMetadata, AttentionMetadataBuilder
from ..ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
)
from ..ops.triton_unified_attention import (
    can_use_3d,
    make_3d_workspace,
    unified_attention,
)
from ...triton_utils import triton


#: 上游 `MIN_LAUNCH_GRID_SIZE_2D = 128`；低于这个 batch 时 3D 分段才有意义。
_MIN_LAUNCH_GRID_SIZE_2D = 128
_NUM_PAR_SOFTMAX_SEGMENTS = 16


def _device_key(device: torch.device | str) -> tuple[str, int]:
    device = torch.device(device)
    index = device.index
    return device.type, -1 if index is None else int(index)


class TritonAttentionWorkspacePool:
    """一个 metadata builder（= 一个 Runner/引擎）持有的 3D 工作区池。

    087 之前工作区放在**进程级** `_WORKSPACE_CACHE` 里：同一模型不同层能共享是省了显存，
    但引擎 shutdown 后池仍被模块字典引用，工作区永远不释放，也不属于任何一个引擎。
    现在所有权落到 builder：

    * builder 在 `__init__` 里建池，`build()` 把同一对象挂到每份 metadata 上；
    * Impl 只从 metadata 借池，不再自己缓存；
    * Runner shutdown 调 `release()`，池和它的 tensor 一起被回收。

    池按 `(device, num_heads, head_size, seq_threshold_3D, segments)` 分桶。同一模型的
    28 个 sliding 层共用一个桶、7 个 full 层共用一个桶；图捕获时的 warmup 会把桶建好，
    capture/replay 读到的就是同一批地址。
    """

    def __init__(self) -> None:
        self._workspaces: dict[
            tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}

    def get(
        self,
        device: torch.device | str,
        num_heads: int,
        head_size: int,
        seq_threshold_3D: int,
        num_par_softmax_segments: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (_device_key(device), int(num_heads), int(head_size),
               int(seq_threshold_3D), int(num_par_softmax_segments))
        workspace = self._workspaces.get(key)
        if workspace is None:
            workspace = make_3d_workspace(
                device, num_heads, head_size, seq_threshold_3D,
                num_par_softmax_segments)
            self._workspaces[key] = workspace
        return workspace

    def release(self) -> None:
        """丢弃所有工作区；只清本池，不碰别的引擎。"""
        self._workspaces.clear()

    def __len__(self) -> int:
        return len(self._workspaces)

    def keys(self) -> tuple[tuple, ...]:
        return tuple(self._workspaces)

    def bucket_bytes(self, key: tuple) -> int:
        """返回已有桶的字节数；key 不存在时明确报错，不顺手新建。"""
        try:
            workspace = self._workspaces[key]
        except KeyError:
            raise KeyError(f"没有这个 3D 工作区桶：{key!r}") from None
        return sum(t.numel() * t.element_size() for t in workspace)

    def num_bytes(self) -> int:
        return sum(self.bucket_bytes(key) for key in self._workspaces)


class TritonAttentionMetadataBuilder(AttentionMetadataBuilder):
    """与 Torch builder 同字段；图能力 ALWAYS。

    3D workspace 由**本 builder** 持有，不在每层 Impl 里缓存，也不是进程级全局：
    metadata 每步重建，但 `workspace_pool` 是同一个对象；Impl 从 metadata 借池。
    这样 capture 时的 warmup 先建桶，重放/换 seq_len 都读同一批地址；Runner
    shutdown 调 `release()` 就能把工作区真正还掉。
    """

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS
    supports_draft_decode_metadata_update = True

    def __init__(self, block_size: int) -> None:
        super().__init__(block_size)
        self.workspace_pool = TritonAttentionWorkspacePool()

    def build(self, query_start_loc: torch.Tensor, seq_lens: torch.Tensor,
              block_table: torch.Tensor, slot_mapping: torch.Tensor,
              num_reqs: int, uniform_query_len: int | None = None,
              causal: bool = True, max_query_len: int | None = None,
              sliding_window: int | None = None) -> AttentionMetadata:
        metadata = super().build(
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            block_table=block_table,
            slot_mapping=slot_mapping,
            num_reqs=num_reqs,
            uniform_query_len=uniform_query_len,
            causal=causal,
            max_query_len=max_query_len,
            sliding_window=sliding_window,
        )
        metadata.workspace_pool = self.workspace_pool
        return metadata

    def update_draft_decode_metadata(self, metadata: AttentionMetadata) -> None:
        # 所有步相关字段都是常驻输入缓冲的切片，写缓冲的 kernel 已经就地更新。
        return None

    def release(self) -> None:
        """Runner shutdown 调用；只释放本 builder 的 3D 工作区。"""
        self.workspace_pool.release()


class TritonAttentionImpl:
    """统一 Triton 分页 attention 实现。"""

    def __init__(self, num_heads: int, head_size: int, scale: float,
                 num_kv_heads: int | None = None) -> None:
        self.num_heads = int(num_heads)
        self.head_size = int(head_size)
        self.num_kv_heads = (self.num_heads if num_kv_heads is None
                             else int(num_kv_heads))
        if self.num_heads <= 0 or self.num_kv_heads <= 0:
            raise ValueError("attention head 数必须为正")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"GQA 要求 query head 数整除 KV head 数，收到 "
                f"{self.num_heads}/{self.num_kv_heads}")
        if not (1 <= self.head_size <= 512):
            raise ValueError(
                f"TRITON_ATTN 当前支持 head_size 1..512（Gemma4 的 256/512、"
                f"Qwen3 的 128、tiny 模型的 8/16 等），收到 {self.head_size}")
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.scale = float(scale)

        # 单请求时至少让 grid 接近 128 个 program；num_seqs 超过它就退回 2D。
        self.seq_threshold_3D = max(
            1, _MIN_LAUNCH_GRID_SIZE_2D // self.num_kv_heads)
        self.num_par_softmax_segments = _NUM_PAR_SOFTMAX_SEGMENTS

    # -------- 入口 --------

    def forward(self, layer, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor, kv_cache,
                attn_metadata: AttentionMetadata) -> torch.Tensor:
        q = self._as_3d(query, self.num_heads)
        k = self._as_3d(key, self.num_kv_heads)
        v = self._as_3d(value, self.num_kv_heads)
        if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
            raise ValueError(
                f"q/k/v 的 token 数不一致：{q.shape[0]}/{k.shape[0]}/{v.shape[0]}")
        self._write_kv(k, v, kv_cache, attn_metadata.slot_mapping)
        out_flat, out = self._output_for(layer, q)
        self._run(q, kv_cache, out, attn_metadata, layer)
        return out_flat

    def forward_q_only(self, layer, query: torch.Tensor, kv_cache,
                       attn_metadata: AttentionMetadata) -> torch.Tensor:
        """Q-only 层：只读源层 cache，绝不写 K/V。"""
        q = self._as_3d(query, self.num_heads)
        out_flat, out = self._output_for(layer, q)
        self._run(q, kv_cache, out, attn_metadata, layer)
        return out_flat

    def do_kv_cache_update(self, layer, key: torch.Tensor, value: torch.Tensor,
                           kv_cache, slot_mapping: torch.Tensor) -> None:
        k = self._as_3d(key, self.num_kv_heads)
        v = self._as_3d(value, self.num_kv_heads)
        self._write_kv(k, v, kv_cache, slot_mapping)

    # -------- 内部 --------

    def _as_3d(self, x: torch.Tensor, heads: int) -> torch.Tensor:
        if x.ndim == 3:
            if x.shape[1] != heads or x.shape[2] != self.head_size:
                raise ValueError(
                    f"tensor 形状 {tuple(x.shape)} 与 heads/head_size="
                    f"{heads}/{self.head_size} 不符")
            return x
        if x.ndim == 2:
            if x.shape[1] != heads * self.head_size:
                raise ValueError(
                    f"tensor 末维 {x.shape[1]} != {heads}×{self.head_size}")
            # `unflatten` 保留 packed QKV 的 row stride，不触发 copy。
            return x.unflatten(-1, (heads, self.head_size))
        raise ValueError(f"attention tensor 必须是 2D/3D，收到 {tuple(x.shape)}")

    def _output_for(self, layer, q: torch.Tensor):
        n = q.shape[0]
        flat = getattr(layer, "_output_buffer", None)
        width = self.num_heads * self.head_size
        if (flat is not None and flat.ndim == 2 and flat.shape[0] >= n
                and flat.shape[1] == width and flat.dtype == q.dtype
                and flat.device == q.device):
            out_flat = flat[:n]
            return out_flat, out_flat.view(n, self.num_heads, self.head_size)
        out_flat = torch.empty(
            (n, width), dtype=q.dtype, device=q.device)
        return out_flat, out_flat.view(n, self.num_heads, self.head_size)

    def _split_cache(self, kv_cache):
        if torch.is_tensor(kv_cache):
            if kv_cache.ndim != 5 or kv_cache.shape[0] != 2:
                raise ValueError(
                    f"KV cache 应为 [2, blocks, block_size, kv_heads, head_size]，"
                    f"收到 {tuple(kv_cache.shape)}")
            return kv_cache[0], kv_cache[1]
        if isinstance(kv_cache, (tuple, list)) and len(kv_cache) == 2:
            return kv_cache[0], kv_cache[1]
        raise ValueError(
            "TRITON_ATTN 需要 [2, ...] 的 KV tensor 或 (k_cache, v_cache) 两片视图")

    def _write_kv(self, k: torch.Tensor, v: torch.Tensor, kv_cache,
                  slot_mapping: torch.Tensor) -> None:
        k_cache, v_cache = self._split_cache(kv_cache)
        if k_cache.ndim != 4 or v_cache.ndim != 4:
            raise ValueError(
                f"KV cache 必须是 [blocks, block_size, kv_heads, head_size]，"
                f"收到 {tuple(k_cache.shape)} / {tuple(v_cache.shape)}")
        if k_cache.shape != v_cache.shape:
            raise ValueError("K cache 与 V cache 形状不一致")
        if tuple(k_cache.shape[2:]) != (self.num_kv_heads, self.head_size):
            raise ValueError(
                f"KV cache 的 kv_heads/head_size 不符：cache="
                f"{tuple(k_cache.shape[2:])}, impl="
                f"{(self.num_kv_heads, self.head_size)}")
        n = k.shape[0]
        slots = slot_mapping[:n]
        if slots.numel() != n:
            raise ValueError(
                f"slot_mapping 只有 {slots.numel()} 个，少于 {n} 个 token")
        triton_reshape_and_cache_flash(k, v, k_cache, v_cache, slots)

    def _run(self, q: torch.Tensor, kv_cache, out: torch.Tensor,
             attn_metadata: AttentionMetadata, layer) -> None:
        if attn_metadata is None:
            raise RuntimeError("TRITON_ATTN 需要 attn_metadata")
        if triton is None:
            raise RuntimeError("Triton 不可用，不能使用 TRITON_ATTN 后端")

        k_cache, v_cache = self._split_cache(kv_cache)
        if attn_metadata.block_size != k_cache.shape[1]:
            raise ValueError(
                f"metadata.block_size={attn_metadata.block_size} 与 cache 的 "
                f"block_size={k_cache.shape[1]} 不一致")

        window = getattr(layer, "per_layer_sliding_window", None)
        if window is not None:
            window = int(window)
            if window <= 0:
                raise ValueError(f"sliding_window 必须为正，收到 {window}")
            if (attn_metadata.sliding_window is not None
                    and int(attn_metadata.sliding_window) != window):
                raise ValueError(
                    f"层 sliding_window={window} 与 metadata="
                    f"{attn_metadata.sliding_window} 不一致")
            window_size = (window - 1, 0)
        else:
            window_size = (-1, -1)

        max_q = (attn_metadata.uniform_query_len
                 if attn_metadata.uniform_query_len is not None
                 else attn_metadata.max_query_len)
        num_seqs = int(attn_metadata.seq_lens.numel())
        workspace = None
        if can_use_3d(
                max_seqlen_q=max_q,
                num_seqs=num_seqs,
                seq_threshold_3D=self.seq_threshold_3D,
                num_par_softmax_segments=self.num_par_softmax_segments):
            pool = attn_metadata.workspace_pool
            if pool is None:
                raise RuntimeError(
                    "TRITON_ATTN 需要 3D 工作区，但这份 metadata 没有 workspace_pool；"
                    "生产路径必须由 TritonAttentionMetadataBuilder.build() 构造 metadata，"
                    "不能绕过它直接调用 impl")
            workspace = pool.get(
                q.device, self.num_heads, self.head_size,
                self.seq_threshold_3D, self.num_par_softmax_segments)
            if workspace[0].shape[0] < num_seqs:
                raise RuntimeError(
                    f"3D workspace 只有 {workspace[0].shape[0]} 行，"
                    f"但本批有 {num_seqs} 条请求")

        unified_attention(
            q, k_cache, v_cache, out,
            attn_metadata.query_start_loc,
            attn_metadata.seq_lens,
            attn_metadata.block_table,
            softmax_scale=self.scale,
            causal=bool(attn_metadata.causal),
            window_size=window_size,
            max_seqlen_q=max_q,
            seq_threshold_3D=(self.seq_threshold_3D
                              if workspace is not None else None),
            num_par_softmax_segments=(self.num_par_softmax_segments
                                      if workspace is not None else None),
            softmax_segm_output=(workspace[0] if workspace is not None else None),
            softmax_segm_max=(workspace[1] if workspace is not None else None),
            softmax_segm_expsum=(workspace[2] if workspace is not None else None),
        )


class TritonAttentionBackend:
    """TRITON_ATTN backend 入口。"""

    @staticmethod
    def get_name() -> str:
        return "TRITON_ATTN"

    @staticmethod
    def get_builder_cls():
        return TritonAttentionMetadataBuilder

    @staticmethod
    def get_impl_cls():
        return TritonAttentionImpl

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return True

    @classmethod
    def supports_non_causal(cls) -> bool:
        return True

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return 1 <= int(head_size) <= 512

    @staticmethod
    def supports_device_cpu_query_lens_mismatch() -> bool:
        # query_start_loc/seq_lens/block_table 都在 device 上读；不需要用 CPU
        # query 长度做形状规划，也不按 CPU 长度分支。
        return True
