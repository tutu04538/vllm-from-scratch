"""融合 RoPE 算子（对应上游 `RotaryEmbedding.forward_cuda` 的输入/输出约定）。

上游 CUDA 路径调用 `ops.rotary_embedding`，它是**原地**的：按 `positions` 查紧凑
`cos_sin_cache`，把 Q/K 的 `[rotary_dim/2]` 两半配对旋转。这里用本地 Triton
kernel 复现同一输入协议：

    positions       [num_tokens] 绝对位置
    query/key       [num_tokens, num_heads * head_size]
    cos_sin_cache   [max_position, rotary_dim] = [cos 半宽 | sin 半宽]

`key=None` 是合法的（Gemma4 Q-only / 跨层 KV 共享），此时 kernel 只旋转 Q。
查询表、位置读取和写回都在 kernel 内完成，不 `.item()`、不 `.tolist()`、不构造
每层完整 cos/sin gather 中间张量。

**原地语义**：CUDA 路径像上游一样修改并返回传入的 Q/K；CPU 路径保留显式 Torch
参考，返回新张量（与上游 `forward_native` 一致）。模型调用方两种都接得住。

**stride**：2D 输入支持任意 token/列 stride。packed QKV 切片 `qkv[:, :q_size]`
的 token stride 是整条 QKV 宽度，本算子只写 Q/K 的列，不碰相邻 V。
"""

from __future__ import annotations

import torch

try:  # triton 只在 CUDA 路径需要；CPU-only 环境仍可 import 本模块。
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - 本机环境有 triton
    triton = None
    tl = None


if triton is not None:

    # debug=True 只用来让 kernel 内的 `device_assert` 生效（Triton 默认会把它
    # 编译掉）。边界错误在设备端报，不需要 host `.item()`；代价是一个越界分支。
    @triton.jit(debug=True)
    def _rope_qk_kernel(
        positions_ptr,
        query_ptr,
        key_ptr,
        cache_ptr,
        q_stride_token,
        q_stride_dim,
        k_stride_token,
        k_stride_dim,
        q_heads,
        k_heads,
        max_position,
        HEAD_SIZE: tl.constexpr,
        ROTARY_DIM: tl.constexpr,
        IS_NEOX: tl.constexpr,
        HAS_KEY: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        token = tl.program_id(0)
        head = tl.program_id(1)
        position = tl.load(positions_ptr + token).to(tl.int64)
        # 越界位置不能让 kernel 静默读相邻内存；device_assert 是设备端检查，
        # 不需要 host `.item()`，也不会把 decode 路径同步化。
        half: tl.constexpr = ROTARY_DIM // 2
        offsets = tl.arange(0, BLOCK)
        # 标量条件如果每个 lane 都断言，越界会打印一整块线程；只让第 0 个
        # lane 报一次。mask 与下面的向量掩码同一形状。
        tl.device_assert(
            (position >= 0) & (position < max_position),
            "RoPE position out of range",
            mask=offsets == 0,
        )
        mask = offsets < half
        # 紧凑布局：[0:half] 是 cos，[half:ROTARY_DIM] 是 sin。
        cos = tl.load(
            cache_ptr + position * ROTARY_DIM + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        sin = tl.load(
            cache_ptr + position * ROTARY_DIM + half + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        if IS_NEOX:
            first = offsets
            second = offsets + half
        else:
            first = offsets * 2
            second = offsets * 2 + 1

        if head < q_heads:
            q_base = token * q_stride_token + head * HEAD_SIZE * q_stride_dim
            q0 = tl.load(
                query_ptr + q_base + first * q_stride_dim,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            q1 = tl.load(
                query_ptr + q_base + second * q_stride_dim,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            out0 = q0 * cos - q1 * sin
            out1 = q1 * cos + q0 * sin
            tl.store(
                query_ptr + q_base + first * q_stride_dim,
                out0.to(query_ptr.dtype.element_ty),
                mask=mask,
            )
            tl.store(
                query_ptr + q_base + second * q_stride_dim,
                out1.to(query_ptr.dtype.element_ty),
                mask=mask,
            )

        if HAS_KEY:
            if head < k_heads:
                k_base = token * k_stride_token + head * HEAD_SIZE * k_stride_dim
                k0 = tl.load(
                    key_ptr + k_base + first * k_stride_dim,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
                k1 = tl.load(
                    key_ptr + k_base + second * k_stride_dim,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
                out0 = k0 * cos - k1 * sin
                out1 = k1 * cos + k0 * sin
                tl.store(
                    key_ptr + k_base + first * k_stride_dim,
                    out0.to(key_ptr.dtype.element_ty),
                    mask=mask,
                )
                tl.store(
                    key_ptr + k_base + second * k_stride_dim,
                    out1.to(key_ptr.dtype.element_ty),
                    mask=mask,
                )


def _validate(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    cos_sin_cache: torch.Tensor,
    head_size: int,
    rotary_dim: int,
    is_neox_style: bool,
) -> None:
    if not isinstance(head_size, int) or head_size <= 0:
        raise ValueError(f"head_size 必须是正整数，收到 {head_size!r}")
    if not isinstance(rotary_dim, int) or rotary_dim <= 0:
        raise ValueError(f"rotary_dim 必须是正整数，收到 {rotary_dim!r}")
    if rotary_dim % 2 != 0:
        raise ValueError(f"rotary_dim 必须是偶数，收到 {rotary_dim}")
    if rotary_dim > head_size:
        raise ValueError(
            f"rotary_dim={rotary_dim} 不能大于 head_size={head_size}")
    if not isinstance(is_neox_style, bool):
        raise ValueError(f"is_neox_style 必须是 bool，收到 {is_neox_style!r}")

    tensors = {"query": query}
    if key is not None:
        tensors["key"] = key
    for name, tensor in tensors.items():
        if tensor.ndim != 2:
            raise ValueError(
                f"RoPE 的 {name} 必须是 [num_tokens, num_heads * head_size] 二维张量，"
                f"收到 shape={tuple(tensor.shape)}")
        if tensor.shape[1] % head_size != 0:
            raise ValueError(
                f"RoPE 的 {name} 最后一维 {tensor.shape[1]} 不能被 "
                f"head_size={head_size} 整除")
        if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32,
                                torch.float64):
            raise ValueError(f"RoPE 的 {name} 只支持浮点 dtype，收到 {tensor.dtype}")
        if tensor.device != query.device:
            raise ValueError(
                f"RoPE 的 query/key 必须同设备：query={query.device}，{name}={tensor.device}")
        if tensor.dtype != query.dtype:
            raise ValueError(
                f"RoPE 的 query/key 必须同 dtype：query={query.dtype}，{name}={tensor.dtype}")
        if tensor.stride(0) <= 0 or tensor.stride(1) <= 0:
            raise ValueError(
                f"RoPE 的 {name} 需要正的 token/列 stride，"
                f"收到 stride={tensor.stride()}；重叠/广播视图不能原地旋转")
    if key is not None and key.shape[0] != query.shape[0]:
        raise ValueError(
            f"RoPE 的 query/key token 行数必须一致："
            f"{query.shape[0]} vs {key.shape[0]}")

    if positions.ndim not in (1, 2):
        raise ValueError(
            f"positions 必须是 1D 或 [num_tokens, 1]，收到 shape={tuple(positions.shape)}")
    if positions.numel() != query.shape[0]:
        raise ValueError(
            f"positions 数量 {positions.numel()} 与 query 行数 {query.shape[0]} 不一致")
    if positions.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"positions 必须是 int32/int64，收到 {positions.dtype}")
    if positions.device != query.device:
        raise ValueError(
            f"positions 与 query 必须同设备：{positions.device} vs {query.device}")

    if cos_sin_cache.ndim != 2:
        raise ValueError(
            f"cos_sin_cache 必须是二维 [max_position, rotary_dim]，"
            f"收到 shape={tuple(cos_sin_cache.shape)}")
    if cos_sin_cache.shape[1] != rotary_dim:
        raise ValueError(
            f"cos_sin_cache 宽度应为 rotary_dim={rotary_dim}，"
            f"收到 {cos_sin_cache.shape[1]}；本实现使用紧凑半宽布局")
    if cos_sin_cache.shape[0] <= 0:
        raise ValueError("cos_sin_cache 至少要有一个合法位置")
    if cos_sin_cache.device != query.device:
        raise ValueError(
            f"cos_sin_cache 与 query 必须同设备："
            f"{cos_sin_cache.device} vs {query.device}")
    if cos_sin_cache.dtype != query.dtype:
        raise ValueError(
            f"cos_sin_cache dtype={cos_sin_cache.dtype} 与输入 dtype={query.dtype} "
            f"不一致；不要在每层 forward 里临时 .to() 复制共享表，"
            f"应在建表/模型 dtype 转换时统一")
    if not cos_sin_cache.is_contiguous():
        raise ValueError("cos_sin_cache 必须连续；共享表不应是带洞的视图")
    if key is not None and query.data_ptr() == key.data_ptr():
        raise ValueError("query 与 key 指向同一块起始地址；原地旋转会被执行两次")


def apply_rotary_embedding_native(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    cos_sin_cache: torch.Tensor,
    head_size: int,
    rotary_dim: int,
    is_neox_style: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """显式 Torch 参考：返回新张量，不修改输入（与上游 `forward_native` 同义）。"""
    _validate(positions, query, key, cos_sin_cache, head_size, rotary_dim,
              is_neox_style)
    positions = positions.reshape(-1)
    cos_sin = cos_sin_cache.index_select(0, positions)
    cos, sin = cos_sin.chunk(2, dim=-1)
    cos = cos.unsqueeze(1).to(query.dtype)
    sin = sin.unsqueeze(1).to(query.dtype)

    def rotate(x: torch.Tensor) -> torch.Tensor:
        x_heads = x.reshape(x.shape[0], -1, head_size)
        x_rot = x_heads[..., :rotary_dim]
        x_pass = x_heads[..., rotary_dim:]
        if is_neox_style:
            x1, x2 = x_rot.chunk(2, dim=-1)
            o1 = x1 * cos - x2 * sin
            o2 = x2 * cos + x1 * sin
            x_rot = torch.cat((o1, o2), dim=-1)
        else:
            x1 = x_rot[..., ::2]
            x2 = x_rot[..., 1::2]
            o1 = x1 * cos - x2 * sin
            o2 = x2 * cos + x1 * sin
            x_rot = torch.stack((o1, o2), dim=-1).flatten(-2)
        if x_pass.shape[-1]:
            out = torch.cat((x_rot, x_pass), dim=-1)
        else:
            out = x_rot
        return out.reshape(x.shape)

    return rotate(query), (None if key is None else rotate(key))


def apply_rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    cos_sin_cache: torch.Tensor,
    head_size: int,
    rotary_dim: int,
    is_neox_style: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """生产入口：CUDA 用原地 Triton kernel，CPU 用显式 Torch 参考。"""
    _validate(positions, query, key, cos_sin_cache, head_size, rotary_dim,
              is_neox_style)
    if not query.is_cuda:
        return apply_rotary_embedding_native(
            positions, query, key, cos_sin_cache, head_size, rotary_dim,
            is_neox_style)
    if triton is None:
        raise RuntimeError(
            "CUDA 上的融合 RoPE 需要 triton；当前环境没有 triton。"
            "没有静默回退到 Torch，请安装 triton 或显式调用 forward_native")

    positions = positions.reshape(-1)
    if not positions.is_contiguous():
        positions = positions.contiguous()
    q_heads = query.shape[1] // head_size
    k_heads = 0 if key is None else key.shape[1] // head_size
    if q_heads <= 0:
        raise ValueError("query 的 head 数为 0")
    if key is not None and k_heads <= 0:
        raise ValueError("key 的 head 数为 0")

    half = rotary_dim // 2
    block = triton.next_power_of_2(half)
    # 2D 张量：stride(1) 既是列 stride，也是 head 内元素 stride。
    grid = (query.shape[0], max(q_heads, k_heads))
    _rope_qk_kernel[grid](
        positions,
        query,
        key if key is not None else query,
        cos_sin_cache,
        query.stride(0),
        query.stride(1),
        key.stride(0) if key is not None else 0,
        key.stride(1) if key is not None else 0,
        q_heads,
        k_heads,
        cos_sin_cache.shape[0],
        HEAD_SIZE=head_size,
        ROTARY_DIM=rotary_dim,
        IS_NEOX=is_neox_style,
        HAS_KEY=key is not None,
        BLOCK=block,
    )
    return query, key
