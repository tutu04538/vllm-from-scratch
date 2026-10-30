"""RMSNorm 的融合算子（需求 85）。

对应上游 `vllm/model_executor/layers/layernorm.py` 的两个类：

    RMSNorm.forward_native       ir.ops.rms_norm / fused_add_rms_norm
    GemmaRMSNorm.forward_native   weight = (1 + w)，其余同 RMSNorm

本文件只做**一个 logical row 的 RMSNorm**，不把模型结构塞进来。CUDA 路径用本地
Triton kernel：一行一个 program，FP32 归约、一次写回；CPU 路径用显式 Torch 参考。
两条路径共享同一套 shape/stride/dtype 契约。

支持的行布局（最后一维必须是连续的）：

    1D  [D]                  单行
    2D  [rows, D]            任意正的 row stride（packed QKV 切片就是这种）
    3D  [batch, heads, D]    Q/K norm；任意正的 batch/head stride

`residual` 与 `x` 同形同 dtype；融合分支先算 `x + residual`，再把结果写到
**新的 residual 张量**并归一化。返回值与上游 `forward_native` 一样是新张量，
不改 caller 的输入；这样调用方不会因为某层原地写 residual 而拿到半更新状态。
"""

from __future__ import annotations

import torch

try:  # triton 只在 CUDA 路径需要；CPU-only 环境仍可 import 本模块。
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - 本机环境有 triton
    triton = None
    tl = None


_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


if triton is not None:

    @triton.jit
    def _rms_norm_kernel(
        x_ptr,
        residual_ptr,
        weight_ptr,
        out_ptr,
        residual_out_ptr,
        x_stride0,
        x_stride1,
        r_stride0,
        r_stride1,
        D,
        eps,
        HEIGHT: tl.constexpr,
        NDIM3: tl.constexpr,
        HAS_RESIDUAL: tl.constexpr,
        HAS_WEIGHT: tl.constexpr,
        ADD_ONE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        if NDIM3:
            batch = row // HEIGHT
            head = row % HEIGHT
            x_offset = batch * x_stride0 + head * x_stride1
            r_offset = batch * r_stride0 + head * r_stride1
        else:
            x_offset = row * x_stride0
            r_offset = row * r_stride0

        offsets = tl.arange(0, BLOCK)
        mask = offsets < D
        x = tl.load(x_ptr + x_offset + offsets, mask=mask, other=0.0).to(tl.float32)
        if HAS_RESIDUAL:
            residual = tl.load(
                residual_ptr + r_offset + offsets, mask=mask, other=0.0
            ).to(tl.float32)
            x = x + residual
            # residual_out 是新建的连续张量：按 logical row 写，不能复用输入
            # residual 的 stride（输入可能是带洞视图）。
            tl.store(
                residual_out_ptr + row * D + offsets,
                x.to(residual_out_ptr.dtype.element_ty),
                mask=mask,
            )

        variance = tl.sum(x * x, axis=0) / D
        x = x * tl.rsqrt(variance + eps)
        if HAS_WEIGHT:
            weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
            if ADD_ONE:
                # GemmaRMSNorm 在上游先把 weight.float()+1，再走 rms_norm；
                # 即使参数被模型 `.to(bf16)`，有效权重仍是 FP32。
                x = x * (weight.to(tl.float32) + 1.0)
            else:
                # 上游 native 的次序是 `x.to(weight.dtype) * weight`。先把归一化
                # 结果 cast 到 weight dtype，乘法结果也落回该 dtype，最后才写
                # output dtype；这样 bf16 权重只经历一次 bf16 舍入。
                x = x.to(weight_ptr.dtype.element_ty) * weight
                x = x.to(weight_ptr.dtype.element_ty)
        tl.store(
            out_ptr + row * D + offsets,
            x.to(out_ptr.dtype.element_ty),
            mask=mask,
        )


def _row_layout(tensor: torch.Tensor, name: str) -> tuple[int, int, int, int]:
    """返回 `(rows, D, stride0, stride1)`；1D/2D 的 stride1 没有意义，填 0。"""
    if tensor.ndim not in (1, 2, 3):
        raise ValueError(
            f"{name} 只支持 1D/2D/3D（最后一维为 hidden/head_dim），"
            f"收到 shape={tuple(tensor.shape)}")
    if tensor.ndim == 1:
        if tensor.shape[0] <= 0:
            raise ValueError(f"{name} 最后一维必须大于 0，收到 {tuple(tensor.shape)}")
        return 1, tensor.shape[0], 0, 0
    if tensor.shape[-1] <= 0:
        raise ValueError(f"{name} 最后一维必须大于 0，收到 {tuple(tensor.shape)}")
    if tensor.stride(-1) != 1:
        raise ValueError(
            f"{name} 最后一维必须连续（stride(-1)==1），收到 stride={tensor.stride()}；"
            f"带洞的最后一维不能由一个 row kernel 安全处理")
    if tensor.ndim == 2:
        rows, dim = tensor.shape
        if rows > 1 and tensor.stride(0) <= 0:
            raise ValueError(
                f"{name} 的 row stride 必须为正，收到 stride={tensor.stride()}")
        return rows, dim, tensor.stride(0), 0
    batch, height, dim = tensor.shape
    for axis, size in ((0, batch), (1, height)):
        if size > 1 and tensor.stride(axis) <= 0:
            raise ValueError(
                f"{name} 的 batch/head stride 必须为正，收到 stride={tensor.stride()}")
    return batch * height, dim, tensor.stride(0), tensor.stride(1)


def _validate(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    residual: torch.Tensor | None,
    eps: float,
    add_one: bool,
) -> tuple[int, int, int, int]:
    if not torch.is_floating_point(x):
        raise ValueError(f"RMSNorm 的 x 必须是浮点张量，收到 {x.dtype}")
    if x.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"RMSNorm 只支持 float16/bfloat16/float32，收到 {x.dtype}")
    if not isinstance(eps, (int, float)) or eps < 0:
        raise ValueError(f"RMSNorm 的 eps 必须是非负数，收到 {eps!r}")
    if not isinstance(add_one, bool):
        raise ValueError(f"add_one 必须是 bool，收到 {add_one!r}")
    if add_one and weight is None:
        raise ValueError("add_one=True（GemmaRMSNorm）必须提供 weight")

    rows, dim, stride0, stride1 = _row_layout(x, "RMSNorm 的 x")
    if weight is not None:
        if weight.ndim != 1 or weight.numel() != dim:
            raise ValueError(
                f"RMSNorm 的 weight 必须是一维且长度等于最后一维 {dim}，"
                f"收到 shape={tuple(weight.shape)}")
        if weight.device != x.device:
            raise ValueError(
                f"RMSNorm 的 x/weight 必须同设备：{x.device} vs {weight.device}")
        if weight.dtype not in _SUPPORTED_DTYPES:
            raise ValueError(
                f"RMSNorm 的 weight 只支持 float16/bfloat16/float32，"
                f"收到 {weight.dtype}")

    if residual is not None:
        if tuple(residual.shape) != tuple(x.shape):
            raise ValueError(
                f"RMSNorm 的 residual 必须与 x 同形：x={tuple(x.shape)}，"
                f"residual={tuple(residual.shape)}")
        if residual.dtype != x.dtype:
            raise ValueError(
                f"RMSNorm 的 residual 必须与 x 同 dtype：{x.dtype} vs {residual.dtype}")
        if residual.device != x.device:
            raise ValueError(
                f"RMSNorm 的 residual 必须与 x 同设备：{x.device} vs {residual.device}")
        if residual.stride(-1) != 1:
            raise ValueError(
                f"RMSNorm 的 residual 最后一维必须连续，收到 stride={residual.stride()}")
    return rows, dim, stride0, stride1


def rms_norm_native(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
    residual: torch.Tensor | None = None,
    *,
    add_one: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """显式 Torch 参考；与上游 `forward_native` 的 cast 次序一致。

    `add_one=True` 表示 GemmaRMSNorm：权重先 `float() + 1`，再做 `x.to(weight.dtype) * weight`。
    """
    _rows, _dim, _s0, _s1 = _validate(x, weight, residual, eps, add_one)

    x32 = x.to(torch.float32)
    if residual is not None:
        total = x32 + residual.to(torch.float32)
        residual_out = total.to(residual.dtype)
    else:
        total = x32
        residual_out = None

    variance = total.pow(2).mean(dim=-1, keepdim=True)
    normed = total * torch.rsqrt(variance + eps)
    if weight is not None:
        effective = weight.float() + 1.0 if add_one else weight
        # 上游先 x.to(weight.dtype)，再 * weight；最后回 orig_dtype。
        normed = normed.to(effective.dtype) * effective
    out = normed.to(x.dtype)
    if residual_out is None:
        return out
    return out, residual_out


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
    residual: torch.Tensor | None = None,
    *,
    add_one: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """生产入口：CUDA 用一行一 program 的 Triton kernel，CPU 用 native 参考。"""
    rows, dim, stride0, stride1 = _validate(x, weight, residual, eps, add_one)

    if not x.is_cuda:
        return rms_norm_native(x, weight, eps, residual, add_one=add_one)
    if triton is None:
        raise RuntimeError(
            "CUDA 上的融合 RMSNorm 需要 triton；当前环境没有 triton。"
            "没有静默回退到 Torch，请安装 triton 或显式调用 rms_norm_native")

    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    if residual is None:
        residual_out = out  # 占位；HAS_RESIDUAL=False 时 kernel 不会读写它
        r_stride0 = 0
        r_stride1 = 0
    else:
        residual_out = torch.empty(residual.shape, dtype=residual.dtype,
                                   device=residual.device)
        r_stride0, r_stride1 = _row_layout(residual, "RMSNorm 的 residual")[2:]

    if rows == 0:
        return out if residual is None else (out, residual_out)

    height = x.shape[1] if x.ndim == 3 else 1
    block = triton.next_power_of_2(dim)
    if block > 32768:
        raise ValueError(
            f"RMSNorm 的最后一维 {dim} 太大：单个 program 的 BLOCK={block} 超过 32768")
    num_warps = 4 if block <= 2048 else 8
    _rms_norm_kernel[(rows,)](
        x,
        residual if residual is not None else x,
        weight if weight is not None else x,
        out,
        residual_out,
        stride0,
        stride1,
        r_stride0,
        r_stride1,
        dim,
        float(eps),
        HEIGHT=height,
        NDIM3=x.ndim == 3,
        HAS_RESIDUAL=residual is not None,
        HAS_WEIGHT=weight is not None,
        ADD_ONE=add_one,
        BLOCK=block,
        num_warps=num_warps,
        num_stages=1,
    )
    if residual is None:
        return out
    return out, residual_out


__all__ = ["rms_norm", "rms_norm_native"]
