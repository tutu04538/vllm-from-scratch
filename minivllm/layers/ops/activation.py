"""GeGLU / SwiGLU 的 `...AndMul` 融合算子（需求 85）。

输入是已经打包的 `[gate | up]`，最后一维为偶数；输出最后一维减半：

    GeluAndMul: out = gelu(gate) * up
    SiluAndMul: out = silu(gate) * up

对应上游 `vllm/model_executor/layers/activation.py` 的同名类。CUDA 用一个 Triton
kernel 完成“切分 + 激活 + 乘 + 写回”，CPU 保留显式 Torch 参考。输入支持
1D/2D/3D、最后一维连续、外层任意正 stride；输出总是连续张量。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - 本机环境有 triton
    triton = None
    tl = None


_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

if triton is not None:

    @triton.jit
    def _act_and_mul_kernel(
        x_ptr,
        out_ptr,
        x_stride0,
        x_stride1,
        D,
        HEIGHT: tl.constexpr,
        NDIM3: tl.constexpr,
        GELU: tl.constexpr,
        APPROX_TANH: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        col_block = tl.program_id(1)
        if NDIM3:
            batch = row // HEIGHT
            head = row % HEIGHT
            x_offset = batch * x_stride0 + head * x_stride1
        else:
            x_offset = row * x_stride0

        half = D // 2
        offsets = col_block * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < half
        gate = tl.load(
            x_ptr + x_offset + offsets, mask=mask, other=0.0
        ).to(tl.float32)
        up = tl.load(
            x_ptr + x_offset + half + offsets, mask=mask, other=0.0
        ).to(tl.float32)

        if GELU:
            if APPROX_TANH:
                # 与 torch F.gelu(approximate="tanh") 同一公式；tanh 用 sigmoid 表示，
                # 避免依赖不同 Triton 版本的 libdevice 绑定。
                z = 0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)
                tanh_z = 2.0 * tl.sigmoid(2.0 * z) - 1.0
                activated = 0.5 * gate * (1.0 + tanh_z)
            else:
                activated = 0.5 * gate * (
                    1.0 + tl.erf(gate * 0.7071067811865476))
        else:
            activated = gate * tl.sigmoid(gate)

        value = activated * up
        tl.store(
            out_ptr + row * half + offsets,
            value.to(out_ptr.dtype.element_ty),
            mask=mask,
        )


def _row_layout(x: torch.Tensor, name: str) -> tuple[int, int, int, int]:
    if x.ndim not in (1, 2, 3):
        raise ValueError(
            f"{name} 只支持 1D/2D/3D（最后一维是 gate|up），收到 shape={tuple(x.shape)}")
    if x.shape[-1] <= 0:
        raise ValueError(f"{name} 最后一维必须大于 0，收到 {tuple(x.shape)}")
    if x.stride(-1) != 1:
        raise ValueError(
            f"{name} 最后一维必须连续（stride(-1)==1），收到 stride={x.stride()}")
    if x.ndim == 1:
        return 1, x.shape[0], 0, 0
    if x.ndim == 2:
        rows, dim = x.shape
        if rows > 1 and x.stride(0) <= 0:
            raise ValueError(
                f"{name} 的 row stride 必须为正，收到 stride={x.stride()}")
        return rows, dim, x.stride(0), 0
    batch, height, dim = x.shape
    for axis, size in ((0, batch), (1, height)):
        if size > 1 and x.stride(axis) <= 0:
            raise ValueError(
                f"{name} 的 batch/head stride 必须为正，收到 stride={x.stride()}")
    return batch * height, dim, x.stride(0), x.stride(1)


def _validate(x: torch.Tensor, name: str) -> tuple[int, int, int, int, int]:
    if not torch.is_floating_point(x):
        raise ValueError(f"{name} 的输入必须是浮点张量，收到 {x.dtype}")
    if x.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"{name} 只支持 float16/bfloat16/float32，收到 {x.dtype}")
    rows, dim, stride0, stride1 = _row_layout(x, name)
    if dim % 2 != 0:
        raise ValueError(
            f"{name} 需要最后一维是偶数（gate|up 各占一半），"
            f"收到 {tuple(x.shape)}")
    return rows, dim, dim // 2, stride0, stride1


def activation_and_mul_native(
    x: torch.Tensor, *, gelu: bool, approximate: str = "none"
) -> torch.Tensor:
    """显式 Torch 参考；与上游 `forward_native` 逐表达式一致。"""
    name = "GeluAndMul" if gelu else "SiluAndMul"
    _rows, _dim, half, _s0, _s1 = _validate(x, name)
    gate = x[..., :half]
    up = x[..., half:]
    if gelu:
        return F.gelu(gate, approximate=approximate) * up
    return F.silu(gate) * up


def activation_and_mul(
    x: torch.Tensor, *, gelu: bool, approximate: str = "none"
) -> torch.Tensor:
    """生产入口：CUDA 用一个融合 Triton kernel，CPU 用 native 参考。"""
    name = "GeluAndMul" if gelu else "SiluAndMul"
    rows, _dim, half, stride0, stride1 = _validate(x, name)
    if gelu and approximate not in ("none", "tanh"):
        raise ValueError(
            f"GeluAndMul 的 approximate 只能是 'none' 或 'tanh'，收到 {approximate!r}")
    if not gelu and approximate != "none":
        raise ValueError("SiluAndMul 没有 approximate 参数")

    if not x.is_cuda:
        return activation_and_mul_native(
            x, gelu=gelu, approximate=approximate)
    if triton is None:
        raise RuntimeError(
            f"CUDA 上的融合 {name} 需要 triton；当前环境没有 triton。"
            "没有静默回退到 Torch，请安装 triton 或显式调用 native 参考")

    out = torch.empty(
        x.shape[:-1] + (half,), dtype=x.dtype, device=x.device)
    if rows == 0:
        return out

    height = x.shape[1] if x.ndim == 3 else 1
    # 激活没有跨列归约：decode（行很少）按列切块，避免一行只占一个 CTA；
    # prefill 行数已经够多，一行一个 CTA 反而少一层调度开销。
    if rows <= 8 and half > 4096:
        block = 4096
        grid = (rows, triton.cdiv(half, block))
        num_warps = 4
    else:
        block = triton.next_power_of_2(half)
        if block > 32768:
            raise ValueError(
                f"{name} 的中间维度 {half} 太大：BLOCK={block} 超过 32768")
        grid = (rows,)
        num_warps = 4 if block <= 2048 else 8
    _act_and_mul_kernel[grid](
        x,
        out,
        stride0,
        stride1,
        x.shape[-1],
        HEIGHT=height,
        NDIM3=x.ndim == 3,
        GELU=gelu,
        APPROX_TANH=approximate == "tanh",
        BLOCK=block,
        num_warps=num_warps,
        num_stages=1,
    )
    return out


__all__ = [
    "activation_and_mul",
    "activation_and_mul_native",
]
