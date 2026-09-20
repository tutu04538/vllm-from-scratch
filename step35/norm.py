"""融合的 RMSNorm：一个 Triton kernel 顶掉原来的一串小算子。

Torch 版本
    x_fp32 = x.float()
    v      = x_fp32.pow(2).mean(-1, keepdim=True)
    y      = ((x_fp32 * rsqrt(v + eps)) * weight.float()).to(x.dtype)

数学上是一件事，执行时却要转精度、平方、求均值、加 eps、rsqrt、两次乘、再转回来，
每一步都是一个 kernel 加一个中间 Tensor。这里把它压成「一个 program 负责一行」：
读该行与 weight，在寄存器里算完，写回该行。中间量从不落到显存。

归一化的维度统一是**最后一维**，所以任意连续张量都能看成 [行数, 行宽]：
hidden [3, 1024] 是 3 行×1024，Q [3, 16, 128] 是 48 行×128 —— 每个 head 各算
自己的均方值，不跨 head。
"""

import torch

import triton
import triton.language as tl


@triton.jit
def _rms_norm_kernel(x_ptr, w_ptr, out_ptr, eps,
                     stride_row,
                     N: tl.constexpr, BLOCK: tl.constexpr):
    # 一个 program = 一行；BLOCK 是补齐到 2 的幂后的宽度（tl.arange 只能取 2 的幂）
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(x_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)
    # 除以**实际宽度** N，不是补齐后的 BLOCK：补出来的列是 0，但它们不能进分母
    mean_sq = tl.sum(x * x, axis=0) / N
    rstd = tl.rsqrt(mean_sq + eps)

    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # 累计和乘权重都在 FP32，最后一次性写回运行精度（FP32 时是空操作）
    y = x * rstd * w
    tl.store(out_ptr + row * stride_row + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


def rms_norm(x, weight, eps):
    """返回与 x 同 shape、同 dtype 的新 Tensor；不改动 x 和 weight。"""
    if not x.is_cuda:
        raise ValueError(f"融合 RMSNorm 需要 CUDA 张量，当前在 {x.device}")
    if x.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(f"融合 RMSNorm 只支持 float32 / bfloat16，收到 {x.dtype}")
    if weight.shape != (x.shape[-1],):
        raise ValueError(f"weight 形状 {tuple(weight.shape)} 与归一化维度 {x.shape[-1]} 不符")
    if weight.dtype != x.dtype:
        raise ValueError(f"weight 与 x 的 dtype 不一致：{weight.dtype} vs {x.dtype}")
    # kernel 用 row * stride_row + column 寻址，等于认定整个张量是「行挨着行」的一整片。
    # 只查最后两维是不够的：base[::2] 那种切片的最后两维各自连续，但组与组之间有空洞，
    # 按紧凑布局去读就会读到本应跳过的数据。weight 同理，它被当成相邻的 [0..N) 来读。
    # 本关不要求支持任意非连续布局，遇到就直接拒绝，而不是悄悄算错。
    if not x.is_contiguous():
        raise ValueError(f"融合 RMSNorm 要求 x 整块连续，当前 shape={tuple(x.shape)}、"
                         f"stride={tuple(x.stride())}")
    if not weight.is_contiguous():
        raise ValueError(f"融合 RMSNorm 要求 weight 整块连续，当前 shape={tuple(weight.shape)}、"
                         f"stride={tuple(weight.stride())}")

    width = x.shape[-1]

    out = torch.empty_like(x)
    num_rows = x.numel() // width
    # tl.arange 只能取 2 的幂，所以行宽向上取整补齐；多出来的列由 mask 挡住
    block = triton.next_power_of_2(width)
    # 大约每 256 个元素给一个 warp，夹在 [1, 8]。
    # 宽度 128 这一档只给 1 个是有原因的：整行都在同一个 warp 里时，
    # tl.sum 走 warp 内的 shfl 蝶形规约，不用共享内存也不用 barrier；
    # 一开到 2 个 warp 以上，各部分和就得经共享内存汇合，还要两次 bar.sync
    # （实测 shared 0 → 16 字节、bar.sync 0 → 2）。128 个元素本来每线程才 4 个，
    # 换成每线程 1 个换不来多少并行度，抵不过同步开销。
    # 但行数很多、GPU 被填满时反过来是多给 warp 更快，所以按宽度分档而不是一律给 1。
    num_warps = min(8, max(1, block // 256))
    _rms_norm_kernel[(num_rows,)](
        x, weight, out, eps,
        x.stride(-2) if x.dim() > 1 else width,
        N=width, BLOCK=block, num_warps=num_warps,
    )
    return out
