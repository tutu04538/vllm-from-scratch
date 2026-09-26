"""融合的 RoPE：一次 forward 用一个 Triton kernel 完成旋转。

Torch 版本

    cos = cos_table[positions]                 # 按位置查表        [N, D/2]
    cos = cat((cos, cos), dim=-1)              # 拼成 D 宽          [N, D]
    rotated = x.float() * cos + rotate_half(x).float() * sin
    return rotated.to(x.dtype)

数学上就是一组二维旋转，执行时却要经过查表、拼接、拆前后半段、配对、两次乘、一次加、
两次转精度——每一步都是一个 GPU kernel 加一个中间 Tensor。这里压成
「一个 program 负责一行的一个 head」：直接按地址读配对的两个分量和对应的 cos/sin，
在寄存器里算完写回。中间不产生 `rotate_half(x)`，也不产生拼接后的角度表。

配对是**前后半段**，不是相邻两个元素：

    x = [a, b, c, d]     配对是 (a, c)、(b, d)
    y[i]     = x[i]     * cos[i] - x[i + D/2] * sin[i]
    y[i+D/2] = x[i+D/2] * cos[i] + x[i]     * sin[i]

所以 cos/sin 表存的是**半宽** D/2，正好每个角度只存一份，拼接那一步也就不需要了。

寻址与分页 attention 的 kernel 同一套约定：**入和出各按自己的 stride 寻址**。
out 由 `empty_like` 产生，它给的 stride 不一定和 x 一样（换序但稠密的布局会保留 stride，
切片切出空洞的会补成连续），共用一套就会写错地址，所以 x 与 out 各传三个 stride；
角度表与 positions 是模块自己的 buffer / 视图，恒为连续，按单位列 stride 读。
x 因此可以是任意稠密布局。
"""

import torch

import triton
import triton.language as tl


@triton.jit
def _rope_kernel(x_ptr, out_ptr, cos_ptr, sin_ptr, pos_ptr,
                 stride_xn, stride_xh, stride_xd,
                 stride_on, stride_oh, stride_od,
                 D_HALF: tl.constexpr, BLOCK: tl.constexpr):
    # 一个 program = 一行的一个 head。行与 head 都由 grid 展开。
    row = tl.program_id(0)
    head = tl.program_id(1)

    # 位置必须查表得来：打包的两行可能分别属于两个请求，真实位置可能是 [100, 7]，
    # 拿打包行号 [0, 1] 当位置会算错
    pos = tl.load(pos_ptr + row)

    offs = tl.arange(0, BLOCK)
    # D/2 未必是 2 的幂（例如 head_dim=14 → 7），补齐出来的列由 mask 挡住
    mask = offs < D_HALF

    # 入和出**各按自己的 stride**寻址。不能共用一套：out 是 empty_like 出来的，
    # 它给的 stride 不一定和 x 一样，共用就会写到错误地址。
    # （分页 attention 的 kernel 也是这么分的：stride_q* 与 stride_o* 各一套。）
    x_base = row * stride_xn + head * stride_xh
    o_base = row * stride_on + head * stride_oh

    # 前后半段配对：偏移相差 D_HALF
    x0 = tl.load(x_ptr + x_base + offs * stride_xd, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x_ptr + x_base + (D_HALF + offs) * stride_xd,
                 mask=mask, other=0.0).to(tl.float32)
    # 角度表是模块自己的 buffer，恒为连续，按单位列 stride 读
    # 角度表恒为 FP32，不从输入精度走
    c = tl.load(cos_ptr + pos * D_HALF + offs, mask=mask, other=0.0)
    s = tl.load(sin_ptr + pos * D_HALF + offs, mask=mask, other=0.0)

    # 旋转全程 FP32
    y0 = x0 * c - x1 * s
    y1 = x1 * c + x0 * s

    # 只在写回时做一次舍入，回到输入精度（FP32 输入时是空操作）
    out_ty = out_ptr.dtype.element_ty
    tl.store(out_ptr + o_base + offs * stride_od, y0.to(out_ty), mask=mask)
    tl.store(out_ptr + o_base + (D_HALF + offs) * stride_od, y1.to(out_ty), mask=mask)


def rope(x, positions, cos_table, sin_table, fp_fusion=False):
    """返回与 x 同 shape、同 dtype、同设备的新 Tensor；不改动 x、positions 或角度表。

    fp_fusion：是否允许编译器把 `x0*c - x1*s` 收缩成一次乘加（FMA）。
    默认 False，这样每一步乘法/加法各自舍入一次，与 Torch 参考路径的舍入次数一致。
    开着更快，但会让计算结果与 FP32 参考产生 1 ULP 量级的差别——不是不能开，
    而是开了就必须把这件事讲清楚，不能靠放宽误差容限蒙过去。
    """
    if not x.is_cuda:
        raise ValueError(f"融合 RoPE 需要 CUDA 张量，当前在 {x.device}")
    if x.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(f"融合 RoPE 只支持 float32 / bfloat16，收到 {x.dtype}")
    if x.dim() != 3:
        raise ValueError(f"融合 RoPE 要求输入是 [N, heads, head_dim]，收到 shape={tuple(x.shape)}")
    head_dim = x.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError(f"融合 RoPE 要求 head_dim 是偶数（前后半段配对），收到 {head_dim}")
    for name, t in (("cos_table", cos_table), ("sin_table", sin_table)):
        if t.dtype != torch.float32:
            raise ValueError(f"{name} 必须是 FP32（角度表不跟着模型精度走），收到 {t.dtype}")
        if t.shape[-1] != head_dim // 2:
            raise ValueError(f"{name} 的宽度 {t.shape[-1]} 与 head_dim/2={head_dim // 2} 不符")
    if positions.dim() != 1 or positions.shape[0] != x.shape[0]:
        raise ValueError(f"positions 形状 {tuple(positions.shape)} 与输入行数 {x.shape[0]} 不符")
    # x 不做连续性要求：kernel 按传入的 stride 寻址，任意稠密布局都对。
    # 下面这三个是模块自己的 buffer / 视图，本来就是连续的，kernel 也按单位列 stride 读，
    # 所以照旧要求连续——这不是能力限制，只是把隐含前提写出来。
    for name, t in (("cos_table", cos_table), ("sin_table", sin_table),
                    ("positions", positions)):
        if not t.is_contiguous():
            raise ValueError(f"融合 RoPE 要求 {name} 整块连续，当前 shape={tuple(t.shape)}、"
                             f"stride={tuple(t.stride())}")
    # 位置的下标边界由调用方保证：Engine 已经在推进长度之前校验过区间。
    # 这里不做 min/max 检查——那要在热路径上对 GPU 张量取 .item()，强制一次同步。
    # 后果要说清楚：越界的位置**不会报错**，kernel 会照着 pos 去读角度表外面的内存，
    # 算出一组看着像样的垃圾。这是拿掉检查换来的，不是可以忽略的细节。

    out = torch.empty_like(x)
    num_rows, num_heads, _ = x.shape
    if num_rows == 0 or num_heads == 0:
        return out

    d_half = head_dim // 2
    block = triton.next_power_of_2(d_half)
    # 每个 program 只搬 2*BLOCK 个元素，宽块才需要更多 warp
    num_warps = max(1, min(4, block // 64))
    _rope_kernel[(num_rows, num_heads)](
        x, out, cos_table, sin_table, positions,
        x.stride(0), x.stride(1), x.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        D_HALF=d_half, BLOCK=block,
        num_warps=num_warps, enable_fp_fusion=fp_fusion,
    )
    return out
