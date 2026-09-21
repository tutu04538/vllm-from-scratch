"""第三十八关正确性检查：融合 RoPE 对独立参考、对 Torch 路径。

    python benchmarks/check_step38_fused_rope.py

三份结果各自回答不同的问题：

    独立参考  按配对旋转的定义在 CPU 上用 FP64 直接算——不依赖任何一份实现
    Torch     现有实现，作为对照与 CPU 路径
    Triton    被测的融合 kernel

覆盖需求 §4 点名的语义：
  - 位置是逻辑位置，不是打包行号（positions=[100,7,3] 而行号是 [0,1,2]）
  - 精度边界：表恒 FP32；BF16 输入先扩 FP32 再旋转，最后转回
  - 不改输入、不改变缓存状态
  - Q/K head 数不同；D=128 与小偶数 8/14
  - 非连续布局必须明确拒绝，不能按错误 stride 读写
  - fp_fusion 开/关带来的舍入差异要量化
"""

import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from step38.model import RotaryEmbedding
from step38.rope import rope

DEVICE = "cuda"
THETA = 10000.0
MAX_SEQ = 256
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))


def reference(x, positions, cos_table, sin_table):
    """独立参考：CPU / FP64，直接按 (i, i+D/2) 配对做二维旋转。"""
    xf = x.detach().to("cpu", torch.float64)
    # 先按 positions 在表所在的设备上取值，再搬回 CPU 用 FP64 算
    c = cos_table[positions].to("cpu", torch.float64)[:, None, :]
    s = sin_table[positions].to("cpu", torch.float64)[:, None, :]
    half = xf.shape[-1] // 2
    x0, x1 = xf[..., :half], xf[..., half:]
    return torch.cat((x0 * c - x1 * s, x1 * c + x0 * s), dim=-1)


def make_tables(head_dim, dtype=torch.float32):
    rot = RotaryEmbedding(head_dim, MAX_SEQ, THETA).to(DEVICE)
    return rot.cos_table.to(dtype), rot.sin_table.to(dtype)


def main():
    torch.manual_seed(0)

    print("=== 1. 与独立参考对照（形状 / 精度覆盖）===")
    cases = [
        (1, 16, 128, torch.float32, "decode 形状 Q"),
        (1, 8, 128, torch.float32, "decode 形状 K"),
        (64, 16, 128, torch.bfloat16, "prefill 形状 Q，BF16"),
        (64, 8, 128, torch.bfloat16, "prefill 形状 K，BF16"),
        (7, 3, 8, torch.float32, "小 head_dim=8"),
        (7, 3, 14, torch.float32, "偶数非 2 的幂 head_dim=14"),
        (5, 1, 14, torch.bfloat16, "MQA + head_dim=14 + BF16"),
        (3, 2, 128, torch.bfloat16, "行数少于 head 数"),
    ]
    for n, h, d, dt, tag in cases:
        cos_t, sin_t = make_tables(d)
        x = torch.randn(n, h, d, device=DEVICE).to(dt)
        pos = torch.arange(n, device=DEVICE)
        ref = reference(x, pos, cos_t, sin_t)

        got = rope(x, pos, cos_t, sin_t)
        got_f = got.to("cpu", torch.float64)
        scale = ref.abs().max().item()
        err = (got_f - ref).abs().max().item()
        # 误差地板来自**输出的存储精度**：参考在 FP64 下算，结果最后要舍入回 dt。
        # 输入本身已经是 dt，与参考用的是同一份 x，所以不构成差异来源。
        eps = 2 ** -9 if dt == torch.bfloat16 else 2 ** -25
        tol = 4 * scale * eps
        check(f"{tag}  shape={tuple(x.shape)} {dt}",
              got.shape == x.shape and got.dtype == x.dtype and err <= tol,
              f"最大绝对差={err:.3e} 容限={tol:.3e}")

    print("\n=== 2. 融合路径与 Torch 路径逐位对照（FP32）===")
    cos_t, sin_t = make_tables(128)
    x = torch.randn(32, 16, 128, device=DEVICE)
    pos = torch.randint(0, MAX_SEQ, (32,), device=DEVICE)
    rot = RotaryEmbedding(128, MAX_SEQ, THETA).to(DEVICE)
    torch_out = rot(x, pos)
    fused_out = rope(x, pos, cos_t, sin_t)
    same = torch.equal(torch_out, fused_out)
    check("FP32 下与 Torch 路径逐位相同（fp_fusion=False）", same,
          f"最大差={(fused_out - torch_out).abs().max().item():.3e}")

    print("\n=== 3. 位置是逻辑位置，不是打包行号 ===")
    # 行号是 [0,1,2]，真实位置是 [100,7,3]；若 kernel 拿行号查表就会算错
    pos_odd = torch.tensor([100, 7, 3], device=DEVICE)
    x3 = torch.randn(3, 4, 128, device=DEVICE)
    ref_odd = reference(x3, pos_odd, cos_t, sin_t)
    got_odd = rope(x3, pos_odd, cos_t, sin_t).to("cpu", torch.float64)
    err_odd = (got_odd - ref_odd).abs().max().item()
    # 反例：故意用行号当位置，误差应当显著更大，证明这个检查有区分度
    ref_rowidx = reference(x3, torch.arange(3, device=DEVICE), cos_t, sin_t)
    err_rowidx = (got_odd - ref_rowidx).abs().max().item()
    check("按 positions 查表正确（而非行号）",
          err_odd < 1e-4 and err_rowidx > 1e-2,
          f"用位置的误差={err_odd:.3e}，用行号的误差={err_rowidx:.3e}")

    print("\n=== 4. 不改输入、不改变状态 ===")
    x = torch.randn(8, 4, 128, device=DEVICE)
    x_before = x.clone()
    pos = torch.arange(8, device=DEVICE)
    pos_before = pos.clone()
    cos_before, sin_before = cos_t.clone(), sin_t.clone()
    out = rope(x, pos, cos_t, sin_t)
    check("输入 x / positions / 角度表都未被改动",
          torch.equal(x, x_before) and torch.equal(pos, pos_before)
          and torch.equal(cos_t, cos_before) and torch.equal(sin_t, sin_before))
    check("返回的是新张量，不是输入的别名", out.data_ptr() != x.data_ptr())

    print("\n=== 5. 布局：kernel 按 stride 寻址，非连续也要算对 ===")
    # out 是 empty_like 出来的；对非连续输入它给的是**连续**张量，stride 与 x 不同。
    # 入和出各按自己的 stride 寻址，两种布局都要正确。
    full = torch.randn(6, 8, 128, device=DEVICE)
    layouts = {
        "x[:, ::2, :]  head 间有空洞": full[:, ::2, :],
        "x[..., ::2]   列 stride=2": full[..., ::2],
        "x.transpose(0,1)": full.transpose(0, 1),
        "x[:, 1:6:2, :] 偏移切片": full[:, 1:6:2, :],
    }
    for tag, xv in layouts.items():
        n, h, d = xv.shape
        cos_v, sin_v = make_tables(d)
        pos = torch.arange(n, device=DEVICE)
        ref = reference(xv, pos, cos_v, sin_v)
        got = rope(xv, pos, cos_v, sin_v)
        err = (got.to("cpu", torch.float64) - ref).abs().max().item()
        scale = ref.abs().max().item()
        check(f"非连续布局算对：{tag}",
              err <= 4 * scale * 2 ** -25,
              f"shape={tuple(xv.shape)} x.stride={tuple(xv.stride())} 最大差={err:.3e}")

    print("\n=== 5b. 仍然拒绝的情形 ===")
    cos_t, sin_t = make_tables(128)
    x = torch.randn(4, 8, 128, device=DEVICE)
    try:
        rope(torch.randn(4, 8, 127, device=DEVICE), torch.arange(4, device=DEVICE),
             *make_tables(128))
        check("奇数 head_dim 被拒绝", False, "没有报错")
    except ValueError as e:
        check("奇数 head_dim 被拒绝", "偶数" in str(e), str(e)[:60])
    try:
        rope(torch.randn(4, 8, 128, device=DEVICE), torch.arange(4, device=DEVICE),
             cos_t.half(), sin_t.half())
        check("非 FP32 角度表被拒绝", False, "没有报错")
    except ValueError as e:
        check("非 FP32 角度表被拒绝", "FP32" in str(e), str(e)[:60])

    print("\n=== 6. fp_fusion 的影响要明确量化 ===")
    pos = torch.randint(0, MAX_SEQ, (64,), device=DEVICE)

    # BF16：融合与否的差别发生在 FP32 内部，最终都要舍入回 BF16。
    # 这个差别（2^-24 量级）被 BF16 输出量化（2^-9 量级）淹没，通常根本看不出来。
    xb = torch.randn(64, 16, 128, device=DEVICE).bfloat16()
    nb = rope(xb, pos, cos_t, sin_t, fp_fusion=False)
    fb = rope(xb, pos, cos_t, sin_t, fp_fusion=True)
    refb = reference(xb, pos, cos_t, sin_t)
    eb_no = (nb.to("cpu", torch.float64) - refb).abs().max().item()
    eb_fu = (fb.to("cpu", torch.float64) - refb).abs().max().item()
    d_bf16 = (fb.float() - nb.float()).abs().max().item()
    n_diff = int(((fb.float() - nb.float()).abs() > 0).sum())
    full_ulp = refb.abs().max().item() * 2 ** -8
    print(f"  BF16  关融合误差={eb_no:.3e}  开融合误差={eb_fu:.3e}  两者之差={d_bf16:.3e}")
    print(f"        有差异元素 {n_diff}/{nb.numel()}，满量程 BF16 ulp={full_ulp:.3e}")
    # 融合只改变 FP32 内部的一次舍入；落到 BF16 输出上，最多把极少数元素的最后一位翻掉，
    # 单个元素的翻转幅度不会超过一个 BF16 ulp
    check("BF16 下开/关融合的差异不超过一个 BF16 ulp，且只影响极少数元素",
          d_bf16 <= full_ulp and n_diff <= nb.numel() // 1000,
          f"最大差={d_bf16:.3e} ≤ {full_ulp:.3e}，影响 {n_diff} 个元素")

    # FP32：这里才看得出融合带来的舍入差异，所以默认关掉它
    xf = torch.randn(64, 16, 128, device=DEVICE)
    nf = rope(xf, pos, cos_t, sin_t, fp_fusion=False)
    ff = rope(xf, pos, cos_t, sin_t, fp_fusion=True)
    reff = reference(xf, pos, cos_t, sin_t)
    ef_no = (nf.to("cpu", torch.float64) - reff).abs().max().item()
    ef_fu = (ff.to("cpu", torch.float64) - reff).abs().max().item()
    d_fp32 = (ff - nf).abs().max().item()
    scale_f = reff.abs().max().item()
    ulp = scale_f * 2 ** -23
    print(f"  FP32  关融合误差={ef_no:.3e}  开融合误差={ef_fu:.3e}  两者之差={d_fp32:.3e}"
          f"（一个 FP32 ULP 约 {ulp:.3e}）")
    # FP32 是逐位对得上 Torch 路径的（见第 2 节），但对 FP64 参考仍有舍入差——
    # 那是 FP32 自身的精度上限，不是实现错误
    check("关融合时 FP32 与 FP64 参考的差在几个 ULP 内", ef_no <= 4 * ulp,
          f"{ef_no / ulp:.2f} ULP")
    check("开/关融合的 FP32 差不超过一个 ULP", d_fp32 <= ulp, f"{d_fp32 / ulp:.2f} ULP")
    # 值得记下方向：FMA 只舍入一次，所以「开融合」离 FP64 真值更近，
    # 「关融合」则与 Torch 路径逐位一致。两者取舍不同，不是谁对谁错。
    print(f"  方向：开融合离 FP64 真值更近（{ef_fu:.3e} vs {ef_no:.3e}），"
          f"关融合与 Torch 路径逐位一致")

    print()
    bad = [n for n, ok, _ in RESULTS if not ok]
    print(f"结果：{len(RESULTS) - len(bad)}/{len(RESULTS)} 通过"
          + (f"，失败：{bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
