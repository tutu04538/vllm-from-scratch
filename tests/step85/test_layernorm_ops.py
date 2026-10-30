"""需求 85：融合 RMSNorm / GemmaRMSNorm 的数值与契约测试。

覆盖：
* 2D/3D、FP32/BF16、有/无 residual、有/无 weight；
* CPU native 与上游 `vllm.ir.ops.rms_norm` 逐值对照；
* CUDA Triton 与同一 native 参考对照；
* packed QKV 切片式的带洞 view、Q/K norm 式 3D view；
* residual 输入不被原地改写；
* CUDA Graph capture/replay 基础兼容。
"""

from __future__ import annotations

import pytest
import torch

from minivllm.layers import GemmaRMSNorm, RMSNorm
from minivllm.layers.ops.layernorm import rms_norm, rms_norm_native

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="step85 Norm GPU 测试需要 CUDA")


def _reference(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
    residual: torch.Tensor | None = None,
    *,
    add_one: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    x32 = x.float()
    if residual is not None:
        total = x32 + residual.float()
        residual_out = total.to(residual.dtype)
    else:
        total = x32
        residual_out = None
    var = total.pow(2).mean(-1, keepdim=True)
    normed = total * torch.rsqrt(var + eps)
    if weight is not None:
        effective = weight.float() + 1.0 if add_one else weight
        normed = normed.to(effective.dtype) * effective
    out = normed.to(x.dtype)
    return out if residual_out is None else (out, residual_out)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(5, 64), (3, 4, 64)])
@requires_cuda
def test_cuda_rms_norm_matches_native(dtype, shape):
    generator = torch.Generator(device="cuda").manual_seed(101)
    x = torch.randn(*shape, dtype=dtype, device="cuda", generator=generator)
    weight = torch.randn(shape[-1], dtype=dtype, device="cuda", generator=generator)
    got = rms_norm(x, weight, 1e-6)
    want = rms_norm_native(x, weight, 1e-6)
    atol = 2e-6 if dtype == torch.float32 else 0.0
    torch.testing.assert_close(got, want, rtol=0, atol=atol)
    assert got.is_contiguous()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@requires_cuda
def test_cuda_fused_residual_matches_native_and_does_not_mutate(dtype):
    generator = torch.Generator(device="cuda").manual_seed(102)
    x = torch.randn(7, 128, dtype=dtype, device="cuda", generator=generator)
    residual = torch.randn(7, 128, dtype=dtype, device="cuda", generator=generator)
    weight = torch.randn(128, dtype=dtype, device="cuda", generator=generator)
    residual_before = residual.clone()
    x_before = x.clone()

    got, got_residual = rms_norm(x, weight, 1e-6, residual)
    want, want_residual = rms_norm_native(x, weight, 1e-6, residual)
    atol = 2e-6 if dtype == torch.float32 else 0.0
    torch.testing.assert_close(got, want, rtol=0, atol=atol)
    torch.testing.assert_close(got_residual, want_residual, rtol=0, atol=0)
    assert torch.equal(x, x_before)
    assert torch.equal(residual, residual_before)
    assert got_residual.is_contiguous()


@requires_cuda
def test_cuda_has_weight_false_and_gemma_add_one():
    generator = torch.Generator(device="cuda").manual_seed(103)
    x = torch.randn(4, 32, dtype=torch.bfloat16, device="cuda", generator=generator)
    # 无权重：不能偷偷乘全 1，也不能要求 weight 参数存在。
    got = rms_norm(x, None, 1e-6)
    want = rms_norm_native(x, None, 1e-6)
    torch.testing.assert_close(got, want, rtol=0, atol=0)

    # Gemma：(1+w) 且在 FP32 里乘；weight 是 FP32 参数。
    weight = torch.randn(32, dtype=torch.float32, device="cuda", generator=generator)
    got = rms_norm(x, weight, 1e-6, add_one=True)
    want = rms_norm_native(x, weight, 1e-6, add_one=True)
    torch.testing.assert_close(got, want, rtol=0, atol=0)


@requires_cuda
def test_cuda_packed_qkv_slice_and_3d_head_view():
    """QKV 打包后的 q/k/v 切片是带 offset 的 view；Q/K norm 还要求 3D。"""
    generator = torch.Generator(device="cuda").manual_seed(104)
    base = torch.randn(5, 1024, dtype=torch.bfloat16, device="cuda",
                       generator=generator)
    # 模拟 qkv_proj 的 q 段，再拆成 [tokens, heads, head_dim]。
    q = base[:, :512].view(5, 4, 128)
    assert not q.is_contiguous()
    weight = torch.randn(128, dtype=torch.bfloat16, device="cuda",
                         generator=generator)
    got = rms_norm(q, weight, 1e-6)
    want = rms_norm_native(q, weight, 1e-6)
    torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert got.is_contiguous()
    assert got.stride() == (512, 128, 1)


@requires_cuda
def test_cuda_graph_capture_replay_norm():
    generator = torch.Generator(device="cuda").manual_seed(105)
    x = torch.randn(4, 256, dtype=torch.float32, device="cuda",
                    generator=generator)
    weight = torch.randn(256, dtype=torch.float32, device="cuda",
                         generator=generator)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            warm = rms_norm(x, weight, 1e-6)
    torch.cuda.current_stream().wait_stream(side)
    assert torch.isfinite(warm).all()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = rms_norm(x, weight, 1e-6)

    x.mul_(1.25)
    graph.replay()
    torch.cuda.synchronize()
    expected = rms_norm_native(x, weight, 1e-6)
    torch.testing.assert_close(captured, expected, rtol=0, atol=2e-6)


@requires_cuda
def test_module_classes_keep_weight_semantics():
    x = torch.randn(3, 16, dtype=torch.float32, device="cuda",
                    generator=torch.Generator(device="cuda").manual_seed(106))
    rms = RMSNorm(16).cuda()
    with torch.no_grad():
        rms.weight.normal_()
    torch.testing.assert_close(
        rms(x), rms_norm(x, rms.weight, rms.variance_epsilon), rtol=0, atol=0)

    gemma = GemmaRMSNorm(16).cuda()
    assert torch.count_nonzero(gemma.weight) == 0
    torch.testing.assert_close(
        gemma(x), rms_norm(x, gemma.weight, gemma.variance_epsilon, add_one=True),
        rtol=0, atol=0)

    no_weight = RMSNorm(16, has_weight=False).cuda()
    assert no_weight.weight is None
    torch.testing.assert_close(
        no_weight(x), rms_norm(x, None, no_weight.variance_epsilon),
        rtol=0, atol=0)


@requires_cuda
def test_invalid_layouts_are_rejected():
    x = torch.randn(4, 8, device="cuda")
    with pytest.raises(ValueError, match="最后一维必须连续"):
        rms_norm(x[:, ::2], torch.ones(4, device="cuda"), 1e-6)
    with pytest.raises(ValueError, match="只支持 1D/2D/3D"):
        rms_norm(torch.randn(2, 2, 2, 2, device="cuda"), None, 1e-6)
    with pytest.raises(ValueError, match="同形"):
        rms_norm(x, None, 1e-6, torch.randn(1, 8, device="cuda"))
    with pytest.raises(ValueError, match="长度等于最后一维"):
        rms_norm(x, torch.ones(7, device="cuda"), 1e-6)


def test_cpu_native_matches_upstream_ir():
    """CPU 路径必须与上游 `ir.ops.rms_norm` 同值；不是“能跑就行”。"""
    from vllm.ir.ops.layernorm import rms_norm as upstream_rms_norm

    for dtype in (torch.float32, torch.bfloat16):
        x = torch.randn(4, 32, dtype=dtype,
                        generator=torch.Generator().manual_seed(107))
        weight = torch.randn(32, dtype=dtype,
                             generator=torch.Generator().manual_seed(108))
        torch.testing.assert_close(
            rms_norm_native(x, weight, 1e-6),
            upstream_rms_norm(x, weight, 1e-6),
            rtol=0, atol=0,
        )
        gemma_weight = torch.randn(
            32, dtype=torch.float32,
            generator=torch.Generator().manual_seed(109))
        torch.testing.assert_close(
            rms_norm_native(x, gemma_weight, 1e-6, add_one=True),
            upstream_rms_norm(x, gemma_weight.float() + 1.0, 1e-6),
            rtol=0, atol=0,
        )


def test_cpu_native_residual_matches_reference_without_mutation():
    x = torch.randn(3, 32, dtype=torch.float32,
                    generator=torch.Generator().manual_seed(110))
    residual = torch.randn(3, 32, dtype=torch.float32,
                           generator=torch.Generator().manual_seed(111))
    weight = torch.randn(32, dtype=torch.float32,
                         generator=torch.Generator().manual_seed(112))
    x0, r0 = x.clone(), residual.clone()
    got = rms_norm_native(x, weight, 1e-6, residual)
    want = _reference(x, weight, 1e-6, residual)
    torch.testing.assert_close(got[0], want[0], rtol=0, atol=0)
    torch.testing.assert_close(got[1], want[1], rtol=0, atol=0)
    assert torch.equal(x, x0)
    assert torch.equal(residual, r0)
