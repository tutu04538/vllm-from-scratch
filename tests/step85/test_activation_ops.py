"""需求 85：共享 GeluAndMul / SiluAndMul 的数值与契约测试。

GELU 的 exact erf 与 tanh 近似不能互换；这里分别对照上游 `forward_native`，
CUDA 再对照“FP32 计算后按输入 dtype 写回”的参考。
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from minivllm.layers import GeluAndMul, SiluAndMul, get_act_and_mul_fn
from minivllm.layers.ops.activation import (activation_and_mul,
                                            activation_and_mul_native)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="step85 激活 GPU 测试需要 CUDA")


def _upstream_gelu(x: torch.Tensor, approximate: str) -> torch.Tensor:
    from vllm.model_executor.layers.activation import GeluAndMul as UpGelu

    # forward_native 只读 self.approximate；不需要构造 CustomOp / VllmConfig。
    return UpGelu.forward_native(
        type("Ref", (), {"approximate": approximate})(), x)


def _fp32_reference(x: torch.Tensor, *, gelu: bool,
                    approximate: str) -> torch.Tensor:
    gate, up = x.float().chunk(2, dim=-1)
    value = (F.gelu(gate, approximate=approximate) if gelu
             else F.silu(gate)) * up
    return value.to(x.dtype)


def test_cpu_matches_upstream_native():
    from vllm.model_executor.layers.activation import SiluAndMul as UpSilu

    x = torch.randn(4, 64, generator=torch.Generator().manual_seed(201))
    for approximate in ("none", "tanh"):
        torch.testing.assert_close(
            GeluAndMul(approximate)(x),
            _upstream_gelu(x, approximate),
            rtol=0, atol=0,
        )
    torch.testing.assert_close(
        SiluAndMul()(x), UpSilu.forward_native(x), rtol=0, atol=0)


def test_cpu_3d_and_odd_errors():
    x = torch.randn(2, 3, 64, generator=torch.Generator().manual_seed(202))
    assert GeluAndMul("tanh")(x).shape == (2, 3, 32)
    assert SiluAndMul()(x).shape == (2, 3, 32)
    with pytest.raises(ValueError, match="偶数"):
        SiluAndMul()(torch.randn(2, 63))
    with pytest.raises(ValueError, match="approximate"):
        GeluAndMul("erf")
    with pytest.raises(ValueError, match="未适配"):
        get_act_and_mul_fn("relu")


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(6, 128), (2, 4, 128)])
def test_cuda_matches_fp32_reference(dtype, shape):
    x = torch.randn(*shape, dtype=dtype, device="cuda",
                    generator=torch.Generator(device="cuda").manual_seed(203))
    for gelu, approximate in ((True, "none"), (True, "tanh"), (False, "none")):
        got = activation_and_mul(x, gelu=gelu, approximate=approximate)
        want = _fp32_reference(x, gelu=gelu, approximate=approximate)
        # FP32 的 erf/sigmoid 实现与 Triton 有末位归约差；BF16 写回后应逐位相同。
        if dtype == torch.float32:
            torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6)
        else:
            torch.testing.assert_close(got, want, rtol=0, atol=0)


@requires_cuda
@pytest.mark.parametrize("gelu,approximate", [(True, "none"), (True, "tanh"),
                                               (False, "none")])
def test_cuda_noncontiguous_2d_and_3d(gelu, approximate):
    generator = torch.Generator(device="cuda").manual_seed(204)
    base = torch.randn(5, 1024, dtype=torch.bfloat16, device="cuda",
                       generator=generator)
    x2 = base[:, :512]
    assert not x2.is_contiguous()
    got = activation_and_mul(x2, gelu=gelu, approximate=approximate)
    want = _fp32_reference(x2, gelu=gelu, approximate=approximate)
    torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert got.is_contiguous()

    x3 = base[:, :512].view(5, 4, 128)
    got = activation_and_mul(x3, gelu=gelu, approximate=approximate)
    want = _fp32_reference(x3, gelu=gelu, approximate=approximate)
    torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert got.shape == (5, 4, 64)


@requires_cuda
def test_cuda_matches_shared_module_and_native():
    x = torch.randn(3, 256, dtype=torch.float32, device="cuda",
                    generator=torch.Generator(device="cuda").manual_seed(205))
    for module in (GeluAndMul("none"), GeluAndMul("tanh"), SiluAndMul()):
        torch.testing.assert_close(
            module(x), activation_and_mul_native(
                x, gelu=isinstance(module, GeluAndMul),
                approximate=getattr(module, "approximate", "none")),
            rtol=1e-6, atol=1e-6)


def test_invalid_layouts_are_rejected():
    with pytest.raises(ValueError, match="最后一维必须连续"):
        activation_and_mul_native(torch.randn(2, 64)[:, ::2], gelu=False)
    with pytest.raises(ValueError, match="只支持 1D/2D/3D"):
        activation_and_mul_native(torch.randn(2, 2, 2, 64), gelu=False)
    with pytest.raises(ValueError, match="偶数"):
        activation_and_mul(torch.randn(2, 63), gelu=False)


@requires_cuda
def test_cuda_graph_capture_replay_activation():
    x = torch.randn(4, 512, dtype=torch.float32, device="cuda",
                    generator=torch.Generator(device="cuda").manual_seed(206))

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            activation_and_mul(x, gelu=True, approximate="tanh")
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = activation_and_mul(x, gelu=True, approximate="tanh")

    x.mul_(1.5)
    graph.replay()
    torch.cuda.synchronize()
    want = _fp32_reference(x, gelu=True, approximate="tanh")
    torch.testing.assert_close(captured, want, rtol=1e-6, atol=1e-6)
