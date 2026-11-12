"""CUDA RMSNorm's in-place residual lifetime follows the frozen operator."""
import pytest
import torch
import vllm._custom_ops

from minivllm.layers.layernorm import RMSNorm


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('strided', [False, True])
def test_fused_rmsnorm_updates_input_storage(dtype, strided):
    assert torch.cuda.is_available()
    gen = torch.Generator(device='cuda').manual_seed(108)
    shape = (3, 64 if strided else 32)
    x = torch.randn(shape, device='cuda', dtype=dtype, generator=gen)[:, :32]
    residual = torch.randn(shape, device='cuda', dtype=dtype, generator=gen)[:, :32]
    weight = torch.randn(32, device='cuda', dtype=dtype, generator=gen)
    ref_x, ref_residual = x.contiguous().clone(), residual.contiguous().clone()
    torch.ops._C.fused_add_rms_norm(ref_x, ref_residual, weight, 1e-6)
    layer = RMSNorm(32).to(device='cuda', dtype=dtype)
    with torch.no_grad():
        layer.weight.copy_(weight)
        alias = residual
        got, got_residual = layer(x, residual)
    assert got.data_ptr() == x.data_ptr()
    assert got_residual.data_ptr() == residual.data_ptr() == alias.data_ptr()
    torch.testing.assert_close(got, ref_x, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(alias, ref_residual, rtol=0, atol=0)
