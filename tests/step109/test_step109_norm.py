"""Large/small row-count CUDA reduction must match the frozen operator."""
import pytest
import torch
from minivllm.layers.ops.layernorm import rms_norm


@pytest.mark.parametrize('rows', [255, 256, 464])
@pytest.mark.parametrize('fused', [False, True])
def test_bf16_norm_reduction_matches_frozen(rows, fused):
    from vllm import _custom_ops as ops
    torch.manual_seed(109)
    x = torch.randn(rows, 4096, device='cuda', dtype=torch.bfloat16)
    weight = torch.randn(4096, device='cuda', dtype=torch.bfloat16)
    residual = torch.randn_like(x) if fused else None
    reference_x = x.clone()
    if fused:
        reference_residual = residual.clone()
        ops.fused_add_rms_norm(reference_x, reference_residual, weight, 1e-6)
        actual_x, actual_residual = rms_norm(x, weight, 1e-6, residual, inplace=True)
        assert torch.equal(actual_residual, reference_residual)
    else:
        expected = torch.empty_like(reference_x)
        ops.rms_norm(expected, reference_x, weight, 1e-6)
        reference_x = expected
        actual_x = rms_norm(x, weight, 1e-6, inplace=True)
    assert torch.equal(actual_x, reference_x)
