"""CPU 精度转换小例，只验证数值语义，不代表真实产物验收。"""
import importlib.util
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[2] / 'benchmarks/step114_convert_fp8.py'
spec = importlib.util.spec_from_file_location('step114_fp8_cpu_converter', SCRIPT)
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


def test_fp8_quantization_scale_reconstruction_and_zero():
    values = torch.tensor([[0., -1., 2., 4.]], dtype=torch.bfloat16)
    q, scale = converter.fp8_quantize(values)
    assert q.dtype == torch.float8_e4m3fn and scale.dtype == torch.float32
    torch.testing.assert_close(scale, torch.tensor(4 / 448), rtol=0, atol=0)
    torch.testing.assert_close(q.float() * scale, values.float(), rtol=0.05, atol=0.02)
    q, scale = converter.fp8_quantize(torch.zeros((2, 4), dtype=torch.bfloat16))
    assert scale.item() == 1.0 and torch.count_nonzero(q.float()).item() == 0


def test_fp8_conversion_rejects_invalid_source_and_skips_non_linear():
    with pytest.raises(ValueError, match='non-finite'):
        converter.fp8_quantize(torch.tensor([[float('nan')]], dtype=torch.bfloat16))
    with pytest.raises(ValueError, match='BF16'):
        converter.fp8_quantize(torch.ones((1, 2), dtype=torch.float32))
    assert converter.is_linear_weight('model.layers.0.self_attn.q_proj.weight')
    assert not converter.is_linear_weight('model.layers.0.self_attn.q_norm.weight')
    assert not converter.is_linear_weight('lm_head.weight')


def test_fp8_conversion_is_deterministic_on_independent_tensors():
    weights = [torch.arange(-16, 16, dtype=torch.bfloat16).reshape(4, 8),
               torch.tensor([[0., 0.125], [-0.25, 0.75]], dtype=torch.bfloat16)]
    for weight in weights:
        first, first_scale = converter.fp8_quantize(weight)
        second, second_scale = converter.fp8_quantize(weight.clone())
        assert torch.equal(first.view(torch.uint8), second.view(torch.uint8))
        assert torch.equal(first_scale.view(torch.int32), second_scale.view(torch.int32))
