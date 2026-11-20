"""CPU/meta 检查，不能代替 GPU GEMM 正确性。"""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from minivllm.layers.quantization import marlin
from minivllm.quant_ops import ops

ROOT = Path(__file__).resolve().parents[2]


def test_scalar_type_id_matches_frozen():
    source = ROOT / 'learning_notes/14_vllm_from_scratch/量化完整需求/参考资料/vllm_0_28_0/vllm/scalar_type.py'
    spec = importlib.util.spec_from_file_location('step122_frozen_scalar_type', source)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    exec(compile(source.read_text(), str(source), "exec"), mod.__dict__)
    assert ops.MARLIN_UINT4_ID == mod.scalar_types.uint4.id


def fake_layer(k=256, n=72, pn=128):
    def param(data):
        p = torch.nn.Parameter(data, requires_grad=False)
        p.layout_state = 'kernel'
        return p
    return SimpleNamespace(
        qweight=param(torch.empty((k // 16, pn * 2), device='meta', dtype=torch.int32)),
        qzeros=param(torch.empty((k // 128, pn // 8), device='meta', dtype=torch.int32)),
        scales=param(torch.empty((k // 128, pn), device='meta', dtype=torch.bfloat16)),
        g_idx=torch.empty(0, device='meta', dtype=torch.int32),
        g_idx_sort_indices=torch.empty(0, device='meta', dtype=torch.int32),
        marlin_logical_nk=(n, k), marlin_padded_nk=(pn, k), marlin_is_k_full=True,
    )


@pytest.mark.parametrize('shape', [(1, 256), (2, 256), (7, 256), (16, 256), (32, 256), (128, 256), (2, 3, 256)])
def test_local_shape_bias_and_call_options(monkeypatch, shape):
    layer = fake_layer()
    layer.bias = torch.empty(72, device='meta', dtype=torch.bfloat16)
    layer.marlin_bias = torch.empty(128, device='meta', dtype=torch.bfloat16)
    workspace = torch.empty(128, device='meta', dtype=torch.int32)
    calls = []
    def fake_gemm(a, weight, scales, zeros, g_idx, perm, ws, **options):
        calls.append((a, weight, scales, zeros, g_idx, perm, ws, options))
        return torch.empty((a.shape[0], options['size_n']), device='meta', dtype=a.dtype)
    monkeypatch.setattr(marlin, 'marlin_gemm_w4a16', fake_gemm)
    x = torch.empty(shape, device='meta', dtype=torch.bfloat16)
    out = marlin.apply_marlin_w4a16(layer, x, workspace, layer.bias)
    assert out.shape == shape[:-1] + (72,)
    a, weight, scales, zeros, gi, perm, ws, options = calls[0]
    assert a.shape == (x.numel() // 256, 256)
    assert ws is workspace and gi is layer.g_idx and perm is layer.g_idx_sort_indices
    assert options['bias'] is layer.marlin_bias
    assert options['use_atomic_add'] is False and options['use_fp32_reduce'] is True
    assert options['size_n'] == 128 and options['size_k'] == 256
    marlin.apply_marlin_w4a16(layer, x, workspace, None)
    assert calls[-1][-1]['bias'] is None
    assert layer.bias.shape == (72,)


def test_wrong_input_and_unprepared_bias_rejected():
    layer = fake_layer()
    x = torch.empty((2, 255), device='meta', dtype=torch.bfloat16)
    ws = torch.empty(128, device='meta', dtype=torch.int32)
    with pytest.raises(ValueError, match='逻辑 K'):
        marlin.apply_marlin_w4a16(layer, x, ws)
    x = torch.empty((2, 256), device='meta', dtype=torch.float16)
    with pytest.raises(ValueError, match='dtype'):
        marlin.apply_marlin_w4a16(layer, x, ws)
    x = x.to(torch.bfloat16)
    with pytest.raises(ValueError, match='bias'):
        marlin.apply_marlin_w4a16(layer, x, ws, torch.empty(72, device='meta', dtype=x.dtype))


def test_atomic_strategy_matches_frozen(monkeypatch):
    dev = torch.device('cuda:0')
    monkeypatch.delenv('VLLM_MARLIN_USE_ATOMIC_ADD', raising=False)
    assert not marlin.should_use_atomic_add_reduce(1, 128, 4096, dev, torch.float16)
    monkeypatch.setenv('VLLM_MARLIN_USE_ATOMIC_ADD', '1')
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device: (8, 9))
    assert marlin.should_use_atomic_add_reduce(1, 128, 4096, dev, torch.float16)
    assert not marlin.should_use_atomic_add_reduce(1, 128, 4096, dev, torch.bfloat16)
    assert not marlin.should_use_atomic_add_reduce(1, 2048, 4096, dev, torch.float16)
    assert not marlin.should_use_atomic_add_reduce(1, 128, 1024, dev, torch.float16)


def test_meta_operator_and_invalid_inputs():
    ops.prepare_full_extension()
    layer = fake_layer()
    a = torch.empty((7, 256), device='meta', dtype=torch.bfloat16)
    ws = torch.empty(128, device='meta', dtype=torch.int32)
    def invoke(a=a, weight=None, zeros=None, workspace=ws):
        return ops.marlin_gemm_w4a16(a, layer.qweight if weight is None else weight,
                 layer.scales, layer.qzeros if zeros is None else zeros, layer.g_idx,
                 layer.g_idx_sort_indices, workspace, size_n=128, size_k=256)
    assert invoke().shape == (7, 128)
    with pytest.raises(ValueError, match='weight'):
        invoke(weight=torch.empty((16, 255), device='meta', dtype=torch.int32))
    with pytest.raises(ValueError, match='zeros'):
        invoke(zeros=torch.empty((2, 15), device='meta', dtype=torch.int32))
    with pytest.raises(ValueError, match='workspace'):
        invoke(workspace=torch.empty(0, device='meta', dtype=torch.int32))
    with pytest.raises(ValueError, match='stride'):
        invoke(a=torch.empty((7, 512), device='meta', dtype=torch.bfloat16)[:, ::2])


def test_bias_kernel_copy_keeps_logical_skip_bias_order():
    bias = torch.arange(72, dtype=torch.bfloat16)
    before = bias.clone()
    kernel_bias = marlin.marlin_permute_bias(bias, 128)
    assert torch.equal(bias, before) and bias.shape == (72,)
    assert kernel_bias.shape == (128,)
    assert not torch.equal(kernel_bias[:72], bias)
    # 期望按冻结 bias permutation 置换，逻辑返回仍保留原顺序。
    perm = [2 * i + j for i in range(4) for j in (0, 1, 8, 9, 16, 17, 24, 25)]
    expected = torch.nn.functional.pad(bias, (0, 56)).reshape(-1, 32)[:, perm].reshape(128)
    assert torch.equal(kernel_bias, expected)
