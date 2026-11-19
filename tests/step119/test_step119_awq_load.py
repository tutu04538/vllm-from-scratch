"""AWQ 表示加载基础检查；不执行量化 GEMM。"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch import nn

from minivllm.layers.linear import ColumnParallelLinear, QKVParallelLinear, MergedColumnParallelLinear
from minivllm.layers.quantization import resolve_quantization_config
from minivllm.model_loader.post_load import process_weights_after_loading

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'step118'))
import step118_w4_reference as reference

CONFIG = {'quant_method': 'awq', 'bits': 4, 'group_size': 128, 'zero_point': True, 'version': 'gemm'}


def config():
    cfg = resolve_quantization_config(None, CONFIG)
    cfg.packed_modules_mapping.update({'qkv_proj': ['q_proj', 'k_proj', 'v_proj'],
                                       'gate_up_proj': ['gate_proj', 'up_proj']})
    return cfg


def values(rows, cols):
    v = torch.arange(rows * cols, dtype=torch.int32).reshape(rows, cols)
    return (v * 12345679) ^ 0x76543210


def test_column_awq_load_convert_and_reject_forward(tmp_path):
    layer = ColumnParallelLinear(256, 16, prefix='proj', quant_config=config())
    qw, qz = values(256, 2), values(2, 2)
    scales = torch.arange(32, dtype=torch.float32).reshape(2, 16) / 16
    for param, tensor in ((layer.qweight, qw), (layer.qzeros, qz), (layer.scales, scales)):
        param.weight_loader(param, tensor)
    model = nn.Sequential(layer)
    process_weights_after_loading(model, None, 'cpu')
    expected_qw, expected_qz, _, _ = reference.awq_to_standard(qw, qz)
    assert torch.equal(layer.qweight, expected_qw)
    assert torch.equal(layer.qzeros, expected_qz)
    assert torch.equal(layer.scales, scales)
    assert layer.qweight.shape == (32, 16) and layer.qweight.packed_dim == 0
    assert layer.qzeros.shape == (2, 2) and layer.qzeros.packed_dim == 0
    assert layer.qweight.layout_state == layer.qzeros.layout_state == layer.scales.layout_state == 'standard'
    assert model._quant_postload_state == 'representation_ready'
    with pytest.raises(NotImplementedError, match='AWQ 量化执行尚未开放'):
        layer(torch.ones(1, 256))
    with pytest.raises(RuntimeError, match='只能执行一次'):
        process_weights_after_loading(model, None, 'cpu')
    # 冻结上游独立进程读取同一物理输入；比较真实生产转换，而非只做自身往返。
    inp, out = tmp_path / 'in.json', tmp_path / 'out.json'
    inp.write_text(json.dumps({'cases': [{'name': 'test', 'qweight': qw.tolist(), 'qzeros': qz.tolist()}]}))
    probe = Path(__file__).resolve().parents[1] / 'step118/step118_frozen_probe.py'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    subprocess.run([sys.executable, str(probe), '--input', str(inp), '--output', str(out)],
                   env=env, check=True, timeout=120, capture_output=True)
    upstream = json.loads(out.read_text())['cases'][0]
    assert torch.equal(layer.qweight, torch.tensor(upstream['qweight'], dtype=torch.int32))
    assert torch.equal(layer.qzeros, torch.tensor(upstream['qzeros'], dtype=torch.int32))


@pytest.mark.parametrize('kind', ['qkv', 'merged'])
def test_fused_awq_each_parameter_tracks_shards(kind):
    if kind == 'qkv':
        layer = QKVParallelLinear(256, 16, 16, 4, prefix='qkv_proj', quant_config=config())
        shards = [('v', 64), ('q', 256), ('k', 64)]
    else:
        layer = MergedColumnParallelLinear(256, [128, 128], prefix='gate_up_proj', quant_config=config())
        shards = [(1, 128), (0, 128)]
    for name in ('qweight', 'qzeros', 'scales'):
        param = getattr(layer, name)
        for shard, width in shards:
            rows = 256 if name == 'qweight' else 2
            tensor = values(rows, width // 8) if name != 'scales' else torch.ones(rows, width)
            param.weight_loader(param, tensor, shard)
        param.validate_loaded()
    with pytest.raises(ValueError, match='重复'):
        layer.scales.weight_loader(layer.scales, torch.ones(2, shards[0][1]), shards[0][0])
    process_weights_after_loading(nn.Sequential(layer), None, 'cpu')


def test_awq_group_and_pack_alignment_are_not_silently_changed():
    with pytest.raises(ValueError, match='group'):
        ColumnParallelLinear(192, 16, prefix='bad', quant_config=config())
    with pytest.raises(ValueError, match='输出逻辑 shard'):
        ColumnParallelLinear(256, 12, prefix='bad', quant_config=config())


def test_missing_scale_shard_prevents_postprocess():
    layer = MergedColumnParallelLinear(256, [128, 128], prefix='gate_up_proj', quant_config=config())
    for name in ('qweight', 'qzeros', 'scales'):
        param = getattr(layer, name)
        for shard in ([0] if name == 'scales' else [0, 1]):
            rows = 256 if name == 'qweight' else 2
            tensor = values(rows, 16) if name != 'scales' else torch.ones(rows, 128)
            param.weight_loader(param, tensor, shard)
    model = nn.Sequential(layer)
    with pytest.raises(ValueError, match='覆盖不完整'):
        process_weights_after_loading(model, None, 'cpu')
    assert model._quant_postload_state == 'failed'


def test_real_stacked_mapping_routes_all_quant_parameter_suffixes():
    from minivllm.model_loader.auto_weights_loader import AutoWeightsLoader, WeightsMapper
    parent = nn.Module()
    parent.self_attn = nn.Module()
    parent.mlp = nn.Module()
    parent.self_attn.qkv_proj = QKVParallelLinear(256, 16, 16, 4, prefix='self_attn.qkv_proj', quant_config=config())
    parent.mlp.gate_up_proj = MergedColumnParallelLinear(256, [128, 128], prefix='mlp.gate_up_proj', quant_config=config())
    mapper = WeightsMapper(orig_to_new_stacked={'.q_proj': ('.qkv_proj', 'q'),
                                               '.k_proj': ('.qkv_proj', 'k'),
                                               '.v_proj': ('.qkv_proj', 'v'),
                                               '.gate_proj': ('.gate_up_proj', 0),
                                               '.up_proj': ('.gate_up_proj', 1)})
    parts = [('self_attn.q_proj', 256), ('self_attn.k_proj', 64), ('self_attn.v_proj', 64),
             ('mlp.gate_proj', 128), ('mlp.up_proj', 128)]
    weights = []
    for prefix, width in parts:
        weights.extend([(prefix + '.qweight', values(256, width // 8)),
                        (prefix + '.qzeros', values(2, width // 8)),
                        (prefix + '.scales', torch.ones(2, width))])
    loaded = AutoWeightsLoader(parent).load_weights(sorted(weights), mapper=mapper)
    assert len(loaded) == 6
    process_weights_after_loading(parent, None, 'cpu')
    assert parent._quant_postload_state == 'representation_ready'


def test_row_tp2_loads_its_own_group_and_converts(monkeypatch):
    from types import SimpleNamespace
    import minivllm.layers.linear as linear
    qw, qz = values(512, 2), values(4, 2)
    scales = torch.arange(64, dtype=torch.float32).reshape(4, 16)
    for rank in (0, 1):
        monkeypatch.setattr(linear, '_linear_tp_group', lambda disabled, rank=rank:
                            (SimpleNamespace(), rank, 2))
        layer = linear.RowParallelLinear(512, 16, prefix='tp_row', quant_config=config())
        for name, tensor in [('qweight', qw), ('qzeros', qz), ('scales', scales)]:
            param = getattr(layer, name)
            param.weight_loader(param, tensor)
        assert torch.equal(layer.qweight, qw[rank*256:(rank+1)*256])
        assert torch.equal(layer.qzeros, qz[rank*2:(rank+1)*2])
        process_weights_after_loading(nn.Sequential(layer), None, 'cpu')
        w_expected, z_expected, _, _ = reference.awq_to_standard(
            qw[rank*256:(rank+1)*256], qz[rank*2:(rank+1)*2])
        assert torch.equal(layer.qweight, w_expected)
        assert torch.equal(layer.qzeros, z_expected)
        assert torch.equal(layer.scales, scales[rank*2:(rank+1)*2])


def test_representation_entry_rejects_unquantized_model(monkeypatch):
    from minivllm.model_loader.base_loader import BaseModelLoader
    loader = BaseModelLoader()
    model = nn.Sequential()
    model._quant_postload_state = 'ready'
    monkeypatch.setattr(loader, '_initialize_model', lambda *args, **kwargs: model)
    monkeypatch.setattr(loader, '_load_initialized_model', lambda *args: model)
    with pytest.raises(ValueError, match='没有量化线性层'):
        loader.load_model_for_representation(None, 'cpu')
