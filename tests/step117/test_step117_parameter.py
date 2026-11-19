"""量化参数小型 CPU fixture；不代表 GPU 量化 kernel 正确性。"""
import pytest
import torch
from torch import nn

from minivllm.layers.parameter import (PackedvLLMParameter, GroupQuantScaleParameter,
                                       PerTensorScaleParameter, permute_param_layout_)
from minivllm.layers.linear import ColumnParallelLinear, RowParallelLinear, MergedColumnParallelLinear
from minivllm.model_loader.post_load import process_weights_after_loading


def packed(shape, layer, required=None):
    return PackedvLLMParameter(torch.empty(shape, dtype=torch.int32), layer.weight_loader,
                               input_dim=0, output_dim=1, packed_dim=0, packed_factor=8,
                               required_shards=required, name='test.qweight')


def test_row_and_column_shard_weights_and_scales():
    w = torch.arange(32 * 256, dtype=torch.int32).reshape(32, 256)
    scales = torch.arange(2 * 256, dtype=torch.float32).reshape(2, 256)
    for rank in (0, 1):
        col = ColumnParallelLinear(256, 256)
        col.tp_rank, col.tp_size = rank, 2
        p = packed((32, 128), col)
        s = GroupQuantScaleParameter(torch.empty(2, 128), col.weight_loader, input_dim=0, output_dim=1)
        col.weight_loader(p, w)
        col.weight_loader(s, scales)
        assert torch.equal(p, w[:, rank * 128:(rank + 1) * 128])
        assert torch.equal(s, scales[:, rank * 128:(rank + 1) * 128])
        p.validate_loaded(); s.validate_loaded()
        row = RowParallelLinear(256, 256)
        row.tp_rank, row.tp_size = rank, 2
        p = packed((16, 256), row)
        s = GroupQuantScaleParameter(torch.empty(1, 256), row.weight_loader, input_dim=0, output_dim=1)
        row.weight_loader(p, w)
        row.weight_loader(s, scales)
        assert torch.equal(p, w[rank * 16:(rank + 1) * 16])
        assert torch.equal(s, scales[rank:rank + 1])


def test_fused_scale_out_of_order_and_duplicate():
    layer = MergedColumnParallelLinear(8, [4, 4])
    param = PerTensorScaleParameter(torch.empty(2), layer.weight_loader, required_shards={0, 1})
    layer.weight_loader(param, torch.tensor(3.), 1)
    with pytest.raises(ValueError, match='覆盖不完整'):
        param.validate_loaded()
    layer.weight_loader(param, torch.tensor(2.), 0)
    assert torch.equal(param, torch.tensor([2., 3.]))
    param.validate_loaded()
    with pytest.raises(ValueError, match='重复'):
        layer.weight_loader(param, torch.tensor(2.), 0)


def test_layout_permutation_cannot_repack():
    layer = ColumnParallelLinear(8, 8)
    p = packed((2, 4), layer)
    permute_param_layout_(p, input_dim=1, output_dim=0, packed_dim=1)
    assert p.shape == (4, 2) and p.packed_dim == 1
    with pytest.raises(ValueError, match='repack'):
        permute_param_layout_(p, input_dim=0, output_dim=1, packed_dim=1)


def test_postload_once_and_failure_state():
    model = nn.Sequential(ColumnParallelLinear(2, 2))
    process_weights_after_loading(model, None, 'cpu')
    assert model._quant_postload_state == 'ready'
    with pytest.raises(RuntimeError, match='只能执行一次'):
        process_weights_after_loading(model, None, 'cpu')
    broken = nn.Sequential(ColumnParallelLinear(2, 2))
    def fail(layer):
        raise ValueError('failed')
    broken[0].quant_method.process_weights_after_loading = fail
    with pytest.raises(ValueError, match='failed'):
        process_weights_after_loading(broken, None, 'cpu')
    assert broken._quant_postload_state == 'failed'
    with pytest.raises(RuntimeError, match='尚未完成加载'):
        broken[0](torch.ones(1, 2))
    with pytest.raises(RuntimeError, match='只能执行一次'):
        process_weights_after_loading(broken, None, 'cpu')


def test_qkv_packed_load_order_and_kv_replication():
    from minivllm.layers.linear import QKVParallelLinear
    layer = QKVParallelLinear(16, 8, 4, 1)
    # 固定示例：TP2 的 q 两份，k/v 每份复制；仅用于验证加载坐标。
    layer.tp_rank, layer.tp_size, layer.num_kv_head_replicas = 1, 2, 2
    layer.q_size, layer.kv_size = 16, 8
    p = PackedvLLMParameter(torch.empty((2, 32), dtype=torch.int32), layer.weight_loader,
                            input_dim=0, output_dim=1, packed_dim=0, packed_factor=8,
                            required_shards={'q', 'k', 'v'})
    q = torch.arange(2 * 32, dtype=torch.int32).reshape(2, 32)
    k = torch.full((2, 8), 111, dtype=torch.int32)
    v = torch.full((2, 8), 222, dtype=torch.int32)
    for shard, source in [('v', v), ('q', q), ('k', k)]:
        layer.weight_loader(p, source, shard)
    p.validate_loaded()
    assert torch.equal(p, torch.cat([q[:, 16:], k, v], dim=1))
    p.advance_layout('standard')
    p.advance_layout('kernel')
    with pytest.raises(RuntimeError, match='无效布局转换'):
        p.advance_layout('standard')


def test_wrong_shape_and_dtype_are_rejected():
    layer = ColumnParallelLinear(16, 16)
    p = packed((2, 16), layer)
    with pytest.raises(ValueError, match='shape'):
        layer.weight_loader(p, torch.zeros(2, 17, dtype=torch.int32))
    with pytest.raises(ValueError, match='dtype'):
        layer.weight_loader(p, torch.zeros(2, 16))
    with pytest.raises(ValueError, match='覆盖不完整'):
        p.validate_loaded()


def test_fused_checkpoint_splits_logical_shards():
    layer = MergedColumnParallelLinear(16, [8, 8])
    p = packed((2, 16), layer, required={0, 1})
    source = torch.arange(32, dtype=torch.int32).reshape(2, 16)
    layer.weight_loader(p, source)
    p.validate_loaded()
    assert torch.equal(p, source)


def test_weights_mapper_checks_missing_and_duplicate_fused_shards():
    from minivllm.model_loader.auto_weights_loader import AutoWeightsLoader, WeightsMapper
    model = nn.Module()
    model.gate_up_proj = MergedColumnParallelLinear(16, [8, 8], prefix='gate_up_proj')
    layer = model.gate_up_proj
    del layer.weight
    layer.register_parameter('qweight', packed((2, 16), layer, required={0, 1}))
    layer.register_parameter('scales', GroupQuantScaleParameter(
        torch.empty(1, 16), layer.weight_loader, input_dim=0, output_dim=1,
        required_shards={0, 1}, name='gate_up_proj.scales'))
    mapper = WeightsMapper(orig_to_new_stacked={'.gate_proj': ('.gate_up_proj', 0),
                                                '.up_proj': ('.gate_up_proj', 1)})
    weights = [('m.gate_proj.qweight', torch.ones(2, 8, dtype=torch.int32)),
               ('m.up_proj.qweight', torch.full((2, 8), 2, dtype=torch.int32)),
               ('m.gate_proj.scales', torch.ones(1, 8))]
    root = nn.Module()
    root.m = model
    with pytest.raises(ValueError, match='没有被加载|覆盖不完整'):
        AutoWeightsLoader(root).load_weights(weights, mapper=mapper)
