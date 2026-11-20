"""AWQ checkpoint layout partitioning with distinguishable source values."""
from types import SimpleNamespace

import pytest
import torch

from minivllm.layers import linear
from minivllm.layers.quantization import resolve_quantization_config

AWQ = {'quant_method': 'awq', 'bits': 4, 'group_size': 128,
       'zero_point': True, 'version': 'gemm'}


def config():
    return resolve_quantization_config(None, AWQ)


def set_rank(monkeypatch, rank, size):
    monkeypatch.setattr(linear, '_linear_tp_group',
                        lambda disable_tp: (SimpleNamespace(), rank, size))


def src(rows, cols, offset=0, dtype=torch.int32):
    return (torch.arange(rows * cols).reshape(rows, cols) + offset).to(dtype)


@pytest.mark.parametrize('size,rank', [(size, rank) for size in (1, 2, 4) for rank in range(size)])
def test_column_and_row_partition(monkeypatch, size, rank):
    set_rank(monkeypatch, rank, size)
    for cls in (linear.ColumnParallelLinear, linear.RowParallelLinear):
        layer = cls(512, 256, bias=False, prefix='layer', quant_config=config(),
                    params_dtype=torch.float32)
        for name, shape in [('qweight', (512, 32)), ('qzeros', (4, 32)), ('scales', (4, 256))]:
            param = getattr(layer, name)
            value = src(*shape, dtype=param.dtype)
            param.weight_loader(param, value)
            dim = 1 if cls is linear.ColumnParallelLinear else 0
            width = shape[dim] // size
            assert torch.equal(param, value.narrow(dim, rank * width, width))
            param.validate_loaded()


@pytest.mark.parametrize('size', [1, 2, 4])
def test_qkv_kv_replication_and_merged_partition(monkeypatch, size):
    for rank in range(size):
        set_rank(monkeypatch, rank, size)
        qkv = linear.QKVParallelLinear(512, 32, 8, 1, prefix='qkv', quant_config=config())
        merged = linear.MergedColumnParallelLinear(512, [256, 512], bias=False,
                                                   prefix='merged', quant_config=config())
        for name in ('qweight', 'qzeros', 'scales'):
            qparam, mparam = getattr(qkv, name), getattr(merged, name)
            rows = 512 if name == 'qweight' else 4
            pack = 1 if name == 'scales' else 8
            for shard, width, offset in [('q', 256, 0), ('k', 32, 10000), ('v', 32, 20000)]:
                value = src(rows, width // pack, offset, qparam.dtype)
                qparam.weight_loader(qparam, value, shard)
                local_width = width // (size if shard == 'q' else 1) // pack
                target_offset = {'q': 0, 'k': 256 // size // pack,
                                 'v': (256 // size + 32) // pack}[shard]
                source_rank = rank if shard == 'q' else 0
                assert torch.equal(qparam.narrow(1, target_offset, local_width),
                                   value.narrow(1, source_rank * local_width, local_width))
            for shard, width in enumerate((256, 512)):
                value = src(rows, width // pack, 30000 * shard, mparam.dtype)
                mparam.weight_loader(mparam, value, shard)
                local_width = width // size // pack
                target_offset = 0 if shard == 0 else 256 // size // pack
                assert torch.equal(mparam.narrow(1, target_offset, local_width),
                                   value.narrow(1, rank * local_width, local_width))
            qparam.validate_loaded()
            mparam.validate_loaded()


def test_row_cannot_split_quantization_group(monkeypatch):
    set_rank(monkeypatch, 0, 4)
    with pytest.raises(ValueError, match='group'):
        linear.RowParallelLinear(768, 256, prefix='bad', quant_config=config())
