"""量化权重的布局和加载，语义对照冻结 vLLM parameter.py 的受限子集。"""
from __future__ import annotations

import torch
from torch.nn import Parameter


class BasevLLMParameter(Parameter):
    def __new__(cls, data, weight_loader=None, **kwargs):
        return super().__new__(cls, data, requires_grad=False)

    def __init__(self, data, weight_loader, *, required_shards=None, name='parameter', **kwargs):
        if kwargs:
            raise TypeError(f'未知 Parameter 属性: {sorted(kwargs)}')
        self.weight_loader = weight_loader
        self.tp_rank, self.tp_size = 0, 1
        self.required_shards = None if required_shards is None else frozenset(required_shards)
        self._loaded_shards = set()
        self.logical_name = name
        self.layout_state = 'checkpoint'

    def update_tp(self, rank, size):
        if type(rank) is not int or type(size) is not int or size < 1 or not 0 <= rank < size:
            raise ValueError(f'{self.logical_name}: 无效 TP 状态 {rank}/{size}')
        self.tp_rank, self.tp_size = rank, size

    def _copy(self, target, source, shard):
        if self.layout_state != 'checkpoint':
            raise RuntimeError(f'{self.logical_name}: 后处理后不能继续写入 checkpoint 权重')
        if shard in self._loaded_shards:
            raise ValueError(f'{self.logical_name}: 重复加载 shard {shard!r}')
        if target.shape != source.shape or target.dtype != source.dtype:
            raise ValueError(f'{self.logical_name}: shard {shard!r} shape/dtype 不匹配: '
                             f'{tuple(target.shape)}/{target.dtype} != {tuple(source.shape)}/{source.dtype}')
        with torch.no_grad():
            target.copy_(source)
        self._loaded_shards.add(shard)

    def validate_loaded(self):
        required = self.required_shards if self.required_shards is not None else frozenset({None})
        if self._loaded_shards != required:
            raise ValueError(f'{self.logical_name}: shard 覆盖不完整，期望 {sorted(map(str, required))}，'
                             f'实际 {sorted(map(str, self._loaded_shards))}')

    def advance_layout(self, target):
        transitions = {'checkpoint': 'standard', 'standard': 'kernel'}
        if transitions.get(self.layout_state) != target:
            raise RuntimeError(f'{self.logical_name}: 无效布局转换 {self.layout_state} → {target}')
        if self.layout_state == 'checkpoint':
            self.validate_loaded()
        self.layout_state = target

    def load_column_parallel_weight(self, loaded_weight):
        self._copy(self.data, loaded_weight, None)

    def load_row_parallel_weight(self, loaded_weight):
        self._copy(self.data, loaded_weight, None)

    def load_merged_column_weight(self, loaded_weight, **kwargs):
        self._copy(self.data, loaded_weight, kwargs.get('shard_id'))

    def load_qkv_weight(self, loaded_weight, **kwargs):
        self._copy(self.data, loaded_weight, kwargs.get('shard_id'))


class ColumnvLLMParameter(BasevLLMParameter):
    def __init__(self, data, weight_loader, *, output_dim, **kwargs):
        self._output_dim = _axis(output_dim, data.ndim, 'output_dim')
        super().__init__(data, weight_loader, **kwargs)

    @property
    def output_dim(self):
        return self._output_dim

    def load_column_parallel_weight(self, loaded_weight):
        size = self.shape[self.output_dim]
        self._validate_source(loaded_weight, self.output_dim, size * self.tp_size)
        self._copy(self.data, loaded_weight.narrow(self.output_dim, self.tp_rank * size, size), None)

    def _validate_source(self, loaded_weight, dim, width):
        expected = list(self.shape)
        expected[dim] = width
        if tuple(loaded_weight.shape) != tuple(expected):
            raise ValueError(f'{self.logical_name}: 原始 shard shape 不匹配: '
                             f'{tuple(loaded_weight.shape)} != {tuple(expected)}')

    def _fused(self, loaded_weight, shard_id, shard_offset, shard_size, source_rank, source_shards):
        if isinstance(self, (PackedColumnParameter, PackedvLLMParameter)) and self.packed_dim == self.output_dim:
            shard_size, shard_offset = self.adjust_shard_indexes_for_packing(shard_size, shard_offset)
        self._validate_source(loaded_weight, self.output_dim, shard_size * source_shards)
        target = self.data.narrow(self.output_dim, shard_offset, shard_size)
        source = loaded_weight.narrow(self.output_dim, source_rank * shard_size, shard_size)
        self._copy(target, source, shard_id)

    def load_merged_column_weight(self, loaded_weight, *, shard_id, shard_offset, shard_size):
        self._fused(loaded_weight, shard_id, shard_offset, shard_size, self.tp_rank, self.tp_size)

    def load_qkv_weight(self, loaded_weight, *, shard_id, shard_offset, shard_size, num_heads):
        rank = self.tp_rank if shard_id == 'q' else self.tp_rank // num_heads
        self._fused(loaded_weight, shard_id, shard_offset, shard_size, rank,
                    self.tp_size if shard_id == 'q' else self.tp_size // num_heads)


class RowvLLMParameter(BasevLLMParameter):
    def __init__(self, data, weight_loader, *, input_dim, **kwargs):
        self._input_dim = _axis(input_dim, data.ndim, 'input_dim')
        super().__init__(data, weight_loader, **kwargs)

    @property
    def input_dim(self):
        return self._input_dim

    def load_row_parallel_weight(self, loaded_weight):
        size = self.shape[self.input_dim]
        expected = list(self.shape)
        expected[self.input_dim] = size * self.tp_size
        if tuple(loaded_weight.shape) != tuple(expected):
            raise ValueError(f'{self.logical_name}: 原始 shard shape 不匹配: '
                             f'{tuple(loaded_weight.shape)} != {tuple(expected)}')
        self._copy(self.data, loaded_weight.narrow(self.input_dim, self.tp_rank * size, size), None)


class ModelWeightParameter(ColumnvLLMParameter, RowvLLMParameter):
    def __init__(self, data, weight_loader, *, input_dim, output_dim, **kwargs):
        if input_dim == output_dim:
            raise ValueError('input_dim 和 output_dim 不能相同')
        super().__init__(data, weight_loader, input_dim=input_dim, output_dim=output_dim, **kwargs)


class GroupQuantScaleParameter(ModelWeightParameter):
    pass


class PerTensorScaleParameter(BasevLLMParameter):
    def _load_scale(self, loaded_weight, shard_id):
        idx = {'q': 0, 'k': 1, 'v': 2}.get(shard_id, shard_id)
        if type(idx) is not int or not 0 <= idx < self.shape[0]:
            raise ValueError(f'{self.logical_name}: 无效 scale shard {shard_id!r}')
        source = loaded_weight.reshape(()) if loaded_weight.numel() == 1 else loaded_weight
        self._copy(self.data[idx], source, shard_id)

    def load_merged_column_weight(self, loaded_weight, *, shard_id, **kwargs):
        self._load_scale(loaded_weight, shard_id)

    def load_qkv_weight(self, loaded_weight, *, shard_id, **kwargs):
        self._load_scale(loaded_weight, shard_id)


class _PackedMixin:
    def __init__(self, data, weight_loader, *, packed_factor, packed_dim, marlin_tile_size=None, **kwargs):
        if type(packed_factor) is not int or packed_factor <= 0:
            raise ValueError('packed_factor 必须是正整数')
        if marlin_tile_size is not None and (type(marlin_tile_size) is not int or marlin_tile_size <= 0):
            raise ValueError('marlin_tile_size 必须是正整数')
        self._packed_factor = packed_factor
        self._packed_dim = _axis(packed_dim, data.ndim, 'packed_dim')
        self._marlin_tile_size = marlin_tile_size
        super().__init__(data, weight_loader, **kwargs)

    @property
    def packed_dim(self):
        return self._packed_dim

    @property
    def packed_factor(self):
        return self._packed_factor

    @property
    def marlin_tile_size(self):
        return self._marlin_tile_size

    def adjust_shard_indexes_for_packing(self, shard_size, shard_offset):
        if shard_size % self.packed_factor or shard_offset % self.packed_factor:
            raise ValueError(f'{self.logical_name}: shard 边界未对齐打包倍数 {self.packed_factor}')
        tile = self.marlin_tile_size or 1
        return shard_size // self.packed_factor * tile, shard_offset // self.packed_factor * tile


class PackedvLLMParameter(_PackedMixin, ModelWeightParameter):
    pass


class PackedColumnParameter(_PackedMixin, ColumnvLLMParameter):
    pass


def _axis(axis, ndim, name):
    if type(axis) is not int or not 0 <= axis < ndim:
        raise ValueError(f'{name}={axis!r} 对 {ndim} 维参数无效')
    return axis


def permute_param_layout_(param, input_dim, output_dim, *, packed_dim=None):
    """调整维度顺序；不改变打包语义，也不执行 repack。"""
    ndim = param.ndim
    _axis(input_dim, ndim, 'input_dim')
    _axis(output_dim, ndim, 'output_dim')
    if input_dim == output_dim:
        raise ValueError('input_dim 和 output_dim 不能相同')
    old_in, old_out = getattr(param, 'input_dim', None), getattr(param, 'output_dim', None)
    if old_in is None or old_out is None:
        if ndim != 2 or old_in is None and old_out is None:
            raise ValueError('缺失维度信息，无法推导布局')
        old_in = 1 - old_out if old_in is None else old_in
        old_out = 1 - old_in if old_out is None else old_out
    perm = [i for i in range(ndim) if i not in (old_in, old_out)]
    perm.insert(input_dim, old_in)
    perm.insert(output_dim, old_out)
    if packed_dim is not None:
        _axis(packed_dim, ndim, 'packed_dim')
        if not hasattr(param, 'packed_dim') or param.packed_dim != perm[packed_dim]:
            raise ValueError('布局变换不支持 repack')
    param.data = param.data.permute(*perm)
    if hasattr(param, '_input_dim'):
        param._input_dim = input_dim
    if hasattr(param, '_output_dim'):
        param._output_dim = output_dim
    if packed_dim is not None:
        param._packed_dim = packed_dim
    return param
