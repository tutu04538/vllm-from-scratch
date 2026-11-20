"""AWQ W4/G128 checkpoint 参数及标准 packed 布局转换。"""
import torch

from .methods import LinearMethodBase
from ..parameter import GroupQuantScaleParameter, PackedvLLMParameter


class AutoAWQMarlinLinearMethod(LinearMethodBase):
    """加载 AWQ checkpoint；Marlin kernel 接入前仅提供表示审计。"""
    post_load_layout = 'standard'
    supports_execution = False

    def __init__(self, config, prefix):
        self.config = config
        self.prefix = prefix

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes,
                       input_size, output_size, params_dtype, **attrs):
        from ..linear import MergedColumnParallelLinear, QKVParallelLinear
        group, pack = self.config.group_size, 8
        if (self.config.weight_bits, group, self.config.zero_point) != (4, 128, True):
            raise NotImplementedError(f'{self.prefix}: 仅支持 AWQ W4/G128/zero_point')
        if input_size_per_partition % group or input_size_per_partition % pack:
            raise ValueError(f'{self.prefix}: 本地输入维 {input_size_per_partition} 与 group {group} 不对齐')
        if any(n <= 0 or n % pack for n in output_partition_sizes):
            raise ValueError(f'{self.prefix}: 输出逻辑 shard {output_partition_sizes} 与打包倍数 {pack} 不对齐')
        n = sum(output_partition_sizes)
        from .kernel_dispatch import MPLinearLayerConfig
        self.kernel_config = MPLinearLayerConfig(
            full_weight_shape=(input_size, output_size),
            partition_weight_shape=(input_size_per_partition, n),
            weight_type='uint4', act_type=params_dtype or torch.get_default_dtype(),
            group_size=group, zero_points=True, has_g_idx=False)
        groups = input_size_per_partition // group
        loader = attrs['weight_loader']
        if isinstance(layer, QKVParallelLinear):
            required = {'q', 'k', 'v'}
        elif isinstance(layer, MergedColumnParallelLinear):
            required = set(range(len(output_partition_sizes)))
        else:
            required = None
        common = dict(weight_loader=loader, input_dim=0, output_dim=1,
                      required_shards=required)
        layer.register_parameter('qweight', PackedvLLMParameter(
            torch.empty((input_size_per_partition, n // pack), dtype=torch.int32),
            packed_dim=1, packed_factor=pack, name=f'{self.prefix}.qweight', **common))
        layer.register_parameter('qzeros', PackedvLLMParameter(
            torch.empty((groups, n // pack), dtype=torch.int32), packed_dim=1,
            packed_factor=pack, name=f'{self.prefix}.qzeros', **common))
        layer.register_parameter('scales', GroupQuantScaleParameter(
            torch.empty((groups, n), dtype=params_dtype), name=f'{self.prefix}.scales', **common))

    def process_weights_after_loading(self, layer):
        _convert_awq_to_standard_format(layer)
        if layer.qweight.device.type == "cuda":
            from .kernel_dispatch import can_plan_marlin
            capability = torch.cuda.get_device_capability(layer.qweight.device)
            ok, reason = can_plan_marlin(self.kernel_config, platform="cuda",
                                         capability=10 * capability[0] + capability[1])
            if not ok:
                raise NotImplementedError(reason)
            from .marlin import prepare_marlin_weights
            self.workspace = prepare_marlin_weights(layer, self.kernel_config, getattr(self, "workspace", None))
            self.post_load_layout = "kernel"

    def apply(self, layer, x, bias=None):
        raise NotImplementedError(f'AWQ 量化执行尚未开放: 层 {self.prefix}；需要后续 Marlin kernel')


def _convert_awq_to_standard_format(layer):
    """对本 rank 完整 AWQ 参数调整位序并重打包；不反量化。"""
    qw, qz, scales = layer.qweight, layer.qzeros, layer.scales
    for param in (qw, qz, scales):
        param.validate_loaded()
        if param.layout_state != 'checkpoint':
            raise RuntimeError(f'{param.logical_name}: 重复 AWQ 布局转换')
    k, n_packed = qw.shape
    groups, z_n_packed = qz.shape
    if k % 8 or n_packed != z_n_packed or scales.shape != (groups, n_packed * 8):
        raise ValueError(f'{layer.prefix}: AWQ 参数布局不匹配')
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=qw.device)
    reverse_order = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], dtype=torch.long, device=qw.device)
    logical_q = ((qw.data.unsqueeze(-1) >> shifts) & 15)[:, :, reverse_order].reshape(k, n_packed * 8)
    standard_qw = (logical_q.reshape(k // 8, 8, n_packed * 8) << shifts[None, :, None]).sum(
        dim=1, dtype=torch.int32).contiguous()
    del logical_q
    logical_z = ((qz.data.unsqueeze(-1) >> shifts) & 15)[:, :, reverse_order].reshape(groups, n_packed * 8)
    standard_qz = (logical_z.T.reshape(n_packed, 8, groups) << shifts[None, :, None]).sum(
        dim=1, dtype=torch.int32).contiguous()
    del logical_z
    new_qw = PackedvLLMParameter(standard_qw, qw.weight_loader, input_dim=0, output_dim=1,
                                 packed_dim=0, packed_factor=8, required_shards=qw.required_shards,
                                 name=qw.logical_name)
    new_qz = PackedvLLMParameter(standard_qz, qz.weight_loader, input_dim=1, output_dim=0,
                                 packed_dim=0, packed_factor=8, required_shards=qz.required_shards,
                                 name=qz.logical_name)
    for old, new in ((qw, new_qw), (qz, new_qz)):
        new.update_tp(old.tp_rank, old.tp_size)
        new._loaded_shards = old._loaded_shards.copy()
        new.advance_layout('standard')
    scales.advance_layout('standard')
    layer.qweight = new_qw
    layer.qzeros = new_qz
