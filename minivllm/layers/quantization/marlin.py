"""冻结 Marlin W4A16/G128 的加载后布局转换；不执行 GEMM。"""
import math

import torch
import torch.nn.functional as F

from ...quant_ops.ops import gptq_marlin_repack, marlin_gemm_w4a16, prepare_full_extension


def marlin_padded_nk(size_n, size_k, group_size):
    if size_n <= 0 or size_k <= 0 or group_size != 128 or size_k % group_size:
        raise ValueError('W4/G128 要求正尺寸且 K 不跨 group 边界')
    def up(n, d):
        return (n + d - 1) // d * d
    candidates = ((up(size_n, 64), up(size_k, math.lcm(128, group_size))),
                  (up(size_n, 128), up(size_k, math.lcm(64, group_size))))
    return min(candidates, key=lambda x: (x[0] * x[1], sum(x)))


def marlin_pad_qweight(qweight, size_n, size_k, padded_n, padded_k):
    if qweight.dtype != torch.int32 or tuple(qweight.shape) != (size_k // 8, size_n):
        raise ValueError('标准 qweight 必须是 [K/8,N] int32')
    if size_k % 8 or padded_k < size_k or padded_n < size_n or (padded_k - size_k) % 8:
        raise ValueError('无效 qweight padding')
    if (padded_n, padded_k) == (size_n, size_k):
        return qweight
    return F.pad(qweight, (0, padded_n - size_n, 0, (padded_k - size_k) // 8))


def marlin_pad_scales(scales, size_n, size_k, padded_n, padded_k, group_size):
    if group_size != 128 or size_k % group_size or padded_k % group_size:
        raise ValueError('scale 的 K 必须与 group 对齐')
    if tuple(scales.shape) != (size_k // group_size, size_n):
        raise ValueError('scale 必须为 [K/group,N]')
    if padded_n < size_n or padded_k < size_k:
        raise ValueError('padding 不得缩小')
    if (padded_n, padded_k) == (size_n, size_k):
        return scales
    return F.pad(scales, (0, padded_n - size_n, 0, (padded_k - size_k) // group_size))


def _scale_perms():
    return ([i + 8 * j for i in range(8) for j in range(8)],
            [2 * i + j for i in range(4) for j in (0, 1, 8, 9, 16, 17, 24, 25)])


def marlin_permute_scales(scales, size_k, size_n, group_size):
    perm, single = _scale_perms()
    selected = perm if group_size < size_k else single
    return scales.reshape(-1, len(selected))[:, selected].reshape(-1, size_n).contiguous()


def marlin_permute_bias(bias, padded_n):
    _, perm = _scale_perms()
    return F.pad(bias, (0, padded_n - bias.numel())).reshape(-1, 32)[:, perm].reshape(padded_n).contiguous()


def marlin_zero_points(zero_points, size_n):
    perm, _ = _scale_perms()
    z = zero_points.reshape(-1, 64)[:, perm]
    z = z.reshape(-1, 8)[:, [0, 2, 4, 6, 1, 3, 5, 7]].reshape(-1, size_n)
    shifts = torch.arange(0, 32, 4, device=z.device, dtype=torch.int32)
    return (z.reshape(z.shape[0], size_n // 8, 8) << shifts).sum(dim=-1, dtype=torch.int32).contiguous()


def marlin_unpack_standard_zeros(qzeros, groups, size_n):
    if qzeros.dtype != torch.int32 or tuple(qzeros.shape) != (size_n // 8, groups):
        raise ValueError('标准 zero point 必须为 [N/8,K/group] int32')
    shifts = torch.arange(0, 32, 4, device=qzeros.device, dtype=torch.int32)
    return ((qzeros.t().contiguous().unsqueeze(-1) >> shifts) & 15).reshape(groups, size_n)


def marlin_make_workspace_new(device, max_blocks_per_sm=1, existing=None):
    device = torch.device(device)
    if device.type != 'cuda' or type(max_blocks_per_sm) is not int or max_blocks_per_sm <= 0:
        raise ValueError('Marlin workspace 要求 CUDA 设备和正数 blocks/SM')
    device = torch.device('cuda', torch.cuda.current_device() if device.index is None else device.index)
    size = torch.cuda.get_device_properties(device).multi_processor_count * max_blocks_per_sm
    if existing is not None:
        if existing.device != device or existing.dtype != torch.int32 or existing.numel() != size:
            raise ValueError('既有 Marlin workspace 与设备、dtype 或容量不匹配')
        return existing.zero_()
    return torch.zeros(size, dtype=torch.int32, device=device)


def prepare_marlin_weights(layer, config, existing_workspace=None):
    """原地将已加载的标准参数转换为 kernel 布局；失败不开放 forward。"""
    if config.group_size != 128 or config.has_g_idx or not config.zero_points or config.weight_type != 'uint4':
        raise NotImplementedError('仅支持无 act-order 的 AWQ W4/G128')
    qw, qz, scales = layer.qweight, layer.qzeros, layer.scales
    if qw.device.type != 'cuda' or any(p.device != qw.device for p in (qz, scales)):
        raise ValueError('Marlin 后处理要求所有参数在同一 CUDA 设备')
    if any(p.layout_state != 'standard' for p in (qw, qz, scales)):
        raise RuntimeError('Marlin 后处理要求 standard 布局')
    k, n = config.partition_weight_shape
    if qw.dtype != torch.int32 or qz.dtype != torch.int32 or scales.dtype != config.act_type:
        raise ValueError('量化参数 dtype 不匹配')
    pn, pk = marlin_padded_nk(n, k, config.group_size)
    # 提前加载完整扩展：只加载 repack 扩展后无法再注册同名 GEMM schema。
    prepare_full_extension()
    workspace = marlin_make_workspace_new(qw.device, existing=existing_workspace)
    empty_idx = torch.nn.Parameter(torch.empty(0, dtype=torch.int32, device=qw.device), requires_grad=False)
    sort_idx = torch.nn.Parameter(torch.empty(0, dtype=torch.int32, device=qw.device), requires_grad=False)
    zeros = marlin_unpack_standard_zeros(qz.data, k // 128, n)
    zeros = marlin_pad_scales(zeros, n, k, pn, pk, 128)
    zeros = marlin_zero_points(zeros, pn)
    packed = marlin_pad_qweight(qw.data.contiguous(), n, k, pn, pk)
    packed = gptq_marlin_repack(packed, sort_idx, pk, pn)
    new_scales = marlin_permute_scales(marlin_pad_scales(scales.data.contiguous(), n, k, pn, pk, 128), pk, pn, 128)
    bias = getattr(layer, 'bias', None)
    if bias is not None:
        # 保留逻辑 bias，供 skip_bias_add/return_bias 返回；kernel 使用单独布局。
        marlin_bias = marlin_permute_bias(bias.data, pn)
        layer.register_buffer("marlin_bias", marlin_bias, persistent=False)
    qw.data, qz.data, scales.data = packed, zeros, new_scales
    for param in (qw, qz, scales):
        param.advance_layout('kernel')
    layer.g_idx, layer.g_idx_sort_indices = empty_idx, sort_idx
    layer.marlin_logical_nk = (n, k)
    layer.marlin_padded_nk = (pn, pk)
    layer.marlin_is_k_full = True
    return workspace


def should_use_atomic_add_reduce(m, n, k, device, dtype):
    """冻结上游默认策略；只在用户显式设置上游变量时启用。"""
    import os
    if n >= 2048 or k < 2048 or device.type != 'cuda':
        return False
    if os.environ.get('VLLM_MARLIN_USE_ATOMIC_ADD', '0') != '1':
        return False
    if torch.cuda.get_device_capability(device)[0] < 9 and dtype == torch.bfloat16:
        return False
    return True


def apply_marlin_w4a16(layer, x, workspace, bias=None):
    """仅准备 local W4A16 GEMM；是否开放由 LinearMethod 的验收门禁决定。"""
    if x.ndim < 1:
        raise ValueError('输入至少需要一个 K 维度')
    n, k = layer.marlin_logical_nk
    pn, pk = layer.marlin_padded_nk
    if x.shape[-1] != k:
        raise ValueError('输入末维与本地逻辑 K 不一致')
    if x.dtype != layer.scales.dtype or x.device != layer.qweight.device:
        raise ValueError('输入与量化权重设备或 dtype 不一致')
    if any(p.layout_state != 'kernel' for p in (layer.qweight, layer.qzeros, layer.scales)):
        raise RuntimeError('Marlin GEMM 要求 kernel 权重布局')
    flat = x.reshape(-1, k)
    if pk != k:
        flat = F.pad(flat, (0, pk - k))
    kernel_bias = None
    if bias is not None:
        if bias is not getattr(layer, 'bias', None) or not hasattr(layer, 'marlin_bias'):
            raise ValueError('bias 必须是该层已准备的逻辑 bias')
        kernel_bias = layer.marlin_bias
    result = marlin_gemm_w4a16(
        flat, layer.qweight, layer.scales, layer.qzeros, layer.g_idx,
        layer.g_idx_sort_indices, workspace, size_n=pn, size_k=pk, bias=kernel_bias,
        is_k_full=layer.marlin_is_k_full,
        use_atomic_add=should_use_atomic_add_reduce(flat.shape[0], pn, pk, flat.device, flat.dtype),
        use_fp32_reduce=True)
    if pn != n:
        result = result[..., :n].contiguous()
    return result.reshape(x.shape[:-1] + (n,))
