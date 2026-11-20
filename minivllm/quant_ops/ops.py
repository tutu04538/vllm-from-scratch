"""项目自有 Marlin operator 的受限接口；推理路径尚未启用。"""
import torch

from .build import load_full_extension, load_repack_extension

_fake_registered = False


def _check_repack(qweight, perm, size_k, size_n, num_bits, is_a_8bit):
    if qweight.ndim != 2 or perm.ndim != 1:
        raise ValueError('qweight 必须二维、perm 必须一维')
    if qweight.dtype != torch.int32 or perm.dtype != torch.int32:
        raise ValueError('qweight 和 perm 必须是 int32')
    if qweight.device != perm.device or qweight.device.type not in ('cuda', 'meta'):
        raise ValueError('qweight 和 perm 必须位于同一 CUDA 设备或 meta 设备')
    if not qweight.is_contiguous() or not perm.is_contiguous():
        raise ValueError('qweight 和 perm 必须连续')
    if num_bits != 4 or is_a_8bit:
        raise NotImplementedError('当前只支持 W4A16 的 repack schema')
    if size_k <= 0 or size_n <= 0 or size_k % 16 or size_n % 64:
        raise ValueError('repack 要求 K 是 16、N 是 64 的正整数倍')
    if tuple(qweight.shape) != (size_k // 8, size_n):
        raise ValueError('qweight 与 K、N 不一致')
    if perm.numel() not in (0, size_k):
        raise ValueError('perm 必须为空或含 K 个元素')


def _register_fake():
    global _fake_registered
    if _fake_registered:
        return
    @torch.library.register_fake('minivllm_quant::gptq_marlin_repack')
    def fake_repack(qweight, perm, size_k, size_n, num_bits, is_a_8bit):
        _check_repack(qweight, perm, size_k, size_n, num_bits, is_a_8bit)
        return qweight.new_empty((size_k // 16, size_n * 2))

    @torch.library.register_fake('minivllm_quant::marlin_gemm')
    def fake_gemm(a, c, b, bias, scales, a_scales, global_scale, zeros, g_idx, perm,
                  workspace, b_type_id, size_m, size_n, size_k, is_k_full,
                  use_atomic_add, use_fp32_reduce, is_zp_float):
        if a.ndim != 2 or tuple(a.shape) != (size_m, size_k):
            raise ValueError('输入 a 与 M/K 不一致')
        if a.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError('W4A16 输入必须是 FP16/BF16')
        if size_m <= 0 or size_n <= 0 or size_k <= 0 or size_k % 16 or size_n % 64:
            raise ValueError('M/N/K 必须为正且满足 Marlin tile 对齐')
        if a.device.type not in ('cuda', 'meta') or b.device != a.device or scales.device != a.device or workspace.device != a.device:
            raise ValueError('a、b、scales、workspace 必须在相同 CUDA 或 meta 设备')
        if b.dtype != torch.int32 or b.ndim != 2 or tuple(b.shape) != (size_k // 16, size_n * 2):
            raise ValueError('Marlin 权重必须为 int32 并匹配 repack 后形状')
        if scales.ndim != 2 or scales.shape[1] != size_n or scales.dtype != a.dtype:
            raise ValueError('scales 的 shape/dtype 不匹配')
        if workspace.dtype != torch.int32 or workspace.ndim != 1:
            raise ValueError('workspace 必须是一维 int32')
        for name, item in [('bias', bias), ('zeros', zeros), ('g_idx', g_idx), ('perm', perm)]:
            if item is not None and item.device != a.device:
                raise ValueError(f'{name} 必须与 a 位于同一设备')
        if bias is not None and (bias.dtype != a.dtype or tuple(bias.shape) != (size_n,)):
            raise ValueError('bias 的 shape/dtype 不匹配')
        if zeros is not None and zeros.dtype != torch.int32:
            raise ValueError('W4A16 zero point 必须为 int32')
        if c is not None:
            if tuple(c.shape) != (size_m, size_n) or c.dtype != a.dtype or c.device != a.device:
                raise ValueError('输出缓冲 c 的 shape/dtype/device 不匹配')
            return c
        return a.new_empty((size_m, size_n))
    _fake_registered = True


def prepare_repack_extension():
    result = load_repack_extension()
    _register_fake()
    return result


def prepare_full_extension():
    result = load_full_extension()
    _register_fake()
    return result


def gptq_marlin_repack(qweight, perm, size_k, size_n, num_bits=4, is_a_8bit=False):
    _check_repack(qweight, perm, size_k, size_n, num_bits, is_a_8bit)
    if not _fake_registered:
        prepare_repack_extension()
    return torch.ops.minivllm_quant.gptq_marlin_repack(qweight, perm, size_k, size_n, num_bits, is_a_8bit)
