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


# ScalarType.uint(4): exponent=0, mantissa=4, signed=0, bias=0,
# finite_only=0, nan_repr=IEEE_754 (bit 50). 对应冻结 C++ ScalarType::id。
MARLIN_UINT4_ID = (1 << 50) | (4 << 8)


def marlin_gemm_w4a16(a, weight, scales, zeros, g_idx, perm, workspace, *,
                      size_n, size_k, bias=None, is_k_full=True,
                      use_atomic_add=False, use_fp32_reduce=True):
    """受限 AWQ W4A16 本地算子调用；本身不开放 Linear 推理。"""
    if a.ndim != 2 or a.shape[1] != size_k or a.shape[0] <= 0:
        raise ValueError('输入必须是非空二维 [M,K]')
    if a.device.type not in ('cuda', 'meta') or a.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError('输入必须是 CUDA/meta 上的 FP16 或 BF16')
    if size_k <= 0 or size_n <= 0 or size_k % 128 or size_n % 64:
        raise ValueError('W4/G128 的 K/N 未对齐')
    if a.stride(1) != 1 or a.stride(0) % 8:
        raise ValueError('输入 stride 不满足 Marlin 的 16 字节加载要求')
    if a.device.type == 'cuda' and a.data_ptr() % 16:
        raise ValueError('输入地址未按 16 字节对齐')
    for name, tensor in [('weight', weight), ('scales', scales), ('zeros', zeros),
                         ('g_idx', g_idx), ('perm', perm), ('workspace', workspace)]:
        if tensor.device != a.device or not tensor.is_contiguous():
            raise ValueError(f'{name} 必须与输入同设备且连续')
    if weight.dtype != torch.int32 or tuple(weight.shape) != (size_k // 16, size_n * 2):
        raise ValueError('weight 必须是对应 K/N 的 Marlin int32 布局')
    if scales.dtype != a.dtype or tuple(scales.shape) != (size_k // 128, size_n):
        raise ValueError('scales 必须与输入 dtype 一致且形状为 [K/128,N]')
    if zeros.dtype != torch.int32 or tuple(zeros.shape) != (size_k // 128, size_n // 8):
        raise ValueError('zeros 必须为 [K/128,N/8] int32')
    if any(t.dtype != torch.int32 or t.ndim != 1 or t.numel() for t in (g_idx, perm)):
        raise ValueError('当前仅支持 CUDA 上空的 int32 g_idx 和 perm')
    if workspace.dtype != torch.int32 or workspace.ndim != 1:
        raise ValueError('workspace 必须是一维 int32')
    if a.device.type == 'cuda':
        sms = torch.cuda.get_device_properties(a.device).multi_processor_count
        if workspace.numel() < sms:
            raise ValueError('workspace 容量小于 SM 数')
    elif workspace.numel() < 1:
        raise ValueError('workspace 不得为空')
    if bias is not None and (bias.device != a.device or bias.dtype != a.dtype or
                             not bias.is_contiguous() or tuple(bias.shape) != (size_n,)):
        raise ValueError('bias 必须与输入同设备同 dtype 且为连续 [N]')
    if type(is_k_full) is not bool or type(use_atomic_add) is not bool or type(use_fp32_reduce) is not bool:
        raise ValueError('归约选项必须为 bool')
    if not _fake_registered:
        prepare_full_extension()
    if a.device.type == 'cuda' and not torch._C._dispatch_has_kernel_for_dispatch_key(
            'minivllm_quant::marlin_gemm', 'CUDA'):
        raise RuntimeError('当前进程仅加载了 repack 扩展；需要在干净进程先加载完整 Marlin 扩展')
    return torch.ops.minivllm_quant.marlin_gemm(
        a, None, weight, bias, scales, None, None, zeros, g_idx, perm,
        workspace, MARLIN_UINT4_ID, a.shape[0], size_n, size_k,
        is_k_full, use_atomic_add, use_fp32_reduce, False)
