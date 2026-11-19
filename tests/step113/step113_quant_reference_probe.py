"""独立上游进程探针；仅测试用途，不由 minivllm 生产代码调用。"""
import argparse
import json
import traceback
from pathlib import Path

import torch
from vllm import _custom_ops as ops
from vllm.model_executor.kernels.linear import (
    _POSSIBLE_KERNELS, choose_mp_linear_kernel, init_fp8_linear_kernel,
)
from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import MPLinearLayerConfig
from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import awq_marlin_quantize, marlin_quantize
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8DynamicTensorSym, kFp8DynamicTokenSym, kFp8StaticTensorSym,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types


def attempt(fn):
    try:
        return {'status': 'passed', 'result': fn()}
    except Exception:
        return {'status': 'failed', 'traceback': traceback.format_exc()}


def mp_dispatch(quant_type, zero_point):
    cfg = MPLinearLayerConfig(full_weight_shape=(4096, 4096), partition_weight_shape=(4096, 4096),
                              weight_type=quant_type, act_type=torch.bfloat16, group_size=128,
                              zero_points=zero_point, has_g_idx=False)
    cc = torch.cuda.get_device_capability()
    capability = 10 * cc[0] + cc[1]
    candidates = []
    for cls in _POSSIBLE_KERNELS.get(current_platform._enum, []):
        check = attempt(lambda: cls.can_implement(cfg))
        minimum = attempt(cls.get_min_capability)
        candidates.append({'name': cls.__name__, 'can_implement': check,
                           'minimum_capability': minimum,
                           'capability_ok': capability >= minimum['result']
                           if minimum['status'] == 'passed' else None})
    return {'selected': choose_mp_linear_kernel(cfg).__name__, 'candidates': candidates}


def marlin_gemm(m, zero_point):
    torch.manual_seed(113)
    n = k = 4096
    dtype = torch.bfloat16
    x = torch.randn((m, k), device='cuda', dtype=dtype)
    weight = torch.randn((k, n), device='cuda', dtype=dtype)
    if zero_point:
        quant_type = scalar_types.uint4
        expected_weight, qweight, scales, zero = awq_marlin_quantize(weight, quant_type, 128, input_dtype=dtype)
        g_idx = perm = None
    else:
        quant_type = scalar_types.uint4b8
        expected_weight, qweight, scales, g_idx, perm, _ = marlin_quantize(
            weight, quant_type, 128, False, input_dtype=dtype)
        zero = None
    workspace = marlin_make_workspace_new(weight.device)
    output = ops.marlin_gemm(x, None, qweight, None, scales, None, None, zero, g_idx, perm,
                             workspace, quant_type, m, n, k, is_k_full=True, use_fp32_reduce=True)
    expected = x @ expected_weight
    torch.cuda.synchronize()
    relative_mean = (output.float() - expected.float()).abs().mean() / expected.float().abs().mean()
    if not torch.isfinite(output).all() or relative_mean.item() >= 0.04:
        raise AssertionError(f'Marlin relative mean error {relative_mean.item()} >= 0.04')
    return {'m': m, 'k': k, 'n': n, 'input_dtype': str(x.dtype), 'qweight_dtype': str(qweight.dtype),
            'qweight_shape': list(qweight.shape), 'output_dtype': str(output.dtype),
            'relative_mean_error': relative_mean.item(), 'op': 'torch.ops._C.marlin_gemm'}


def fp8_gemm(m):
    torch.manual_seed(113)
    n = k = 4096
    x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    w = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
    cc = torch.cuda.get_device_capability()
    capability = 10 * cc[0] + cc[1]
    supported = bool(ops.cutlass_scaled_mm_supports_fp8(capability))
    if not supported:
        raise RuntimeError(f'installed CUTLASS FP8 op reports SM{capability} unsupported')
    xq, xs = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
    wq, ws = ops.scaled_fp8_quant(w)
    y = ops.cutlass_scaled_mm(xq, wq.t(), xs, ws, out_dtype=torch.bfloat16)
    reference = (xq.float() * xs) @ (wq.float() * ws).t()
    torch.cuda.synchronize()
    error = (y.float() - reference).abs().mean() / reference.abs().mean()
    if not torch.isfinite(y).all() or error.item() >= 0.04:
        raise AssertionError(f'FP8 GEMM relative mean error {error.item()} >= 0.04')
    return {'m': m, 'k': k, 'n': n, 'a_dtype': str(xq.dtype), 'b_dtype': str(wq.dtype),
            'a_scale_shape': list(xs.shape), 'b_scale_shape': list(ws.shape),
            'output_dtype': str(y.dtype), 'relative_mean_error': error.item(),
            'op': 'torch.ops._C.cutlass_scaled_mm', 'path': 'W8A8'}


def fp8_dispatch():
    selected = {}
    for name, activation in [('per_tensor', kFp8DynamicTensorSym), ('per_token', kFp8DynamicTokenSym)]:
        result = attempt(lambda: type(init_fp8_linear_kernel(
            activation, kFp8StaticTensorSym, torch.bfloat16, torch.bfloat16, (4096, 4096))).__name__)
        selected[name] = result
    failures = {name: value for name, value in selected.items() if value['status'] != 'passed'}
    if failures:
        raise RuntimeError('FP8 dispatch failure: ' + json.dumps(failures))
    return selected


def fp8_cache():
    x = torch.tensor([-2., -1., 0., 1., 2.], device='cuda', dtype=torch.bfloat16)
    packed = torch.empty(x.shape, device='cuda', dtype=torch.uint8)
    restored = torch.empty_like(x)
    ops.convert_fp8(packed, x, scale=1.0, kv_dtype='fp8')
    ops.convert_fp8(restored, packed, scale=1.0, kv_dtype='fp8')
    torch.cuda.synchronize()
    torch.testing.assert_close(restored, x, atol=0, rtol=0)
    return {'input_dtype': str(x.dtype), 'cache_dtype': str(packed.dtype),
            'restored_dtype': str(restored.dtype), 'op': 'torch.ops._C_cache_ops.convert_fp8', 'scale': 1.0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    cc = torch.cuda.get_device_capability()
    results = {'awq_dispatch': attempt(lambda: mp_dispatch(scalar_types.uint4, True)),
               'gptq_dispatch': attempt(lambda: mp_dispatch(scalar_types.uint4b8, False)),
               'fp8_dispatch': attempt(fp8_dispatch),
               'awq_m1': attempt(lambda: marlin_gemm(1, True)),
               'awq_m128': attempt(lambda: marlin_gemm(128, True)),
               'gptq_m1': attempt(lambda: marlin_gemm(1, False)),
               'gptq_m128': attempt(lambda: marlin_gemm(128, False)),
               'fp8_m1': attempt(lambda: fp8_gemm(1)),
               'fp8_m128': attempt(lambda: fp8_gemm(128)),
               'fp8_cache': attempt(fp8_cache)}
    status = 'passed' if all(value['status'] == 'passed' for value in results.values()) else 'failed'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'status': status, 'device': torch.cuda.get_device_name(),
                                      'capability': list(cc), 'results': results}, indent=2) + '\n')
    return 0 if status == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
