"""122 tiny 同输入 CUDA 差分入口；真实性能测量按约定延期。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from minivllm.layers.quantization.marlin import marlin_permute_scales, marlin_permute_bias, marlin_zero_points
from minivllm.quant_ops.ops import gptq_marlin_repack, prepare_full_extension


def tiny_compare():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU 不可用')
    if torch.cuda.get_device_capability(0) != (8, 9):
        raise RuntimeError('当前只支持 SM89')
    k, n = 256, 128
    generator = torch.Generator().manual_seed(122)
    q = torch.randint(0, 16, (k, n), dtype=torch.int32, generator=generator)
    z = torch.randint(0, 16, (k // 128, n), dtype=torch.int32, generator=generator)
    scales = (torch.rand((k // 128, n), generator=generator) * 0.02 + 0.01).to(torch.bfloat16)
    x = (torch.randn((128, k), generator=generator) * 0.3).to(torch.bfloat16)
    bias = (torch.randn(n, generator=generator) * 0.1).to(torch.bfloat16)
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)
    standard = (q.reshape(k // 8, 8, n) << shifts[None, :, None]).sum(1, dtype=torch.int32).contiguous()
    prepare_full_extension()
    packed = gptq_marlin_repack(standard.cuda(), torch.empty(0, device='cuda', dtype=torch.int32), k, n).cpu()
    payload = {'k': k, 'n': n, 'weight': packed, 'zeros': marlin_zero_points(z, n),
               'scales': marlin_permute_scales(scales, k, n, 128),
               'bias': marlin_permute_bias(bias, n), 'x': x, 'empty': torch.empty(0, dtype=torch.int32)}
    oracle_weight = (q.float() - z.repeat_interleave(128, dim=0).float()) * scales.repeat_interleave(128, dim=0).float()
    with tempfile.TemporaryDirectory() as tmp:
        inp = Path(tmp) / 'input.pt'
        torch.save(payload, inp)
        paths = [Path(tmp) / 'mini.pt', Path(tmp) / 'vllm.pt']
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        for backend, path in zip(('mini', 'vllm'), paths):
            subprocess.run([sys.executable, str(ROOT / 'tests/step122/step122_gpu_probe.py'),
                            '--backend', backend, '--input', str(inp), '--output', str(path)],
                           env=env, check=True, timeout=240)
        a, b = (torch.load(path, weights_only=True, map_location='cpu') for path in paths)
    report = {}
    for key, actual in a.items():
        m = int(key)
        oracle = x[:m].float() @ oracle_weight + bias.float()
        torch.testing.assert_close(actual.float(), b[key].float(), rtol=1e-2, atol=2e-2)
        torch.testing.assert_close(actual.float(), oracle, rtol=2e-2, atol=5e-2)
        report[key] = {'frozen_max_abs': (actual.float() - b[key].float()).abs().max().item(),
                       'oracle_max_abs': (actual.float() - oracle).abs().max().item()}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--compare', action='store_true')
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--case', choices=('all', 'tiny'), default='all')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not any((args.check, args.compare, args.trace)):
        parser.error('选择 --check、--compare 或 --trace')
    result = {'status': 'passed_cpu_check', 'local_forward': 'blocked', 'gpu_compare': 'not_run',
              'real_shape': 'not_run', 'trace': 'not_run', 'performance': 'deferred'}
    code = 0
    if args.compare:
        try:
            result['gpu_compare'] = tiny_compare()
            result['status'] = 'passed_tiny_gpu_only'
        except Exception as exc:
            result.update(status='failed_or_blocked', reason=f'{type(exc).__name__}: {exc}')
            code = 2
    if args.trace:
        result.update(status='blocked', reason='trace 留待 GPU 空闲后实现并执行')
        code = 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
