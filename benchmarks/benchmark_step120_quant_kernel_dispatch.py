"""W4A16 选择计划和独立扩展身份检查；不运行 GPU 算子或性能测试。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from minivllm.layers.quantization.kernel_dispatch import MPLinearLayerConfig, plan_mp_kernel
from minivllm.quant_ops.build import build_identity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check', action='store_true')
    p.add_argument('--compare', action='store_true')
    p.add_argument('--trace', action='store_true')
    p.add_argument('--case', choices=['all'], default='all')
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    if not any((args.check, args.compare, args.trace)):
        p.error('select --check, --compare or --trace')
    if args.trace or args.compare:
        result, code = {'status': 'not_run', 'reason': 'GPU 对照和性能测试留到后续'}, 2
    else:
        cfg = MPLinearLayerConfig((4096, 4096), (4096, 4096), 'uint4', torch.bfloat16, 128, True, False)
        plan = plan_mp_kernel(cfg, platform='cuda', capability=89)
        identity, digest = build_identity('full')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        probe = subprocess.run([sys.executable, str(ROOT / 'tests/step120/step120_full_isolation_probe.py')],
                               capture_output=True, text=True, check=True, timeout=240, env=env)
        result = {'status': 'passed_meta_only', 'plan': plan, 'identity_sha256': digest,
                  'build_identity': identity, 'isolated_probe': json.loads(probe.stdout.splitlines()[-1]),
                  'gpu_kernel_execution': 'not_run', 'performance': 'not_run'}
        code = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'output': str(args.output)}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
