"""检查 121 前置或隔离对照 tiny CUDA repack；真实抽层计时尚待实现。"""
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
from minivllm.layers.quantization.marlin import marlin_padded_nk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--compare', action='store_true')
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--case', choices=('all', 'tiny'), default='all')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if not (args.check or args.compare or args.trace):
        parser.error('至少指定一种操作')
    result = {'status': 'passed_cpu_check', 'padded_nk_72_256': marlin_padded_nk(72, 256, 128),
              'gpu_comparison': 'not_run', 'real_layer_timing': 'not_run', 'trace': 'not_run'}
    code = 0
    if args.compare:
        if not torch.cuda.is_available():
            result.update(status='blocked', reason='CUDA GPU 不可用')
            code = 2
        else:
            with tempfile.TemporaryDirectory() as d:
                paths = [Path(d) / 'mini.pt', Path(d) / 'vllm.pt']
                env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
                for backend, path in zip(('mini', 'vllm'), paths):
                    subprocess.run([sys.executable, str(ROOT / 'tests/step121/step121_gpu_probe.py'),
                                    '--backend', backend, '--output', str(path)], check=True, env=env)
                a, b = (torch.load(path, weights_only=True, map_location='cpu') for path in paths)
                cases = {key: bool(torch.equal(a[key], b[key])) for key in a}
                result.update(status='passed_tiny_gpu' if all(cases.values()) else 'failed',
                              gpu_comparison=cases)
                code = 0 if all(cases.values()) else 1
    if args.trace:
        result.update(status='blocked', reason='真实抽层 trace 尚未实现')
        code = 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
