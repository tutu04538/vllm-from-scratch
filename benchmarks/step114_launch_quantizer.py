"""单卡离线量化启动器：未确认 GPU 空闲时不启动子进程。"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from benchmark_step113_quant_capability import gpu_occupancy

ROOT = Path(__file__).resolve().parents[1]


def launch(*, device, model, tool_python, data, output_root, result_path, method='awq'):
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result = {'method': method, 'model': model, 'device': device, 'occupancy': gpu_occupancy(device)}
    if result['occupancy']['status'] != 'idle':
        result.update(status='not_run', reason='GPU not confirmed idle')
        result_path.write_text(json.dumps(result, indent=2) + '\n')
        return 2
    uv = shutil.which('uv')
    if uv is None or not tool_python.is_file():
        result.update(status='failed', reason='isolated tool environment or uv unavailable')
        result_path.write_text(json.dumps(result, indent=2) + '\n')
        return 1
    freeze = subprocess.run([uv, 'pip', 'freeze', '--python', str(tool_python)], cwd=ROOT,
                            capture_output=True, text=True, timeout=60)
    if freeze.returncode:
        result.update(status='failed', reason='cannot freeze tool environment', stderr=freeze.stderr)
        result_path.write_text(json.dumps(result, indent=2) + '\n')
        return 1
    command = [str(tool_python), str(ROOT / f'benchmarks/step114_quantize_{method}.py'),
               '--model', model, '--data', str(data), '--output-root', str(output_root)]
    log = result_path.with_suffix('.log')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(device), HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONNOUSERSITE='1')
    result.update(status='running', command=command, tool_python=str(tool_python),
                  tool_packages=freeze.stdout.splitlines(), log=str(log))
    result_path.write_text(json.dumps(result, indent=2) + '\n')
    try:
        with log.open('w') as stream:
            proc = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        result.update(status='finished_not_validated' if proc.returncode == 0 else 'failed',
                      exit_code=proc.returncode)
    except OSError as exc:
        result.update(status='failed', reason=f'{type(exc).__name__}: {exc}')
    result_path.write_text(json.dumps(result, indent=2) + '\n')
    return 0 if result['status'] == 'finished_not_validated' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--method', choices=['awq', 'gptq'], default='awq')
    parser.add_argument('--model', choices=['8B', '14B'], required=True)
    parser.add_argument('--tool-python', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, default=Path('/data/syh/models/converted/Qwen'))
    parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    if args.device < 0:
        parser.error('device must be non-negative')
    return launch(device=args.device, model=args.model, tool_python=args.tool_python,
                  data=args.data, output_root=args.output_root, result_path=args.result, method=args.method)


if __name__ == '__main__':
    sys.exit(main())
