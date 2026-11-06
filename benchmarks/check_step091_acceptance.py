#!/usr/bin/env python3
"""091 配置矩阵验收：独立进程、超时和 JSON manifest。"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from minivllm.config import ModelConfig, ParallelConfig, VllmConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('all', 'matrix', 'oracle'), default='all')
    parser.add_argument('--timeout', type=int, default=120)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('--timeout 必须大于 0')
    selected = {'all': 'tests/step091', 'matrix': 'tests/step091/test_step091_parallel.py',
                'oracle': 'tests/step091/test_step091_parallel.py::test_frozen_source_matches_installed'}[args.case]
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    command = [sys.executable, '-m', 'pytest', selected, '-q', '-ra']
    start = time.monotonic()
    result = {'case': args.case, 'command': command, 'status': 'failed', 'timeout_seconds': args.timeout,
              'support_matrix': {'single': 'supported', 'tp_pp_dp_multi': 'not implemented',
                                 'eagle3_pp': 'rejected', 'async_scheduling': 'rejected'}, 'layouts': []}
    import torch
    result['environment'] = {'python': sys.version.split()[0], 'torch': torch.__version__,
                             'cuda_visible_device_count': torch.cuda.device_count(),
                             'devices': [torch.cuda.get_device_name(i)
                                         for i in range(torch.cuda.device_count())]}
    manifest = ROOT / 'learning_notes/14_vllm_from_scratch/分布式并行完整需求/参考资料/源码清单.json'
    result['source_manifest_sha256'] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    effective = VllmConfig(ModelConfig('manifest-only'))
    result['effective_single_config'] = {
        'parallel_config': asdict(effective.parallel_config),
        'distributed_executor_backend': effective.parallel_config.resolved_backend,
        'world_size': effective.parallel_config.world_size,
        'total_gpu_count': effective.parallel_config.total_gpu_count,
        'draft_tensor_parallel_size': effective.effective_draft_tensor_parallel_size,
        'model_runner': 'v2' if effective.use_v2_model_runner else 'v1',
    }
    for tp, pp, dp in [(1, 1, 1), (2, 1, 2), (2, 2, 1), (4, 1, 2)]:
        p = ParallelConfig(tensor_parallel_size=tp, pipeline_parallel_size=pp, data_parallel_size=dp)
        result['layouts'].append({'config': asdict(p), 'world_size': p.world_size,
                                  'total_gpu_count': p.total_gpu_count, 'resolved_backend': p.resolved_backend})
    try:
        proc = subprocess.run(command, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, timeout=args.timeout)
        result.update(exit_code=proc.returncode, output=proc.stdout,
                      status='passed' if proc.returncode == 0 else 'failed')
    except subprocess.TimeoutExpired as exc:
        result.update(exit_code=124, output=str(exc), status='timeout')
    result['elapsed_seconds'] = time.monotonic() - start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(result['output'])
    return result['exit_code']


if __name__ == '__main__':
    sys.exit(main())
