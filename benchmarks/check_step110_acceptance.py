"""Run step110 support-boundary and frozen-source acceptance tests."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['all', 'boundary', 'frozen'], default='all')
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('timeout must be positive')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log = args.output.with_suffix('.log')
    selected = {
        'boundary': ['tests/step110/test_step110_boundary.py'],
        'frozen': [
            'tests/step091/test_step091_parallel.py::test_frozen_draft_tp_oracle',
            'tests/step109/test_step109_graph.py::test_frozen_speculator_sources_match_installed_vllm',
        ],
        'all': ['tests/step110', 'tests/step091', 'tests/step108/test_step108_config.py'],
    }[args.case]
    cmd = [sys.executable, '-m', 'pytest', *selected, '-q', '-ra']
    start = time.monotonic()
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with log.open('w') as stream:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = proc.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            code = 124
    result = {'status': 'passed' if code == 0 else 'failed', 'case': args.case,
              'command': cmd, 'exit_code': code, 'elapsed_seconds': time.monotonic() - start,
              'log': str(log)}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    return int(code != 0)


if __name__ == '__main__':
    sys.exit(main())
