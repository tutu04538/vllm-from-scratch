#!/usr/bin/env python3
"""095 worker failure injection and cleanup; real models are not exercised."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / 'tests/step095/step095_scenarios.py'
CASES = ('init', 'load', 'initialize', 'capture', 'execute', 'sample', 'propose', 'output', 'abrupt', 'abrupt4', 'collective', 'kill_escalation', 'engine', 'fresh')


def execute(case, output, timeout):
    output.parent.mkdir(parents=True, exist_ok=True)
    log = output.with_suffix('.log')
    cmd = [sys.executable, str(SCENARIO), '--case', case, '--output', str(output)]
    started = time.monotonic()
    with log.open('w') as stream:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                start_new_session=True,
                                env={**os.environ, 'OMP_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1'})
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            code = 124
    value = json.loads(output.read_text()) if output.exists() else {'status':'failed', 'error':'no result'}
    expected = 0 if case in ('engine', 'fresh') else 1
    clean_threads = 'Exception in thread' not in log.read_text()
    return {'case':case, 'command':cmd, 'exit_code':code, 'expected_exit_code':expected,
            'passed':code == expected and value['status'] == 'passed' and clean_threads,
            'clean_threads':clean_threads,
            'elapsed_seconds':time.monotonic()-started, 'result':value, 'log':str(log)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', choices=('all', *CASES), default='all')
    p.add_argument('--timeout', type=int, default=120)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.timeout <= 0:
        p.error('--timeout must be positive')
    cases = CASES if a.case == 'all' else (a.case,)
    root = a.output.parent / (a.output.stem + '_cases')
    values = [execute(case, root / f'{case}.json', a.timeout) for case in cases]
    result = {'status':'passed' if all(v['passed'] for v in values) else 'failed', 'cases':values}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'status':result['status'], 'cases':len(values), 'output':str(a.output)}, ensure_ascii=False))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    sys.exit(main())
