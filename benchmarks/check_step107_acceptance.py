"""Run 107 tests or real frozen/reference draft comparisons, serially."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DRAFT = Path('/data/syh/models/converted/AngelSlim')
REVISIONS = {'8b': 'Qwen3-8B_eagle3/9629dfce7a4a10564dd48d3e5485c3976095653c',
             '14b': 'Qwen3-14B_eagle3/3d13517724e81cb409ddf1d4650772ec52f1e18e'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', choices=['unit', '8b-tp2', '8b-tp4', '14b-tp2', '14b-tp4', 'all'], required=True)
    p.add_argument('--timeout', type=int, default=1800, help='per subprocess timeout in seconds')
    p.add_argument('--gpu-memory-utilization', type=float, default=.75,
                   help='explicit frozen reference memory reservation fraction')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.timeout <= 0:
        p.error('timeout must be positive')
    out = args.output.resolve()
    folder = out.parent / (out.stem + '_runs')
    folder.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    executions = []

    def run(name, command):
        log = folder / f'{name}.log'
        start = time.monotonic()
        with log.open('w') as stream:
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
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
        executions.append({'name': name, 'command': command, 'exit_code': code,
                           'elapsed_seconds': time.monotonic()-start, 'log': str(log)})
        if code:
            raise RuntimeError(f'{name} failed ({code}); see {log}')

    def script(name, *arguments):
        return [sys.executable, str(ROOT / 'benchmarks' / name), *map(str, arguments)]

    try:
        if args.case in ('unit', 'all'):
            run('unit', [sys.executable, '-m', 'pytest', 'tests/step107', '-q', '-ra'])
        selected = ['8b-tp2', '8b-tp4', '14b-tp2', '14b-tp4'] if args.case == 'all' else [args.case]
        for case in selected:
            if case == 'unit':
                continue
            model, tp_string = case.split('-tp')
            reference, mini = folder / f'{case}_ref', folder / f'{case}_mini'
            fixture, contract = folder / f'{case}_fixture.pt', folder / f'{case}_contract.json'
            run(f'{case}_reference', script('step107_frozen_probe.py', '--model', model, '--tp', tp_string,
                                            '--gpu-memory-utilization', args.gpu_memory_utilization,
                                            '--output', reference))
            run(f'{case}_register', script('step107_register_reference.py', '--reference', reference,
                                           '--draft', DRAFT/REVISIONS[model],
                                           '--fixture', fixture, '--contract', contract))
            run(f'{case}_mini', [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                                 f'--nproc-per-node={tp_string}', str(ROOT/'benchmarks/step107_draft_probe.py'),
                                 '--draft', str(DRAFT/REVISIONS[model]), '--fixture', str(fixture),
                                 '--output', str(mini)])
            run(f'{case}_compare', script('step107_compare.py', '--reference', reference,
                                          '--contract', contract, '--mini', mini,
                                          '--output', folder/f'{case}_comparison.json'))
        status = 'passed'
    except RuntimeError as exc:
        status = 'failed'
        print(str(exc), file=sys.stderr)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'status': status, 'case': args.case, 'executions': executions},
                              ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'status': status, 'case': args.case, 'subprocesses': len(executions)}))
    return int(status != 'passed')


if __name__ == '__main__':
    sys.exit(main())
