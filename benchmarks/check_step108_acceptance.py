"""Run the 108 TP speculative matrix and frozen GPU rejection differential."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', choices=['all', 'unit', 'smoke', 'frozen', 'trace'], required=True)
    p.add_argument('--timeout', type=int, default=1800)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.timeout <= 0:
        p.error('timeout must be positive')
    out = args.output.resolve()
    folder = out.parent / (out.stem + '_runs')
    folder.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', TRITON_F32_DEFAULT='ieee')
    entries = []

    def run(name, cmd):
        log = folder / f'{name}.log'; start = time.monotonic()
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
        entries.append({'name': name, 'command': cmd, 'exit_code': code,
                        'elapsed_seconds': time.monotonic()-start, 'log': str(log)})
        if code:
            raise RuntimeError(f'{name} failed ({code}); {log}')

    try:
        if args.case in ('unit', 'all'):
            run('tests', [sys.executable, '-m', 'pytest', 'tests/step108', '-q', '-ra'])
        if args.case in ('frozen', 'all'):
            run('frozen_rejection', [sys.executable, '-m', 'pytest',
                'tests/step106/test_step106_forced_rejection.py', '-q', '-ra'])
        if args.case in ('smoke', 'all'):
            for tp, k in ((2, 2), (4, 4)):
                run(f'mixed_tp{tp}_k{k}', [sys.executable, str(ROOT/'tests/step108/step108_tiny_probe.py'),
                    '--tp', str(tp), '--k', str(k), '--batch', '2', '--mixed',
                    '--detect-divergence', '--output', str(folder/f'mixed_tp{tp}_k{k}.json')])
            for tp in (2, 4):
                run(f'probability_tp{tp}', [sys.executable, str(ROOT/'tests/step108/step108_tiny_probe.py'),
                    '--tp', str(tp), '--k', '2', '--batch', '2', '--probabilistic',
                    '--output', str(folder/f'probability_tp{tp}.json')])
        if args.case in ('trace', 'all'):
            probe = str(ROOT/'tests/step108/step108_reference_probe.py')
            compare = str(ROOT/'tests/step108/step108_compare_reference.py')
            for tp in (2, 4):
                for k in (1, 2, 4):
                    name = f'trace_tp{tp}_k{k}'
                    fixture, ref, repeat, mini = (folder/f'{name}_{suffix}' for suffix in
                                                   ('fixture', 'reference', 'repeat', 'mini'))
                    contract = folder/f'{name}_contract.json'
                    for implementation, directory, label in (('frozen', ref, 'reference'),
                                                              ('frozen', repeat, 'repeat')):
                        run(f'{name}_{label}', [sys.executable, probe, '--implementation', implementation,
                            '--tp', str(tp), '--k', str(k), '--fixture', str(fixture), '--output', str(directory)])
                    run(f'{name}_register', [sys.executable, compare, '--reference', str(ref),
                        '--other', str(repeat), '--contract', str(contract), '--register',
                        '--output', str(folder/f'{name}_repeat.json')])
                    run(f'{name}_mini', [sys.executable, probe, '--implementation', 'mini',
                        '--tp', str(tp), '--k', str(k), '--fixture', str(fixture), '--output', str(mini)])
                    run(f'{name}_compare', [sys.executable, compare, '--reference', str(ref),
                        '--other', str(mini), '--contract', str(contract),
                        '--output', str(folder/f'{name}_comparison.json')])
        status = 'passed'
    except RuntimeError as exc:
        status = 'failed'
        print(str(exc), file=sys.stderr)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'status': status, 'case': args.case, 'executions': entries},
                              ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'status': status, 'case': args.case, 'subprocesses': len(entries)}))
    return int(status != 'passed')


if __name__ == '__main__':
    sys.exit(main())
