"""106 验收驱动：独立测试及真实 EAGLE3 单卡参考与逐轮对照。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['all', 'unit', '8b', '14b'], default='all')
    parser.add_argument('--timeout', type=int, default=1800, help='每个子进程的超时秒数')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    run_dir = output.parent / (output.stem + '_runs')
    run_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy(); env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    results = []

    def run(name, command):
        start = time.monotonic()
        log = run_dir / f'{name}.log'
        try:
            with log.open('w') as stream:
                completed = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                           stderr=subprocess.STDOUT, timeout=args.timeout)
            code = completed.returncode
        except subprocess.TimeoutExpired:
            code = 124
        results.append({'name': name, 'command': command, 'exit_code': code,
                        'elapsed_seconds': time.monotonic()-start, 'log': str(log)})
        if code:
            raise RuntimeError(f'{name} failed with exit code {code}; log: {log}')

    def script(name, *arguments):
        return [sys.executable, str(ROOT / 'benchmarks' / name), *map(str, arguments)]

    try:
        if args.case in ('all', 'unit'):
            run('unit', [sys.executable, '-m', 'pytest', 'tests/step106', '-q', '-ra'])
        if args.case in ('all', '8b', '14b'):
            run('audit_drafts', script('step106_audit_drafts.py', '--output', run_dir/'draft_audit.json'))
        if args.case in ('all', '8b'):
            k0_ref = run_dir/'reference_k0'; k0_mini = run_dir/'mini_k0'
            run('8b_ref_k0', script('step103_frozen_probe.py', '--model', '8b', '--tp', 1,
                                    '--batch', 1, '--output', k0_ref))
            run('8b_register_k0', script('step106_compare_k0.py', '--reference', k0_ref, '--other',
                                         k0_ref, '--register', '--output', run_dir/'contract_k0.json'))
            run('8b_mini_k0', script('step103_mini_probe.py', '--model', '8b', '--tp', 1,
                                     '--batch', 1, '--output', k0_mini))
            run('8b_compare_k0', script('step106_compare_k0.py', '--reference', k0_ref, '--other',
                                        k0_mini, '--output', run_dir/'compare_k0.json'))
            for k in (1, 2, 4):
                ref = run_dir / f'reference_k{k}'; mini = run_dir / f'mini_k{k}'
                run(f'8b_ref_k{k}', script('step106_reference_probe.py', '--k', k, '--output', ref))
                run(f'8b_contract_k{k}', script('check_step106_reference.py', '--reference', ref,
                                                '--output', run_dir / f'contract_k{k}.json'))
                if k == 2:
                    draft = '/data/syh/models/converted/AngelSlim/Qwen3-8B_eagle3/9629dfce7a4a10564dd48d3e5485c3976095653c'
                    run('8b_fp32_k2', script('step106_reference_fp32.py', '--reference', ref/'run0.pt',
                                              '--draft', draft, '--output', run_dir/'fp32_k2.json'))
                run(f'8b_mini_k{k}', script('step106_mini_probe.py', '--k', k, '--output', mini))
                run(f'8b_compare_k{k}', script('check_step106_compare.py', '--reference', ref,
                                               '--mini', mini, '--require-kv', '--output',
                                               run_dir / f'compare_k{k}.json'))
                forced_ref = run_dir / f'forced_reference_k{k}'
                forced_mini = run_dir / f'forced_mini_k{k}'
                run(f'8b_forced_ref_k{k}', script('step106_reference_probe.py', '--k', k,
                                                  '--repeats', 1, '--forced-from', ref,
                                                  '--output', forced_ref))
                run(f'8b_forced_mini_k{k}', script('step106_mini_probe.py', '--k', k,
                                                   '--forced-from', ref, '--output', forced_mini))
                run(f'8b_forced_compare_k{k}', script('check_step106_compare.py', '--reference', forced_ref,
                                                      '--mini', forced_mini, '--require-kv', '--output',
                                                      run_dir / f'forced_compare_k{k}.json'))
        if args.case in ('all', '14b'):
            ref = run_dir/'draft14_ref'; repeat = run_dir/'draft14_ref_repeat'; mini = run_dir/'draft14_mini'
            for name, directory in [('14b_ref', ref), ('14b_ref_repeat', repeat)]:
                run(name, script('step106_draft_local.py', '--implementation', 'reference', '--output', directory))
            run('14b_register', script('step106_compare_local.py', '--reference', ref,
                                       '--other', repeat, '--output', run_dir/'draft14_contract.json', '--register'))
            run('14b_mini', script('step106_draft_local.py', '--implementation', 'mini', '--output', mini))
            run('14b_compare', script('step106_compare_local.py', '--reference', ref,
                                      '--other', mini, '--output', run_dir/'draft14_compare.json'))
        status = 'passed'
    except RuntimeError as exc:
        status = 'failed'
        print(str(exc), file=sys.stderr)
    payload = {'status': status, 'case': args.case, 'results': results,
               'limits': ['14B target not executed', '8B comparison does not cover full KV history']}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': status, 'case': args.case, 'cases': len(results)}))
    return int(status != 'passed')


if __name__ == '__main__':
    raise SystemExit(main())
