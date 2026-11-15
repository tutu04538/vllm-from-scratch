"""需求 112：审计生产导入、包摘要与固定资料；不把静态结果当作 GPU 验收。"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'minivllm'
sys.path.insert(0, str(ROOT))
from benchmarks.step111_matrix import compute_pids


def selected_gpu_processes(visible):
    """按本次测试使用的设备检查计算进程，并恢复调用进程的环境。"""
    previous = os.environ.get('CUDA_VISIBLE_DEVICES')
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = visible
        return sorted(compute_pids(len(visible.split(','))))
    finally:
        if previous is None:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = previous


def package_hash(package=PACKAGE):
    digest = hashlib.sha256()
    for path in sorted(package.rglob('*.py'), key=lambda x: x.relative_to(package).as_posix()):
        rel = path.relative_to(package).as_posix()
        digest.update((rel + '\0' + hashlib.sha256(path.read_bytes()).hexdigest() + '\n').encode())
    return digest.hexdigest()


def upstream_imports(package=PACKAGE):
    violations = []
    for path in sorted(package.rglob('*.py')):
        if 'testing' in path.relative_to(package).parts:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import): names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom): names = [node.module or '']
            elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
                fn = node.func
                if ((isinstance(fn, ast.Name) and fn.id == '__import__') or
                    (isinstance(fn, ast.Attribute) and fn.attr == 'import_module')):
                    names = [node.args[0].value] if isinstance(node.args[0].value, str) else []
            for name in names:
                if name == 'vllm' or name.startswith('vllm.'):
                    violations.append({'file':str(path.relative_to(package)), 'line':node.lineno, 'module':name})
    return violations


def regression_groups(root=ROOT):
    legacy = [f'tests/step{n}' for n in range(58, 89)]
    parallel = [f'tests/step{n:03d}' for n in range(89, 112)
                if (root/f'tests/step{n:03d}').is_dir()]
    return [('pytest', ['tests/step112'], '2,3,0,1'),
            ('legacy_final', legacy, '0'),
            ('parallel_final', parallel, '2,3,0,1')]


def regressions_match_package(regression, current_hash):
    runs = regression.get('runs', [])
    return (regression.get('status') == 'passed_regressions'
            and regression.get('package_sha256') == current_hash
            and regression.get('package_sha256_end') == current_hash
            and len(runs) == 3
            and {item.get('name') for item in runs} == {'pytest', 'legacy_final', 'parallel_final'}
            and all(item.get('exit_code') == 0 for item in runs))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def audit_model_identity(identity, *, draft=False):
    """按已登记的 hash 检查快照与转换产物，缺失或不一致都明确报告。"""
    source = Path(identity['local_path'])
    checks = [{'path':str(source), 'expected_revision':identity['revision'],
               'passed':source.name == identity['revision']}]
    def check(path, expected):
        path = Path(path)
        actual = file_sha256(path) if path.is_file() else None
        checks.append({'path':str(path), 'expected':expected, 'actual':actual,
                       'passed':actual is not None and actual == expected})
    if draft:
        converted = Path(identity['converted_path'])
        check(source/'pytorch_model.bin', identity['source_sha256'])
        check(converted/'model.safetensors', identity['converted_sha256'])
        manifest_path = Path(identity['conversion_manifest'])
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        checks.append({'path':str(manifest_path), 'passed':
                       manifest.get('source_sha256') == identity['source_sha256'] and
                       manifest.get('converted_sha256') == identity['converted_sha256'] and
                       manifest.get('identity') == {'repo': identity['repo'], 'revision': identity['revision']}})
        if manifest.get('config_sha256'):
            check(converted/'config.json', manifest['config_sha256'])
        else:
            checks.append({'path':str(converted/'config.json'), 'passed':False})
    else:
        for name, digest in identity['weights_sha256'].items():
            check(source/name, digest)
        check(source/'tokenizer.json', identity['tokenizer_sha256'])
    return checks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', choices=('all', 'static', 'models', 'regressions', 'model-worker'), default='all')
    p.add_argument('--model', choices=('8b', '14b', '8b-draft', '14b-draft'))
    p.add_argument('--timeout', type=int, default=600)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.timeout <= 0: p.error('timeout must be positive')
    start = time.monotonic()
    results = json.loads((ROOT/'docs/results.json').read_text())
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.case == 'regressions':
        runs = []
        regression_package = package_hash()
        for name, paths, gpu in regression_groups():
            log = ROOT/'benchmarks/results/step112'/f'{name}.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable,'-m','pytest',*paths,'-q','-ra']
            env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
            if gpu is not None: env['CUDA_VISIBLE_DEVICES'] = gpu
            started = time.monotonic()
            visible = env.get('CUDA_VISIBLE_DEVICES') or '0,1,2,3'
            env['CUDA_VISIBLE_DEVICES'] = visible
            try:
                busy = selected_gpu_processes(visible)
                error = None
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                busy, error = [], str(exc)
            if busy or error:
                runs.append({'name':name, 'command':cmd, 'exit_code':125, 'log':None,
                             'summary':None, 'seconds':time.monotonic()-started,
                             'cuda_visible_devices':visible, 'status':'unrun',
                             'occupied_gpu_pids':busy, 'occupancy_error':error})
                a.output.write_text(json.dumps({'status':'running','runs':runs,'package_sha256':regression_package}, indent=2)+'\n')
                continue
            with log.open('w') as stream:
                proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                        start_new_session=True)
                try: code = proc.wait(timeout=a.timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try: proc.wait(timeout=10)
                    except subprocess.TimeoutExpired: os.killpg(proc.pid, signal.SIGKILL); proc.wait()
                    code = 124
            summaries = [line for line in log.read_text(errors='replace').splitlines() if re.search(r'\b\d+ passed\b', line)]
            runs.append({'name':name,'command':cmd,'exit_code':code,'log':str(log),
                         'summary':summaries[-1] if summaries else None,'seconds':time.monotonic()-started,
                         'cuda_visible_devices':env.get('CUDA_VISIBLE_DEVICES')})
            a.output.write_text(json.dumps({'status':'running','runs':runs,'package_sha256':regression_package}, indent=2)+'\n')
        regression_package_end = package_hash()
        ok = (regression_package_end == regression_package and
              all(item['exit_code'] == 0 and item['summary'] is not None for item in runs))
        report = {'status':'passed_regressions' if ok else 'failed','runs':runs,
                  'package_sha256':regression_package, 'package_sha256_end':regression_package_end}
        a.output.write_text(json.dumps(report, indent=2)+'\n'); print(json.dumps({'status':report['status']}))
        return int(not ok)
    if a.case == 'all':
        executions = []
        for mode in ('static', 'models'):
            destination = a.output.with_name(a.output.stem + '_' + mode + '.json')
            destination.unlink(missing_ok=True)
            command = [sys.executable, __file__, '--case', mode, '--timeout', str(a.timeout),
                       '--output', str(destination)]
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       text=True, start_new_session=True)
            try:
                _, error = process.communicate(timeout=a.timeout * (5 if mode == 'models' else 1))
                code = process.returncode
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try: process.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); process.communicate()
                code, error = 124, 'timed out'
            value = json.loads(destination.read_text()) if destination.is_file() else None
            executions.append({'mode':mode, 'exit_code':code, 'result':value, 'stderr':error})
        evidence = {}
        for step in ('step110', 'step111'):
            status = results.get(step, {}).get('results', {}).get('status')
            evidence[step] = {'status':status, 'passed':isinstance(status, str) and status.startswith('passed')}
        for name in ('pytest', 'legacy_final', 'parallel_final'):
            log = ROOT/'benchmarks/results/step112'/f'{name}.log'
            content = log.read_text(errors='replace') if log.is_file() else ''
            summaries = [line for line in content.splitlines() if re.search(r'\b\d+ passed\b', line)]
            last = summaries[-1] if summaries else ''
            evidence[name] = {'path':str(log), 'summary':last or None,
                              'passed':bool(last) and not re.search(r'\b\d+ (?:failed|error|errors)\b', last)}
        regression_path = ROOT/'benchmarks/results/step112/regressions.json'
        regression = json.loads(regression_path.read_text()) if regression_path.is_file() else {}
        evidence['regressions_exit_codes'] = {'path':str(regression_path),
                                               'passed':regressions_match_package(regression, package_hash())}
        ok = all(item['exit_code'] == 0 and item['result'] is not None for item in executions)
        ok = ok and all(item['passed'] for item in evidence.values())
        report = {'status':'passed' if ok else 'failed_or_unrun', 'executions':executions,
                  'evidence':evidence, 'elapsed_seconds':time.monotonic()-start}
        a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        print(json.dumps({'status':report['status'], 'evidence':evidence}))
        return int(not ok)
    if a.case == 'model-worker':
        if not a.model: p.error('--model is required for model-worker')
        identity = results['step090']['models']['identities'][a.model]
        checks = audit_model_identity(identity, draft=a.model.endswith('-draft'))
        if a.model.endswith('-draft'):
            key = a.model.replace('-', '_')
            expected = results['step109']['models'][key]['manifest_sha256']
            path = Path(identity['conversion_manifest'])
            actual = file_sha256(path) if path.is_file() else None
            checks.append({'path':str(path), 'expected':expected, 'actual':actual,
                           'passed':actual == expected})
        ok = all(item['passed'] for item in checks)
        a.output.write_text(json.dumps({'model':a.model, 'checks':checks, 'passed':ok}, indent=2)+'\n')
        return int(not ok)
    if a.case == 'models':
        names = [a.model] if a.model else ['8b', '14b', '8b-draft', '14b-draft']
        executions = []
        for name in names:
            result_path = a.output.with_name(a.output.stem + '_' + name + '.json')
            result_path.unlink(missing_ok=True)
            command = [sys.executable, __file__, '--case', 'model-worker', '--model', name,
                       '--output', str(result_path), '--timeout', str(a.timeout)]
            try:
                completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=a.timeout)
                code = completed.returncode
                error = completed.stderr
            except subprocess.TimeoutExpired:
                code = 124; error = 'timed out'
            value = json.loads(result_path.read_text()) if result_path.is_file() else None
            executions.append({'model':name, 'exit_code':code, 'result':value, 'stderr':error})
        ok = all(item['exit_code'] == 0 and item['result'] is not None for item in executions)
        result = {'status':'passed_models_only' if ok else 'failed', 'executions':executions,
                  'elapsed_seconds':time.monotonic()-start}
        a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
        print(json.dumps({'status':result['status'], 'models':len(executions)}))
        return int(not ok)
    imports = upstream_imports(); actual_hash = package_hash()
    expected = results['current_package']['sha256']
    checker = ROOT/'learning_notes/14_vllm_from_scratch/分布式并行完整需求/参考资料/check_requirements.py'
    try:
        completed = subprocess.run([sys.executable, str(checker)], cwd=ROOT, capture_output=True,
                                   text=True, timeout=a.timeout)
        check = {'exit_code':completed.returncode, 'output':completed.stdout, 'stderr':completed.stderr}
    except subprocess.TimeoutExpired:
        check = {'exit_code':124, 'output':'', 'stderr':'timed out'}
    ok = not imports and actual_hash == expected and check['exit_code'] == 0
    result = {'status':'passed_static_only' if ok else 'failed', 'production_imports':imports,
              'package_sha256':actual_hash, 'recorded_sha256':expected, 'requirements_check':check,
              'elapsed_seconds':time.monotonic()-start,
              'not_checked':['runtime state ownership', 'full model loading peaks', 'GPU/real model evidence', '111 performance']}
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'status':result['status'], 'package_sha256':actual_hash}))
    return int(not ok)

if __name__ == '__main__': raise SystemExit(main())
