"""step113 身份与能力探针；未执行的 GPU 检查不会被记为通过。"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / 'learning_notes/14_vllm_from_scratch/量化完整需求/参考资料'
PARALLEL = ROOT / 'learning_notes/14_vllm_from_scratch/分布式并行完整需求/参考资料'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_manifest(manifest: dict, base: Path) -> dict:
    errors = []
    paths = set()
    for item in manifest['files']:
        relative = Path(item['path'])
        if relative.is_absolute() or '..' in relative.parts or item['path'] in paths:
            errors.append(f'invalid or duplicate path: {item["path"]}')
            continue
        paths.add(item['path'])
        path = base / relative
        if not path.is_file():
            errors.append(f'missing: {relative}')
        elif path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            errors.append(f'mismatch: {relative}')
    return {'status': 'failed' if errors else 'passed', 'count': len(paths), 'errors': errors}


def package_hash(root: Path) -> dict:
    digest = hashlib.sha256()
    files = sorted(root.rglob('*.py'), key=lambda p: p.relative_to(root).as_posix())
    for path in files:
        digest.update(f'{path.relative_to(root).as_posix()}\0{sha(path)}\n'.encode())
    return {'sha256': digest.hexdigest(), 'python_files': len(files)}


def command(argv: list[str], *, timeout=15) -> dict:
    try:
        result = subprocess.run(argv, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
        return {'exit_code': result.returncode, 'stdout': result.stdout.strip(),
                'stderr': result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'exit_code': None, 'error': f'{type(exc).__name__}: {exc}'}


def installed_check(manifest: dict) -> dict:
    spec = importlib.util.find_spec('vllm')
    if spec is None or spec.origin is None:
        return {'status': 'failed', 'errors': ['installed vllm not found'], 'count': 0}
    base = Path(spec.origin).parent
    rows = [r for r in manifest['files'] if r['path'].startswith('vllm/') and r['path'].endswith('.py')]
    errors = []
    for row in rows:
        path = base / row['path'].removeprefix('vllm/')
        if not path.is_file() or sha(path) != row['sha256']:
            errors.append(str(path))
    return {'status': 'failed' if errors else 'passed', 'count': len(rows),
            'installed_root': str(base), 'errors': errors}


def parallel_check() -> dict:
    manifest = json.loads((PARALLEL / '源码清单.json').read_text())
    spec = importlib.util.find_spec('vllm')
    if spec is None or spec.origin is None:
        return {'status': 'failed', 'count': 0, 'errors': ['installed vllm not found']}
    installed = Path(spec.origin).parent
    frozen = PARALLEL / 'vllm_0_28_0'
    errors = []
    for row in manifest['files']:
        expected = row['sha256']
        for path in (frozen / row['path'], installed / row['path']):
            if not path.is_file() or sha(path) != expected:
                errors.append(str(path))
    return {'status': 'failed' if errors else 'passed',
            'count': len(manifest['files']), 'errors': errors}


def working_tree() -> dict:
    """记录当前有变化文件的内容摘要，不把 dirty tree 误认成 HEAD。"""
    result = subprocess.run(['git', 'status', '--porcelain=v1', '-z', '--untracked-files=all'],
                            cwd=ROOT, capture_output=True, check=True)
    entries = []
    records = iter(result.stdout.split(b'\0'))
    for record in records:
        if not record:
            continue
        state, raw_path = record[:2].decode(), record[3:]
        name = os.fsdecode(raw_path)
        item = {'status': state, 'path': name}
        if 'R' in state or 'C' in state:
            item['original_path'] = os.fsdecode(next(records))
        path = ROOT / name
        if path.is_file():
            item['sha256'] = sha(path)
        entries.append(item)
    diffs = {}
    for label, argv in (('working', ['git', 'diff', '--binary']),
                        ('staged', ['git', 'diff', '--binary', '--cached'])):
        data = subprocess.check_output(argv, cwd=ROOT)
        diffs[label + '_diff_sha256'] = hashlib.sha256(data).hexdigest()
    return {'files': entries, **diffs}


def versions() -> dict:
    result = {'python': sys.version}
    for name in ('torch', 'triton', 'vllm', 'transformers'):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def collect_baseline() -> dict:
    manifest_path = REF / '源码清单.json'
    manifest = json.loads(manifest_path.read_text())
    source = check_manifest(manifest, REF / 'vllm_0_28_0')
    installed = installed_check(manifest)
    parallel = parallel_check()
    gpu = command(['nvidia-smi', '--query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu',
                   '--format=csv,noheader,nounits'])
    git_head = command(['git', 'rev-parse', 'HEAD'])
    git_status = command(['git', 'status', '--porcelain=v1', '--untracked-files=all'])
    checks = (source, installed, parallel)
    status = 'passed' if all(v['status'] == 'passed' for v in checks) else 'failed'
    return {'kind': 'cpu_source_and_identity_only', 'status': status,
            'captured_at_utc': datetime.now(timezone.utc).isoformat(),
            'git_head': git_head, 'git_status': git_status, 'working_tree': working_tree(),
            'package': package_hash(ROOT / 'minivllm'),
            'source_manifest_sha256': sha(manifest_path), 'upstream_commit': manifest['upstream_commit'],
            'source_manifest': source, 'installed_python': installed, 'historical_parallel': parallel,
            'versions': versions(), 'nvcc': command(['nvcc', '--version']), 'gpu_snapshot': gpu,
            'cutlass_tag': command(['git', 'ls-remote', 'https://github.com/NVIDIA/cutlass.git', 'refs/tags/v4.4.2'], timeout=20),
            'environment': {key: os.environ.get(key) for key in
                            ('CUDA_HOME', 'CUDA_VISIBLE_DEVICES', 'VLLM_BATCH_INVARIANT',
                             'VLLM_DISABLED_KERNELS', 'TORCH_CUDA_ARCH_LIST', 'MINIVLLM_UVA_BACKEND')},
            'gpu_numerical_probes': {'status': 'not_run',
                                     'reason': 'not part of --check; requires independent idle-GPU probe'},
            'minivllm_quantized_execution': 'not_implemented'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--compare', action='store_true')
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--case', choices=('all', 'identity'), default='all')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not any((args.check, args.compare, args.trace)):
        parser.error('select --check, --compare, or --trace')
    if args.compare or args.trace:
        result = {'kind': 'gpu_probe', 'status': 'not_run',
                  'reason': 'GPU probe is not implemented; no GPU result is implied'}
        code = 2
    else:
        result = collect_baseline()
        code = 0 if result['status'] == 'passed' else 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'kind': result['kind'], 'output': str(args.output)}, ensure_ascii=False))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
