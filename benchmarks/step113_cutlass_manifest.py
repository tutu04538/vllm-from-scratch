"""冻结 CUTLASS 源身份及头文件/许可证，不表示已编译量化算子。"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root = args.source.resolve()
    revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain'], text=True).strip()
    if revision != 'da5e086dab31d63815acafdac9a9c5893b1c69e2' or dirty:
        raise ValueError('CUTLASS checkout differs from clean v4.4.2 source')
    files = sorted((p for p in root.rglob('*') if p.is_file() and '.git' not in p.relative_to(root).parts
                    and p.suffix in ('.h', '.hpp', '.cuh')), key=lambda p:p.relative_to(root).as_posix())
    digest = hashlib.sha256()
    for file in files:
        digest.update(f'{file.relative_to(root).as_posix()}\0{file_hash(file)}\n'.encode())
    license_file = root / 'LICENSE.txt'
    if not license_file.is_file():
        license_file = root / 'LICENSE'
    result = {'status': 'passed', 'scope': 'clean source identity and headers only',
              'commit': revision, 'source': str(root), 'header_count': len(files),
              'header_tree_sha256': digest.hexdigest(), 'license': str(license_file),
              'license_sha256': file_hash(license_file), 'cuda_kernel_build': 'not_run'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
