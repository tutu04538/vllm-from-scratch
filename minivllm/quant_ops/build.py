"""按源码和工具链身份构建独立的 Marlin CUDA 扩展。"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import shutil
import sys

import torch

ROOT = Path(__file__).resolve().parents[1] / 'csrc'
MARLIN = ROOT / 'quantization' / 'marlin'
CUDA_FLAGS = ('-O2', '-std=c++17', '-DUSE_CUDA', '-DTORCH_TARGET_VERSION=0x020B000000000000ULL', '-gencode=arch=compute_89,code=sm_89',
              '-static-global-template-stub=false')
CPP_FLAGS = ('-O2', '-std=c++17', '-DUSE_CUDA', '-DTORCH_TARGET_VERSION=0x020B000000000000ULL')
_lock = threading.Lock()
_loaded = None


def build_identity(component="repack"):
    if component not in ("repack", "full"):
        raise ValueError("未知扩展组件")
    cuda_home = os.environ.get('CUDA_HOME')
    if not cuda_home:
        raise RuntimeError('CUDA_HOME 未设置；请 source scripts/activate-4090.sh')
    nvcc = Path(cuda_home) / 'bin/nvcc'
    if not nvcc.is_file() or not os.access(nvcc, os.X_OK):
        raise RuntimeError(f'无法执行 nvcc: {nvcc}')
    try:
        version = subprocess.run([str(nvcc), '--version'], check=True, capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f'nvcc 版本检查失败: {exc}') from exc
    compiler = os.environ.get('CXX', 'c++')
    try:
        host_version = subprocess.run([compiler, '--version'], check=True, capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f'C++ 编译器检查失败: {compiler}: {exc}') from exc
    paths = sorted(p for p in ROOT.rglob('*') if p.is_file() and p.suffix in ('.cu', '.cuh', '.h', '.hpp', '.cpp', '.py'))
    files = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    identity = {'sources': files, 'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
                'torch_cxx11_abi': torch.compiled_with_cxx11_abi(), 'cuda_home': str(Path(cuda_home).resolve()),
                'nvcc_version': version, 'host_compiler': compiler, 'host_compiler_version': host_version, 'cuda_flags': CUDA_FLAGS, 'cpp_flags': CPP_FLAGS,
                'arch': 'sm_89', 'component': component,
                'builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'generator_profile': 'w4a16-sm89-group128-and-single-group' if component == 'full' else None}
    encoded = json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()
    return identity, hashlib.sha256(encoded).hexdigest()


def load_repack_extension():
    """仅按显式调用编译/加载；失败直接报错，绝不回退到上游。"""
    global _loaded
    with _lock:
        identity, digest = build_identity()
        if _loaded is not None:
            if _loaded['identity_sha256'] != digest:
                raise RuntimeError('同一进程中检测到量化扩展构建身份变化，需启动新进程')
            return _loaded
        from torch.utils.cpp_extension import load
        try:
            lib = load(name='minivllm_quant_' + digest[:16],
                       sources=[str(MARLIN / 'bindings.cpp'), str(MARLIN / 'gptq_marlin_repack.cu')],
                       extra_include_paths=[str(ROOT)], extra_cflags=list(CPP_FLAGS),
                       extra_cuda_cflags=list(CUDA_FLAGS), with_cuda=True,
                       is_python_module=False, verbose=False)
        except Exception as exc:
            raise RuntimeError(f'minivllm Marlin 扩展构建/加载失败（身份 {digest}）：{exc}') from exc
        _loaded = {'identity': identity, 'identity_sha256': digest, 'library': str(lib)}
        return _loaded


def load_full_extension():
    """编译同源 Marlin GEMM 和 repack；不等同于已通过 121/122 验收。"""
    global _loaded
    with _lock:
        identity, digest = build_identity('full')
        if _loaded is not None:
            if _loaded['identity_sha256'] != digest:
                raise RuntimeError('进程已有其他量化扩展，必须使用干净进程加载完整扩展')
            return _loaded
        from torch.utils.cpp_extension import load, get_default_build_root
        build_dir = Path(get_default_build_root()) / ('minivllm_quant_full_' + digest[:16])
        gen = build_dir / 'generated'
        gen.mkdir(parents=True, exist_ok=True)
        try:
            stamp_path = gen / 'generation_manifest.json'
            lock_path = build_dir / 'generation.lock'
            with lock_path.open('w') as lock_file:
                fcntl.flock(lock_file, fcntl.LOCK_EX)
                stamp = None
                if stamp_path.is_file():
                    try:
                        stamp = json.loads(stamp_path.read_text())
                    except (ValueError, OSError):
                        stamp = None
                valid = (isinstance(stamp, dict) and stamp.get('identity_sha256') == digest
                         and isinstance(stamp.get('files'), dict)
                         and all((gen / name).is_file() and hashlib.sha256((gen / name).read_bytes()).hexdigest() == value
                                 for name, value in stamp['files'].items()))
                if not valid:
                    for source in MARLIN.iterdir():
                        if source.is_file():
                            shutil.copy2(source, gen / source.name)
                    # 原生成器只在专用缓存目录写入；过滤到 W4A16、G128 与单 group 的实例。
                    generator = gen / 'generate_kernels.py'
                    text = generator.read_text()
                    marker = '\n\ndef remove_old_kernels():'
                    if text.count(marker) != 1:
                        raise ValueError('冻结生成器结构变化，无法应用实例过滤')
                    profile = ("\n\nQUANT_CONFIGS = [dict(c, group_blocks=[-1, 8], "
                               "a_type=['kFloat16', 'kBFloat16']) for c in QUANT_CONFIGS "
                               "if c['b_type'] in ('kU4', 'kU4B8') and 'a_type' not in c]\n")
                    generator.write_text(text.replace(marker, profile + marker))
                    subprocess.run([sys.executable, str(generator), '8.9'], cwd=gen,
                                   check=True, capture_output=True, text=True, timeout=120)
                    generated = sorted(gen.glob('sm80_kernel_*.cu'))
                    if len(generated) != 4 or not (gen / 'kernel_selector.h').is_file():
                        raise RuntimeError(f'W4A16 生成器输出不完整: {len(generated)}')
                    tracked = [p for p in gen.iterdir() if p.is_file() and p != stamp_path]
                    record = {'identity_sha256': digest, 'files': {
                        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}}
                    stamp_tmp = gen / 'generation_manifest.tmp'
                    stamp_tmp.write_text(json.dumps(record, sort_keys=True))
                    os.replace(stamp_tmp, stamp_path)
                generated = sorted(gen.glob('sm80_kernel_*.cu'))
            sources = [gen / 'bindings.cpp', gen / 'gptq_marlin_repack.cu', gen / 'marlin.cu', *generated]
            lib = load(name='minivllm_quant_full_' + digest[:16], sources=[str(p) for p in sources],
                       extra_include_paths=[str(ROOT), str(gen)], extra_cflags=list(CPP_FLAGS),
                       extra_cuda_cflags=list(CUDA_FLAGS), with_cuda=True, is_python_module=False,
                       build_directory=str(build_dir), verbose=False)
        except Exception as exc:
            raise RuntimeError(f'minivllm 完整 Marlin 扩展构建/加载失败（身份 {digest}）：{exc}') from exc
        _loaded = {'identity': identity, 'identity_sha256': digest, 'library': str(lib),
                   'generated_sources': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in generated},
                   'kernel_selector_sha256': hashlib.sha256((gen / 'kernel_selector.h').read_bytes()).hexdigest()}
        return _loaded
