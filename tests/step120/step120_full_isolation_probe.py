"""干净进程装载完整自有扩展并验证 fake schema；不启动 GPU 算子。"""
import hashlib
import importlib.abc
import subprocess
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class NoVllm(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'vllm' or fullname.startswith('vllm.'):
            raise ImportError('测试禁止导入 vllm')
        return None


sys.meta_path.insert(0, NoVllm())
import torch
from minivllm.quant_ops.ops import prepare_full_extension

identity = prepare_full_extension()
for name in ('gptq_marlin_repack', 'marlin_gemm'):
    assert torch._C._dispatch_has_kernel_for_dispatch_key(f'minivllm_quant::{name}', 'CUDA')
q = torch.empty((16, 64), dtype=torch.int32, device='meta')
perm = torch.empty((0,), dtype=torch.int32, device='meta')
repack = torch.ops.minivllm_quant.gptq_marlin_repack(q, perm, 128, 64, 4, False)
a = torch.empty((4, 128), dtype=torch.bfloat16, device='meta')
scale = torch.empty((1, 64), dtype=torch.bfloat16, device='meta')
workspace = torch.empty((128,), dtype=torch.int32, device='meta')
out = torch.ops.minivllm_quant.marlin_gemm(a, None, repack, None, scale, None, None,
                                          None, None, None, workspace, 0,
                                          4, 64, 128, True, False, True, False)
assert list(repack.shape) == [8, 128] and list(out.shape) == [4, 64]
try:
    bad = torch.empty((8, 127), dtype=torch.int32, device='meta')
    torch.ops.minivllm_quant.marlin_gemm(a, None, bad, None, scale, None, None,
                                        None, None, None, workspace, 0,
                                        4, 64, 128, True, False, True, False)
except ValueError:
    pass
else:
    raise AssertionError('错误形状没有被拒绝')
needed = subprocess.run(['readelf', '-d', identity['library']], check=True, capture_output=True, text=True).stdout
assert not any('vllm' in line.lower() for line in needed.splitlines() if '(NEEDED)' in line)
loaded_vllm = [x for x in sys.modules if x == 'vllm' or x.startswith('vllm.')]
maps = open('/proc/self/maps').read()
vllm_maps = [line for line in maps.splitlines() if '/vllm/' in line and '.so' in line]
assert not loaded_vllm and not vllm_maps
print(json.dumps({'status': 'passed_meta_only', 'repack_shape': list(repack.shape),
                  'gemm_shape': list(out.shape), 'gemm_dtype': str(out.dtype),
                  'identity_sha256': identity['identity_sha256'], 'library': identity['library'],
                  'library_sha256': hashlib.sha256(Path(identity['library']).read_bytes()).hexdigest(),
                  'needed_libraries': [line.strip() for line in needed.splitlines() if '(NEEDED)' in line],
                  'vllm_modules': loaded_vllm, 'vllm_shared_objects': vllm_maps}))
