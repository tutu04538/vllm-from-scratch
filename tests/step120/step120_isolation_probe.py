"""干净子进程禁止 import vllm，并检查本地 operator 注册和 meta 执行。"""
import importlib.abc
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
from minivllm.quant_ops.ops import prepare_repack_extension, gptq_marlin_repack

identity = prepare_repack_extension()
q = torch.empty((16, 64), dtype=torch.int32, device='meta')
p = torch.empty((0,), dtype=torch.int32, device='meta')
out = gptq_marlin_repack(q, p, 128, 64)
loaded_vllm = [x for x in sys.modules if x == 'vllm' or x.startswith('vllm.')]
maps = open('/proc/self/maps').read()
vllm_maps = [line for line in maps.splitlines() if '/vllm/' in line and '.so' in line]
assert not loaded_vllm and not vllm_maps
print(json.dumps({'status': 'passed_meta_only', 'shape': list(out.shape), 'dtype': str(out.dtype),
                  'library': identity['library'], 'identity_sha256': identity['identity_sha256'],
                  'vllm_modules': loaded_vllm, 'vllm_shared_objects': vllm_maps}))
