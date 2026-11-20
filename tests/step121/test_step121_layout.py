"""CPU 变换检查；真实 repack 与 workspace 必须另在 GPU 检验。"""
import ast
from pathlib import Path
import subprocess
import sys

import numpy
import pytest
import torch

from minivllm.layers.quantization.marlin import (
    marlin_pad_qweight, marlin_pad_scales, marlin_padded_nk,
    marlin_permute_scales, marlin_unpack_standard_zeros, marlin_zero_points,
    marlin_make_workspace_new,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'learning_notes/14_vllm_from_scratch/量化完整需求/参考资料/vllm_0_28_0/vllm/model_executor/layers/quantization/utils/marlin_utils.py'


def frozen_functions():
    # 从冻结文件读取原函数 AST；避免导入生产 vLLM。pack_cols 是上游算法的独立 numpy 实现。
    tree = ast.parse(SOURCE.read_text())
    wanted = {'get_scale_perms', 'marlin_permute_scales', 'marlin_zero_points'}
    module = ast.Module(body=[x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name in wanted], type_ignores=[])
    def pack_cols(q, bits, groups, n):
        a = q.numpy().astype(numpy.uint32)
        result = numpy.zeros((groups, n // 8), dtype=numpy.uint32)
        for i in range(8):
            result |= a[:, i::8] << (4 * i)
        return torch.from_numpy(result.view(numpy.int32).copy())
    env = {'torch': torch, 'numpy': numpy, 'pack_cols': pack_cols}
    exec(compile(module, str(SOURCE), 'exec'), env)
    return env


@pytest.mark.parametrize('k,n', [(128, 64), (256, 64), (256, 128), (384, 192)])
def test_scale_and_zero_permutation_matches_frozen(k, n):
    ref = frozen_functions()
    groups = k // 128
    scales = torch.arange(groups * n, dtype=torch.float32).reshape(groups, n) + 0.25
    zeros = (torch.arange(groups * n, dtype=torch.int32).reshape(groups, n) * 7 + 3) % 16
    expected_s = ref['marlin_permute_scales'](scales, k, n, 128)
    expected_z = ref['marlin_zero_points'](zeros, groups, n, 4)
    assert torch.equal(marlin_permute_scales(scales, k, n, 128), expected_s)
    assert torch.equal(marlin_zero_points(zeros, n), expected_z)
    standard = torch.empty((n // 8, groups), dtype=torch.int32)
    for col in range(n // 8):
        for group in range(groups):
            word = sum(int(zeros[group, 8 * col + i]) << (4 * i) for i in range(8))
            standard[col, group] = word if word < 2**31 else word - 2**32
    assert torch.equal(marlin_unpack_standard_zeros(standard, groups, n), zeros)


def test_padding_preserves_logical_values_and_rejects_group_cut():
    k, n = 256, 72
    pn, pk = marlin_padded_nk(n, k, 128)
    assert (pn, pk) == (128, 256)
    q = torch.arange(k // 8 * n, dtype=torch.int32).reshape(k // 8, n)
    s = torch.arange(k // 128 * n, dtype=torch.float32).reshape(k // 128, n)
    padded_q = marlin_pad_qweight(q, n, k, pn, pk)
    padded_s = marlin_pad_scales(s, n, k, pn, pk, 128)
    assert torch.equal(padded_q[:, :n], q) and not padded_q[:, n:].any()
    assert torch.equal(padded_s[:, :n], s) and not padded_s[:, n:].any()
    with pytest.raises(ValueError, match='group'):
        marlin_padded_nk(n, 192, 128)


def test_workspace_refuses_cpu():
    with pytest.raises(ValueError, match='CUDA'):
        marlin_make_workspace_new('cpu')


def test_import_does_not_depend_on_vllm():
    code = "import builtins; old=builtins.__import__; exec('def blocked(name,*a,**k):\\n if name == \"vllm\" or name.startswith(\"vllm.\"): raise RuntimeError(name)\\n return old(name,*a,**k)'); builtins.__import__=blocked; import minivllm.layers.quantization.marlin"
    subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True)
