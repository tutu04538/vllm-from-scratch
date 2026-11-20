"""选择边界和本地 operator schema 的 CPU/meta 基础检查。"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from minivllm.layers.quantization.kernel_dispatch import MPLinearLayerConfig, can_plan_marlin, plan_mp_kernel, require_mp_kernel
from minivllm.quant_ops.build import build_identity


def cfg(**changes):
    args = {'full_weight_shape': (4096, 4096), 'partition_weight_shape': (4096, 4096),
            'weight_type': 'uint4', 'act_type': torch.bfloat16, 'group_size': 128,
            'zero_points': True, 'has_g_idx': False}
    args.update(changes)
    return MPLinearLayerConfig(**args)


def test_sm89_plan_does_not_open_execution():
    plan = plan_mp_kernel(cfg(), platform='cuda', capability=89)
    assert plan['backend'] == 'MarlinLinearKernel' and not plan['execution_available']
    with pytest.raises(NotImplementedError, match='121/122'):
        require_mp_kernel(cfg(), platform='cuda', capability=89)


@pytest.mark.parametrize('changes', [
    {'group_size': 96}, {'partition_weight_shape': (2048, 4095)},
    {'weight_type': 'uint4b8'}, {'has_g_idx': True}, {'act_type': torch.float32},
    {'partition_weight_shape': (2000, 4096)}, {'zero_points': False},
])
def test_unsupported_configs_fail_explicitly(changes):
    ok, reason = can_plan_marlin(cfg(**changes), platform='cuda', capability=89)
    assert not ok and reason


def test_other_platform_or_capability_is_not_guessed():
    assert not can_plan_marlin(cfg(), platform='cpu', capability=89)[0]
    assert not can_plan_marlin(cfg(), platform='cuda', capability=90)[0]


def test_missing_toolchain_does_not_fallback(monkeypatch):
    monkeypatch.setenv('CUDA_HOME', '/nonexistent/minivllm_cuda')
    with pytest.raises(RuntimeError, match='nvcc'):
        build_identity()


def test_frozen_upstream_capability_subset_in_separate_process():
    path = Path(__file__).with_name('step120_frozen_capability_probe.py')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    out = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, check=True, timeout=120, env=env)
    upstream = json.loads(out.stdout.splitlines()[-1])
    cases = {'awq': cfg(), 'gptq': cfg(partition_weight_shape=(2048, 4096), weight_type='uint4b8', zero_points=False),
             'group_crosses_rank': cfg(partition_weight_shape=(2000, 4096)),
             'bad_type_zp': cfg(weight_type='uint4b8'), 'bad_group': cfg(group_size=96)}
    for name, case in cases.items():
        assert can_plan_marlin(case, platform='cuda', capability=89)[0] == upstream[name]['ok']


def test_clean_process_loads_local_schema_and_meta_without_vllm():
    path = Path(__file__).with_name('step120_isolation_probe.py')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    out = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, check=True, timeout=180, env=env)
    result = json.loads(out.stdout.splitlines()[-1])
    assert result['shape'] == [8, 128] and result['dtype'] == 'torch.int32'
    assert result['status'] == 'passed_meta_only' and not result['vllm_modules'] and not result['vllm_shared_objects']
    assert '/torch_extensions/' in result['library'] and 'minivllm_quant_' in result['library']


def test_full_extension_meta_and_clean_process():
    path = Path(__file__).with_name('step120_full_isolation_probe.py')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    proc = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, timeout=240, env=env)
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout.splitlines()[-1])
    assert result['status'] == 'passed_meta_only'
    assert result['repack_shape'] == [8, 128] and result['gemm_shape'] == [4, 64]
    assert result['gemm_dtype'] == 'torch.bfloat16'
    assert not result['vllm_modules'] and not result['vllm_shared_objects']
    assert 'minivllm_quant_full_' in result['library']


def test_repack_schema_rejects_invalid_shape_and_dtype_before_build():
    from minivllm.quant_ops.ops import _check_repack
    q = torch.empty((16, 64), dtype=torch.int32, device='meta')
    p = torch.empty((0,), dtype=torch.int32, device='meta')
    _check_repack(q, p, 128, 64, 4, False)
    with pytest.raises(ValueError, match='qweight 与'):
        _check_repack(q, p, 256, 64, 4, False)
    with pytest.raises(ValueError, match='int32'):
        _check_repack(q.to(torch.float32), p, 128, 64, 4, False)
    with pytest.raises(ValueError, match='perm 必须'):
        _check_repack(q, torch.empty((4,), dtype=torch.int32, device='meta'), 128, 64, 4, False)


def test_imported_marlin_sources_match_frozen_except_operator_namespace():
    root = Path(__file__).resolve().parents[2]
    reference = root / 'learning_notes/14_vllm_from_scratch/量化完整需求/参考资料/vllm_0_28_0/csrc'
    local = root / 'minivllm/csrc'
    names = ['gptq_marlin_repack.cu', 'marlin.cu', 'marlin.cuh', 'marlin_dtypes.cuh',
             'marlin_mma.h', 'marlin_template.h', 'dequant.h', 'kernel.h', 'generate_kernels.py']
    for name in names:
        old = (reference / 'libtorch_stable/quantization/marlin' / name).read_bytes()
        new = (local / 'quantization/marlin' / name).read_bytes()
        if name in ('gptq_marlin_repack.cu', 'marlin.cu'):
            old = old.replace(b'STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, m)',
                              b'STABLE_TORCH_LIBRARY_IMPL(minivllm_quant, CUDA, m)')
        assert new == old, name
    for name in ('libtorch_stable/torch_utils.h', 'core/scalar_type.hpp'):
        assert (local / name).read_bytes() == (reference / name).read_bytes()


def test_awq_method_carries_full_and_partition_kernel_config():
    from minivllm.layers.linear import RowParallelLinear
    from minivllm.layers.quantization import resolve_quantization_config
    quant = resolve_quantization_config(None, {'quant_method': 'awq', 'bits': 4,
                                               'group_size': 128, 'zero_point': True})
    layer = RowParallelLinear(256, 64, params_dtype=torch.bfloat16, prefix='layer', quant_config=quant)
    actual = layer.quant_method.kernel_config
    assert actual.full_weight_shape == (256, 64)
    assert actual.partition_weight_shape == (256, 64)
    assert actual.weight_type == 'uint4' and actual.zero_points and not actual.has_g_idx
    assert actual.act_type == torch.bfloat16 and actual.group_size == 128
    assert plan_mp_kernel(actual, platform='cuda', capability=89)['backend'] == 'MarlinLinearKernel'
    with pytest.raises(NotImplementedError, match='量化执行尚未开放'):
        layer(torch.empty((1, 256), dtype=torch.bfloat16))
