"""Independent real-GPU Graph collectives and failure tests."""
import importlib.util
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('step104_acceptance',ROOT/'benchmarks/check_step104_acceptance.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)

@pytest.mark.parametrize('world,failure',[(2,False),(4,False),(2,True)])
def test_graph_collectives(world,failure,tmp_path):
    result=runner.execute(world,failure,tmp_path,120)
    assert result['passed'],Path(result['log']).read_text()[-15000:]
    if not failure:
        assert all(len(r['cycles'])==2 for r in result['ranks'])
        assert all(c['oracle'] and len(c['graphs'])==2 for r in result['ranks'] for c in r['cycles'])


def test_executor_capture_failure(tmp_path):
    result=runner.execute_executor(tmp_path,120)
    assert result['passed'],Path(result['log']).read_text()[-15000:]


def test_sequential_engines(tmp_path):
    result=runner.execute_executor(tmp_path,120,lifecycle=True)
    assert result['passed'],Path(result['log']).read_text()[-15000:]


def test_frozen_sources_match_installed_vllm():
    import hashlib
    import importlib.util
    import json
    ref=ROOT/'learning_notes/14_vllm_from_scratch/分布式并行完整需求/参考资料'
    manifest=json.loads((ref/'源码清单.json').read_text())
    hashes={x['path']:x['sha256'] for x in manifest['files']}
    installed=Path(importlib.util.find_spec('vllm').origin).parent
    for name in ('distributed/parallel_state.py','distributed/device_communicators/pynccl.py',
                 'distributed/device_communicators/pynccl_wrapper.py',
                 'distributed/device_communicators/cuda_communicator.py',
                 'distributed/device_communicators/custom_all_reduce.py'):
        assert hashlib.sha256((ref/'vllm_0_28_0'/name).read_bytes()).hexdigest()==hashes[name]==hashlib.sha256((installed/name).read_bytes()).hexdigest()
