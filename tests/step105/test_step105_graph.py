"""Independent TP model eager/Graph tests; no pre-run benchmark required."""
import importlib.util
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('step105_acceptance',ROOT/'benchmarks/check_step105_acceptance.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)

@pytest.mark.parametrize('tp,dynamic',[(1,False),(2,False),(4,False),(2,True)])
def test_tiny_model_graph(tp,dynamic,tmp_path):
 result=runner.execute(tp,dynamic,tmp_path,180)
 assert result['status']=='passed',result


def test_tp_graph_support_boundary():
 from types import SimpleNamespace as N
 from minivllm.config import ModelConfig,CUDAGraphMode
 from minivllm.testing.tiny_models import tiny_qwen3_config
 from minivllm.worker.worker import Worker
 def make(mode,*,prefix=False,spec=None,pp=1):
  hf=tiny_qwen3_config('tiny_gqa')
  cfg=N(parallel_config=N(world_size=2,tensor_parallel_size=2,pipeline_parallel_size=pp,data_parallel_size=1),
        model_config=ModelConfig('test',hf_config=hf,max_model_len=64),use_v2_model_runner=True,
        speculative_config=spec,compilation_config=N(cudagraph_mode=mode),device_config=N(device='cuda'),
        attention_backend='triton',cache_config=N(enable_prefix_caching=prefix))
  return Worker(cfg,rank=0,device_index=0)
 make(CUDAGraphMode.FULL_DECODE_ONLY)._validate_tp_model()
 for worker in (make(CUDAGraphMode.FULL),make(CUDAGraphMode.PIECEWISE),make(CUDAGraphMode.FULL_AND_PIECEWISE),
                make(CUDAGraphMode.FULL_DECODE_ONLY,prefix=True),make(CUDAGraphMode.FULL_DECODE_ONLY,spec=object()),
                make(CUDAGraphMode.FULL_DECODE_ONLY,pp=2)):
  with pytest.raises(NotImplementedError):worker._validate_tp_model()


@pytest.mark.parametrize('name',['engine_lifecycle','capture_failure'])
def test_model_graph_lifecycle(name,tmp_path):
 result=runner.execute_lifecycle(name,tmp_path,300)
 assert result['status']=='passed',Path(result['stage']['log']).read_text()[-15000:]


def test_frozen_graph_sources_match_installed_vllm():
 import hashlib
 import importlib.util
 import json
 ref=ROOT/'learning_notes/14_vllm_from_scratch/分布式并行完整需求/参考资料'
 hashes={x['path']:x['sha256'] for x in json.loads((ref/'源码清单.json').read_text())['files']}
 installed=Path(importlib.util.find_spec('vllm').origin).parent
 for name in ('v1/worker/gpu/cudagraph_utils.py','v1/worker/gpu/model_runner.py','compilation/cuda_graph.py','config/compilation.py'):
  assert hashlib.sha256((ref/'vllm_0_28_0'/name).read_bytes()).hexdigest()==hashes[name]==hashlib.sha256((installed/name).read_bytes()).hexdigest()


def test_random_constraints_same_tp_rank_state(tmp_path):
 result=runner.execute_random(tmp_path,180)
 assert result['status']=='passed',result
