"""Run each tiny inference configuration in its own process."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '1'
import torch
from minivllm import CacheConfig, DeviceConfig, LLMEngine, ModelConfig, SamplingParams, SchedulerConfig, UniProcExecutor, VllmConfig
from minivllm.config import CompilationConfig, ParallelConfig
from minivllm.executor.multiproc_executor import MultiprocExecutor
from minivllm.testing.step098_worker import Step098Worker
from minivllm.testing.tiny_models import tiny_qwen3_config, tiny_qwen3_dir


def run(tp, output, caps=None):
    if caps:
        os.environ['STEP098_CAPS'] = caps
    cfg = VllmConfig(model_config=ModelConfig(model=tiny_qwen3_dir('tiny_gqa'), dtype='float32', max_model_len=64, hf_config=tiny_qwen3_config('tiny_gqa')),
        cache_config=CacheConfig(block_size=16, num_gpu_blocks=8 if tp > 1 else 6),
        scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=64),
        parallel_config=ParallelConfig(tensor_parallel_size=tp), device_config=DeviceConfig(device='cuda'),
        compilation_config=CompilationConfig(cudagraph_mode='none'), attention_backend='triton')
    if tp == 1:
        executor = UniProcExecutor(cfg, Step098Worker(cfg))
    else:
        executor = MultiprocExecutor(cfg, _worker_cls=Step098Worker)
    engine = LLMEngine(cfg, executor, tokenizer=None)
    try:
        core = engine.engine_core.engine_core
        cap = core.kv_cache_manager.kv_cache_config.num_gpu_blocks
        engine.add_request('a', [1,2,3,5,6,7,8,9,10,1,2,3,5,6,7,8,9], SamplingParams(temperature=0, max_tokens=3, ignore_eos=True))
        snapshots=[]
        block_ids=[]
        for _ in range(2):
            engine.step()
            if tp == 1:
                kv = [{k:v.detach().cpu() for k,v in executor.driver_worker.model_runner.kv_caches.items()}]
            else:
                kv = executor.collective_rpc('inspect_kv')
            snapshots.append(kv)
            block_ids.append(core.kv_cache_manager.get_blocks('a').get_block_ids())
        engine.abort_request(['a'])
        engine.add_request('b', [10,9,8,7,6,5,3,2,1,10,9,8,7,6,5,3,2], SamplingParams(temperature=0, max_tokens=3, ignore_eos=True))
        engine.step()
        block_ids.append(core.kv_cache_manager.get_blocks('b').get_block_ids())
        if tp == 1:
            kv = [{k:v.detach().cpu() for k,v in executor.driver_worker.model_runner.kv_caches.items()}]
        else:
            kv = executor.collective_rpc('inspect_kv')
        snapshots.append(kv)
        inputs = executor.collective_rpc('inspect_attention_inputs') if tp > 1 else [executor.driver_worker.inspect_attention_inputs()]
        budgets = executor.worker_kv_available_memory if tp > 1 else []
        if tp > 1:
            assert all(executor.collective_rpc('test_padding'))
        torch.save({'snapshots':snapshots, 'capacity':cap, 'block_ids':block_ids, 'inputs':inputs, 'budgets':budgets}, output)
        print(json.dumps({'tp':tp,'capacity':cap,'ok':True}))
    finally:
        engine.shutdown()


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--tp', type=int, required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--caps')
    a=p.parse_args()
    run(a.tp,a.output,a.caps)
