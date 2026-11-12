"""Measure instrumented proposal GPU time and accepted draft-token counts."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['VLLM_USE_V2_MODEL_RUNNER']='1'
from benchmarks.check_step090_models import resolve_snapshot
from benchmarks.step103_workload import prompts


def install_timer(worker):
    import torch
    spec=worker.model_runner.speculator
    original=spec.propose
    worker.model_runner._step109_timers=[]
    def propose(*args,**kwargs):
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record();result=original(*args,**kwargs);end.record()
        worker.model_runner._step109_timers.append((start,end))
        return result
    spec.propose=propose
    return True


def timer_result(worker):
    import torch
    import torch.distributed as dist
    torch.cuda.synchronize(worker.device)
    times=[start.elapsed_time(end) for start,end in worker.model_runner._step109_timers]
    target=worker.model_runner.model;draft=worker.model_runner.speculator.model
    def storages(model):
        unique={}
        for tensor in list(model.parameters()) + [layer.self_attn.attn.kv_cache for layer in model.model.layers]:
            storage=tensor.untyped_storage()
            unique[(str(tensor.device),storage.data_ptr())]=storage.nbytes()
        return unique
    t=storages(target);d=storages(draft);shared=t.keys() & d.keys()
    return {'rank':dist.get_rank() if dist.is_initialized() else 0,'count':len(times),
            'sum_gpu_ms':sum(times),'mean_gpu_ms':sum(times)/len(times) if times else None,
            'target_parameter_and_kv_unique_bytes':sum(t.values()),
            'draft_parameter_and_kv_unique_bytes':sum(d.values()),
            'shared_unique_bytes':sum(t[key] for key in shared),
            'combined_unique_bytes':sum(t.values())+sum(d.values())-sum(t[key] for key in shared),
            'allocator_peak_bytes':torch.cuda.max_memory_allocated(worker.device)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['8b','14b'],required=True)
    p.add_argument('--mode',choices=['eager','graph'],required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    from minivllm import CacheConfig,DeviceConfig,LLMEngine,ModelConfig,SamplingParams,SchedulerConfig,SpeculativeConfig,VllmConfig
    from minivllm.config import CompilationConfig,ParallelConfig
    from minivllm.executor.multiproc_executor import MultiprocExecutor
    target,_=resolve_snapshot(a.model)
    revisions={'8b':'9629dfce7a4a10564dd48d3e5485c3976095653c','14b':'3d13517724e81cb409ddf1d4650772ec52f1e18e'}
    draft=Path('/data/syh/models/converted/AngelSlim')/f'Qwen3-{a.model.upper()}_eagle3'/revisions[a.model]
    cfg=VllmConfig(model_config=ModelConfig(str(target),dtype='bfloat16',max_model_len=2048,hf_config=json.loads((target/'config.json').read_text()),seed=0),
        cache_config=CacheConfig(block_size=16,num_gpu_blocks=192,enable_prefix_caching=False),scheduler_config=SchedulerConfig(max_num_seqs=1,max_num_batched_tokens=128,async_scheduling=False),
        device_config=DeviceConfig('cuda'),parallel_config=ParallelConfig(tensor_parallel_size=2),attention_backend='triton',
        compilation_config=CompilationConfig(cudagraph_mode='full_decode_only' if a.mode=='graph' else 'none',
                                              cudagraph_capture_sizes=[1,2,3,4,6,12] if a.mode=='graph' else None),
        speculative_config=SpeculativeConfig(method='eagle3',num_speculative_tokens=2,
            draft_model_config=ModelConfig(str(draft),dtype='bfloat16',max_model_len=2048,hf_config=json.loads((draft/'config.json').read_text()))))
    executor=MultiprocExecutor(cfg);engine=None
    try:
        engine=LLMEngine(cfg,executor,tokenizer=None);executor.collective_rpc(install_timer)
        engine.add_request('a',prompts(a.model,1)[0],SamplingParams(temperature=0,max_tokens=32,ignore_eos=True))
        totals={'drafted':0,'accepted':0};output=[];start=time.perf_counter()
        while engine.has_unfinished_requests():
            for item in engine.step():output=list(item.token_ids)
            stats=engine.last_scheduler_stats.spec_decoding_stats
            if stats is not None:
                totals['drafted']+=stats.num_draft_tokens;totals['accepted']+=stats.num_accepted_tokens
        wall=time.perf_counter()-start
        times=executor.collective_rpc(timer_result)
        assert len(output)==32 and totals['drafted']>0
        result={'model':a.model,'mode':a.mode,'tp':2,'k':2,'prompt_tokens':128,'output_tokens':32,
                'wall_seconds':wall,'timers':times,'drafted':totals['drafted'],'accepted':totals['accepted'],
                'acceptance_rate':totals['accepted']/totals['drafted'],
                'note':'proposal GPU stream interval and parent wall time, instrumented correctness workload; not serving throughput. Unique storage includes parameters and KV, while allocator peak includes other workspace/graphs and cannot be split by model.'}
        a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    finally:
        if engine is None:executor.shutdown()
        else:engine.shutdown()

if __name__=='__main__':main()
