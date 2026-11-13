"""Capture first-divergence logits in an isolated, diagnostic-only engine run."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from benchmarks.check_step090_models import resolve_snapshot
from benchmarks.step111_parallel_worker import run_once


def install(worker, targets):
    import torch
    runner=worker.model_runner
    runner._step111_logit_rows=[]
    runner._step111_seen={}
    original_logits=runner.model.compute_logits
    original_sample=runner.sample
    def logits(hidden):
        result=original_logits(hidden)
        runner._step111_logits_dtype=str(result.dtype)
        runner._step111_raw_logits=result.detach()
        values,ids=result.float().topk(2,dim=-1)
        runner._step111_top=(values.detach().cpu(),ids.detach().cpu())
        return result
    def sample(hidden,batch,grammar):
        result=original_sample(hidden,batch,grammar)
        sampled,counts,_=result
        counts=counts.detach().cpu().tolist()
        sampled_ids=sampled.sampled_token_ids.detach().cpu().tolist()
        values,ids=runner._step111_top
        for row,req in enumerate(batch.req_ids):
            req=str(req).split('-',1)[0]
            if not counts[row]: continue
            step=runner._step111_seen.get(req,0);runner._step111_seen[req]=step+int(counts[row])
            raw = runner._step111_raw_logits[row].cpu().float().tolist() if targets.get(req) == step else None
            runner._step111_logit_rows.append({'request':req,'step':step,'top_ids':ids[row].tolist(),
                  'top_values':values[row].tolist(),'dtype':runner._step111_logits_dtype,
                  'sampled':int(sampled_ids[row][0]), 'full_logits':raw})
        return result
    runner.model.compute_logits=logits
    runner.sample=sample
    return True


def flush(worker):
    import torch
    import torch.distributed as dist
    return {'rank':dist.get_rank() if dist.is_initialized() else 0,
            'device_uuid':str(torch.cuda.get_device_properties(worker.device).uuid),
            'events':worker.model_runner._step111_logit_rows}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--implementation',choices=('mini','frozen'),required=True)
    p.add_argument('--gate-report',type=Path,required=True)
    p.add_argument('--case-id')
    p.add_argument('--output-tokens',type=int,choices=(32,128),default=32)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--targets',required=True)
    a=p.parse_args()
    cases=json.loads(a.gate_report.read_text())['results']
    matches=[row for row in cases if a.case_id is None or row.get('case_id')==a.case_id]
    if len(matches)!=1:p.error('select exactly one case')
    source=matches[0]
    case=source['case'];tp=case['tp'];model=case['model'];batch=case['batch']
    if case['k']:p.error('diagnostic currently supports K0 only')
    graph=case['mode']=='graph'
    prompts=source['gate']['frozen_ref']['result']['prompt_ids']
    target,_=resolve_snapshot(model);hf=json.loads((target/'config.json').read_text())
    if a.implementation=='mini':
        from minivllm import CacheConfig,DeviceConfig,LLMEngine,ModelConfig,SchedulerConfig,VllmConfig
        from minivllm.config import CompilationConfig,ParallelConfig
        from minivllm.executor.multiproc_executor import MultiprocExecutor
        from minivllm.executor.uniproc_executor import UniProcExecutor
        cfg=VllmConfig(model_config=ModelConfig(str(target),dtype='bfloat16',max_model_len=2048,hf_config=hf,seed=0),
            cache_config=CacheConfig(block_size=16,num_gpu_blocks=800,enable_prefix_caching=False),
            scheduler_config=SchedulerConfig(max_num_seqs=batch,max_num_batched_tokens=128,async_scheduling=False),
            device_config=DeviceConfig('cuda'),parallel_config=ParallelConfig(tensor_parallel_size=tp),
            compilation_config=CompilationConfig(cudagraph_mode='full_decode_only' if graph else 'none',
                cudagraph_capture_sizes=[1,2,4] if graph else None),attention_backend='triton')
        executor=MultiprocExecutor(cfg,_startup_timeout=1500) if tp>1 else UniProcExecutor(cfg);engine=None
        def rpc(fn,*args): return executor.collective_rpc(fn,args=args) if tp>1 else [fn(executor.driver_worker,*args)]
        try:
            engine=LLMEngine(cfg,executor,tokenizer=None)
            run_once(engine,prompts,a.output_tokens,False,warmup=True);rpc(install,json.loads(a.targets))
            run=run_once(engine,prompts,a.output_tokens,False);observed=rpc(flush)
        finally:
            if engine is None:executor.shutdown()
            else:engine.shutdown()
    else:
        os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION']='1'
        from benchmarks.check_step090_reference import verify_oracle
        verify_oracle()
        from vllm import LLM
        heads=max(1,hf['num_key_value_heads']//tp)
        block_bytes=2*16*heads*hf['head_dim']*2*hf['num_hidden_layers']
        llm=LLM(model=str(target),skip_tokenizer_init=True,dtype='bfloat16',tensor_parallel_size=tp,
                enforce_eager=not graph,async_scheduling=False,max_model_len=2048,max_num_seqs=batch,max_num_batched_tokens=128,
                block_size=16,gpu_memory_utilization=.9,kv_cache_memory_bytes=800*block_bytes,
                attention_config={'backend':'TRITON_ATTN'},enable_prefix_caching=False,disable_log_stats=True,seed=0,
                compilation_config={'mode':0,'cudagraph_mode':'FULL_DECODE_ONLY','cudagraph_capture_sizes':[1,2,4]} if graph else {'mode':0})
        try:
            engine=llm.llm_engine
            run_once(engine,prompts,a.output_tokens,True,warmup=True);llm.collective_rpc(install,args=(json.loads(a.targets),))
            run=run_once(engine,prompts,a.output_tokens,True);observed=llm.collective_rpc(flush)
        finally:
            import gc
            del llm;gc.collect()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({'implementation':a.implementation,'case':case,'prompt_ids':prompts,
                                    'tokens':run['tokens'],'ranks':observed},indent=2)+'\n')
    return 0

if __name__=='__main__':raise SystemExit(main())
