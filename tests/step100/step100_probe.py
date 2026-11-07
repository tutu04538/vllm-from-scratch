"""真实多 Worker 的 100 定点场景，父进程不创建 CUDA 模型。"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ['VLLM_USE_V2_MODEL_RUNNER']='1'
from minivllm import CacheConfig, DeviceConfig, LLMEngine, ModelConfig, SamplingParams, SchedulerConfig, VllmConfig
from minivllm.config import CompilationConfig, ParallelConfig
from minivllm.executor.multiproc_executor import MultiprocExecutor
from minivllm.testing.step100_worker import Step100Worker
from minivllm.testing.tiny_models import tiny_qwen3_config, tiny_qwen3_dir


def run(tp, mode, output):
    hf=tiny_qwen3_config('tiny_gqa')
    cfg=VllmConfig(model_config=ModelConfig(model=tiny_qwen3_dir('tiny_gqa'),dtype='float32',max_model_len=64,hf_config=hf,seed=123),
        cache_config=CacheConfig(block_size=16,num_gpu_blocks=4 if mode=='preempt' else 8),
        scheduler_config=SchedulerConfig(max_num_seqs=2,max_num_batched_tokens=64),
        parallel_config=ParallelConfig(tensor_parallel_size=tp),device_config=DeviceConfig(device='cuda'),
        compilation_config=CompilationConfig(cudagraph_mode='none'),attention_backend='triton')
    ex=MultiprocExecutor(cfg,_worker_cls=Step100Worker)
    engine=LLMEngine(cfg,ex,tokenizer=None)
    try:
        core=engine.engine_core.engine_core
        values=dict(temperature=0 if mode=='greedy' else 0.8,max_tokens=24 if mode=='preempt' else 8,
                    ignore_eos=True,logprobs=2)
        if mode!='greedy':values.update(top_k=7,top_p=0.9,min_p=0.05,repetition_penalty=1.1,presence_penalty=0.2,frequency_penalty=0.1)
        for req,seed in [('a',17),('b',None if mode=='preempt' else 23)]:
            engine.add_request(req,[1,2,3,5,6,7,8,9,10,1,2,3,5,6,7,8,9],SamplingParams(seed=seed,**values))
        rounds=[]
        limit=36 if mode=='preempt' else 3
        for step in range(limit):
            if not engine.has_unfinished_requests():
                break
            if step==1 and mode!='preempt':core.scheduler.running.reverse()
            if step==2 and mode!='preempt':
                engine.abort_request(['a'])
                engine.add_request('c',[9,8,7,6],SamplingParams(seed=None,**values))
            outputs=engine.step()
            ranks=ex.collective_rpc('inspect_sampling')
            rounds.append({'ranks':ranks,'outputs':{o.request_id:o.token_ids for o in outputs},
                           'scheduler':{req:list(obj.output_token_ids) for req,obj in core.scheduler.requests.items()},
                           'preemptions':core.scheduler.num_preemptions})
        Path(output).write_text(json.dumps({'tp':tp,'mode':mode,'rounds':rounds})+'\n')
        print(json.dumps({'tp':tp,'mode':mode,'rounds':len(rounds),'preemptions':core.scheduler.num_preemptions}))
    finally:engine.shutdown()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tp',type=int,required=True);p.add_argument('--mode',choices=['greedy','random','preempt'],required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.tp,a.mode,a.output)
