"""Run 103 workloads in production MiniVLLM workers in an isolated process."""
import argparse,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['VLLM_USE_V2_MODEL_RUNNER']='1'
from minivllm import CacheConfig,DeviceConfig,LLMEngine,ModelConfig,SamplingParams,SchedulerConfig,VllmConfig
from minivllm.config import CompilationConfig,ParallelConfig
from minivllm.executor.multiproc_executor import MultiprocExecutor
from minivllm.executor.uniproc_executor import UniProcExecutor
from benchmarks.step103_workload import prompts
from benchmarks.step103_hooks import install,flush
from benchmarks.check_step090_models import resolve_snapshot


def main():
 p=argparse.ArgumentParser();p.add_argument('--model',choices=['8b','14b'],required=True)
 p.add_argument('--tp',type=int,choices=[1,2,4],required=True);p.add_argument('--batch',type=int,choices=[1,4],required=True)
 p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.model=='14b' and a.tp==1:p.error('14B TP1 exceeds card memory')
 a.output.mkdir(parents=True,exist_ok=True)
 path,_=resolve_snapshot(a.model);hf=json.loads((path/'config.json').read_text())
 cfg=VllmConfig(model_config=ModelConfig(str(path),dtype='bfloat16',max_model_len=2048,hf_config=hf,seed=0),
     cache_config=CacheConfig(block_size=16,num_gpu_blocks=192,enable_prefix_caching=False),
     scheduler_config=SchedulerConfig(max_num_seqs=a.batch,max_num_batched_tokens=128),
     device_config=DeviceConfig(device='cuda'),parallel_config=ParallelConfig(tensor_parallel_size=a.tp),
     compilation_config=CompilationConfig(cudagraph_mode='none'),attention_backend='triton')
 start=time.monotonic();ex=MultiprocExecutor(cfg,_startup_timeout=1500) if a.tp>1 else UniProcExecutor(cfg);engine=None
 def rpc(fn,*args):
  return ex.collective_rpc(fn,args=args) if a.tp>1 else [fn(ex.driver_worker,*args)]
 try:
  engine=LLMEngine(cfg,ex,tokenizer=None)
  initial=rpc(install,False)
  prompt_ids=prompts(a.model,a.batch);run_id='run0'
  for i,prompt in enumerate(prompt_ids):
   engine.add_request(str(i),prompt,SamplingParams(temperature=0,max_tokens=32,ignore_eos=True))
  outputs={};finished={};steps=0
  while engine.has_unfinished_requests():
   step_outputs=engine.step();steps+=1
   for o in step_outputs:
    outputs[o.request_id]=list(o.token_ids)
    if o.finished:finished[o.request_id]=steps
   if steps>256:raise RuntimeError('generation failed to finish within 256 scheduler steps')
  ranks=rpc(flush,str(a.output.resolve()),run_id)
  ordered=[{'request_id':str(i),'tokens':outputs[str(i)],'finish_step':finished[str(i)]} for i in range(a.batch)]
  assert all(len(o['tokens'])==32 for o in ordered)
  result={'status':'passed','model':a.model,'tp':a.tp,'batch':a.batch,'prompts':prompt_ids,
          'initial_ranks':initial,'runs':[{'outputs':ordered,'ranks':ranks}],
          'scheduler_steps':steps,'elapsed_seconds':time.monotonic()-start}
  (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
  print(json.dumps({'status':'passed','model':a.model,'tp':a.tp,'batch':a.batch,'steps':steps}))
 finally:
  if engine is not None:engine.shutdown()
  else:ex.shutdown()
if __name__=='__main__':main()
