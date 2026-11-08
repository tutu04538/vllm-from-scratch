"""Isolated real Qwen3 workload with per-rank workspace and optional constrained sampling."""
import argparse,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['VLLM_USE_V2_MODEL_RUNNER']='1';os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION']='1'
from benchmarks.step103_workload import prompts
from benchmarks.step103_memory_hooks import snapshot
from benchmarks.step103_hooks import install,flush
from benchmarks.check_step090_models import resolve_snapshot


def main():
 p=argparse.ArgumentParser();p.add_argument('--engine',choices=['mini','frozen'],required=True);p.add_argument('--model',choices=['8b','14b'],required=True);p.add_argument('--tp',type=int,choices=[1,2,4],required=True);p.add_argument('--batch',type=int,choices=[1,4],required=True);p.add_argument('--random',action='store_true');p.add_argument('--forced',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.model=='14b' and a.tp==1:p.error('14B TP1 exceeds card memory')
 a.output.mkdir(parents=True,exist_ok=True);path,_=resolve_snapshot(a.model);hf=json.loads((path/'config.json').read_text());ids=prompts(a.model,a.batch)
 if a.random and a.forced:p.error('random and forced are separate diagnostics')
 sample={'temperature':0.8 if a.random else 0,'max_tokens':4 if a.forced else 32,'ignore_eos':True}
 histories=None
 if a.forced:
  ref=ROOT/f'benchmarks/results/step103/reference_{a.model}_tp{a.tp}_b{a.batch}/summary.json'
  histories=[x['tokens'][:4] for x in json.loads(ref.read_text())['runs'][0]['outputs']]
  from benchmarks.step103_forced_hooks import install as force_install,flush as force_flush
 if a.random:sample.update(seed=123,top_k=5,top_p=0.95,min_p=0.05)
 start=time.monotonic()
 if a.engine=='frozen':
  from benchmarks.check_step090_reference import verify_oracle
  verify_oracle()
  from vllm import LLM,SamplingParams
  budget=2*16*(hf['num_key_value_heads']//a.tp)*hf['head_dim']*2*hf['num_hidden_layers']*192
  llm=LLM(model=str(path),skip_tokenizer_init=True,dtype='bfloat16',tensor_parallel_size=a.tp,enforce_eager=True,max_model_len=2048,max_num_seqs=a.batch,max_num_batched_tokens=128,kv_cache_memory_bytes=budget,gpu_memory_utilization=.86,attention_config={'backend':'TRITON_ATTN'},enable_prefix_caching=False,disable_log_stats=True,seed=0)
  before=llm.collective_rpc(snapshot,args=(True,));llm.collective_rpc(install,args=(True,))
  if a.forced:llm.collective_rpc(force_install,args=(histories,))
  generated=llm.generate([{'prompt_token_ids':x} for x in ids],SamplingParams(**sample),use_tqdm=False)
  after=llm.collective_rpc(snapshot,args=(True,));ranks=llm.collective_rpc(flush,args=(str(a.output.resolve()),'run0'))
  forced_ranks=llm.collective_rpc(force_flush,args=(str(a.output.resolve()),)) if a.forced else []
  output=[{'request_id':o.request_id,'tokens':list(o.outputs[0].token_ids)} for o in generated]
 else:
  from minivllm import CacheConfig,DeviceConfig,LLMEngine,ModelConfig,SamplingParams,SchedulerConfig,VllmConfig
  from minivllm.config import CompilationConfig,ParallelConfig
  from minivllm.executor.multiproc_executor import MultiprocExecutor
  from minivllm.executor.uniproc_executor import UniProcExecutor
  cfg=VllmConfig(model_config=ModelConfig(str(path),dtype='bfloat16',max_model_len=2048,hf_config=hf,seed=0),cache_config=CacheConfig(block_size=16,num_gpu_blocks=192),scheduler_config=SchedulerConfig(max_num_seqs=a.batch,max_num_batched_tokens=128),device_config=DeviceConfig(device='cuda'),parallel_config=ParallelConfig(tensor_parallel_size=a.tp),compilation_config=CompilationConfig(cudagraph_mode='none'),attention_backend='triton')
  ex=MultiprocExecutor(cfg,_startup_timeout=1500) if a.tp>1 else UniProcExecutor(cfg);engine=None
  def rpc(fn,*args):return ex.collective_rpc(fn,args=args) if a.tp>1 else [fn(ex.driver_worker,*args)]
  try:
   engine=LLMEngine(cfg,ex,tokenizer=None);before=rpc(snapshot,False);rpc(install,False)
   if a.forced:rpc(force_install,histories)
   for i,prompt in enumerate(ids):engine.add_request(str(i),prompt,SamplingParams(**sample))
   outputs={}
   while engine.has_unfinished_requests():
    for o in engine.step():outputs[o.request_id]=list(o.token_ids)
   after=rpc(snapshot,False);ranks=rpc(flush,str(a.output.resolve()),'run0')
   forced_ranks=rpc(force_flush,str(a.output.resolve())) if a.forced else []
   output=[{'request_id':str(i),'tokens':outputs[str(i)]} for i in range(a.batch)]
  finally:
   if engine is not None:engine.shutdown()
   else:ex.shutdown()
 assert all(len(x['tokens'])==sample['max_tokens'] for x in output)
 if a.forced:assert [x['tokens'] for x in output]==histories
 result={'status':'passed','engine':a.engine,'model':a.model,'tp':a.tp,'batch':a.batch,'random':a.random,
         'sampling':sample,'forced':a.forced,'forced_ranks':forced_ranks,'before':before,'after':after,'ranks':ranks,'outputs':output,'elapsed_seconds':time.monotonic()-start}
 (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'passed','engine':a.engine,'model':a.model,'tp':a.tp,'batch':a.batch,'random':a.random}))
if __name__=='__main__':main()
