"""Run the same 8B TP2 B1 fixture on one engine with test-only layer hooks."""
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['VLLM_USE_V2_MODEL_RUNNER']='1';os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION']='1'
from benchmarks.step103_workload import prompts
from benchmarks.step103_diagnostic_hooks import install,flush
from benchmarks.check_step090_models import resolve_snapshot


def main():
 p=argparse.ArgumentParser();p.add_argument('--engine',choices=['mini','frozen'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--model',choices=['8b','14b'],default='8b');p.add_argument('--tp',type=int,choices=[2,4],default=2);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 path,_=resolve_snapshot(a.model);hf=json.loads((path/'config.json').read_text());prompt=prompts(a.model,1)[0]
 if a.engine=='frozen':
  from benchmarks.check_step090_reference import verify_oracle
  verify_oracle()
  from vllm import LLM,SamplingParams
  budget=2*16*(hf['num_key_value_heads']//a.tp)*hf['head_dim']*2*hf['num_hidden_layers']*192
  llm=LLM(model=str(path),skip_tokenizer_init=True,dtype='bfloat16',tensor_parallel_size=a.tp,enforce_eager=True,max_model_len=2048,max_num_seqs=1,max_num_batched_tokens=128,kv_cache_memory_bytes=budget,gpu_memory_utilization=.86,attention_config={'backend':'TRITON_ATTN'},enable_prefix_caching=False,disable_log_stats=True,seed=0)
  llm.collective_rpc(install)
  result=llm.generate([{'prompt_token_ids':prompt}],SamplingParams(temperature=0,max_tokens=4,ignore_eos=True),use_tqdm=False)
  tokens=list(result[0].outputs[0].token_ids);ranks=llm.collective_rpc(flush,args=(str(a.output.resolve()),))
 else:
  from minivllm import CacheConfig,DeviceConfig,LLMEngine,ModelConfig,SamplingParams,SchedulerConfig,VllmConfig
  from minivllm.config import CompilationConfig,ParallelConfig
  from minivllm.executor.multiproc_executor import MultiprocExecutor
  cfg=VllmConfig(model_config=ModelConfig(str(path),dtype='bfloat16',max_model_len=2048,hf_config=hf,seed=0),cache_config=CacheConfig(block_size=16,num_gpu_blocks=192),scheduler_config=SchedulerConfig(max_num_seqs=1,max_num_batched_tokens=128),device_config=DeviceConfig(device='cuda'),parallel_config=ParallelConfig(tensor_parallel_size=a.tp),compilation_config=CompilationConfig(cudagraph_mode='none'),attention_backend='triton')
  ex=MultiprocExecutor(cfg);engine=None
  try:
   engine=LLMEngine(cfg,ex,tokenizer=None);ex.collective_rpc(install)
   engine.add_request('0',prompt,SamplingParams(temperature=0,max_tokens=4,ignore_eos=True));tokens=[]
   while engine.has_unfinished_requests():
    for out in engine.step():tokens=list(out.token_ids)
   ranks=ex.collective_rpc(flush,args=(str(a.output.resolve()),))
  finally:
   if engine is not None:engine.shutdown()
   else:ex.shutdown()
 (a.output/'summary.json').write_text(json.dumps({'tokens':tokens,'ranks':ranks},indent=2)+'\n')
 print(json.dumps({'engine':a.engine,'tokens':tokens}))
if __name__=='__main__':main()
