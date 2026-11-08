"""Production Qwen3 Worker load/forward probes, no Worker override."""
import argparse,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).parent))
os.environ['VLLM_USE_V2_MODEL_RUNNER']='1'
from minivllm import CacheConfig,DeviceConfig,LLMEngine,ModelConfig,SamplingParams,SchedulerConfig,VllmConfig
from minivllm.config import CompilationConfig,ParallelConfig
from minivllm.executor.multiproc_executor import MultiprocExecutor
from minivllm.executor.uniproc_executor import UniProcExecutor
from step101_hooks import manifest,install,flush


def tiny(path,head_dim=8):
 import torch
 from safetensors.torch import save_file
 from minivllm.testing.tiny_models import tiny_qwen3_config,_weights
 cfg=tiny_qwen3_config('tiny_gqa');cfg.update(num_hidden_layers=4,layer_types=['full_attention']*4,head_dim=head_dim,intermediate_size=64,vocab_size=129)
 path.mkdir(parents=True,exist_ok=True)
 (path/'config.json').write_text(json.dumps(cfg,indent=2)+'\n');save_file(_weights(cfg,seed=0),str(path/'model.safetensors'))
 return cfg


def rpc(ex,tp,fn,*args):
 return ex.collective_rpc(fn,args=args) if tp>1 else [fn(ex.driver_worker,*args)]


def main():
 p=argparse.ArgumentParser();p.add_argument('--model',choices=['tiny','8b','14b'],required=True);p.add_argument('--tp',type=int,choices=[1,2,4],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--load-only',action='store_true');p.add_argument('--head-dim',type=int,default=8);p.add_argument('--pause-layer',type=int);a=p.parse_args()
 if a.model=='14b' and a.tp==1:p.error('14B BF16 TP1 超出单卡容量')
 a.output.mkdir(parents=True,exist_ok=True)
 if a.model=='tiny':path=a.output/'tiny';hf=tiny(path,a.head_dim);dtype='float32';maxlen=64;budget=64;blocks=8;prompt=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17];steps=4
 else:
  from benchmarks.check_step090_models import resolve_snapshot
  path,_=resolve_snapshot(a.model);hf=json.loads((path/'config.json').read_text());dtype='bfloat16';maxlen=2048;budget=128;blocks=128
  prompt=json.loads((ROOT/'benchmarks/results/step090/numerics_contract.json').read_text())['models']['8b']['prompt_token_ids'];steps=1
 cfg=VllmConfig(model_config=ModelConfig(str(path),dtype=dtype,max_model_len=maxlen,hf_config=hf),
  cache_config=CacheConfig(block_size=16,num_gpu_blocks=blocks),scheduler_config=SchedulerConfig(max_num_seqs=1,max_num_batched_tokens=budget),
  device_config=DeviceConfig(device='cuda'),parallel_config=ParallelConfig(tensor_parallel_size=a.tp),
  compilation_config=CompilationConfig(cudagraph_mode='none'),attention_backend='triton')
 start=time.monotonic();ex=MultiprocExecutor(cfg,_startup_timeout=1500) if a.tp>1 else UniProcExecutor(cfg)
 try:
  m=rpc(ex,a.tp,manifest,str(a.output.resolve())); result={'model':a.model,'tp':a.tp,'load_seconds':time.monotonic()-start,'manifest':m}
  if not a.load_only:
   engine=LLMEngine(cfg,ex,tokenizer=None)
   try:
    rpc(ex,a.tp,install,a.model=='tiny',a.pause_layer)
    engine.add_request('a',prompt,SamplingParams(temperature=0,max_tokens=steps,ignore_eos=True))
    for step in range(steps):
     out=engine.step();rpc(ex,a.tp,flush,str(a.output.resolve()),step)
    result['output_tokens']=out[-1].token_ids
   finally:engine.shutdown()
  result['status']='passed';(a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
 finally:ex.shutdown()
if __name__=='__main__':main()
