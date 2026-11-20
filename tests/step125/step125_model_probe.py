"""Run 103 workloads in production MiniVLLM workers in an isolated process."""
import argparse,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
os.environ['VLLM_USE_V2_MODEL_RUNNER']='1'
from minivllm import CacheConfig,DeviceConfig,LLMEngine,ModelConfig,SamplingParams,SchedulerConfig,VllmConfig
from minivllm.config import CompilationConfig,ParallelConfig
from minivllm.executor.multiproc_executor import MultiprocExecutor
from minivllm.executor.uniproc_executor import UniProcExecutor
from benchmarks.step103_workload import prompts
from benchmarks.step103_hooks import install,flush
from benchmarks.step125_awq_audit import audit
from benchmarks.step123_layer_hooks import install_layers, flush_layers
from benchmarks.check_step090_models import resolve_snapshot


def main():
 p=argparse.ArgumentParser();p.add_argument('--model',choices=['8b','14b'],required=True)
 p.add_argument('--tp',type=int,choices=[1,2,4],required=True);p.add_argument('--batch',type=int,choices=[1,4],required=True)
 p.add_argument('--output',type=Path,required=True);p.add_argument('--layer-audit',action='store_true');p.add_argument('--teacher-forcing',action='store_true');p.add_argument('--teacher-index',type=int,choices=range(4));a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=True)
 path=Path('/data/syh/.cache/huggingface/hub/models--Qwen--Qwen3-'+a.model.upper()+'-AWQ/snapshots')
 path=path / {'8b': '4da05a8edb55c6046cce958586c33b61da07bb79', '14b': '31c69efc29464b6bb0aee1398b5a7b50a99340c3'}[a.model];hf=json.loads((path/'config.json').read_text())
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
  methods=rpc(audit)
  if a.layer_audit: rpc(install_layers)
  prompt_ids=prompts(a.model,a.batch);run_id='run0'
  if a.teacher_forcing:
   if a.teacher_index is None and a.batch != 4: p.error('teacher forcing requires batch 4 or teacher index')
   fixed=[198,14880,11622,105321]
   sources=prompts(a.model,4)
   prompt_ids=[sources[i][:32]+fixed[:i] for i in (range(4) if a.teacher_index is None else [a.teacher_index])]
  for i,prompt in enumerate(prompt_ids):
   engine.add_request(str(i),prompt,SamplingParams(temperature=0,max_tokens=1 if a.teacher_forcing else 128,ignore_eos=True))
  outputs={};finished={};steps=0
  while engine.has_unfinished_requests():
   step_outputs=engine.step();steps+=1
   for o in step_outputs:
    outputs[o.request_id]=list(o.token_ids)
    if o.finished:finished[o.request_id]=steps
   if steps>256:raise RuntimeError('generation failed to finish within 256 scheduler steps')
  ranks=rpc(flush,str(a.output.resolve()),run_id)
  if a.layer_audit: rpc(flush_layers,str(a.output.resolve()))
  ordered=[{'request_id':str(i),'tokens':outputs[str(i)],'finish_step':finished[str(i)]} for i in range(a.batch)]
  assert all(len(o['tokens'])==(1 if a.teacher_forcing else 128) for o in ordered)
  result={'status':'passed','model':a.model,'tp':a.tp,'batch':a.batch,'prompts':prompt_ids,
          'kv_num_blocks_requested':cfg.cache_config.num_gpu_blocks,
          'initial_ranks':initial,'awq_methods':methods,'runs':[{'outputs':ordered,'ranks':ranks}],
          'scheduler_steps':steps,'elapsed_seconds':time.monotonic()-start}
  (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
  print(json.dumps({'status':'passed','model':a.model,'tp':a.tp,'batch':a.batch,'steps':steps}))
 finally:
  if engine is not None:engine.shutdown()
  else:ex.shutdown()
if __name__=='__main__':main()
