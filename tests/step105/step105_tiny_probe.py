"""Real multiprocess tiny Qwen3 execution for 105 Graph bring-up."""
import argparse,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests/step101'))
os.environ['VLLM_USE_V2_MODEL_RUNNER']='1'
from minivllm import CacheConfig,DeviceConfig,LLMEngine,ModelConfig,SamplingParams,SchedulerConfig,VllmConfig
from minivllm.config import CompilationConfig,ParallelConfig
from minivllm.executor.multiproc_executor import MultiprocExecutor
from minivllm.executor.uniproc_executor import UniProcExecutor
from step101_probe import tiny
from step105_hooks import install,flush


def graph_stats(worker):
 import torch
 import torch.distributed as dist
 r=worker.model_runner;m=r.target_cudagraph_manager
 return {'rank':dist.get_rank() if dist.is_initialized() else 0,'uuid':str(torch.cuda.get_device_properties(worker.device).uuid),
         'mode':str(r.cudagraph_mode),'capture_stats':r.cudagraph_capture_stats,
         'selections':[] if m is None else list(m.selections),'replays':0 if m is None else m.num_replays,
         'graph_count':0 if m is None else len(m.graphs),'buckets':[] if m is None else m.captured_token_counts(),
         'addresses':{'input_ids':r.input_buffers.input_ids.data_ptr(),'positions':r.input_buffers.positions.data_ptr(),
          'seq_lens':r.input_buffers.seq_lens.data_ptr(),'slot_mapping':r.block_tables.slot_mappings.data_ptr(),
          'block_tables':[t.data_ptr() for t in r.block_tables.input_block_tables]}}


def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['eager','graph'],required=True);p.add_argument('--tp',type=int,choices=[1,2,4],required=True);p.add_argument('--batch',type=int,default=4);p.add_argument('--output',type=Path,required=True);p.add_argument('--poison-padding',action='store_true');p.add_argument('--dynamic',action='store_true');p.add_argument('--random',action='store_true');a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=True);path=a.output/'tiny';hf=tiny(path,32)
 cfg=VllmConfig(model_config=ModelConfig(str(path),dtype='float32',max_model_len=64,hf_config=hf,seed=0),
  cache_config=CacheConfig(block_size=16,num_gpu_blocks=64),scheduler_config=SchedulerConfig(max_num_seqs=4,max_num_batched_tokens=64),device_config=DeviceConfig(device='cuda'),parallel_config=ParallelConfig(tensor_parallel_size=a.tp),
  compilation_config=CompilationConfig(cudagraph_mode='full_decode_only' if a.mode=='graph' else 'none',cudagraph_capture_sizes=[1,2,4] if a.mode=='graph' else None),attention_backend='triton')
 ex=MultiprocExecutor(cfg,_startup_timeout=180) if a.tp>1 else UniProcExecutor(cfg);engine=None
 def rpc(fn,*args):return ex.collective_rpc(fn,args=args) if a.tp>1 else [fn(ex.driver_worker,*args)]
 try:
  engine=LLMEngine(cfg,ex,tokenizer=None);before=rpc(graph_stats);rpc(install,a.poison_padding)
  initial=3 if a.dynamic else a.batch
  sampling=dict(temperature=0.8,top_k=5,top_p=0.95,min_p=0.05,seed=123) if a.random else dict(temperature=0)
  for i in range(initial):engine.add_request(str(i),list(range(1,16+i if a.dynamic else 9+i)),SamplingParams(max_tokens=20 if a.dynamic else 6,ignore_eos=True,**sampling))
  outputs={};step=0
  while engine.has_unfinished_requests():
   if a.dynamic and step==3:engine.add_request('3',list(range(1,19)),SamplingParams(temperature=0,max_tokens=16,ignore_eos=True))
   if a.dynamic and step==6:engine.abort_request(['1'])
   if a.dynamic and step==9:
    engine.abort_request(['0'])
    engine.add_request('4',list(range(1,20)),SamplingParams(temperature=0,max_tokens=12,ignore_eos=True))
   for out in engine.step():outputs[out.request_id]=out.token_ids
   step+=1
   if step>50:raise RuntimeError('too many steps')
  ranks=rpc(graph_stats);traces=rpc(flush,str(a.output.resolve()))
  result={'status':'passed','mode':a.mode,'tp':a.tp,'batch':a.batch,'outputs':outputs,'steps':step,'before':before,'ranks':ranks,'traces':traces,'dynamic':a.dynamic,'poison_padding':a.poison_padding,'random':a.random}
  (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
 finally:
  if engine is not None:engine.shutdown()
  else:ex.shutdown()
if __name__=='__main__':main()
