"""Run 105 FULL decode Graph workloads in production MiniVLLM workers."""
import argparse,json,os,sys,time,threading
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


def graph_stats(worker):
 import torch
 import torch.distributed as dist
 runner=worker.model_runner;manager=runner.target_cudagraph_manager
 return {'rank':dist.get_rank() if dist.is_initialized() else 0,
         'uuid':str(torch.cuda.get_device_properties(worker.device).uuid),
         'mode':str(runner.cudagraph_mode),'captured':0 if manager is None else len(manager.graphs),
         'buckets':[] if manager is None else manager.captured_token_counts(),
         'selections':[] if manager is None else list(manager.selections),
         'replays':0 if manager is None else manager.num_replays}


def enable_stack_dump(worker):
 import faulthandler,signal
 faulthandler.register(signal.SIGUSR2,all_threads=True)
 return worker.rank


def main():
 p=argparse.ArgumentParser();p.add_argument('--model',choices=['8b','14b'],required=True)
 p.add_argument('--tp',type=int,choices=[1,2,4],required=True);p.add_argument('--batch',type=int,choices=[1,4],required=True)
 p.add_argument('--output',type=Path,required=True);p.add_argument('--shutdown-diagnostics',action='store_true');a=p.parse_args()
 if a.model=='14b' and a.tp==1:p.error('14B TP1 exceeds card memory')
 a.output.mkdir(parents=True,exist_ok=True)
 (a.output/'summary.json').unlink(missing_ok=True)
 path,_=resolve_snapshot(a.model);hf=json.loads((path/'config.json').read_text())
 cfg=VllmConfig(model_config=ModelConfig(str(path),dtype='bfloat16',max_model_len=2048,hf_config=hf,seed=0),
     cache_config=CacheConfig(block_size=16,num_gpu_blocks=192,enable_prefix_caching=False),
     scheduler_config=SchedulerConfig(max_num_seqs=a.batch,max_num_batched_tokens=128),
     device_config=DeviceConfig(device='cuda'),parallel_config=ParallelConfig(tensor_parallel_size=a.tp),
     compilation_config=CompilationConfig(cudagraph_mode='full_decode_only',cudagraph_capture_sizes=[1,2,4]),attention_backend='triton')
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
  graph=rpc(graph_stats)
  if a.shutdown_diagnostics and a.tp>1:rpc(enable_stack_dump)
  ordered=[{'request_id':str(i),'tokens':outputs[str(i)],'finish_step':finished[str(i)]} for i in range(a.batch)]
  assert all(len(o['tokens'])==32 for o in ordered)
  result={'status':'passed','model':a.model,'tp':a.tp,'batch':a.batch,'prompts':prompt_ids,
          'initial_ranks':initial,'graph':graph,'runs':[{'outputs':ordered,'ranks':ranks}],
          'scheduler_steps':steps,'elapsed_seconds':time.monotonic()-start}
 finally:
  # Diagnostics only: sample Worker kernel wait states during bounded shutdown.
  # These are OS process states, not synchronization inserted into production.
  stop=threading.Event();samples=[];monitor_start=time.monotonic();signalled=set()
  workers=getattr(ex,'workers',[])
  def monitor():
   while not stop.wait(.1):
    row={'time':time.monotonic(),'workers':[]}
    for worker in workers:
     pid=worker.process.pid;proc=Path(f'/proc/{pid}')
     try:
      status=(proc/'status').read_text().splitlines()
      state=next(x for x in status if x.startswith('State:'))
      wait=(proc/'wchan').read_text().strip()
      row['workers'].append({'rank':worker.rank,'pid':pid,'state':state,'wchan':wait})
      if a.shutdown_diagnostics and time.monotonic()-monitor_start>2 and pid not in signalled:
       import signal
       os.kill(pid,signal.SIGUSR2);signalled.add(pid)
     except (OSError,StopIteration):
      row['workers'].append({'rank':worker.rank,'pid':pid,'state':'exited'})
    samples.append(row)
  thread=threading.Thread(target=monitor,daemon=True);thread.start()
  try:
   if engine is not None:engine.shutdown()
   else:ex.shutdown()
  finally:
   stop.set();thread.join(timeout=2)
   (a.output/'shutdown_states.json').write_text(json.dumps(samples,indent=2)+'\n')
 (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'status':'passed','model':a.model,'tp':a.tp,'batch':a.batch,'steps':steps}))
if __name__=='__main__':main()
