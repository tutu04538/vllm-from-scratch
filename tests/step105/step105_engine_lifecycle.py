"""Two live TP Graph engines; closing one must not affect the other."""
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests/step101'));sys.path.insert(0,str(ROOT/'tests/step105'))
os.environ['VLLM_USE_V2_MODEL_RUNNER']='1'
from minivllm import CacheConfig,DeviceConfig,LLMEngine,ModelConfig,SamplingParams,SchedulerConfig,VllmConfig
from minivllm.config import CompilationConfig,ParallelConfig
from minivllm.executor.multiproc_executor import MultiprocExecutor
from step101_probe import tiny
from step105_tiny_probe import graph_stats


def shutdown_and_audit(worker):
 import gc
 import weakref
 from minivllm.distributed import get_tp_group
 runner=worker.model_runner
 refs={'manager':weakref.ref(runner.target_cudagraph_manager),
       'graphs':[weakref.ref(graph) for graph in runner.target_cudagraph_manager.graphs.values()],
       'workspace':weakref.ref(runner.attn_metadata_builder.workspace_pool),
       'communicator':weakref.ref(get_tp_group().device_communicator)}
 del runner
 worker.shutdown()
 gc.collect()
 alive={'manager':refs['manager']() is not None,
        'graphs':sum(ref() is not None for ref in refs['graphs']),
        'workspace':refs['workspace']() is not None,
        'communicator':refs['communicator']() is not None}
 assert not any(alive.values()),alive
 return alive


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 path=(a.output/'tiny').resolve();hf=tiny(path,32);engines=[];executors=[]
 try:
  for devices in ((0,1),(2,3)):
   cfg=VllmConfig(model_config=ModelConfig(str(path),dtype='float32',max_model_len=64,hf_config=hf,seed=0),
    cache_config=CacheConfig(block_size=16,num_gpu_blocks=64),scheduler_config=SchedulerConfig(max_num_seqs=4,max_num_batched_tokens=64),
    device_config=DeviceConfig(device='cuda'),parallel_config=ParallelConfig(tensor_parallel_size=2,device_ids=devices),
    compilation_config=CompilationConfig(cudagraph_mode='full_decode_only',cudagraph_capture_sizes=[1,2,4]),attention_backend='triton')
   ex=MultiprocExecutor(cfg,_startup_timeout=180);executors.append(ex);engines.append(LLMEngine(cfg,ex,tokenizer=None))
  for engine in engines:engine.add_request('a',list(range(1,18)),SamplingParams(temperature=0,max_tokens=6,ignore_eos=True))
  bfirst=engines[1].step();assert bfirst
  afull=[]
  while engines[0].has_unfinished_requests():afull.extend(engines[0].step())
  a_output=afull[-1].token_ids
  before=executors[1].collective_rpc(graph_stats);assert all(x['graph_count']==3 for x in before)
  released=executors[0].collective_rpc(shutdown_and_audit)
  engines[0].shutdown();assert all(not w.process.is_alive() for w in executors[0].workers)
  bout=list(bfirst)
  while engines[1].has_unfinished_requests():bout.extend(engines[1].step())
  after=executors[1].collective_rpc(graph_stats);assert all(x['replays']>0 and x['graph_count']==3 for x in after)
  assert all(x['addresses']==y['addresses'] for x,y in zip(before,after))
  assert bout[-1].token_ids==a_output
  engines[1].shutdown();assert all(not w.process.is_alive() for w in executors[1].workers)
  result={'status':'passed','tokens':a_output,'alive_engine_replays':[r['replays'] for r in after],
          'before':before,'after':after,'released':released,'worker_pids_reaped':True}
  (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'passed','tokens':a_output}))
 finally:
  for engine in engines:engine.shutdown()
  for executor in executors:executor.shutdown()
if __name__=='__main__':main()
