"""Inject rank failure into actual tiny model Graph capture."""
import argparse,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests/step101'))
os.environ['VLLM_USE_V2_MODEL_RUNNER']='1'
from minivllm import CacheConfig,DeviceConfig,LLMEngine,ModelConfig,SchedulerConfig,VllmConfig
from minivllm.config import CompilationConfig,ParallelConfig
from minivllm.executor.multiproc_executor import MultiprocExecutor
from minivllm.worker.worker import Worker
from step101_probe import tiny


class CaptureFailureWorker(Worker):
 def load_model(self):
  super().load_model()
  if self.rank==1:
   import torch
   def fail(module,args):
    if torch.cuda.is_current_stream_capturing():raise RuntimeError('step105 injected actual model capture failure')
   self.model_runner.model.register_forward_pre_hook(fail)


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 path=(a.output/'tiny').resolve();hf=tiny(path,32)
 cfg=VllmConfig(model_config=ModelConfig(str(path),dtype='float32',max_model_len=64,hf_config=hf,seed=0),cache_config=CacheConfig(block_size=16,num_gpu_blocks=32),scheduler_config=SchedulerConfig(max_num_seqs=4,max_num_batched_tokens=64),device_config=DeviceConfig(device='cuda'),parallel_config=ParallelConfig(tensor_parallel_size=2),compilation_config=CompilationConfig(cudagraph_mode='full_decode_only',cudagraph_capture_sizes=[1,2,4]),attention_backend='triton')
 executor=MultiprocExecutor(cfg,_worker_cls=CaptureFailureWorker,_startup_timeout=180);start=time.monotonic()
 try:
  try:LLMEngine(cfg,executor)
  except RuntimeError as exc:
   assert 'rank 1' in str(exc) and 'step105 injected actual model capture failure' in str(exc),str(exc)
  else:raise AssertionError('capture failure not propagated')
  assert executor.failure is not None and executor._closed
  assert all(not worker.process.is_alive() for worker in executor.workers)
  result={'status':'passed','elapsed_seconds':round(time.monotonic()-start,3),'workers_reaped':True}
  (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
 finally:executor.shutdown()
if __name__=='__main__':main()
