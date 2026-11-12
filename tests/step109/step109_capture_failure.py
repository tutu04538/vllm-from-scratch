"""Inject a rank failure during actual EAGLE3 draft CUDA Graph capture."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ['VLLM_USE_V2_MODEL_RUNNER']='1'
from minivllm import CacheConfig,DeviceConfig,LLMEngine,ModelConfig,SchedulerConfig,SpeculativeConfig,VllmConfig
from minivllm.config import CompilationConfig,ParallelConfig
from minivllm.executor.multiproc_executor import MultiprocExecutor
from minivllm.testing.tiny_models import tiny_eagle3_dir,tiny_qwen3_dir,tiny_qwen3_config
from minivllm.worker.worker import Worker


class Step109CaptureFailureWorker(Worker):
    def load_model(self):
        super().load_model()
        if self.rank == 1:
            import torch
            def fail(module,args):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError('step109 injected draft graph capture failure')
            self.model_runner.speculator.model.register_forward_pre_hook(fail)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    draft=Path(tiny_eagle3_dir('tiny_gqa'))
    cfg=VllmConfig(model_config=ModelConfig(tiny_qwen3_dir('tiny_gqa'),dtype='float32',max_model_len=64,hf_config=tiny_qwen3_config('tiny_gqa')),
        cache_config=CacheConfig(block_size=16,num_gpu_blocks=32),scheduler_config=SchedulerConfig(max_num_seqs=2,max_num_batched_tokens=64),
        device_config=DeviceConfig('cuda'),parallel_config=ParallelConfig(tensor_parallel_size=2),attention_backend='triton',
        compilation_config=CompilationConfig(cudagraph_mode='full_decode_only',cudagraph_capture_sizes=[1,2,3,6]),
        speculative_config=SpeculativeConfig(method='eagle3',num_speculative_tokens=2,
            draft_model_config=ModelConfig(str(draft),dtype='float32',max_model_len=64,hf_config=json.loads((draft/'config.json').read_text()))))
    ex=MultiprocExecutor(cfg,_worker_cls=Step109CaptureFailureWorker,_startup_timeout=180);start=time.monotonic()
    try:
        try: LLMEngine(cfg,ex,tokenizer=None)
        except RuntimeError as exc:
            assert 'rank 1' in str(exc) and 'step109 injected draft graph capture failure' in str(exc),str(exc)
        else: raise AssertionError('draft capture failure did not propagate')
        assert ex.failure is not None and ex._closed
        assert all(not worker.process.is_alive() for worker in ex.workers)
        a.output.write_text(json.dumps({'status':'passed','elapsed_seconds':time.monotonic()-start,'workers_reaped':True})+'\n')
    finally: ex.shutdown()

if __name__=='__main__':main()
