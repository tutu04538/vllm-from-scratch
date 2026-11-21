"""Inject real AWQ loading failure or a dead rank during NCCL collective."""
import argparse
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '1'
from minivllm import CacheConfig, DeviceConfig, ModelConfig, SchedulerConfig, VllmConfig
from minivllm.config import CompilationConfig, ParallelConfig
from minivllm.executor.multiproc_executor import MultiprocExecutor
from minivllm.worker.worker import Worker

class FaultWorker(Worker):
    def load_model(self):
        if self.rank == 1 and os.environ.get('STEP125_FAULT') == 'load':
            from minivllm.model_loader.default_loader import DefaultModelLoader
            original = DefaultModelLoader.get_all_weights
            def fail_after_read(loader, model):
                for index, item in enumerate(original(loader, model)):
                    if index == 4:
                        raise RuntimeError('step125 injected AWQ checkpoint read failure')
                    yield item
            DefaultModelLoader.get_all_weights = fail_after_read
        return super().load_model()


def fail_during_collective(worker):
    import torch
    import torch.distributed as dist
    if worker.rank == 1:
        time.sleep(1)
        os._exit(72)
    value = torch.ones(1, device=worker.device)
    dist.all_reduce(value)
    return value.item()


def hang_rank_one(worker):
    if worker.rank == 1:
        time.sleep(30)
    return worker.rank


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--case', choices=('load','collective','timeout'), required=True)
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args()
    os.environ['STEP125_FAULT']=a.case
    path=Path('/data/syh/.cache/huggingface/hub/models--Qwen--Qwen3-8B-AWQ/snapshots/4da05a8edb55c6046cce958586c33b61da07bb79')
    hf=json.loads((path/'config.json').read_text())
    cfg=VllmConfig(ModelConfig(str(path),dtype='bfloat16',max_model_len=64,hf_config=hf),
        cache_config=CacheConfig(block_size=16,num_gpu_blocks=8),
        scheduler_config=SchedulerConfig(max_num_seqs=1,max_num_batched_tokens=64),
        device_config=DeviceConfig('cuda'),parallel_config=ParallelConfig(tensor_parallel_size=4),
        compilation_config=CompilationConfig(cudagraph_mode='none'),attention_backend='triton')
    start=time.monotonic()
    if a.case=='load':
        try:
            MultiprocExecutor(cfg,_worker_cls=FaultWorker,_startup_timeout=150)
        except RuntimeError as exc:
            assert 'rank 1' in str(exc) and 'step125 injected AWQ checkpoint read failure' in str(exc), str(exc)
            assert not multiprocessing.active_children(), multiprocessing.active_children()
            result={'error':str(exc),'children_reaped':True}
        else:
            raise AssertionError('injected loading failure did not propagate')
    else:
        ex=MultiprocExecutor(cfg,_startup_timeout=150)
        rendezvous=Path(ex._rendezvous.name)
        names=list(ex._response_shm_names)
        try:
            method = fail_during_collective if a.case == 'collective' else hang_rank_one
            future=ex.collective_rpc(method, timeout=2 if a.case == 'timeout' else None,
                                     non_block=True)
            started=time.monotonic()
            try:
                future.result()
            except (RuntimeError, TimeoutError) as exc:
                error=str(exc)
                assert ('rank 1' in error if a.case == 'collective' else '超时' in error), error
            else:
                raise AssertionError('dead rank did not propagate')
            elapsed=time.monotonic()-started
            assert elapsed < 15, elapsed
            assert ex.failure is not None and future.done()
            try:
                ex.collective_rpc(fail_during_collective)
            except RuntimeError:
                pass
            else:
                raise AssertionError('failed executor accepted another RPC')
            result={'error':error,'future_done':True,'failure_seconds':elapsed}
        finally:
            ex.shutdown()
            ex.shutdown()
        assert all(not worker.process.is_alive() for worker in ex.workers)
        assert not rendezvous.exists()
        assert all(not (Path('/dev/shm')/name).exists() for name in names)
        result['children_reaped']=True
        result['rendezvous_and_queues_removed']=True
    result.update(status='passed',case=a.case,total_seconds=time.monotonic()-start)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
if __name__ == '__main__': main()
