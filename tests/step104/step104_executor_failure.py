"""Inject a real CUDA capture failure through MultiprocExecutor's RPC."""
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'tests/step094'))
from step094_rpc_scenarios import config
from minivllm.executor import MultiprocExecutor
from minivllm.testing.step094_probe import RPCProbeWorker


class CaptureFailureWorker(RPCProbeWorker):
    def capture_failure(self):
        import torch
        from minivllm.distributed import get_tp_group
        group=get_tp_group();x=torch.ones(4,device=group.device)
        with group.graph_capture() as context:
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=context.stream):
                if self.rank==1:
                    raise RuntimeError('step104 injected executor capture failure')
                group.all_reduce(x)


def main():
    executor=MultiprocExecutor(config(tp=2,pp=1),_worker_cls=CaptureFailureWorker,_startup_timeout=60)
    start=time.monotonic()
    try:
        future=executor.collective_rpc('capture_failure',non_block=True)
        try:future.result()
        except RuntimeError as exc:
            assert 'rank 1' in str(exc) and 'step104 injected executor capture failure' in str(exc),str(exc)
        else:raise AssertionError('capture failure not propagated')
        assert executor.failure is not None
        try:executor.collective_rpc('rank_identity')
        except RuntimeError:pass
        else:raise AssertionError('failed executor accepted another RPC')
    finally:
        executor.shutdown()
    assert all(not worker.process.is_alive() for worker in executor.workers)
    print(json.dumps({'status':'passed','elapsed_seconds':round(time.monotonic()-start,3),'workers_reaped':True}))

if __name__=='__main__':main()
