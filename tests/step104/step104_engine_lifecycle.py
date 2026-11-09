"""Two sequential production engines with test-only Worker graph capture."""
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'tests/step094'))
from step094_rpc_scenarios import config
from minivllm.engine import LLMEngine
from minivllm.executor import MultiprocExecutor
from minivllm.testing.step094_probe import RPCProbeWorker


class GraphProbeWorker(RPCProbeWorker):
    def graph_probe(self):
        import torch
        from minivllm.distributed import get_tp_group
        group=get_tp_group();x=torch.tensor([float(self.rank+1)],device=group.device)
        warm=group.all_reduce(x)
        assert warm.item()==3
        with group.graph_capture() as context:
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=context.stream):
                result=group.all_reduce(x)
        torch.cuda.current_stream(group.device).wait_stream(context.stream)
        x.fill_(float(self.rank+4));graph.replay();torch.cuda.synchronize(group.device)
        assert result.item()==9
        del graph,result,x,warm
        torch.cuda.synchronize(group.device)
        return self.rank


def main():
    for cycle in range(2):
        cfg=config(tp=2,pp=1)
        executor=MultiprocExecutor(cfg,_worker_cls=GraphProbeWorker,_startup_timeout=60)
        engine=LLMEngine(cfg,executor)
        try:assert executor.collective_rpc('graph_probe')==[0,1]
        finally:engine.shutdown()
        assert all(not worker.process.is_alive() for worker in executor.workers)
    print(json.dumps({'status':'passed','sequential_engines':2,'workers_reaped':True}))

if __name__=='__main__':main()
