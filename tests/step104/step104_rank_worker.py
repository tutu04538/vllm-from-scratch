"""104: real multi-process capture/replay with rank-local evidence."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import time
import traceback

import torch
import torch.distributed as dist
from minivllm.distributed import (init_distributed_environment, initialize_model_parallel,
    get_tp_group, destroy_model_parallel, destroy_distributed_environment)


def cycle(rank, world, base, number, failure):
    init_distributed_environment(world, rank, (base/f'store{number}').resolve().as_uri(), rank,
                                 timeout=timedelta(seconds=30))
    records=[]
    try:
        initialize_model_parallel(world, 1, timeout=timedelta(seconds=30))
        group=get_tp_group()
        communicator=group.device_communicator
        assert communicator.backend=='pynccl'
        for graph_index in range(2):
            x=torch.full((4,), float(rank+1), device=group.device)
            warm=group.all_reduce(x)
            assert torch.equal(warm,torch.full_like(warm,world*(world+1)/2))
            with group.graph_capture() as context:
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=context.stream):
                    if failure and rank==1:
                        raise RuntimeError('step104 injected rank failure during graph capture')
                    reduced=group.all_reduce(x)
                    gathered=group.all_gather(x,dim=0)
            torch.cuda.current_stream(group.device).wait_stream(context.stream)
            torch.cuda.synchronize(group.device)
            addresses=[x.data_ptr(),reduced.data_ptr(),gathered.data_ptr()]
            for i in range(100):
                value=rank+1+(i%3)*3
                x.fill_(value)
                graph.replay()
                torch.cuda.synchronize(group.device)
                expected=sum(r+1+(i%3)*3 for r in range(world))
                assert torch.equal(reduced,torch.full_like(reduced,expected)), (i,reduced,expected)
                assert torch.equal(gathered,torch.tensor([r+1+(i%3)*3 for r in range(world) for _ in range(4)],device=group.device,dtype=x.dtype))
                assert addresses==[x.data_ptr(),reduced.data_ptr(),gathered.data_ptr()]
            records.append({'graph':graph_index,'replays':100,'addresses':addresses})
            del graph,reduced,gathered,x,warm
            torch.cuda.synchronize(group.device)
        # A real upstream communicator provides an independent same-input oracle.
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator as Ref
        reference=Ref(group.cpu_group,group.device)
        data=torch.tensor([rank+1.],device=group.device)
        ref=reference.all_reduce(data)
        got=group.all_reduce(data)
        torch.cuda.synchronize(group.device)
        assert torch.equal(got,ref)
        del ref,got,data
        reference.destroy()
        props=torch.cuda.get_device_properties(rank)
        return {'rank':rank,'uuid':str(props.uuid),'backend':communicator.backend,
                'custom_reason':communicator.custom_all_reduce_reason,
                'peer_access':{str(r):torch.cuda.can_device_access_peer(rank,r) for r in range(world) if r!=rank},
                'graphs':records,'oracle':True}
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()
        (base/f'cycle{number}_rank{rank}.done').touch()
        if not failure:
            deadline=time.monotonic()+30
            while not all((base/f'cycle{number}_rank{r}.done').exists() for r in range(world)):
                if time.monotonic()>deadline:raise TimeoutError('rank cleanup rendezvous')
                time.sleep(.02)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--failure',action='store_true');a=p.parse_args()
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']); result={'rank':rank,'complete':False,'cycles':[]}
    try:
        for i in range(2):result['cycles'].append(cycle(rank,world,a.out,i,a.failure))
        result['complete']=True
    except BaseException:
        result['error']=traceback.format_exc()
        raise
    finally:
        (a.out/f'rank{rank}.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
