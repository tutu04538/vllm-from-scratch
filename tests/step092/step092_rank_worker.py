"""torchrun rank 进程；为 092 真实 GPU 验收写逐 rank 证据。"""
import argparse
from datetime import timedelta
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
import torch.distributed as dist


def mark(stage):
    print(f"rank={os.environ.get('RANK')} stage={stage}", flush=True)


def run_one(tp, pp, timeout, init_method, borrowed=False, oracle=False):
    from minivllm.distributed import (init_distributed_environment, initialize_model_parallel,
        get_tp_group, get_pp_group, destroy_model_parallel, destroy_distributed_environment)
    rank, local, world = (int(os.environ[x]) for x in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'))
    torch.cuda.set_device(local)
    if borrowed:
        dist.init_process_group(backend='nccl', init_method=init_method, world_size=world, rank=rank,
                                timeout=timedelta(seconds=timeout))
    init_distributed_environment(world, rank, distributed_init_method=init_method,
                                 local_rank=local, timeout=timedelta(seconds=timeout))
    mark(f'init borrowed={borrowed}')
    try:
        initialize_model_parallel(tp, pp, timeout=timedelta(seconds=timeout))
        mark('groups')
        tg, pg = get_tp_group(), get_pp_group()
        assert tg.ranks == list(range((rank // tp) * tp, (rank // tp + 1) * tp))
        assert pg.ranks == [rank % tp + stage * tp for stage in range(pp)]
        data = torch.tensor([rank + 1.], device='cuda')
        reduced = tg.all_reduce(data)
        assert reduced.item() == sum(i + 1 for i in tg.ranks)
        pp_reduced = pg.all_reduce(data)
        assert pp_reduced.item() == sum(i + 1 for i in pg.ranks)
        assert data.item() == rank + 1
        assert (reduced is data) == (tp == 1)
        if oracle and tp > 1:
            base = ROOT / 'learning_notes/14_vllm_from_scratch/分布式并行完整需求/参考资料/vllm_0_28_0'
            upstream_root = Path(importlib.util.find_spec('vllm').origin).parent
            for rel in ('distributed/device_communicators/cuda_communicator.py',
                        'distributed/device_communicators/base_device_communicator.py'):
                assert hashlib.sha256((base / rel).read_bytes()).digest() == hashlib.sha256((upstream_root / rel).read_bytes()).digest()
            from types import SimpleNamespace
            from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
            from vllm.distributed.device_communicators.base_device_communicator import DeviceCommunicatorBase
            ref = SimpleNamespace(pynccl_comm=None, qr_comm=None, fi_ar_comm=None,
                                  aiter_ar_comm=None, ca_comm=None, symm_mem_comm=None,
                                  device_group=tg.device_group, world_size=tp)
            assert torch.equal(CudaCommunicator.all_reduce(ref, data), reduced)
        tensor = (torch.arange(6, device='cuda').reshape(2, 3) + rank * 10).t()
        assert not tensor.is_contiguous()
        for dim in (0, -1):
            gathered = tg.all_gather(tensor, dim)
            expected = torch.cat([(torch.arange(6, device='cuda').reshape(2, 3) + r * 10).t() for r in tg.ranks], dim=dim)
            assert torch.equal(gathered, expected), (dim, gathered, expected)
            assert (gathered is tensor) == (tp == 1)
            if oracle and tp > 1:
                assert torch.equal(DeviceCommunicatorBase.all_gather(ref, tensor.contiguous(), dim), gathered)
        for dim in (0, -1):
            shape = (tp * 2, tp * 3)
            val = torch.arange(shape[0] * shape[1], device='cuda').reshape(shape) + rank * 100
            output = tg.reduce_scatter(val, dim)
            reference = sum(torch.arange(shape[0] * shape[1], device='cuda').reshape(shape) + r * 100 for r in tg.ranks)
            assert torch.equal(output, reference.chunk(tp, dim=dim)[tg.rank_in_group])
            assert (output is val) == (tp == 1)
            if oracle and tp > 1:
                assert torch.equal(DeviceCommunicatorBase.reduce_scatter(ref, val, dim), output)
        if oracle and tp == 2:
            # 冻结实现对 3D dim=2 使用 movedim(0, 2)，与按 dim 切分的语义不一致。
            val3 = torch.arange(48, device='cuda').reshape(2, 4, 6) + rank * 100
            got3 = tg.reduce_scatter(val3, 2)
            ref3 = DeviceCommunicatorBase.reduce_scatter(ref, val3, 2)
            expected3 = sum(torch.arange(48, device='cuda').reshape(2, 4, 6) + r * 100
                            for r in tg.ranks).chunk(tp, dim=2)[tg.rank_in_group]
            assert torch.equal(got3, ref3)
            assert tuple(got3.shape) == (6, 2, 2) and tuple(expected3.shape) == (2, 4, 3)
        b = torch.tensor([rank], device='cuda')
        assert tg.broadcast(b, src=0) is b
        assert b.item() == tg.ranks[0]
        obj = {'group': tg.ranks} if tg.rank_in_group == 0 else None
        assert tg.broadcast_object(obj, src=0) == {'group': tg.ranks}
        mark('before-send')
        if tp > 1:
            if tg.rank_in_group % 2 == 0:
                tg.send(torch.tensor([rank + 77], device='cuda'), dst=tg.rank_in_group + 1)
            else:
                src = tg.rank_in_group - 1
                received = tg.recv((1,), torch.int64, src=src)
                assert received.item() == tg.ranks[src] + 77
            mark('after-send')
        else:
            try:
                tg.send(data)
            except ValueError:
                pass
            else:
                raise AssertionError('单 rank send 必须拒绝')
        props = torch.cuda.get_device_properties(local)
        record = {'rank': rank, 'local_rank': local, 'uuid': str(props.uuid), 'device_name': props.name,
                  'tp_ranks': tg.ranks, 'tp_rank': tg.rank_in_group,
                  'pp_ranks': pg.ranks, 'pp_rank': pg.rank_in_group,
                  'all_reduce': reduced.tolist(), 'pp_all_reduce': pp_reduced.tolist(), 'noncontiguous_all_gather': True,
                  'reduce_scatter_dims': [0, -1], 'broadcast': b.tolist(), 'send_recv': tp > 1,
                  'borrowed_default': borrowed, 'frozen_oracle': oracle and tp > 1}
        dist.barrier(group=tg.cpu_group)
        mark('before-cleanup')
        return record
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()
        mark('after-cleanup')
        if borrowed:
            assert dist.is_initialized()
            dist.barrier()
            dist.destroy_process_group()
        assert not dist.is_initialized()


def main():
    p=argparse.ArgumentParser(); p.add_argument('--tp', type=int, required=True); p.add_argument('--pp', type=int, required=True)
    p.add_argument('--timeout', type=int, default=30); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--failure-rank', type=int, default=-1); p.add_argument('--oracle', action='store_true'); args=p.parse_args()
    rank=int(os.environ['RANK']); path=args.out / f'rank{rank}.json'; path.parent.mkdir(parents=True, exist_ok=True)
    result={'rank':rank, 'complete':False, 'cycles':[]}
    try:
        if args.failure_rank >= 0:
            from minivllm.distributed import (GroupCoordinator, init_distributed_environment,
                                               destroy_distributed_environment)
            local=int(os.environ['LOCAL_RANK']); world=int(os.environ['WORLD_SIZE'])
            init_method=(args.out / 'failure.store').resolve().as_uri()
            init_distributed_environment(world, rank, init_method, local, timeout=timedelta(seconds=args.timeout))
            partial = None
            try:
                partial=GroupCoordinator([list(range(world))], local, timedelta(seconds=args.timeout))
                mark('partial-group-created')
                if rank == args.failure_rank:
                    raise RuntimeError('step092 injected rank failure after partial group creation')
                GroupCoordinator([[r] for r in range(world)], local, timedelta(seconds=args.timeout))
                raise RuntimeError('peer should have failed during partial initialization')
            finally:
                if partial is not None:
                    partial.destroy()
                destroy_distributed_environment()
        for borrowed in (False, True):
            init_method=(args.out / f'rendezvous{int(borrowed)}.store').resolve().as_uri()
            result['cycles'].append(run_one(args.tp, args.pp, args.timeout, init_method,
                                            borrowed=borrowed, oracle=args.oracle))
            # PyTorch 文档要求默认进程组销毁后、重新初始化之前做带外同步。
            marker=args.out / f'cycle{int(borrowed)}-rank{rank}.done'
            marker.touch()
            deadline=time.monotonic()+args.timeout
            while not all((args.out / f'cycle{int(borrowed)}-rank{r}.done').exists()
                          for r in range(int(os.environ['WORLD_SIZE']))):
                if time.monotonic() > deadline:
                    raise TimeoutError('默认组销毁后的带外同步超时')
                time.sleep(0.02)
        result['complete']=True
    except BaseException:
        result['error']=traceback.format_exc()
        raise
    finally:
        path.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__': main()
