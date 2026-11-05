"""Step 089 NCCL probe: parent supervises torchrun; workers measure on GPUs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist

OPS = ('all_reduce', 'all_gather', 'reduce_scatter', 'broadcast', 'send_recv')
SIZES = (4096, 65536, 1048576, 16777216)
GROUPS = ('0,1', '2,3', '0,2', '0,1,2,3')


def positive(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return value


def run_parent(args):
    args.output.mkdir(parents=True, exist_ok=True)
    summary = []
    for group in args.groups:
        ids = group.split(',')
        if len(ids) < 2 or len(set(ids)) != len(ids) or any(not x.isdigit() for x in ids):
            raise ValueError(f'invalid GPU group: {group}')
        tag = '-'.join(ids)
        target = args.output / tag
        target.mkdir(exist_ok=True)
        for rank in range(len(ids)):
            (target / f'rank{rank}.json').unlink(missing_ok=True)
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=group, NCCL_DEBUG='INFO', NCCL_DEBUG_SUBSYS='INIT,GRAPH,NET',
                   TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
        cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
               '--nproc-per-node', str(len(ids)), str(Path(__file__).resolve()), '--worker',
               '--output', str(target.resolve()), '--timeout', str(args.timeout),
               '--warmup', str(args.warmup), '--iterations', str(args.iterations),
               '--repeats', str(args.repeats), '--sizes', *map(str, args.sizes),
               '--ops', *args.ops]
        start = time.monotonic()
        with (target / 'torchrun.log').open('w') as log:
            try:
                proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True)
                try:
                    code = proc.wait(timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    import signal
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                    code = 124
            except OSError as exc:
                log.write(str(exc))
                code = 1
        ranks = []
        for rank in range(len(ids)):
            path = target / f'rank{rank}.json'
            ranks.append(json.loads(path.read_text()) if path.exists() else None)
        if code == 0 and any(r is None or not r.get('complete') for r in ranks):
            code = 1
        summary.append({'physical_devices': ids, 'exit_code': code, 'seconds': time.monotonic()-start,
                        'ranks': ranks, 'log': str(target / 'torchrun.log')})
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
        if code:
            details = {f'rank{i}': (r.get('last') if r else 'not started or no status')
                       for i, r in enumerate(ranks)}
            print(f'group {group} failed ({code}); last rank status: {details}', file=sys.stderr)
            return code
    return 0


def run_worker(args):
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    path = args.output / f'rank{rank}.json'
    status = {'rank': rank, 'local_rank': local_rank, 'device': torch.cuda.get_device_name(device),
              'uuid': str(getattr(torch.cuda.get_device_properties(device), 'uuid', 'unavailable')),
              'pid': os.getpid(), 'complete': False, 'last': 'initializing', 'measurements': []}
    def save():
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(status, indent=2))
        tmp.replace(path)
    save()
    dist.init_process_group('nccl', timeout=timedelta(seconds=args.timeout), device_id=device)
    try:
        status['nccl'] = torch.cuda.nccl.version()
        for size in args.sizes:
            if size % (4 * world):
                raise ValueError('sizes must be divisible by 4 * world size')
            n = size // 4
            for op in args.ops:
                status['last'] = f'{op} bytes={size} correctness'
                save()
                # size means input bytes per rank; gather output and scatter output differ.
                src = torch.full((n,), rank + 1, device=device, dtype=torch.float32)
                out = torch.empty(n * world if op == 'all_gather' else n // world if op == 'reduce_scatter' else n, device=device)
                def call():
                    if op == 'all_reduce': dist.all_reduce(src)
                    elif op == 'all_gather': dist.all_gather_into_tensor(out, src)
                    elif op == 'reduce_scatter': dist.reduce_scatter_tensor(out, src)
                    elif op == 'broadcast': dist.broadcast(src, 0)
                    else:
                        works = dist.batch_isend_irecv([dist.P2POp(dist.isend, src, (rank+1)%world),
                                                       dist.P2POp(dist.irecv, out, (rank-1)%world)])
                        for work in works: work.wait()
                call()
                torch.cuda.synchronize(device)
                if op == 'all_reduce': expected = torch.full_like(src, world*(world+1)//2); actual = src
                elif op == 'all_gather': expected = torch.arange(1, world+1, device=device, dtype=src.dtype).repeat_interleave(n); actual = out
                elif op == 'reduce_scatter': expected = torch.full_like(out, world*(world+1)//2); actual = out
                elif op == 'broadcast': expected = torch.ones_like(src); actual = src
                else: expected = torch.full_like(out, (rank-1)%world+1); actual = out
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                # Zeros keep repeated in-place all-reduces finite; correctness was checked separately.
                src.zero_()
                torch.cuda.synchronize(device)
                status['last'] = f'{op} bytes={size} timing'
                save()
                for _ in range(args.warmup): call()
                torch.cuda.synchronize(device)
                for rep in range(args.repeats):
                    walls, events = [], []
                    for _ in range(args.iterations):
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        torch.cuda.synchronize(device)
                        t0 = time.perf_counter_ns()
                        start.record(); call(); end.record(); end.synchronize()
                        walls.append((time.perf_counter_ns()-t0)/1e6)
                        events.append(start.elapsed_time(end))
                    wall = statistics.mean(walls); event = statistics.mean(events)
                    # Logical result/input payload, not physical link traffic or bus bandwidth.
                    effective_bytes = size * world if op == 'all_gather' else size
                    status['measurements'].append({'op': op, 'input_bytes_per_rank': size, 'repeat': rep,
                        'wall_ms_mean': wall, 'event_ms_mean': event,
                        'effective_bytes': effective_bytes,
                        'effective_gbps_event': effective_bytes / (event * 1e6) if event else None,
                        'correct': True})
                save()
        status['complete'] = True
        status['last'] = 'complete'
        save()
    finally:
        dist.destroy_process_group()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--output', type=Path, required=True, help='output directory')
    p.add_argument('--groups', nargs='+', default=GROUPS)
    p.add_argument('--timeout', type=positive, default=1800, help='seconds per GPU group, including startup')
    p.add_argument('--warmup', type=positive, default=20)
    p.add_argument('--iterations', type=positive, default=100)
    p.add_argument('--repeats', type=positive, default=3)
    p.add_argument('--sizes', nargs='+', type=positive, default=SIZES, help='input bytes per rank')
    p.add_argument('--ops', nargs='+', choices=OPS, default=OPS)
    args = p.parse_args(argv)
    if args.worker: run_worker(args); return 0
    return run_parent(args)

if __name__ == '__main__':
    sys.exit(main())
