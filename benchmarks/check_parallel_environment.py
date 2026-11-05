"""开发环境验收，不是 TP 实现或性能 benchmark。

python benchmarks/check_parallel_environment.py --single
 torchrun --standalone --nproc-per-node=4 benchmarks/check_parallel_environment.py
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import importlib.metadata
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
import triton
import triton.language as tl


@triton.jit
def add_kernel(x, y, out, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out + i, tl.load(x + i, i < N, 0) + tl.load(y + i, i < N, 0), i < N)


def device_checks(device: int) -> dict:
    torch.cuda.set_device(device)
    x = torch.arange(4096, dtype=torch.float32, device=f"cuda:{device}")
    y = torch.ones_like(x)
    out = torch.empty_like(x)
    add_kernel[(16,)](x, y, out, x.numel(), 256)
    torch.testing.assert_close(out, x + y, rtol=0, atol=0)
    a = torch.eye(64, dtype=torch.bfloat16, device=x.device)
    torch.testing.assert_close(a @ a, a, rtol=0, atol=0)
    # Warmup on a side stream before capture; input/output addresses stay stable.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        add_kernel[(16,)](x, y, out, x.numel(), 256)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        add_kernel[(16,)](x, y, out, x.numel(), 256)
    x.add_(3)
    graph.replay()
    torch.testing.assert_close(out, x + y, rtol=0, atol=0)
    return {"device": device, "name": torch.cuda.get_device_name(device),
            "capability": torch.cuda.get_device_capability(device),
            "triton": True, "bf16_matmul": True, "cuda_graph_replay": True}


def main():
    if "--communication" in sys.argv:
        from step089_communication import main as communication_main
        args = sys.argv[1:]
        args.remove("--communication")
        sys.exit(communication_main(args))
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--communication", action="store_true", help="run supervised step089 communication benchmark; remaining options after --")
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    assert torch.cuda.is_available(), "CUDA is required; CPU fallback is not a pass"
    if args.single:
        devices = [device_checks(i) for i in range(torch.cuda.device_count())]
        torch.cuda.set_device(0)
        from minivllm.uva_ext import get_cuda_view_from_cpu_tensor, resolve_backend
        assert resolve_backend() == "cpp", "验收默认 C++ UVA，不以 Python 兜底代替"
        host = torch.arange(32, dtype=torch.int64, pin_memory=True)
        view = get_cuda_view_from_cpu_tensor(host)
        torch.testing.assert_close(view.cpu(), host, rtol=0, atol=0)
        host.add_(7)
        torch.testing.assert_close(view.cpu(), host, rtol=0, atol=0)
        from vllm import _custom_ops
        norm_input = torch.ones((2, 64), device="cuda", dtype=torch.bfloat16)
        norm_output = torch.empty_like(norm_input)
        _custom_ops.rms_norm(norm_output, norm_input, torch.ones_like(norm_input[0]), 1e-6)
        torch.testing.assert_close(norm_output, norm_input, rtol=0, atol=0)
        result = {"devices": devices, "uva_cpp_shared_memory": True,
                  "vllm_cuda_rmsnorm": True,
                  "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
                  "versions": {n: importlib.metadata.version(n) for n in
                               ("vllm", "triton", "transformers", "arctic-inference", "pytest")},
                  "peer_access": [[torch.cuda.can_device_access_peer(i, j) if i != j else None
                                   for j in range(len(devices))] for i in range(len(devices))]}
    else:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", timeout=timedelta(seconds=90),
                                device_id=torch.device("cuda", local_rank))
        try:
            rank, world = dist.get_rank(), dist.get_world_size()
            assert world >= 2, "Use torchrun --nproc-per-node >= 2"
            row = device_checks(local_rank)
            device = torch.device("cuda", local_rank)
            total = world * (world + 1) // 2
            value = torch.full((8,), rank + 1, dtype=torch.float32, device=device)
            dist.all_reduce(value)
            assert torch.all(value == total).item()
            src = torch.full((8,), rank, dtype=torch.float32, device=device)
            gathered = torch.empty(world * 8, dtype=src.dtype, device=device)
            dist.all_gather_into_tensor(gathered, src)
            torch.testing.assert_close(gathered, torch.arange(world, device=device,
                                       dtype=src.dtype).repeat_interleave(8), rtol=0, atol=0)
            scatter_input = torch.full((world * 8,), rank + 1, dtype=src.dtype, device=device)
            dist.reduce_scatter_tensor(value, scatter_input)
            assert torch.all(value == total).item()
            value.fill_(42 if rank == 0 else -1)
            dist.broadcast(value, src=0)
            assert torch.all(value == 42).item()
            received = torch.empty_like(src)
            operations = [dist.P2POp(dist.isend, src, (rank + 1) % world),
                          dist.P2POp(dist.irecv, received, (rank - 1) % world)]
            for work in dist.batch_isend_irecv(operations):
                work.wait()
            assert torch.all(received == (rank - 1) % world).item()
            row.update(rank=rank, all_reduce=True, all_gather=True, reduce_scatter=True,
                       broadcast=True, ring_send_recv=True)
            rows = [None] * world
            dist.all_gather_object(rows, row)
            result = {"backend": dist.get_backend(), "world_size": world,
                      "nccl": torch.cuda.nccl.version(), "ranks": rows}
            dist.barrier()
        finally:
            dist.destroy_process_group()
        if rank != 0:
            return
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
