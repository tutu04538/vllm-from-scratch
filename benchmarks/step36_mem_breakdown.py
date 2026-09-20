"""显存口径修正：为什么不能用父进程的 mem_get_info 比较两个引擎。

本机（WSL2）上 `torch.cuda.mem_get_info()` 只反映**当前进程**所属 CUDA 上下文的
占用，看不到另一个进程里分配的显存。实测：

    父进程看到的已用显存 = 1.312 GiB
    子进程分配 512 MiB 并持有期间，父进程看到的仍然是 1.312 GiB

vLLM 0.28 默认把 GPU 工作放在 EngineCore 子进程里，所以我们之前矩阵里的
`gpu_used_after_gib` 对 vLLM 是无效的（它只是 WSL 的基线值）。

这里的做法：用 vLLM 自己的 `apply_model()` 把查询函数送进它真正执行 GPU 工作的
worker 进程，取该进程的 torch 统计；我们的引擎在同一进程内，直接读同一组统计。
这样两边回答的是同一个问题——「这次工作占了多少显存」。

    python benchmarks/step36_mem_breakdown.py
"""

import json
import pathlib
import sys

PROJECT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from benchmarks.bench_step36_vllm_compare import (CASES, MODEL_DIR, build_mine, build_vllm,
                                                  load_inputs, run_mine, run_vllm, kv_bytes)

GIB = 2 ** 30

def _query_torch_memory(model):
    """在 vLLM 的 worker 进程里执行，返回那个进程自己的 CUDA 统计。"""
    import torch
    free, total = torch.cuda.mem_get_info()
    return {
        "torch_allocated_gib": torch.cuda.memory_allocated() / 2 ** 30,
        "torch_reserved_gib": torch.cuda.memory_reserved() / 2 ** 30,
        "torch_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
        "process_sees_free_gib": free / 2 ** 30,
        "process_sees_total_gib": total / 2 ** 30,
    }


def probe_vllm():
    llm = build_vllm()
    prompts = load_inputs("short_c8")
    run_vllm(llm, prompts, CASES["short_c8"]["gen"])
    got = llm.apply_model(_query_torch_memory)
    if isinstance(got, list):
        got = got[0]
    cc = llm.llm_engine.vllm_config.cache_config
    return dict(engine="vllm", **{k: round(v, 3) for k, v in got.items()},
                kv_blocks=cc.num_gpu_blocks, block_size=cc.block_size,
                kv_cache_gib=round(cc.num_gpu_blocks * cc.block_size
                                   * 28 * 2 * 8 * 128 * 2 / GIB, 3))


def probe_mine():
    import torch
    engine = build_mine(graph=True)
    prompts = load_inputs("short_c8")
    run_mine(engine, prompts, CASES["short_c8"]["gen"], "mem")
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return dict(engine="mine",
                torch_allocated_gib=round(torch.cuda.memory_allocated() / GIB, 3),
                torch_reserved_gib=round(torch.cuda.memory_reserved() / GIB, 3),
                torch_peak_allocated_gib=round(torch.cuda.max_memory_allocated() / GIB, 3),
                process_sees_free_gib=round(free / GIB, 3),
                process_sees_total_gib=round(total / GIB, 3),
                kv_blocks=512, block_size=16,
                kv_cache_gib=round(kv_bytes() / GIB, 3))


def main():
    import torch
    torch.set_num_threads(1)
    out = []
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("both", "vllm"):
        out.append(probe_vllm())
    if which in ("both", "mine"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        out.append(probe_mine())
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
