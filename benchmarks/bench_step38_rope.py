"""第三十八关 A/B：只换 RoPE 后端，其余配置完全相同。

    python benchmarks/bench_step38_rope.py --rope torch  --case decode_c1
    python benchmarks/bench_step38_rope.py --rope triton --case decode_c1

同版 step38、同一份输入、同一 KV 容量、同样开 Triton norm / tiled attention / Graph，
唯一变量是 `rope_backend`。

另有一个纯算子模式（--op），只测一次 RoPE 调用本身的耗时，分别用 prefill 和 decode 形状。
"""

import argparse
import json
import pathlib
import statistics
import sys
import time

PROJECT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from benchmarks.bench_step36_vllm_compare import (BLOCK_SIZE, CASES, KV_BLOCKS, MAX_NUM_SEQS,
                                                  MAX_NUM_BATCHED_TOKENS, MODEL_DIR, load_inputs)


def build(rope_backend, use_cuda_graph=True):
    import torch
    import step38 as m
    engine = m.Engine.from_model_dir(
        MODEL_DIR, device="cuda", dtype=torch.bfloat16, attention_backend="triton",
        norm_backend="triton", rope_backend=rope_backend, use_cuda_graph=use_cuda_graph,
        max_num_seqs=MAX_NUM_SEQS, max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        block_size=BLOCK_SIZE, num_kv_blocks=KV_BLOCKS, enable_prefix_caching=False)
    engine.model.eos_token_ids = ()
    engine.scheduler.eos_token_ids = set()
    # 断言运行时对象的真实值，不从入参推断
    assert engine.model.rotary.backend == rope_backend, (
        f"rope 后端实际是 {engine.model.rotary.backend!r}，请求的是 {rope_backend!r}")
    assert engine.model.norm_backend == "triton"
    return engine


def run(engine, prompts, gen, tag):
    import torch
    got = {}
    engine.scheduler.on_finished = lambda r: got.__setitem__(r["request_id"], len(r["output_ids"]))
    t0 = time.perf_counter()
    for i, ids in enumerate(prompts):
        engine.add_request({"request_id": f"{tag}_{i}", "prompt_ids": ids, "max_new_tokens": gen})
    while engine.has_unfinished_requests():
        engine.step()
    torch.cuda.synchronize()
    return time.perf_counter() - t0, got


def op_benchmark(head_dim=128, num_heads=16, num_rows=None, label="", inner=100, reps=30):
    """纯算子：一次 RoPE 调用真正花在 GPU kernel 上的时间。

    用 profiler 读 self_device_time_total，直接得到 kernel 执行时长：
    - 不用 Event 计时：那样会把 CPU 发射开销算进去，Torch 路径一次 12 个 kernel，
      发射开销会完全盖过 GPU 时间，测出来的差异不是算子本身的差异。
    - 不用 CUDA Graph：捕获会改变分配行为，测出的数会自相矛盾（试过，prefill K 只有
      prefill Q 的 1/18，明显不对）。
    """
    import torch
    from torch.profiler import profile, ProfilerActivity
    from step38.model import RotaryEmbedding
    from step38.rope import rope

    torch.manual_seed(0)
    rot = RotaryEmbedding(head_dim, 4096, 10000.0).to("cuda")
    x = torch.randn(num_rows, num_heads, head_dim, device="cuda").bfloat16()
    pos = torch.arange(num_rows, device="cuda")
    out = {}

    for backend in ("torch", "triton"):
        call = ((lambda: rot(x, pos)) if backend == "torch"
                else (lambda: rope(x, pos, rot.cos_table, rot.sin_table)))
        for _ in range(5):                       # 预热（含 Triton 编译）
            call()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(inner):
                call()
            torch.cuda.synchronize()
        out[backend] = sum(e.self_device_time_total for e in prof.key_averages()) / inner

    print(f"  {label:<16} x=[{num_rows:>4},{num_heads:>2},{head_dim}]  "
          f"torch={out['torch']:7.1f} µs  triton={out['triton']:6.1f} µs  "
          f"提升 {out['torch'] / out['triton']:.2f}×")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rope", choices=("torch", "triton"))
    ap.add_argument("--case", choices=tuple(CASES))
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--op", action="store_true", help="只测纯算子")
    args = ap.parse_args()

    import torch
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False

    if args.op:
        print("纯算子对照（CUDA Event，单进程交替）:")
        op_benchmark(128, 16, 2048, "prefill 形状 Q")
        op_benchmark(128, 8, 2048, "prefill 形状 K")
        op_benchmark(128, 16, 1, "decode 形状 Q")
        op_benchmark(128, 8, 1, "decode 形状 K")
        return

    spec = CASES[args.case]
    prompts = load_inputs(args.case)
    engine = build(args.rope)
    run(engine, prompts, spec["gen"], "warm0")
    run(engine, prompts, spec["gen"], "warm1")

    samples = []
    for r in range(args.reps):
        dt, got = run(engine, prompts, spec["gen"], f"m{r}")
        if not all(v == spec["gen"] for v in got.values()):
            raise RuntimeError(f"输出长度不对: {got}")
        samples.append(dt)

    med = statistics.median(samples)
    print(json.dumps(dict(
        rope=args.rope, case=args.case, spec=spec, graph=True,
        median_s=round(med, 4), samples_s=[round(s, 4) for s in samples],
        throughput_tok_s=round(spec["conc"] * spec["gen"] / med, 1),
        output_tokens=spec["conc"] * spec["gen"],
    ), ensure_ascii=False))


if __name__ == "__main__":
    main()
