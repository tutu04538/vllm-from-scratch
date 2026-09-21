"""第三十九关 A/B：只换投影组织，其余配置完全相同。

    python benchmarks/bench_step39_proj.py --step step38 --case decode_c1
    python benchmarks/bench_step39_proj.py --step step39 --case decode_c1

step38 是每层 7 次投影，step39 是 4 次（QKV 一次、gate/up 一次）。
同一份输入、同一 KV 容量、同开 Triton norm/rope/tiled attention/Graph，
唯一变量就是投影组织。

    --kernels   另测一步 decode 的 kernel 总数与投影 GEMM 数（对照 launch 数）
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


def build(step, use_cuda_graph=True, rope_backend="triton"):
    import torch
    mod = __import__(step)
    engine = mod.Engine.from_model_dir(
        MODEL_DIR, device="cuda", dtype=torch.bfloat16, attention_backend="triton",
        norm_backend="triton", rope_backend=rope_backend, use_cuda_graph=use_cuda_graph,
        max_num_seqs=MAX_NUM_SEQS, max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        block_size=BLOCK_SIZE, num_kv_blocks=KV_BLOCKS, enable_prefix_caching=False)
    engine.model.eos_token_ids = ()
    engine.scheduler.eos_token_ids = set()
    # 断言运行时对象的真实值，不从入参推断
    assert engine.model.norm_backend == "triton" and engine.model.rotary.backend == rope_backend
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


def kernel_count(step):
    """一步 decode 的 kernel 总数与投影 GEMM 数（eager，按 kernel 名分类）。"""
    import torch
    from torch.profiler import profile, ProfilerActivity
    from benchmarks.profile_step36 import classify
    prompts = [load_inputs("decode_c1")[0]]
    e = build(step, use_cuda_graph=False)
    for i in range(4):                       # 前几步含 prefill，先热身
        e.add_request({"request_id": f"w{i}", "prompt_ids": prompts[0], "max_new_tokens": 8})
        for _ in range(4):
            if e.has_unfinished_requests():
                e.step()
        e.scheduler.running.clear(); e.scheduler.waiting.clear()
    e.add_request({"request_id": "m", "prompt_ids": prompts[0], "max_new_tokens": 8})
    for _ in range(3):
        e.step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        e.step()
        torch.cuda.synchronize()
    ev = [x for x in prof.events() if str(getattr(x, "device_type", "")) == "DeviceType.CUDA"]
    gemm = [x for x in ev if classify(x.name) == "GEMM/GEMV"]
    return len(ev), len(gemm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=("step38", "step39", "step40"))
    ap.add_argument("--case", choices=tuple(CASES))
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--kernels", action="store_true")
    ap.add_argument("--paired", action="store_true", help="同进程轮转配对，消掉顺序漂移")
    args = ap.parse_args()

    import torch
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False

    if args.kernels:
        for step in ("step38", "step39"):
            tot, gemm = kernel_count(step)
            print(f"  {step}: 一步 decode 共 {tot} 个 kernel，其中投影 GEMM/GEMV {gemm} 个"
                  f"（28 层 × {gemm // 28}）")
        return

    if args.paired:
        # 同一个进程里轮流测两个版本，逐轮配对求差。
        # 单独跑两个进程时，同一个负载在这个环境里两次能差几十个百分点，
        # 顺序差会被误当成版本差；配对之后每轮只比较相邻的两次，把漂移消掉。
        spec_c = CASES[args.case]
        prompts_c = load_inputs(args.case)
        engines = {s: build(s) for s in ("step39", "step40")}
        for s in engines:
            run(engines[s], prompts_c, spec_c["gen"], f"warm_{s}")
            run(engines[s], prompts_c, spec_c["gen"], f"warm2_{s}")
        pair = []
        for r in range(args.reps):
            order = ("step39", "step40") if r % 2 == 0 else ("step40", "step39")
            row = {}
            for s in order:
                dt, got = run(engines[s], prompts_c, spec_c["gen"], f"{s}_{r}")
                if not all(v == spec_c["gen"] for v in got.values()):
                    raise RuntimeError(f"{s} 输出长度不对: {got}")
                row[s] = dt
            pair.append(row)
            print(f"  轮 {r}: step39={row['step39'] * 1000:.2f} ms  "
                  f"step40={row['step40'] * 1000:.2f} ms  "
                  f"差={(row['step40'] - row['step39']) / row['step39'] * 100:+.1f}%")
        a = [p["step39"] for p in pair]
        b = [p["step40"] for p in pair]
        diffs = [(y - x) / x for x, y in zip(a, b)]
        print(json.dumps(dict(
            case=args.case, paired=True, reps=len(pair),
            median_39=round(statistics.median(a), 4), median_40=round(statistics.median(b), 4),
            paired_median_pct=round(statistics.median(diffs) * 100, 2),
            same_direction=sum(1 for d in diffs if d < 0),
        ), ensure_ascii=False))
        return

    spec = CASES[args.case]
    prompts = load_inputs(args.case)
    engine = build(args.step)
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
        step=args.step, case=args.case, spec=spec, graph=True,
        median_s=round(med, 4), samples_s=[round(s, 4) for s in samples],
        throughput_tok_s=round(spec["conc"] * spec["gen"] / med, 1),
    ), ensure_ascii=False))


if __name__ == "__main__":
    main()
