"""第三十七关 A/B：只换 attention 实现，其余配置完全相同。

    python benchmarks/bench_step37_prefill.py --step step35 --case prefill_c8
    python benchmarks/bench_step37_prefill.py --step step37 --case prefill_c8

step35 是逐行 kernel，step37 是 query 分块 kernel。输入、KV 容量、块大小、norm 后端、
Graph 开关、计时边界全部一致，唯一变量就是 attention 走哪条路。

再单独量一次纯 attention 算子时间（不含模型其余部分），用于确认改的确实是 attention
热点，而不是靠别的地方变快。
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


def build(step, use_cuda_graph=True, norm_backend="triton"):
    import torch
    mod = __import__(step)
    engine = mod.Engine.from_model_dir(
        MODEL_DIR, device="cuda", dtype=torch.bfloat16, attention_backend="triton",
        norm_backend=norm_backend, use_cuda_graph=use_cuda_graph,
        max_num_seqs=MAX_NUM_SEQS, max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        block_size=BLOCK_SIZE, num_kv_blocks=KV_BLOCKS, enable_prefix_caching=False)
    engine.model.eos_token_ids = ()
    engine.scheduler.eos_token_ids = set()
    assert engine.model.norm_backend == norm_backend
    return engine


def run(engine, prompts, gen, tag):
    import torch
    got = {}
    engine.scheduler.on_finished = lambda r: got.__setitem__(r["request_id"], len(r["output_ids"]))
    t0 = time.perf_counter()
    for i, ids in enumerate(prompts):
        engine.add_request({"request_id": f"{tag}_{i}", "prompt_ids": ids, "max_new_tokens": gen})
    steps = 0
    while engine.has_unfinished_requests():
        engine.step()
        steps += 1
    torch.cuda.synchronize()
    return time.perf_counter() - t0, got, steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=("step35", "step37"))
    ap.add_argument("--case", required=True, choices=tuple(CASES))
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--no-graph", action="store_true")
    args = ap.parse_args()

    import torch
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False

    spec = CASES[args.case]
    prompts = load_inputs(args.case)
    engine = build(args.step, use_cuda_graph=not args.no_graph)
    run(engine, prompts, spec["gen"], "warm0")
    run(engine, prompts, spec["gen"], "warm1")

    samples = []
    for r in range(args.reps):
        dt, got, steps = run(engine, prompts, spec["gen"], f"m{r}")
        assert all(v == spec["gen"] for v in got.values()), f"输出长度不对: {got}"
        samples.append(dt)

    tokens = spec["conc"] * spec["gen"]
    med = statistics.median(samples)
    graph_keys = sorted(getattr(engine.model, "graphs", {}).keys(), key=str)
    print(json.dumps(dict(
        step=args.step, case=args.case, spec=spec, graph=not args.no_graph,
        median_s=round(med, 4), samples_s=[round(s, 4) for s in samples],
        throughput_tok_s=round(tokens / med, 1), output_tokens=tokens,
        steps=steps, graph_keys=[str(k) for k in graph_keys],
    ), ensure_ascii=False))


if __name__ == "__main__":
    main()
