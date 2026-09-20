"""单卡同条件基准：step35 引擎 vs 本机 vLLM。

六个测点（长度按 token ID 数）：

    负载            prompt  生成   并发
    短请求            64     32    1、8
    长 prefill       512      1    1、8
    较长 decode       64    128    1、8

两边用**同一份输入 token IDs**（`benchmarks/results/step36_inputs.json`，由下面的
`--make-inputs` 生成，固定种子）。测速走固定长度模式：关闭 EOS 提前停止，生成到预算
为止，这样两边的工作量严格相同。

用法（每个引擎各自独立进程）：

    python benchmarks/bench_step36_vllm_compare.py --make-inputs
    python benchmarks/bench_step36_vllm_compare.py --engine mine  --case short_c1
    python benchmarks/bench_step36_vllm_compare.py --engine vllm  --case short_c1
    python benchmarks/bench_step36_vllm_compare.py --engine mine  --case short_c1 --graph   # 关掉 Graph 做对照

计时边界：从提交这一组请求到全部 token 就绪；不含模型加载、tokenizer、首次编译与图捕获。
"""

import argparse
import json
import os
import pathlib
import random
import statistics
import sys
import time

PROJECT = pathlib.Path(__file__).resolve().parents[1]
MODEL_DIR = "/home/user/proj/KuiperLLama/Qwen/Qwen3-0.6B"
INPUTS = PROJECT / "benchmarks" / "results" / "step36_inputs.json"

# 两边共用的运行参数（对齐口径）
BLOCK_SIZE = 16
MAX_MODEL_LEN = 1024
MAX_NUM_SEQS = 8
MAX_NUM_BATCHED_TOKENS = 2048
KV_BLOCKS = 512          # 我们自己引擎的块数
VOCAB = 151936

CASES = {
    "short_c1":   dict(prompt=64,  gen=32,  conc=1),
    "short_c8":   dict(prompt=64,  gen=32,  conc=8),
    "prefill_c1": dict(prompt=512, gen=1,   conc=1),
    "prefill_c8": dict(prompt=512, gen=1,   conc=8),
    "decode_c1":  dict(prompt=64,  gen=128, conc=1),
    "decode_c8":  dict(prompt=64,  gen=128, conc=8),
}


def kv_bytes():
    """我们自己引擎的 KV 池字节数，用于让 vLLM 用同样的 KV 容量。"""
    sys.path.insert(0, str(PROJECT))
    return KV_BLOCKS * BLOCK_SIZE * 8 * 128 * 2 * 28 * 2      # blocks×slot×kv_heads×dim×K/V×layers×bf16


def make_inputs():
    # 每条 prompt 用不同的一段随机 token，避免共享前缀带来的额外影响
    rng = random.Random(20260920)
    data = {}
    for name, c in CASES.items():
        prompts = []
        for _ in range(c["conc"]):
            prompts.append([rng.randrange(1000, VOCAB - 1000) for _ in range(c["prompt"])])
        data[name] = prompts
    INPUTS.parent.mkdir(parents=True, exist_ok=True)
    INPUTS.write_text(json.dumps(data))
    print(f"写入 {INPUTS}（{len(data)} 个测点）")


def load_inputs(case):
    data = json.loads(INPUTS.read_text())
    return data[case]


# ---------------- 我们的引擎 ----------------

def build_mine(graph=True):
    import torch
    sys.path.insert(0, str(PROJECT))
    import step35 as m
    engine = m.Engine.from_model_dir(
        MODEL_DIR, device="cuda", dtype=torch.bfloat16, attention_backend="triton",
        use_cuda_graph=graph, max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS, block_size=BLOCK_SIZE,
        num_kv_blocks=KV_BLOCKS, enable_prefix_caching=False)
    # 固定长度模式：关掉 EOS 提前停止。这是**测试侧的适配**，没有改实现——
    # 只把两个公开属性清空，正常路径的 EOS 行为不受影响。
    engine.model.eos_token_ids = ()
    engine.scheduler.eos_token_ids = set()
    return engine


def run_mine(engine, prompts, gen, tag):
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


# ---------------- vLLM ----------------

def build_vllm(profiler_dir=None):
    from vllm import LLM
    kwargs = {}
    if profiler_dir is not None:
        # vLLM 0.28 要显式给 ProfilerConfig 才允许 start_profile；
        # ignore_frontend=True 表示 trace 只取真正跑 GPU 的 worker 进程。
        from vllm.config import ProfilerConfig
        kwargs["profiler_config"] = ProfilerConfig(
            profiler="torch", torch_profiler_dir=str(profiler_dir), ignore_frontend=True)
    return LLM(model=MODEL_DIR, dtype="bfloat16", max_model_len=MAX_MODEL_LEN,
               block_size=BLOCK_SIZE, enable_prefix_caching=False,
               max_num_seqs=MAX_NUM_SEQS, max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
               gpu_memory_utilization=0.85, kv_cache_memory_bytes=kv_bytes(),
               disable_log_stats=True, skip_tokenizer_init=True, **kwargs)


def run_vllm(llm, prompts, gen):
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0, max_tokens=gen, ignore_eos=True)
    t0 = time.perf_counter()
    out = llm.generate([{"prompt_token_ids": p} for p in prompts], sampling_params=sp,
                       use_tqdm=False)
    dt = time.perf_counter() - t0
    return dt, {o.request_id: len(o.outputs[0].token_ids) for o in out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-inputs", action="store_true")
    ap.add_argument("--engine", choices=("mine", "vllm"))
    ap.add_argument("--case", choices=tuple(CASES))
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--no-graph", action="store_true", help="我们的引擎不开 CUDA Graph")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.make_inputs:
        return make_inputs()
    if not args.engine or not args.case:
        ap.error("需要 --engine 和 --case（或用 --make-inputs）")

    import torch
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    spec = CASES[args.case]
    prompts = load_inputs(args.case)
    assert len(prompts) == spec["conc"] and len(prompts[0]) == spec["prompt"]

    import torch as _t
    _free, _total = _t.cuda.mem_get_info()
    used_before = (_total - _free) / 2 ** 30
    t_load0 = time.perf_counter()
    if args.engine == "mine":
        engine = build_mine(graph=not args.no_graph)
        run = lambda tag: run_mine(engine, prompts, spec["gen"], tag)
        kv_note = f"num_kv_blocks={KV_BLOCKS} (= {KV_BLOCKS * BLOCK_SIZE} token 槽位)"
    else:
        engine = build_vllm()
        run = lambda tag: (lambda dt, got: (dt, got, None))(*run_vllm(engine, prompts, spec["gen"]))
        cc = engine.llm_engine.vllm_config.cache_config
        kv_note = f"num_gpu_blocks={cc.num_gpu_blocks} (= {cc.num_gpu_blocks * cc.block_size} token 槽位)"
    load_s = time.perf_counter() - t_load0

    # 预热（含首次编译 / 图捕获）
    for w in range(args.warmup):
        _, got, _ = run(f"w{w}")
        assert all(v == spec["gen"] for v in got.values()), f"预热输出长度不对: {got}"

    # 显存用设备级口径：vLLM 的 GPU 工作在 EngineCore 子进程里，
    # 父进程的 torch.cuda.max_memory_allocated() 会是 0，两边没法比
    def used_gib():
        free, total = torch.cuda.mem_get_info()
        return (total - free) / 2 ** 30

    base_used = used_gib()
    samples, peaks = [], []
    for r in range(args.reps):
        dt, got, steps = run(f"m{r}")
        assert all(v == spec["gen"] for v in got.values()), f"第 {r} 次输出长度不对: {got}"
        assert len(got) == spec["conc"]
        samples.append(dt)
        peaks.append(used_gib())

    tokens = spec["conc"] * spec["gen"]
    med = statistics.median(samples)
    q = statistics.quantiles(samples, n=4, method="inclusive") if len(samples) > 1 else [med] * 3
    report = dict(
        engine=args.engine, case=args.case, spec=spec,
        graph=(None if args.engine == "vllm" else (not args.no_graph)),
        load_s=round(load_s, 2), kv=kv_note,
        gpu_used_before_load_gib=round(used_before, 3),
        output_tokens=tokens, samples_s=[round(s, 4) for s in samples],
        median_s=round(med, 4), iqr_over_median=round((q[2] - q[0]) / med, 4),
        throughput_tok_s=round(tokens / med, 1),
        gpu_used_after_gib=round(max(peaks), 3),
        output_lengths_ok=True,
    )
    print(json.dumps(report, ensure_ascii=False))
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
