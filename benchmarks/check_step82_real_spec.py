"""82 关：真实 Gemma4 单卡正确性/性能基准。

用法示例：
  python benchmarks/check_step82_real_spec.py --preflight
  python benchmarks/check_step82_real_spec.py --correctness --max-new-tokens 8
  python benchmarks/check_step82_real_spec.py --perf --max-new-tokens 32 --repeats 1
  python benchmarks/check_step82_real_spec.py --perf --graph --prefix --dynamic

原则：
- K=0 必须是 speculative_config=None，不是同一个 proposer 改宽度；
- 正确性和性能分开，正确性没过不进入主表；
- 累计 token_ids 按增量记账；
- 慢也是结果，不伪造加速比。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

from minivllm import (CacheConfig, DeviceConfig, LLMEngine, ModelConfig,  # noqa: E402
                      SamplingParams, SchedulerConfig, SpeculativeConfig,
                      UniProcExecutor, VllmConfig, Worker)
from minivllm.config import CompilationConfig  # noqa: E402
from spec_decode_metrics import (AcceptanceAggregate, RequestTiming,  # noqa: E402
                                 median_range, summarize_run)

TARGET = ROOT / "models" / "gemma-4-E2B-it"
ASSISTANT = ROOT / "models" / "gemma-4-E2B-it-assistant"
TARGET_REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
ASSISTANT_REVISION = "2d874ef7d29f9a30599a1e4b3c1cbc9595f005df"
TARGET_EOS = 1
TARGET_STOP = [106, 50]

PROMPTS = [
    ("zh", "请用三句话解释什么是 KV cache，以及为什么分页能省显存。"),
    ("en", "Explain in two sentences why speculative decoding can reduce latency."),
    ("code", "Complete this Python function and explain the base case:\n\ndef fib(n):\n"),
    ("long", "下面是需要保留的公共前缀材料。" * 64 + "\n请只回答：公共前缀之后你看到了什么？"),
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _tensor_bytes(path: Path) -> int:
    """只读 safetensors header，不加载张量。"""
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    return sum(v["data_offsets"][1] - v["data_offsets"][0]
               for k, v in header.items() if k != "__metadata__")


def _gpu_info() -> dict:
    info = {"available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        info.update({
            "name": torch.cuda.get_device_name(0),
            "total_bytes": total,
            "free_bytes_at_manifest": free,
            "capability": list(torch.cuda.get_device_capability(0)),
        })
    return info


def _git() -> dict:
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"],
                                             cwd=ROOT, text=True).strip())
        return {"head": head, "dirty": dirty}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def _package_hash() -> dict:
    root = ROOT / "minivllm"
    names = sorted(root.rglob("*.py"), key=lambda p: str(p.relative_to(root)))
    digest = hashlib.sha256("".join(
        f"{p.relative_to(root)}\0{hashlib.sha256(p.read_bytes()).hexdigest()}\n"
        for p in names).encode()).hexdigest()
    return {"root": "minivllm/", "python_files": len(names), "sha256": digest}


def _configs() -> tuple[dict, dict]:
    return (json.loads((TARGET / "config.json").read_text()),
            json.loads((ASSISTANT / "config.json").read_text()))


def _prompt_ids(tokenizer, text: str) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    return tokenizer(rendered, add_special_tokens=False)["input_ids"]


def preflight(args) -> dict:
    target_bytes = _tensor_bytes(TARGET / "model.safetensors")
    assistant_bytes = _tensor_bytes(ASSISTANT / "model.safetensors")
    gpu = _gpu_info()
    return {
        "target": {
            "path": str(TARGET.relative_to(ROOT)),
            "revision": TARGET_REVISION,
            "tensor_bytes": target_bytes,
            "sha256": _sha256(TARGET / "model.safetensors"),
        },
        "assistant": {
            "path": str(ASSISTANT.relative_to(ROOT)),
            "revision": ASSISTANT_REVISION,
            "tensor_bytes": assistant_bytes,
            "sha256": _sha256(ASSISTANT / "model.safetensors"),
        },
        "weights_total_bytes": target_bytes + assistant_bytes,
        "gpu": gpu,
        "budget": {
            "max_model_len": args.max_model_len,
            "block_size": args.block_size,
            "num_gpu_blocks": args.blocks,
            "max_num_seqs": args.batch,
            "max_num_batched_tokens": args.budget,
        },
    }


@dataclass
class EngineRun:
    summary: dict
    tokens: dict[str, list[int]]
    build_s: float
    engine: object
    runner: object


def _build_engine(args, *, k: int, batch: int, graph: bool):
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    target_cfg, assistant_cfg = _configs()
    dtype = args.dtype
    target = ModelConfig(model=str(TARGET), dtype=dtype,
                         max_model_len=args.max_model_len,
                         hf_config=target_cfg, tokenizer=str(TARGET))
    draft = ModelConfig(model=str(ASSISTANT), dtype=dtype,
                        max_model_len=args.max_model_len,
                        hf_config=assistant_cfg)
    spec = None if k == 0 else SpeculativeConfig(
        method=None, num_speculative_tokens=k, draft_model_config=draft)
    mode = "full_decode_only" if graph else "none"
    cfg = VllmConfig(
        model_config=target,
        cache_config=CacheConfig(
            block_size=args.block_size, num_gpu_blocks=args.blocks,
            enable_prefix_caching=args.prefix),
        scheduler_config=SchedulerConfig(
            max_num_seqs=max(batch, 1), max_num_batched_tokens=args.budget),
        device_config=DeviceConfig(device=args.device),
        speculative_config=spec,
        compilation_config=CompilationConfig(
            cudagraph_mode=mode,
            cudagraph_capture_sizes=([1, 2, 3, 4, 6, 8, 12, 16]
                                     if graph else []),
            max_cudagraph_capture_size=(16 if graph else 0)),
        attention_backend=args.attention_backend,
    )
    start = time.perf_counter()
    engine = LLMEngine(cfg, UniProcExecutor(cfg, Worker(cfg)), tokenizer=None)
    build_s = time.perf_counter() - start
    runner = (engine.engine_core.engine_core.model_executor
              .driver_worker.model_runner)
    return engine, runner, build_s


def _run_once(engine, runner, args, *, k: int, batch: int, timing: bool):
    tokenizer = AutoTokenizer.from_pretrained(str(TARGET))
    prompts = [(name, _prompt_ids(tokenizer, text))
               for name, text in PROMPTS[:batch]]
    timings = {name: RequestTiming(name, submitted_at=time.perf_counter())
               for name, _ in prompts}
    acceptance = AcceptanceAggregate()
    for name, prompt in prompts:
        params = SamplingParams(
            max_tokens=args.max_new_tokens, temperature=0.0,
            eos_token_id=(TARGET_EOS if not args.ignore_eos else None),
            stop_token_ids=(TARGET_STOP if not args.ignore_eos else []),
            ignore_eos=args.ignore_eos)
        engine.add_request(name, list(prompt), params)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    steps = 0
    outputs: dict[str, list[int]] = {}
    tokens: dict[str, list[int]] = {}
    for _ in range(20000):
        if not engine.has_unfinished_requests():
            break
        step_outputs = engine.step()
        steps += 1
        now = time.perf_counter()
        for out in step_outputs:
            outputs[out.request_id] = list(out.token_ids)
            timings[out.request_id].observe(now, len(out.token_ids))
            tokens[out.request_id] = list(out.token_ids)
        stats = engine.last_scheduler_stats
        if stats is not None and stats.spec_decoding_stats is not None:
            acceptance.observe(stats.spec_decoding_stats)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall = time.perf_counter() - start
    summary = summarize_run(
        timings=timings, acceptance=acceptance, wall_time_s=wall,
        engine_steps=steps,
        preemptions=int(getattr(engine.engine_core.engine_core.scheduler,
                                "num_preemptions", 0)))
    if torch.cuda.is_available():
        summary["memory_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        summary["memory_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    summary["k"] = k
    summary["batch"] = batch
    summary["graph"] = bool(args.graph)
    summary["prefix"] = bool(args.prefix)
    summary["dtype"] = args.dtype
    if getattr(runner, "cudagraph_capture_stats", None):
        summary["cudagraph_capture_stats"] = runner.cudagraph_capture_stats
    return summary, tokens


def _engine_kwargs(args, *, k: int, batch: int, graph: bool) -> EngineRun:
    engine, runner, build_s = _build_engine(args, k=k, batch=batch, graph=graph)
    return EngineRun({}, {}, build_s, engine, runner)


def correctness(args) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(str(TARGET))
    prompts = [(name, _prompt_ids(tokenizer, text)) for name, text in PROMPTS]
    runs = {}
    diagnostics = {}
    for label, k, graph in (("target_only_eager", 0, False),
                            ("spec_k2_eager", 2, False),
                            ("spec_k2_graph", 2, True)):
        engine, runner, build_s = _build_engine(
            args, k=k, batch=1, graph=graph)
        try:
            tokens = {}
            # 逐 prompt 顺序跑；避免把 correctness 的 batch 语义和 perf 混在一起。
            for name, prompt in prompts:
                engine.add_request(
                    f"{label}-{name}", list(prompt),
                    SamplingParams(max_tokens=args.max_new_tokens,
                                   temperature=0.0, eos_token_id=TARGET_EOS,
                                   stop_token_ids=TARGET_STOP))
                final = {}
                for _ in range(args.max_new_tokens * 20 + 100):
                    if not engine.has_unfinished_requests():
                        break
                    for out in engine.step():
                        final[out.request_id] = list(out.token_ids)
                tokens[name] = final[f"{label}-{name}"]
        finally:
            engine.shutdown()
        runs[label] = tokens
        diagnostics[label] = {"build_s": build_s, "tokens": tokens}
    exact = {
        "target_vs_eager": all(runs["target_only_eager"][n] == runs["spec_k2_eager"][n]
                               for n, _ in prompts),
        "target_vs_graph": all(runs["target_only_eager"][n] == runs["spec_k2_graph"][n]
                               for n, _ in prompts),
        "eager_vs_graph": all(runs["spec_k2_eager"][n] == runs["spec_k2_graph"][n]
                              for n, _ in prompts),
    }
    mismatches = {}
    for name, _ in prompts:
        if runs["target_only_eager"][name] != runs["spec_k2_eager"][name]:
            mismatches[name] = {
                "target": runs["target_only_eager"][name],
                "spec": runs["spec_k2_eager"][name],
                "first_diff": next((i for i, (a, b) in enumerate(
                    zip(runs["target_only_eager"][name],
                        runs["spec_k2_eager"][name])) if a != b), None),
            }
    return {"exact": exact, "runs": diagnostics, "mismatches": mismatches,
            "prompts": {name: ids for name, ids in prompts}}


def perf(args) -> dict:
    results = []
    for batch in args.batch_sizes:
        for k in args.ks:
            engine_run = _engine_kwargs(args, k=k, batch=batch, graph=args.graph)
            engine, runner = engine_run.engine, engine_run.runner
            try:
                # warmup 一次；随后 repeats 次独立测量
                _run_once(engine, runner, args, k=k, batch=batch, timing=True)
                samples = []
                for _ in range(args.repeats):
                    summary, tokens = _run_once(
                        engine, runner, args, k=k, batch=batch, timing=True)
                    samples.append({**summary, "tokens": tokens})
            finally:
                engine.shutdown()
            results.append({
                "batch": batch,
                "k": k,
                "build_s": engine_run.build_s,
                "samples": samples,
                "output_tokens_per_s": median_range(
                    [s["output_tokens_per_s"] for s in samples]),
                "wall_time_s": median_range([s["wall_time_s"] for s in samples]),
                "acceptance_rate": median_range(
                    [s["acceptance"]["acceptance_rate"] for s in samples
                     if s["acceptance"]["acceptance_rate"] is not None]),
            })
    return {"results": results}



def _drain(engine) -> None:
    core = engine.engine_core.engine_core
    for _ in range(20):
        if not core.has_requests():
            return
        engine.step()


def _trace_hits(engine) -> int:
    scheduler = engine.engine_core.engine_core.scheduler
    return sum(sum(record.get("hits", {}).values()) for record in scheduler.trace)


def supplement(args) -> dict:
    """三个定点补充：graph、prefix+chunked、dynamic batch。"""
    import copy

    result = {}
    # 1) Graph：B=1，K=0/2，eager vs FULL_DECODE_ONLY
    rows = []
    for k in (0, 2):
        for graph in (False, True):
            local = copy.copy(args)
            local.graph = graph
            engine, runner, build_s = _build_engine(local, k=k, batch=1,
                                                    graph=graph)
            try:
                _run_once(engine, runner, local, k=k, batch=1, timing=True)
                _drain(engine)
                summary, _ = _run_once(engine, runner, local, k=k, batch=1,
                                       timing=True)
                summary["build_s"] = build_s
                summary["graph_capture"] = getattr(
                    runner, "cudagraph_capture_stats", None)
            finally:
                engine.shutdown()
            rows.append({"k": k, "graph": graph, **summary})
    result["graph"] = rows

    # 2) Prefix + chunked prefill：B=4，K=0/2，冷/热各一次
    rows = []
    for k in (0, 2):
        local = copy.copy(args)
        local.prefix = True
        engine, runner, build_s = _build_engine(local, k=k, batch=4,
                                                graph=False)
        try:
            cold, _ = _run_once(engine, runner, local, k=k, batch=4, timing=True)
            _drain(engine)
            cold_hits = _trace_hits(engine)
            warm, _ = _run_once(engine, runner, local, k=k, batch=4, timing=True)
            _drain(engine)
            warm_hits = _trace_hits(engine) - cold_hits
        finally:
            engine.shutdown()
        rows.append({"k": k, "build_s": build_s,
                     "cold": cold, "cold_hits": cold_hits,
                     "warm": warm, "warm_hits": warm_hits})
    result["prefix"] = rows

    # 3) 动态 batch 4 -> 3 -> 4：K=2，第一条短请求完成后再补一条
    local = copy.copy(args)
    local.max_new_tokens = min(args.max_new_tokens, 16)
    engine, runner, build_s = _build_engine(local, k=2, batch=4, graph=False)
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(TARGET))
        timings = {}
        acceptance = AcceptanceAggregate()
        for index, (name, text) in enumerate(PROMPTS[:4]):
            ids = _prompt_ids(tokenizer, text)
            timings[name] = RequestTiming(name, time.perf_counter())
            engine.add_request(
                name, ids, SamplingParams(
                    max_tokens=(4 if index == 0 else local.max_new_tokens),
                    temperature=0.0, ignore_eos=True))
        start = time.perf_counter()
        added = False
        steps = 0
        for _ in range(20000):
            if not engine.has_unfinished_requests():
                break
            for out in engine.step():
                timings[out.request_id].observe(
                    time.perf_counter(), len(out.token_ids))
            steps += 1
            # 第一条短请求完成后补入第五条，形成 4 -> 3 -> 4
            if not added and "zh" not in engine.engine_core.engine_core.scheduler.requests:
                ids = _prompt_ids(tokenizer, PROMPTS[0][1])
                timings["dyn5"] = RequestTiming("dyn5", time.perf_counter())
                engine.add_request("dyn5", ids, SamplingParams(
                    max_tokens=local.max_new_tokens, temperature=0.0,
                    ignore_eos=True))
                added = True
            stats = engine.last_scheduler_stats
            if stats is not None and stats.spec_decoding_stats is not None:
                acceptance.observe(stats.spec_decoding_stats)
        wall = time.perf_counter() - start
    finally:
        engine.shutdown()
    result["dynamic"] = summarize_run(
        timings=timings, acceptance=acceptance, wall_time_s=wall,
        engine_steps=steps, preemptions=0)
    result["dynamic"]["build_s"] = build_s
    result["dynamic"]["added_after_first_finished"] = added
    return result

def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--perf", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--supplement", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="torch")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--blocks", type=int, default=512)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--prefix", action="store_true")
    parser.add_argument("--ignore-eos", action="store_true", default=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--ks", type=int, nargs="+", default=[0, 2, 4])
    parser.add_argument("--output-dir", default=str(ROOT / "benchmarks/results/step82"))
    args = parser.parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "git": _git(), "package": _package_hash(), "gpu": _gpu_info(),
        "target": {"path": str(TARGET), "revision": TARGET_REVISION},
        "assistant": {"path": str(ASSISTANT), "revision": ASSISTANT_REVISION},
        "args": vars(args),
    }
    _write(out_dir / "manifest.json", manifest)
    if args.preflight:
        pre = preflight(args)
        _write(out_dir / "preflight.json", pre)
        print(json.dumps(pre, ensure_ascii=False, indent=1))
    if args.correctness or args.all:
        result = correctness(args)
        _write(out_dir / "correctness.json", result)
        print("correctness exact:", result["exact"])
    if args.perf or args.all:
        result = perf(args)
        _write(out_dir / "perf.json", result)
        for row in result["results"]:
            print(f"B={row['batch']} K={row['k']} "
                  f"tok/s={row['output_tokens_per_s']['median']} "
                  f"wall={row['wall_time_s']['median']} "
                  f"acc={row['acceptance_rate']['median']}")
    if args.supplement:
        result = supplement(args)
        _write(out_dir / "supplements.json", result)
        print("supplements written", out_dir / "supplements.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
