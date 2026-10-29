"""82/83 关：真实 Gemma4 单卡正确性与性能基准。

用法示例：
  python benchmarks/check_step82_real_spec.py --preflight
  python benchmarks/check_step82_real_spec.py --correctness --max-new-tokens 8
  python benchmarks/check_step82_real_spec.py --perf --max-new-tokens 32 --repeats 1
  python benchmarks/check_step82_real_spec.py --perf --graph --prefix
  python benchmarks/check_step82_real_spec.py --supplement --max-new-tokens 16

83 关起，多配置测量由**父进程编排、一个配置一个子进程**完成：

    父进程：case 配置 -> subprocess 新解释器 -> 收结果 -> 子进程退出 -> 下一个
    子进程：建一个 engine -> 预热 -> 测量 -> 保存 -> shutdown -> 退出

原则：
- K=0 必须是 speculative_config=None，不是同一个 proposer 改宽度；
- 正确性和性能分开，正确性没过不进入主表；
- 累计 token_ids 按增量记账；吞吐只数 committed output token；
- 子进程失败/超时写成失败记录，不返回 0 tok/s 假装成功；
- 慢也是结果，不伪造加速比；
- 加载/捕获与正式测量分阶段记录显存。

历史 step82 结果（`benchmarks/results/step82/`）不重写；它们的 `tpot_s` 是旧口径
（按输出事件数），新结果带 `metric_schema=step83`。
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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

from benchmark_runner import (case_worker_main, run_case,  # noqa: E402
                              run_matrix)
from spec_decode_metrics import (AcceptanceAggregate, RequestTiming,  # noqa: E402
                                 median_range, memory_snapshot,
                                 reset_peak_memory, summarize_run)

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


# ---------------------------------------------------------------------------
# 不初始化 CUDA 的父进程工具
# ---------------------------------------------------------------------------

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


def _parent_gpu_info() -> dict:
    """父进程用 nvidia-smi 记录设备，**不 import torch / 不初始化 CUDA**。"""
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            text=True, capture_output=True, check=True)
        devices = []
        for line in completed.stdout.splitlines():
            name, total, driver = [part.strip() for part in line.split(",")]
            devices.append({"name": name, "total_bytes": int(total) * 1024 * 1024,
                            "driver_version": driver})
        return {"source": "nvidia-smi", "devices": devices}
    except Exception as exc:  # noqa: BLE001
        return {"source": "nvidia-smi", "error": str(exc)}


def _configs() -> tuple[dict, dict]:
    return (json.loads((TARGET / "config.json").read_text()),
            json.loads((ASSISTANT / "config.json").read_text()))


def _args_from_case(case: dict) -> argparse.Namespace:
    return argparse.Namespace(**case["args"])


def _worker_argv() -> list[str]:
    return [sys.executable, str(Path(__file__).resolve())]


def _case_payload(args, kind: str, label: str, **extra) -> dict:
    payload = {
        "kind": kind,
        "label": label,
        "args": vars(args).copy(),
    }
    payload.update(extra)
    return payload


# ---------------------------------------------------------------------------
# 子进程：模型/tokenizer/引擎只在这里出现
# ---------------------------------------------------------------------------

def _load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(TARGET))


def _prompt_ids(tokenizer, text: str) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    return tokenizer(rendered, add_special_tokens=False)["input_ids"]


def _gpu_info() -> dict:
    import torch

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


def preflight(args) -> dict:
    target_bytes = _tensor_bytes(TARGET / "model.safetensors")
    assistant_bytes = _tensor_bytes(ASSISTANT / "model.safetensors")
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
        "gpu": _gpu_info(),
        "budget": {
            "max_model_len": args.max_model_len,
            "block_size": args.block_size,
            "num_gpu_blocks": args.blocks,
            "max_num_seqs": args.batch,
            "max_num_batched_tokens": args.budget,
        },
    }


def _build_engine(args, *, k: int, batch: int, graph: bool):
    from minivllm import (CacheConfig, DeviceConfig, LLMEngine, ModelConfig,
                          SchedulerConfig, SpeculativeConfig, UniProcExecutor,
                          VllmConfig, Worker)
    from minivllm.config import CompilationConfig

    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")
    target_cfg, assistant_cfg = _configs()
    target = ModelConfig(model=str(TARGET), dtype=args.dtype,
                         max_model_len=args.max_model_len,
                         hf_config=target_cfg, tokenizer=str(TARGET))
    draft = ModelConfig(model=str(ASSISTANT), dtype=args.dtype,
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


def _make_prompts(tokenizer, batch: int) -> list[tuple[str, list[int]]]:
    return [(name, _prompt_ids(tokenizer, text)) for name, text in PROMPTS[:batch]]


def _run_once(engine, runner, args, *, k: int, batch: int, timing: bool,
              tokenizer=None, prompts=None):
    import torch

    if tokenizer is None:
        tokenizer = _load_tokenizer()
    if prompts is None:
        prompts = _make_prompts(tokenizer, batch)
    timings = {name: RequestTiming(name, submitted_at=time.perf_counter())
               for name, _ in prompts}
    acceptance = AcceptanceAggregate()
    for name, prompt in prompts:
        params = _sampling_params(args)
        engine.add_request(name, list(prompt), params)
    if timing and torch.cuda.is_available():
        # 峰值由 case 在 warmup 之后重置；这里只保证计时起点前没有未完成工作。
        torch.cuda.synchronize()
    start = time.perf_counter()
    steps = 0
    tokens: dict[str, list[int]] = {}
    for _ in range(20000):
        if not engine.has_unfinished_requests():
            break
        step_outputs = engine.step()
        steps += 1
        now = time.perf_counter()
        for out in step_outputs:
            timings[out.request_id].observe(now, len(out.token_ids))
            tokens[out.request_id] = list(out.token_ids)
        stats = engine.last_scheduler_stats
        if stats is not None and stats.spec_decoding_stats is not None:
            acceptance.observe(stats.spec_decoding_stats)
    if timing and torch.cuda.is_available():
        torch.cuda.synchronize()
    wall = time.perf_counter() - start
    summary = summarize_run(
        timings=timings, acceptance=acceptance, wall_time_s=wall,
        engine_steps=steps,
        preemptions=int(getattr(engine.engine_core.engine_core.scheduler,
                                "num_preemptions", 0)))
    summary["k"] = k
    summary["batch"] = batch
    summary["graph"] = bool(args.graph)
    summary["prefix"] = bool(args.prefix)
    summary["dtype"] = args.dtype
    summary["attention_backend"] = args.attention_backend
    summary["max_new_tokens"] = args.max_new_tokens
    if timing:
        summary["memory"] = memory_snapshot()
    if getattr(runner, "cudagraph_capture_stats", None):
        summary["cudagraph_capture_stats"] = runner.cudagraph_capture_stats
    return summary, tokens


def _sampling_params(args):
    from minivllm import SamplingParams

    return SamplingParams(
        max_tokens=args.max_new_tokens, temperature=0.0,
        eos_token_id=(TARGET_EOS if not args.ignore_eos else None),
        stop_token_ids=(TARGET_STOP if not args.ignore_eos else []),
        ignore_eos=args.ignore_eos)


def _case_perf(case: dict) -> dict:
    import torch

    args = _args_from_case(case)
    k, batch = int(case["k"]), int(case["batch"])
    tokenizer = _load_tokenizer()
    prompts = _make_prompts(tokenizer, batch)
    engine, runner, build_s = _build_engine(
        args, k=k, batch=batch, graph=bool(case.get("graph", args.graph)))
    load_memory = memory_snapshot()
    try:
        warm_start = time.perf_counter()
        _run_once(engine, runner, args, k=k, batch=batch, timing=True,
                  tokenizer=tokenizer, prompts=prompts)
        warmup_s = time.perf_counter() - warm_start
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            reset_peak_memory()
        measure_start_memory = memory_snapshot()
        samples = []
        for _ in range(args.repeats):
            summary, tokens = _run_once(
                engine, runner, args, k=k, batch=batch, timing=True,
                tokenizer=tokenizer, prompts=prompts)
            samples.append({**summary, "tokens": tokens})
        measure_end_memory = memory_snapshot()
        capture_stats = getattr(runner, "cudagraph_capture_stats", None)
    finally:
        engine.shutdown()
    capture_s = ((capture_stats or {}).get("capture_seconds")
                 if isinstance(capture_stats, dict) else None)
    measure_peak = {
        "peak_allocated_bytes": measure_end_memory.get("peak_allocated_bytes"),
        "peak_reserved_bytes": measure_end_memory.get("peak_reserved_bytes"),
        "device_total_bytes": measure_end_memory.get("device_total_bytes"),
    }
    return {
        "batch": batch,
        "k": k,
        "graph": bool(case.get("graph", args.graph)),
        "prefix": bool(args.prefix),
        "build_s": build_s,
        "load_s_estimate": (max(build_s - capture_s, 0.0)
                            if capture_s is not None else build_s),
        "graph_capture_s": capture_s,
        "warmup_s": warmup_s,
        "samples": samples,
        "output_tokens_per_s": median_range(
            [s["output_tokens_per_s"] for s in samples]),
        "wall_time_s": median_range([s["wall_time_s"] for s in samples]),
        "acceptance_rate": median_range(
            [s["acceptance"]["acceptance_rate"] for s in samples
             if s["acceptance"]["acceptance_rate"] is not None]),
        "memory": {
            "load_capture": load_memory,
            "measure_start": measure_start_memory,
            "measure_end": measure_end_memory,
            "measure_peak": measure_peak,
        },
        "cudagraph_capture_stats": capture_stats,
    }


def _case_correctness(case: dict) -> dict:
    args = _args_from_case(case)
    label = case["label"]
    k, graph = int(case["k"]), bool(case["graph"])
    tokenizer = _load_tokenizer()
    prompts = _make_prompts(tokenizer, 1)
    engine, runner, build_s = _build_engine(args, k=k, batch=1, graph=graph)
    load_memory = memory_snapshot()
    try:
        token_map = {}
        # 同一配置下逐 prompt 顺序跑；不同配置之间已经由子进程隔离。
        for name, prompt in prompts:
            request_id = f"{label}-{name}"
            engine.add_request(
                request_id, list(prompt),
                _sampling_params(args))
            final = {}
            for _ in range(args.max_new_tokens * 20 + 100):
                if not engine.has_unfinished_requests():
                    break
                for out in engine.step():
                    final[out.request_id] = list(out.token_ids)
            token_map[name] = final[request_id]
    finally:
        engine.shutdown()
    return {
        "label": label,
        "k": k,
        "graph": graph,
        "build_s": build_s,
        "tokens": token_map,
        "prompts": {name: ids for name, ids in prompts},
        "memory": {"load_capture": load_memory},
    }


def _drain(engine) -> None:
    core = engine.engine_core.engine_core
    for _ in range(20):
        if not core.has_requests():
            return
        engine.step()


def _trace_hits(engine) -> int:
    scheduler = engine.engine_core.engine_core.scheduler
    return sum(sum(record.get("hits", {}).values()) for record in scheduler.trace)


def _case_supplement_graph(case: dict) -> dict:
    import torch

    args = _args_from_case(case)
    k, graph = int(case["k"]), bool(case["graph"])
    tokenizer = _load_tokenizer()
    prompts = _make_prompts(tokenizer, 1)
    engine, runner, build_s = _build_engine(args, k=k, batch=1, graph=graph)
    load_memory = memory_snapshot()
    try:
        warm_start = time.perf_counter()
        _run_once(engine, runner, args, k=k, batch=1, timing=True,
                  tokenizer=tokenizer, prompts=prompts)
        warmup_s = time.perf_counter() - warm_start
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            reset_peak_memory()
        measure_start_memory = memory_snapshot()
        summary, _ = _run_once(engine, runner, args, k=k, batch=1,
                               timing=True, tokenizer=tokenizer, prompts=prompts)
        measure_end_memory = memory_snapshot()
        summary["build_s"] = build_s
        summary["warmup_s"] = warmup_s
        summary["graph_capture"] = getattr(
            runner, "cudagraph_capture_stats", None)
        summary["memory"] = {
            "load_capture": load_memory,
            "measure_start": measure_start_memory,
            "measure_end": measure_end_memory,
        }
        # `_run_once` 用 args.graph 填了默认值；这里以 case 的 graph 为准（同一个
        # args 会复用给 eager/graph 两个 case）。
        summary["graph"] = bool(graph)
        summary["k"] = k
    finally:
        engine.shutdown()
    return summary


def _case_supplement_prefix(case: dict) -> dict:
    args = _args_from_case(case)
    k = int(case["k"])
    tokenizer = _load_tokenizer()
    prompts = _make_prompts(tokenizer, 4)
    engine, runner, build_s = _build_engine(args, k=k, batch=4, graph=False)
    load_memory = memory_snapshot()
    try:
        cold, _ = _run_once(engine, runner, args, k=k, batch=4, timing=True,
                            tokenizer=tokenizer, prompts=prompts)
        _drain(engine)
        cold_hits = _trace_hits(engine)
        warm, _ = _run_once(engine, runner, args, k=k, batch=4, timing=True,
                            tokenizer=tokenizer, prompts=prompts)
        _drain(engine)
        warm_hits = _trace_hits(engine) - cold_hits
    finally:
        engine.shutdown()
    return {"k": k, "build_s": build_s, "cold": cold, "cold_hits": cold_hits,
            "warm": warm, "warm_hits": warm_hits,
            "memory": {"load_capture": load_memory}}


def _case_supplement_dynamic(case: dict) -> dict:
    args = _args_from_case(case)
    args.max_new_tokens = min(args.max_new_tokens, 16)
    tokenizer = _load_tokenizer()
    engine, runner, build_s = _build_engine(args, k=2, batch=4, graph=False)
    load_memory = memory_snapshot()
    try:
        timings = {}
        acceptance = AcceptanceAggregate()
        for index, (name, text) in enumerate(PROMPTS[:4]):
            ids = _prompt_ids(tokenizer, text)
            timings[name] = RequestTiming(name, time.perf_counter())
            from minivllm import SamplingParams

            engine.add_request(
                name, ids, SamplingParams(
                    max_tokens=(4 if index == 0 else args.max_new_tokens),
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
            if (not added
                    and "zh" not in engine.engine_core.engine_core.scheduler.requests):
                from minivllm import SamplingParams

                ids = _prompt_ids(tokenizer, PROMPTS[0][1])
                timings["dyn5"] = RequestTiming("dyn5", time.perf_counter())
                engine.add_request("dyn5", ids, SamplingParams(
                    max_tokens=args.max_new_tokens, temperature=0.0,
                    ignore_eos=True))
                added = True
            stats = engine.last_scheduler_stats
            if stats is not None and stats.spec_decoding_stats is not None:
                acceptance.observe(stats.spec_decoding_stats)
        wall = time.perf_counter() - start
    finally:
        engine.shutdown()
    result = summarize_run(
        timings=timings, acceptance=acceptance, wall_time_s=wall,
        engine_steps=steps, preemptions=0)
    result["build_s"] = build_s
    result["added_after_first_finished"] = added
    result["memory"] = {"load_capture": load_memory}
    return result


_OPS = {
    "preflight": lambda case: preflight(_args_from_case(case)),
    "perf": _case_perf,
    "correctness": _case_correctness,
    "supplement_graph": _case_supplement_graph,
    "supplement_prefix": _case_supplement_prefix,
    "supplement_dynamic": _case_supplement_dynamic,
}


def _dispatch_case(case: dict) -> dict:
    kind = case.get("kind")
    if kind not in _OPS:
        raise ValueError(f"未知 case kind={kind!r}；可用：{sorted(_OPS)}")
    result = _OPS[kind](case)
    result.setdefault("case_label", case.get("label"))
    return result


# ---------------------------------------------------------------------------
# 父进程：只读配置、启动子进程、汇总
# ---------------------------------------------------------------------------

def _run_cases(cases: list[dict]) -> dict:
    return run_matrix(_worker_argv(), cases)


def _successful_results(matrix: dict) -> list[dict]:
    return [record["result"] for record in matrix["cases"]
            if record["status"] == "ok"]


def perf(args) -> dict:
    cases = [
        _case_payload(args, "perf", f"B{batch}K{k}", k=k, batch=batch,
                      graph=bool(args.graph))
        for batch in args.batch_sizes
        for k in args.ks
    ]
    matrix = _run_cases(cases)
    results = _successful_results(matrix)
    return {
        "ok": matrix["ok"],
        "metric_schema": "step83",
        "results": results,
        "failures": matrix["failures"],
    }


def correctness(args) -> dict:
    labels = (("target_only_eager", 0, False),
              ("spec_k2_eager", 2, False),
              ("spec_k2_graph", 2, True))
    cases = [
        _case_payload(args, "correctness", label, k=k, graph=graph)
        for label, k, graph in labels
    ]
    matrix = _run_cases(cases)
    diagnostics = {row["label"]: row for row in _successful_results(matrix)}
    runs = {label: row["tokens"] for label, row in diagnostics.items()}
    prompts = next(iter(diagnostics.values()), {}).get("prompts", {})
    all_present = all(label in runs for label, _, _ in labels)
    exact = {
        "target_vs_eager": (all_present and all(
            runs["target_only_eager"][name] == runs["spec_k2_eager"][name]
            for name in prompts)),
        "target_vs_graph": (all_present and all(
            runs["target_only_eager"][name] == runs["spec_k2_graph"][name]
            for name in prompts)),
        "eager_vs_graph": (all_present and all(
            runs["spec_k2_eager"][name] == runs["spec_k2_graph"][name]
            for name in prompts)),
    }
    mismatches = {}
    if all_present:
        for name in prompts:
            if runs["target_only_eager"][name] != runs["spec_k2_eager"][name]:
                mismatches[name] = {
                    "target": runs["target_only_eager"][name],
                    "spec": runs["spec_k2_eager"][name],
                    "first_diff": next((i for i, (a, b) in enumerate(
                        zip(runs["target_only_eager"][name],
                            runs["spec_k2_eager"][name])) if a != b), None),
                }
    return {
        "ok": matrix["ok"],
        "exact": exact,
        "runs": diagnostics,
        "mismatches": mismatches,
        "prompts": prompts,
        "failures": matrix["failures"],
    }


def supplement(args) -> dict:
    graph_cases = [
        _case_payload(args, "supplement_graph", f"graph_K{k}_{graph}",
                      k=k, graph=graph)
        for k in (0, 2) for graph in (False, True)
    ]
    prefix_args = argparse.Namespace(**vars(args))
    prefix_args.prefix = True
    prefix_cases = [
        _case_payload(prefix_args, "supplement_prefix", f"prefix_K{k}", k=k)
        for k in (0, 2)
    ]
    dynamic_args = argparse.Namespace(**vars(args))
    dynamic_args.max_new_tokens = min(args.max_new_tokens, 16)
    dynamic_cases = [
        _case_payload(dynamic_args, "supplement_dynamic", "dynamic")
    ]
    matrix = _run_cases(graph_cases + prefix_cases + dynamic_cases)
    results = _successful_results(matrix)
    graph_rows = [row for row in results
                  if str(row.get("case_label", "")).startswith("graph_")]
    prefix_rows = [row for row in results
                   if str(row.get("case_label", "")).startswith("prefix_")]
    dynamic_rows = [row for row in results
                    if row.get("case_label") == "dynamic"]
    return {
        "ok": matrix["ok"],
        "graph": graph_rows,
        "prefix": prefix_rows,
        "dynamic": dynamic_rows[0] if dynamic_rows else None,
        "failures": matrix["failures"],
    }


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    os.replace(temporary, path)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # 子进程入口：父进程会在 argv 里追加这两个参数。
    if "--case-json" in argv:
        return case_worker_main(_dispatch_case)

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
    parser.add_argument("--timeout-s", type=float, default=3600)
    parser.add_argument("--output-dir", default=str(ROOT / "benchmarks/results/step82"))
    args = parser.parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "git": _git(), "package": _package_hash(), "gpu": _parent_gpu_info(),
        "target": {"path": str(TARGET), "revision": TARGET_REVISION},
        "assistant": {"path": str(ASSISTANT), "revision": ASSISTANT_REVISION},
        "args": vars(args),
        "isolation": "one-config-per-subprocess",
    }
    _write(out_dir / "manifest.json", manifest)
    exit_code = 0
    if args.preflight:
        result = run_case(_worker_argv(), _case_payload(args, "preflight", "preflight"),
                          timeout_s=args.timeout_s)
        _write(out_dir / "preflight.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=1))
    if args.correctness or args.all:
        result = correctness(args)
        _write(out_dir / "correctness.json", result)
        print("correctness exact:", result["exact"])
        if not result["ok"]:
            print(f"correctness failures: {len(result['failures'])}", file=sys.stderr)
            exit_code = 1
    if args.perf or args.all:
        result = perf(args)
        _write(out_dir / "perf.json", result)
        for row in result["results"]:
            print(f"B={row['batch']} K={row['k']} "
                  f"tok/s={row['output_tokens_per_s']['median']} "
                  f"wall={row['wall_time_s']['median']} "
                  f"acc={row['acceptance_rate']['median']} "
                  f"graph={row['graph']}")
        if not result["ok"]:
            print(f"perf failures: {len(result['failures'])}", file=sys.stderr)
            exit_code = 1
    if args.supplement:
        result = supplement(args)
        _write(out_dir / "supplements.json", result)
        print("supplements written", out_dir / "supplements.json")
        if not result["ok"]:
            print(f"supplement failures: {len(result['failures'])}", file=sys.stderr)
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
