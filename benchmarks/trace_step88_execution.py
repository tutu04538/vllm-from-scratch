"""需求 88：双方相同边界的 profiler 采集。

只选代表配置：

    B=1 K=2 eager   × mini/vLLM
    B=1 K=2 FULL 图 × mini/vLLM
    B=1 K=2 FULL 图 @ 1024-token 上下文 × mini/vLLM（暴露分页 attention）

采集边界统一为**预填完成后的稳态 decode**：prefill 在 profiler 外跑完，只 profile
接下来固定的 decode step 数。这样不会拿“mini 8 个 decode”对比“vLLM 含 prefill 的
9 次迭代”。

mini 用 torch.profiler；vLLM 用 `LLM.start_profile()/stop_profile()`，因为它的
EngineCore 在独立进程。两边原始 trace 都写到 gitignored 的 results 目录，只把统一
摘要写到 summary.json。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BENCH = Path(__file__).resolve().parent
for path in (ROOT, BENCH):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_runner import case_worker_main, run_matrix  # noqa: E402
from compare_step88_execution import (  # noqa: E402
    _load_prompt_ids, _tokenizer, _vllm_build, _vllm_graph_mode, _vllm_rpc,
)
from step88_common import make_case  # noqa: E402
from step88_runtime import worker_snapshot  # noqa: E402

RESULT_DIR = ROOT / "benchmarks" / "results" / "step88" / "trace"
TRACE_CASES = (
    ("minivllm", 1, 2, False, "short"),
    ("minivllm", 1, 2, True, "short"),
    ("vllm", 1, 2, False, "short"),
    ("vllm", 1, 2, True, "short"),
    ("minivllm", 1, 2, True, "context"),
    ("vllm", 1, 2, True, "context"),
)


def trace_cases(decode_steps: int = 8) -> list[dict[str, Any]]:
    out = []
    for engine, batch, k, graph, workload in TRACE_CASES:
        case = make_case(engine, batch, k, graph, workload)
        suffix = "" if workload == "short" else "_context"
        case["label"] = f"{engine}_B{batch}_K{k}_{'graph' if graph else 'eager'}{suffix}_decode"
        case["trace_phase"] = "decode"
        case["decode_steps"] = int(decode_steps)
        out.append(case)
    return out


def _device_name(value: Any) -> str:
    name = getattr(value, "name", None)
    return name if name is not None else str(value)


def summarize_torch_events(events: list[Any]) -> dict[str, Any]:
    """把 torch.profiler events 归一成 kernel/launch 摘要。"""
    kernels: dict[str, dict[str, float]] = {}
    cpu_launches = 0
    graph_launches = 0
    cpu_phase: dict[str, dict[str, float]] = {}
    gpu_events = 0
    cpu_events = 0
    memcpy = {"d2h": 0, "h2d": 0, "d2d": 0}
    sync = {"event": 0, "device": 0}
    for event in events:
        device = _device_name(getattr(event, "device_type", ""))
        name = str(getattr(event, "key", getattr(event, "name", "")))
        count = int(getattr(event, "count", 0) or 0)
        if "Memcpy DtoH" in name:
            memcpy["d2h"] += count
        elif "Memcpy HtoD" in name:
            memcpy["h2d"] += count
        elif "Memcpy DtoD" in name:
            memcpy["d2d"] += count
        if name.startswith("cudaEventSynchronize"):
            sync["event"] += count
        if name.startswith("cudaDeviceSynchronize"):
            sync["device"] += count
        if name.startswith("mini/"):
            # record_function 的 span 可能同时带 CPU/GPU 时间；它是阶段标注，
            # 不是 CUDA kernel，不能混进 kernel 计数和 top kernels。
            item = cpu_phase.setdefault(
                name, {"count": 0, "cpu_us": 0.0, "device_span_us": 0.0})
            item["count"] += count
            item["cpu_us"] += float(
                getattr(event, "self_cpu_time_total",
                        getattr(event, "cpu_time_total", 0.0)) or 0.0)
            item["device_span_us"] += float(
                getattr(event, "self_device_time_total", 0.0) or 0.0)
            continue
        if device == "CUDA":
            gpu_events += count
            if count <= 0:
                continue
            item = kernels.setdefault(
                name, {"count": 0, "device_time_us": 0.0})
            item["count"] += count
            item["device_time_us"] += float(
                getattr(event, "self_device_time_total", 0.0) or 0.0)
        else:
            cpu_events += count
            if name.startswith("cudaLaunchKernel"):
                cpu_launches += count
            if name.startswith("cudaGraphLaunch"):
                graph_launches += count
    top = sorted(kernels.items(),
                 key=lambda item: item[1]["device_time_us"], reverse=True)[:25]
    return {
        "gpu_kernel_events": gpu_events,
        "cpu_event_count": cpu_events,
        "cpu_kernel_launch_events": cpu_launches,
        "cuda_graph_launch_events": graph_launches,
        "gpu_kernel_time_us": sum(item["device_time_us"]
                                  for item in kernels.values()),
        "distinct_gpu_kernels": len(kernels),
        "top_gpu_kernels": [
            {"name": name, **values} for name, values in top],
        "cpu_phases": [
            {"name": name, **values}
            for name, values in sorted(cpu_phase.items())],
        "gpu_memcpy_events": memcpy,
        "cuda_sync_events": sync,
    }


def summarize_chrome_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    """解析 vLLM/torch 都兼容的 Chrome trace 事件。"""
    kernels: dict[str, dict[str, float]] = {}
    cpu_launches = 0
    graph_launches = 0
    gpu_events = 0
    cpu_event_count = 0
    memcpy = {"d2h": 0, "h2d": 0, "d2d": 0}
    sync = {"event": 0, "device": 0}
    for event in events:
        if event.get("ph") != "X":
            continue
        name = str(event.get("name", ""))
        cat = str(event.get("cat", ""))
        duration = float(event.get("dur", 0.0) or 0.0)
        if cat == "gpu_memcpy" or "Memcpy " in name:
            if "DtoH" in name:
                memcpy["d2h"] += 1
            elif "HtoD" in name:
                memcpy["h2d"] += 1
            elif "DtoD" in name:
                memcpy["d2d"] += 1
        if name.startswith("cudaEventSynchronize"):
            sync["event"] += 1
        if name.startswith("cudaDeviceSynchronize"):
            sync["device"] += 1
        if cat == "kernel":
            gpu_events += 1
            item = kernels.setdefault(
                name, {"count": 0, "device_time_us": 0.0})
            item["count"] += 1
            item["device_time_us"] += duration
        elif cat in {"cuda_runtime", "cuda_driver", "runtime"} or \
                name.startswith(("cudaLaunchKernel", "cudaGraphLaunch")):
            cpu_event_count += 1
            if name.startswith("cudaLaunchKernel"):
                cpu_launches += 1
            if name.startswith("cudaGraphLaunch"):
                graph_launches += 1
        elif cat in {"cpu_op", "user_annotation", "python_function"}:
            cpu_event_count += 1
    top = sorted(kernels.items(),
                 key=lambda item: item[1]["device_time_us"], reverse=True)[:25]
    return {
        "gpu_kernel_events": gpu_events,
        "cpu_event_count": cpu_event_count,
        "cpu_kernel_launch_events": cpu_launches,
        "cuda_graph_launch_events": graph_launches,
        "gpu_kernel_time_us": sum(item["device_time_us"]
                                  for item in kernels.values()),
        "distinct_gpu_kernels": len(kernels),
        "top_gpu_kernels": [
            {"name": name, **values} for name, values in top],
        "gpu_memcpy_events": memcpy,
        "cuda_sync_events": sync,
    }


def _install_mini_phase_hooks(runner) -> None:
    """只在 trace 进程安装 CPU 阶段标注，不改生产代码。"""
    from torch.profiler import record_function

    def wrap(obj, attr, label):
        original = getattr(obj, attr, None)
        if original is None:
            return
        def wrapped(*args, **kwargs):
            with record_function(label):
                return original(*args, **kwargs)
        setattr(obj, attr, wrapped)

    wrap(runner, "prepare_inputs", "mini/input_prepare")
    wrap(runner, "execute_model", "mini/target_execute")
    wrap(runner, "sample", "mini/target_sample")
    speculator = getattr(runner, "speculator", None)
    if speculator is not None:
        wrap(speculator, "propose", "mini/draft_propose")


def _prefill_mini(engine, request_id: str, prompt: list[int]) -> int:
    from minivllm import SamplingParams

    engine.add_request(
        request_id, list(prompt),
        SamplingParams(max_tokens=128, temperature=0.0, ignore_eos=True))
    steps = 0
    seen = False
    for _ in range(256):
        if not engine.has_unfinished_requests():
            break
        outputs = engine.step()
        steps += 1
        if outputs:
            seen = True
            break
    if not seen:
        raise RuntimeError("prefill 没有产生任何输出")
    return steps


def _mini_trace(case: dict[str, Any], raw_dir: Path) -> dict[str, Any]:
    import torch
    from torch.profiler import ProfilerActivity, profile, record_function

    from check_step82_real_spec import _build_engine

    args = SimpleNamespace(**case["args"])
    args.graph = bool(case["graph"])
    args.prefix = False
    args.device = "cuda"
    args.batch = int(case["batch"])
    tokenizer = _tokenizer()
    prompt = _load_prompt_ids(tokenizer, case["workload"])
    engine, runner, build_s = _build_engine(
        args, k=int(case["k"]), batch=int(case["batch"]),
        graph=bool(case["graph"]))
    raw_dir.mkdir(parents=True, exist_ok=True)
    try:
        prefill_steps = _prefill_mini(engine, "trace", prompt)
        _install_mini_phase_hooks(runner)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=False, profile_memory=False,
                with_stack=False) as prof:
            decode_steps = 0
            output_events = []
            for _ in range(int(case["decode_steps"])):
                if not engine.has_unfinished_requests():
                    break
                with record_function("mini/engine_step"):
                    outputs = engine.step()
                decode_steps += 1
                output_events.extend(
                    len(item.token_ids) for item in outputs)
        torch.cuda.synchronize()
        wall = time.perf_counter() - started
        raw_path = raw_dir / f"{case['label']}.json"
        prof.export_chrome_trace(str(raw_path))
        summary = summarize_torch_events(list(prof.events()))
        manager = getattr(runner, "target_cudagraph_manager", None)
        mode = getattr(runner, "cudagraph_mode", None)
        summary.update({
            "engine": "minivllm",
            "label": case["label"],
            "workload": case["workload"],
            "graph": bool(case["graph"]),
            "actual_graph_mode": (
                mode.name if hasattr(mode, "name") else str(mode)),
            "prefill_steps": prefill_steps,
            "decode_steps": decode_steps,
            "decode_output_events": len(output_events),
            "profile_wall_s": wall,
            "build_s": build_s,
            "target_graphs": (
                len(manager.graphs) if manager is not None else 0),
            "raw_trace": str(raw_path),
        })
        return summary
    finally:
        engine.shutdown()


def _prefill_vllm(llm, prompt: list[int]) -> tuple[int, str]:
    from vllm import SamplingParams

    engine = llm.llm_engine
    engine.add_request(
        "trace", list(prompt),
        SamplingParams(max_tokens=128, temperature=0.0, ignore_eos=True))
    steps = 0
    seen = False
    for _ in range(256):
        if not engine.has_unfinished_requests():
            break
        outputs = engine.step()
        steps += 1
        if outputs:
            seen = True
            break
    if not seen:
        raise RuntimeError("vLLM prefill 没有产生任何输出")
    return steps, "trace"


def _vllm_trace(case: dict[str, Any], raw_dir: Path) -> dict[str, Any]:
    import torch

    raw_dir.mkdir(parents=True, exist_ok=True)
    llm, build_s = _vllm_build(case, profiler_dir=str(raw_dir))
    try:
        prompt = _load_prompt_ids(_tokenizer(), case["workload"])
        prefill_steps, request_id = _prefill_vllm(llm, prompt)
        snapshot = _vllm_rpc(llm, worker_snapshot)
        torch.cuda.synchronize()
        started = time.perf_counter()
        llm.start_profile(f"step88_{case['label']}")
        engine = llm.llm_engine
        decode_steps = 0
        output_events = 0
        for _ in range(int(case["decode_steps"])):
            if not engine.has_unfinished_requests():
                break
            outputs = engine.step()
            decode_steps += 1
            output_events += len(outputs)
        llm.stop_profile()
        torch.cuda.synchronize()
        wall = time.perf_counter() - started
        trace_paths = sorted(raw_dir.rglob("*.json"))
        events: list[dict[str, Any]] = []
        for path in trace_paths:
            try:
                payload = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            events.extend(payload.get("traceEvents", []))
        summary = summarize_chrome_events(events)
        summary.update({
            "engine": "vllm",
            "label": case["label"],
            "workload": case["workload"],
            "graph": bool(case["graph"]),
            "actual_graph_mode": _vllm_graph_mode(snapshot),
            "prefill_steps": prefill_steps,
            "decode_steps": decode_steps,
            "decode_output_events": output_events,
            "profile_wall_s": wall,
            "build_s": build_s,
            "target_graphs": (
                ((snapshot.get("graph") or {}).get("target") or {})
                .get("num_graphs", 0)),
            "draft_graphs": (
                ((snapshot.get("graph") or {}).get("draft") or {})),
            "raw_traces": [str(path) for path in trace_paths],
        })
        return summary
    finally:
        del llm


def _dispatch(case: dict[str, Any]) -> dict[str, Any]:
    raw_dir = Path(case["raw_dir"])
    if case["engine"] == "minivllm":
        return _mini_trace(case, raw_dir)
    if case["engine"] == "vllm":
        return _vllm_trace(case, raw_dir)
    raise ValueError(f"未知 engine={case['engine']!r}")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--case-json" in argv:
        return case_worker_main(_dispatch)

    parser = argparse.ArgumentParser()
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--timeout-s", type=float, default=3600)
    parser.add_argument("--output-dir", default=str(RESULT_DIR))
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.decode_steps < 1:
        raise SystemExit("--decode-steps 必须 >= 1")
    cases = trace_cases(args.decode_steps)
    if args.labels:
        wanted = set(args.labels)
        cases = [case for case in cases if case["label"] in wanted]
        unknown = wanted - {case["label"] for case in cases}
        if unknown:
            raise SystemExit(f"未知 label：{sorted(unknown)}")
    out_dir = Path(args.output_dir)
    if args.dry_run:
        print(json.dumps({"ok": True, "cases": cases}, indent=1))
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    child_env = {
        "VLLM_WSL2_ENABLE_PIN_MEMORY": "1",
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    for case in cases:
        case["raw_dir"] = str(out_dir / "raw" / case["label"])
    matrix = run_matrix(
        [sys.executable, str(Path(__file__).resolve())],
        cases, timeout_s=args.timeout_s, env=child_env)
    summary = {
        "ok": matrix["ok"],
        "decode_steps": args.decode_steps,
        "cases": [record["result"] for record in matrix["cases"]
                  if record["status"] == "ok"],
        "failures": matrix["failures"],
        "matrix": matrix,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1, default=str))
    for row in summary["cases"]:
        print(f"{row['label']:<44} "
              f"kernels={row['gpu_kernel_events']:>5} "
              f"launch={row['cpu_kernel_launch_events']:>4} "
              f"graph={row['cuda_graph_launch_events']:>3} "
              f"gpu_ms={row['gpu_kernel_time_us'] / 1000:>8.2f}")
    if not summary["ok"]:
        print(f"trace failures: {len(matrix['failures'])}", file=sys.stderr)
        return 1
    print(f"trace summary: {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
