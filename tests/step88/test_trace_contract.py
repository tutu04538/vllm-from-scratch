"""88：trace 代表矩阵与 kernel 摘要（不需要 GPU）。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from benchmarks.trace_step88_execution import (
    summarize_chrome_events,
    trace_cases,
)

ROOT = Path(__file__).resolve().parents[2]


def test_trace_matrix_has_six_representative_decode_cases():
    cases = trace_cases(8)
    assert len(cases) == 6
    assert {case["engine"] for case in cases} == {"minivllm", "vllm"}
    assert {(case["graph"], case["workload"]) for case in cases} == {
        (False, "short"), (True, "short"), (True, "context")}
    assert all(case["trace_phase"] == "decode" for case in cases)
    assert all(case["decode_steps"] == 8 for case in cases)


def test_chrome_summary_counts_graph_launch_and_kernels():
    events = [
        {"ph": "X", "cat": "kernel", "name": "kernel_unified_attention",
         "dur": 100},
        {"ph": "X", "cat": "kernel", "name": "kernel_unified_attention",
         "dur": 300},
        {"ph": "X", "cat": "kernel", "name": "reshape_and_cache_kernel",
         "dur": 50},
        {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel",
         "dur": 10},
        {"ph": "X", "cat": "cuda_runtime", "name": "cudaGraphLaunch",
         "dur": 5},
        {"ph": "X", "cat": "cpu_op", "name": "llm_engine step", "dur": 7},
        {"ph": "M", "cat": "process_name", "name": "EngineCore"},
    ]
    summary = summarize_chrome_events(events)
    assert summary["gpu_kernel_events"] == 3
    assert summary["gpu_kernel_time_us"] == 450
    assert summary["cpu_kernel_launch_events"] == 1
    assert summary["cuda_graph_launch_events"] == 1
    assert summary["cpu_event_count"] == 3
    assert summary["top_gpu_kernels"][0]["name"] == "kernel_unified_attention"
    assert summary["top_gpu_kernels"][0]["count"] == 2


def test_trace_script_dry_run_lists_six_cases():
    completed = subprocess.run(
        [sys.executable, str(ROOT / "benchmarks/trace_step88_execution.py"),
         "--dry-run"],
        cwd=ROOT, text=True, capture_output=True, check=True,
        env={**os.environ, "PYTHONPATH": str(ROOT)})
    payload = json.loads(completed.stdout)
    assert payload["ok"]
    assert len(payload["cases"]) == 6


def test_torch_summary_excludes_phase_annotations_from_kernels():
    from types import SimpleNamespace
    from benchmarks.trace_step88_execution import summarize_torch_events

    events = [
        SimpleNamespace(device_type="CUDA", key="kernel_unified_attention",
                        count=2, self_device_time_total=300.0,
                        self_cpu_time_total=0.0),
        SimpleNamespace(device_type="CPU", key="cudaLaunchKernel",
                        count=1, self_device_time_total=0.0,
                        self_cpu_time_total=10.0),
        SimpleNamespace(device_type="CUDA", key="mini/target_execute",
                        count=1, self_device_time_total=999.0,
                        self_cpu_time_total=50.0),
    ]
    summary = summarize_torch_events(events)
    assert summary["gpu_kernel_events"] == 2
    assert summary["gpu_kernel_time_us"] == 300.0
    assert summary["cpu_kernel_launch_events"] == 1
    assert [item["name"] for item in summary["top_gpu_kernels"]] == [
        "kernel_unified_attention"]
    assert summary["cpu_phases"][0]["name"] == "mini/target_execute"
