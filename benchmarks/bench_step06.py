"""第六关 CPU 生命周期基线；不修改学生实现，计时前检查实际执行轨迹。"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import copy
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.benchmark import Timer

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0]=[str(p) for p in PROJECT.glob("step[0-9][0-9]") if p.is_dir()]+[str(PROJECT)]
import step06

REQUESTS = [
    {"request_id": "A", "prompt_ids": [0], "max_new_tokens": 10},
    {"request_id": "B", "prompt_ids": [3], "max_new_tokens": 10},
    {"request_id": "C", "prompt_ids": [1], "max_new_tokens": 10},
]
TOKENS = {"A": [1, 2, 3, 4], "B": [4], "C": [2, 3, 4]}
ORDER = {1: "ABC", 2: "BAC", 3: "BCA"}
HISTORY = {
    1: [[0], [0, 1], [0, 1, 2], [0, 1, 2, 3], [3], [1], [1, 2], [1, 2, 3]],
    2: [[0], [3], [0, 1], [1], [0, 1, 2], [1, 2], [0, 1, 2, 3], [1, 2, 3]],
    3: [[0], [3], [1], [0, 1], [1, 2], [0, 1, 2], [1, 2, 3], [0, 1, 2, 3]],
}
CALLS_PER_STEP = {1: [1] * 8, 2: [2] * 4, 3: [3, 2, 2, 1]}


def expected(capacity):
    return [{"request_id": name, "output_ids": TOKENS[name]} for name in ORDER[capacity]]


def run_static(capacity):
    # 每个 trial 都使用全新 Engine，初始化成本计入；输入 fixture 不计入。
    engine = step06.Engine(max_num_seqs=capacity)
    for request in REQUESTS:
        engine.add_request(request)
    completed = []
    while engine.has_unfinished_requests():
        completed.extend(engine.step())
    return completed


def check_workload(capacity):
    before = copy.deepcopy(REQUESTS)
    history, calls_per_step = [], []
    init_count = 0
    original_init = step06.DummyModel.__init__
    original_forward = step06.DummyModel.forward
    original_step = step06.Engine.step

    def recording_init(model):
        nonlocal init_count
        init_count += 1
        original_init(model)

    def recording_forward(model, ids):
        history.append(ids.tolist())
        assert len(history) <= 8, "出现额外模型调用"
        return original_forward(model, ids)

    def recording_step(engine):
        assert len(calls_per_step) < 8, "出现额外 step 或无法结束"
        start = len(history)
        result = original_step(engine)
        calls_per_step.append(len(history) - start)
        return result

    with patch.object(step06.DummyModel, "__init__", recording_init), \
         patch.object(step06.DummyModel, "forward", recording_forward), \
         patch.object(step06.Engine, "step", recording_step):
        actual = run_static(capacity)
    assert actual == expected(capacity)
    assert REQUESTS == before
    assert init_count == 1
    assert history == HISTORY[capacity]
    assert calls_per_step == CALLS_PER_STEP[capacity]
    return {"outputs": actual, "model_inputs": history,
            "model_initializations": init_count, "calls_per_step": calls_per_step}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capacity", type=int, choices=[1, 2, 3], default=2)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    source = PROJECT / "step06" / f"step06.py"
    report = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": sys.version, "python_executable": sys.executable, "torch": torch.__version__,
        "platform": platform.platform(), "device": "cpu", "num_threads": 1, "seed": 0,
        "max_num_seqs": args.capacity, "callback": None, "requests": REQUESTS,
        "expected_new_tokens": 8, "completed_requests": 3,
        "boundary": "Engine and dummy model initialization + add_request + step/query loop + result collection; excludes imports, fixtures, assertions and tracing",
        "min_run_time_s": 1.0,
        "comparison_scope": "same lifecycle boundary as step05; not process cold start or GPU throughput",
    }
    try:
        report["workload_check"] = check_workload(args.capacity)
    except Exception as exc:
        report.update(status="correctness_failed", error=f"{type(exc).__name__}: {exc}")
        print("负载检查失败，不测速：", report["error"])
        code = 1
    else:
        result = Timer(stmt="run_static(capacity)",
                       globals={"run_static": run_static, "capacity": args.capacity}, num_threads=1,
                       label=f"step06: CPU full lifecycle, capacity={args.capacity}",
                       description="3 requests, 8 new tokens, includes dummy initialization"
                       ).blocked_autorange(min_run_time=1.0)
        assert run_static(args.capacity) == expected(args.capacity)
        report.update(status="measured", median_us=result.median * 1e6, iqr_us=result.iqr * 1e6,
                      number_per_run=result.number_per_run, raw_times_s=result.raw_times,
                      has_warnings=result.has_warnings)
        print(result)
        code = 0
    assert hashlib.sha256(source.read_bytes()).hexdigest() == report["source_sha256"], "测量期间源码变化"
    directory = PROJECT / "benchmarks/results"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = directory / f"step06_cap{args.capacity}_{stamp}_{os.getpid()}.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("记录已保存：", output)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
