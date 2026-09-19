"""第五关 CPU 基线：含 Engine/假模型初始化、提交、推进与收集结果。"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

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
import step05

REQUESTS = [
    {"request_id": "A", "prompt_ids": [0], "max_new_tokens": 10},
    {"request_id": "B", "prompt_ids": [3], "max_new_tokens": 10},
    {"request_id": "C", "prompt_ids": [1], "max_new_tokens": 10},
]
EXPECTED = [
    {"request_id": "B", "output_ids": [4]},
    {"request_id": "C", "output_ids": [2, 3, 4]},
    {"request_id": "A", "output_ids": [1, 2, 3, 4]},
]


def run_static():
    # 每次测量都新建 Engine，避免已完成状态跨 trial 残留。
    # 当前构造函数会初始化假模型，这项成本明确计入总耗时。
    engine = step05.Engine()
    for request in REQUESTS:
        engine.add_request(request)
    completed = []
    while engine.has_unfinished_requests():
        completed.extend(engine.step())
    return completed


def check_workload():
    before = copy.deepcopy(REQUESTS)
    history = []
    init_count = 0
    original_init = step05.DummyModel.__init__
    original_forward = step05.DummyModel.forward

    def recording_init(model):
        nonlocal init_count
        init_count += 1
        original_init(model)

    def recording_forward(model, ids):
        history.append(ids.tolist())
        assert len(history) <= 8, "出现额外模型调用"
        return original_forward(model, ids)

    with patch.object(step05.DummyModel, "__init__", recording_init), \
         patch.object(step05.DummyModel, "forward", recording_forward):
        actual = run_static()
    assert actual == EXPECTED
    assert REQUESTS == before
    assert init_count == 1
    assert history == [[0], [3], [1], [0, 1], [1, 2], [0, 1, 2], [1, 2, 3], [0, 1, 2, 3]]
    return {"outputs": actual, "model_inputs": history, "model_initializations": init_count}


def main():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    source = PROJECT / "step05" / f"step05.py"
    report = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": sys.version, "python_executable": sys.executable, "torch": torch.__version__,
        "platform": platform.platform(), "device": "cpu", "num_threads": 1, "seed": 0,
        "requests": REQUESTS, "expected_new_tokens": 8, "completed_requests": 3,
        "boundary": "Engine and dummy model initialization + add_request + step/query loop + result collection; excludes imports, fixtures, assertions and tracing",
        "min_run_time_s": 1.0, "comparison_scope": "not directly comparable to step01-step04 generation-only timings",
    }
    try:
        report["workload_check"] = check_workload()
    except Exception as exc:
        report.update(status="correctness_failed", error=f"{type(exc).__name__}: {exc}")
        print("负载检查失败，不测速：", report["error"])
        code = 1
    else:
        result = Timer(stmt="run_static()", globals={"run_static": run_static}, num_threads=1,
                       label="step05: CPU full lifecycle, includes dummy initialization",
                       description="3 requests, 8 new tokens").blocked_autorange(min_run_time=1.0)
        assert run_static() == EXPECTED
        report.update(status="measured", median_us=result.median * 1e6, iqr_us=result.iqr * 1e6,
                      number_per_run=result.number_per_run, raw_times_s=result.raw_times,
                      has_warnings=result.has_warnings)
        print(result)
        code = 0
    assert hashlib.sha256(source.read_bytes()).hexdigest() == report["source_sha256"], "测量期间源码变化"
    directory = PROJECT / "benchmarks/results"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = directory / f"step05_{stamp}_{os.getpid()}.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("记录已保存：", output)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
