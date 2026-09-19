"""第二关验收与 CPU 串行三请求基线；不修改生成实现。"""
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
import step02

REQUESTS = [
    {"request_id": "A", "prompt_ids": [0], "max_new_tokens": 10},
    {"request_id": "B", "prompt_ids": [3], "max_new_tokens": 10},
    {"request_id": "C", "prompt_ids": [1], "max_new_tokens": 10},
]
EXPECTED = [
    {"request_id": "A", "output_ids": [1, 2, 3, 4]},
    {"request_id": "B", "output_ids": [4]},
    {"request_id": "C", "output_ids": [2, 3, 4]},
]


def check_correctness():
    mixed = [
        {"request_id": "short", "prompt_ids": [0], "max_new_tokens": 2},
        {"request_id": "zero", "prompt_ids": [0], "max_new_tokens": 0},
        {"request_id": "end", "prompt_ids": [3], "max_new_tokens": 5},
    ]
    cases = [
        (REQUESTS, EXPECTED, [[0], [0, 1], [0, 1, 2], [0, 1, 2, 3],
                              [3], [1], [1, 2], [1, 2, 3]]),
        ([], [], []),
        (mixed, [{"request_id": "short", "output_ids": [1, 2]},
                 {"request_id": "zero", "output_ids": []},
                 {"request_id": "end", "output_ids": [4]}], [[0], [0, 1], [3]]),
    ]
    records = []
    for requests, expected, expected_history in cases:
        inputs = copy.deepcopy(requests)
        seen = []
        forward = step02.model.forward

        def recording_forward(ids):
            seen.append(ids.tolist() if isinstance(ids, torch.Tensor) else list(ids))
            return forward(ids)

        with patch.object(step02.model, "forward", side_effect=recording_forward), \
             patch.object(step02.DummyModel, "__init__", side_effect=AssertionError("重复初始化模型")):
            actual = step02.generate_many(inputs)
        assert isinstance(actual, list) and actual == expected, (actual, expected)
        assert all(type(token) is int for item in actual for token in item["output_ids"])
        assert inputs == requests, "修改了原始请求或其中的 prompt"
        assert seen == expected_history, (seen, expected_history)
        records.append({"requests": requests, "outputs": actual,
                        "model_inputs": seen, "model_calls": len(seen), "passed": True})

    # step02 复制了第一关实现，因此也检查它自己的单请求接口。
    for prompt, budget, expected in [([0], 10, [1, 2, 3, 4]), ([0], 2, [1, 2]),
                                     ([2], 10, [3, 4]), ([0], 0, []), ([0, 1], 1, [2])]:
        original = prompt.copy()
        actual = step02.generate(prompt, budget)
        assert type(actual) is list and actual == expected
        assert all(type(token) is int for token in actual)
        assert prompt == original
    return records


def main():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    source = PROJECT / "step02" / f"step02.py"
    report = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": sys.version, "python_executable": sys.executable,
        "torch": torch.__version__, "platform": platform.platform(),
        "device": str(step02.model.device), "num_threads": 1, "seed": 0,
        "requests": REQUESTS, "expected_new_tokens": 8, "completed_requests": 3,
        "workload": "step02.generate_many(requests)", "min_run_time_s": 1.0,
        "boundary": "complete three-request call; excludes request fixture creation, imports, checks, printing",
    }
    try:
        assert step02.model.device.type == "cpu"
        report["checks"] = check_correctness()
    except Exception as exc:
        report.update(status="correctness_failed", error=f"{type(exc).__name__}: {exc}")
        print("正确性检查失败，未测速：", report["error"])
        exit_code = 1
    else:
        # 补丁已退出，计时范围不包含验收用的记录/断言开销。
        result = Timer(
            stmt="generate_many(requests)",
            globals={"generate_many": step02.generate_many, "requests": REQUESTS},
            num_threads=1, label="step02: CPU serial requests",
            description="A/B/C: 3 requests, 8 new tokens including EOS",
        ).blocked_autorange(min_run_time=1.0)
        assert step02.generate_many(REQUESTS) == EXPECTED
        assert hashlib.sha256(source.read_bytes()).hexdigest() == report["source_sha256"], "测速期间源码发生变化"
        report.update(status="measured", median_us=result.median * 1e6,
                      iqr_us=result.iqr * 1e6, number_per_run=result.number_per_run,
                      raw_times_s=result.raw_times, has_warnings=result.has_warnings)
        print(result)
        exit_code = 0

    directory = PROJECT / "benchmarks/results"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = directory / f"step02_{stamp}_{os.getpid()}.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("记录已保存：", output)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
