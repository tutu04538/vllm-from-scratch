"""第三关：先验收轮流执行契约，通过后测 CPU 三请求耗时。"""
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
sys.path.insert(0, str(PROJECT))
import step03

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
    cases = [
        ("A/B/C", REQUESTS, EXPECTED,
         [[0], [3], [1], [0, 1], [1, 2], [0, 1, 2], [1, 2, 3], [0, 1, 2, 3]]),
        ("empty", [], [], []),
        ("mixed_budget", [
            {"request_id": "short", "prompt_ids": [0], "max_new_tokens": 2},
            {"request_id": "zero", "prompt_ids": [0], "max_new_tokens": 0},
            {"request_id": "end", "prompt_ids": [3], "max_new_tokens": 5}], [
            {"request_id": "short", "output_ids": [1, 2]},
            {"request_id": "zero", "output_ids": []},
            {"request_id": "end", "output_ids": [4]}], [[0], [3], [0, 1]]),
        ("all_zero", [
            {"request_id": "Z0", "prompt_ids": [0], "max_new_tokens": 0},
            {"request_id": "Z1", "prompt_ids": [1], "max_new_tokens": 0}], [
            {"request_id": "Z0", "output_ids": []},
            {"request_id": "Z1", "output_ids": []}], []),
    ]
    # 单请求时也应兼容前两关的生成结果。
    for i, (prompt, budget, outputs) in enumerate([
        ([0], 10, [1, 2, 3, 4]), ([0], 2, [1, 2]), ([2], 10, [3, 4]),
        ([0], 0, []), ([0, 1], 1, [2]),
    ]):
        cases.append((f"single_{i}",
                      [{"request_id": "S", "prompt_ids": prompt, "max_new_tokens": budget}],
                      [{"request_id": "S", "output_ids": outputs}],
                      [prompt + outputs[:j] for j in range(len(outputs))]))

    records = []
    for name, requests, expected, expected_history in cases:
        inputs = copy.deepcopy(requests)
        seen = []
        forward = step03.model.forward

        def recording_forward(ids):
            seen.append(ids.tolist() if isinstance(ids, torch.Tensor) else list(ids))
            assert len(seen) <= len(expected_history) + 10, "模型调用过多，停止异常执行"
            return forward(ids)

        record = {"name": name, "requests": requests, "expected": expected,
                  "expected_model_inputs": expected_history}
        try:
            with patch.object(step03.model, "forward", side_effect=recording_forward), \
                 patch.object(step03.DummyModel, "__init__", side_effect=AssertionError("重复初始化模型")):
                actual = step03.generate_many(inputs)
            record["actual"] = actual
            assert isinstance(actual, list) and actual == expected, "返回结果不符合约定"
            assert all(type(token) is int for item in actual for token in item["output_ids"])
            assert inputs == requests, "修改了输入"
            assert seen == expected_history, "模型调用顺序/次数/历史不符合约定"
        except Exception as exc:
            record.update(passed=False, error=f"{type(exc).__name__}: {exc}")
        else:
            record["passed"] = True
        record.update(model_inputs=seen, model_calls=len(seen))
        records.append(record)
        print(f"{'PASS' if record['passed'] else 'FAIL'} {name}: {record.get('actual')}")
    return records


def main():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    source = PROJECT / "step03.py"
    report = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": sys.version, "python_executable": sys.executable,
        "torch": torch.__version__, "platform": platform.platform(),
        "device": str(step03.model.device), "num_threads": 1, "seed": 0,
        "requests": REQUESTS, "expected_new_tokens": 8, "completed_requests": 3,
        "workload": "step03.generate_many(requests)", "min_run_time_s": 1.0,
        "boundary": "complete three-request call; excludes request fixture creation, imports, checks, printing",
    }
    assert step03.model.device.type == "cpu"
    report["checks"] = check_correctness()
    if all(case["passed"] for case in report["checks"]):
        result = Timer(
            stmt="generate_many(requests)",
            globals={"generate_many": step03.generate_many, "requests": REQUESTS},
            num_threads=1, label="step03: CPU round-robin requests",
            description="A/B/C: 3 requests, 8 new tokens including EOS",
        ).blocked_autorange(min_run_time=1.0)
        assert step03.generate_many(REQUESTS) == EXPECTED
        report.update(status="measured", median_us=result.median * 1e6, iqr_us=result.iqr * 1e6,
                      number_per_run=result.number_per_run, raw_times_s=result.raw_times,
                      has_warnings=result.has_warnings)
        print(result)
        exit_code = 0
    else:
        report["status"] = "correctness_failed"
        print("正确性未全部通过，不进行性能测量。")
        exit_code = 1
    assert hashlib.sha256(source.read_bytes()).hexdigest() == report["source_sha256"], "执行期间源码发生变化"
    directory = PROJECT / "benchmarks/results"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = directory / f"step03_{stamp}_{os.getpid()}.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("记录已保存：", output)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
