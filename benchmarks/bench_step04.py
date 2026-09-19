"""第四关：回调时机验收；通过后测有/无空操作回调的 CPU 耗时。"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import copy
import hashlib
import inspect
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
import step04
import bench_step03 as previous_checks

REQUESTS = previous_checks.REQUESTS
EXPECTED = previous_checks.EXPECTED


def output(request_id, tokens):
    return {"request_id": request_id, "output_ids": tokens}


def event(request_id, tokens, calls):
    return {"result": output(request_id, tokens), "model_calls": calls}


def check_callbacks():
    # 正式接口检查，不因诊断时使用旧接口而放宽要求。
    records = []
    probe = []
    try:
        result = step04.generate_many(
            [{"request_id": "P", "prompt_ids": [3], "max_new_tokens": 1}],
            on_finished=lambda item: probe.append(copy.deepcopy(item)),
        )
        assert result == [output("P", [4])] and probe == result
    except Exception as exc:
        records.append({"name": "public_callback_contract", "passed": False,
                        "error": f"{type(exc).__name__}: {exc}"})
    else:
        records.append({"name": "public_callback_contract", "passed": True})

    # 兼容旧名字仅为继续诊断时机：会明确记录，不作为正式接口通过的证据。
    params = inspect.signature(step04.generate_many).parameters
    keyword = "on_finished" if "on_finished" in params else "on_finish"
    cases = [
        ("A/B/C", REQUESTS, EXPECTED,
         [[0], [3], [1], [0, 1], [1, 2], [0, 1, 2], [1, 2, 3], [0, 1, 2, 3]],
         [event("B", [4], 2), event("C", [2, 3, 4], 7), event("A", [1, 2, 3, 4], 8)]),
        ("mixed_zero", [
            {"request_id": "short", "prompt_ids": [0], "max_new_tokens": 2},
            {"request_id": "zero", "prompt_ids": [0], "max_new_tokens": 0},
            {"request_id": "end", "prompt_ids": [3], "max_new_tokens": 5}],
         [output("short", [1, 2]), output("zero", []), output("end", [4])],
         [[0], [3], [0, 1]],
         [event("zero", [], 0), event("end", [4], 2), event("short", [1, 2], 3)]),
        ("budget_before_other_request", [
            {"request_id": "A", "prompt_ids": [0], "max_new_tokens": 1},
            {"request_id": "B", "prompt_ids": [1], "max_new_tokens": 2}],
         [output("A", [1]), output("B", [2, 3])], [[0], [1], [1, 2]],
         [event("A", [1], 1), event("B", [2, 3], 3)]),
        ("all_zero", [
            {"request_id": "Z0", "prompt_ids": [0], "max_new_tokens": 0},
            {"request_id": "Z1", "prompt_ids": [1], "max_new_tokens": 0}],
         [output("Z0", []), output("Z1", [])], [], [event("Z0", [], 0), event("Z1", [], 0)]),
        ("empty", [], [], [], []),
    ]
    for name, requests, expected, expected_history, expected_events in cases:
        inputs = copy.deepcopy(requests)
        seen, events, arg_counts = [], [], []
        forward = step04.model.forward

        def recording_forward(ids):
            seen.append(ids.tolist() if isinstance(ids, torch.Tensor) else list(ids))
            assert len(seen) <= len(expected_history) + 10, "模型调用过多"
            return forward(ids)

        def callback(*args):
            arg_counts.append(len(args))
            if len(args) == 1:
                item = args[0]
            elif len(args) == 2:
                item = output(args[0], args[1])  # 仅诊断旧版两参数回调的时机。
            else:
                raise AssertionError("回调参数数量错误")
            events.append({"result": copy.deepcopy(item), "model_calls": len(seen)})

        record = {"name": name, "tested_keyword": keyword, "requests": requests,
                  "expected_events": expected_events, "expected_model_inputs": expected_history}
        try:
            with patch.object(step04.model, "forward", side_effect=recording_forward), \
                 patch.object(step04.DummyModel, "__init__", side_effect=AssertionError("重复初始化模型")):
                actual = step04.generate_many(inputs, **{keyword: callback})
            record["actual"] = actual
            assert actual == expected and inputs == requests, "结果/输入不变约定不满足"
            assert seen == expected_history, "模型调用历史/次数错误"
            assert events == expected_events, "回调顺序/时机/次数/内容错误"
        except Exception as exc:
            record.update(passed=False, error=f"{type(exc).__name__}: {exc}")
        else:
            record["passed"] = True
        record.update(events=events, model_inputs=seen, callback_arg_counts=arg_counts)
        records.append(record)
        print(f"{'PASS' if record['passed'] else 'FAIL'} callback timing {name}: {events}")
    return records


def noop(result):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["no_callback", "callback"], default="callback")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    source = PROJECT / "step04" / f"step04.py"
    report = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "regression_helper_sha256": hashlib.sha256(Path(previous_checks.__file__).read_bytes()).hexdigest(),
        "mode": args.mode, "python": sys.version, "torch": torch.__version__,
        "python_executable": sys.executable, "platform": platform.platform(),
        "device": str(step04.model.device), "num_threads": 1, "requests": REQUESTS,
        "expected_new_tokens": 8, "completed_requests": 3, "min_run_time_s": 1.0,
        "boundary": "complete three-request call; excludes fixtures, imports, checks, tracing, printing",
    }
    assert step04.model.device.type == "cpu"
    with patch.object(previous_checks, "step03", step04):
        report["no_callback_regression"] = previous_checks.check_correctness()
    report["callback_checks"] = check_callbacks()
    checks = report["no_callback_regression"] + report["callback_checks"]
    if all(c["passed"] for c in checks):
        stmt = ("generate_many(requests)" if args.mode == "no_callback" else
                "generate_many(requests, on_finished=noop)")
        result = Timer(stmt=stmt, globals={"generate_many": step04.generate_many,
                       "requests": REQUESTS, "noop": noop}, num_threads=1,
                       label=f"step04: {args.mode}").blocked_autorange(min_run_time=1.0)
        report.update(status="measured", median_us=result.median * 1e6, iqr_us=result.iqr * 1e6,
                      number_per_run=result.number_per_run, raw_times_s=result.raw_times,
                      has_warnings=result.has_warnings)
        print(result)
        exit_code = 0
    else:
        report["status"] = "correctness_failed"
        print("验收未通过，不进行性能测量；旧接口适配只用于诊断，不算接口验收通过。")
        exit_code = 1
    assert hashlib.sha256(source.read_bytes()).hexdigest() == report["source_sha256"], "执行期间源码变化"
    directory = PROJECT / "benchmarks/results"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output_file = directory / f"step04_{args.mode}_{stamp}_{os.getpid()}.json"
    output_file.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("记录已保存：", output_file)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
