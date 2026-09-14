"""第一关 CPU 基准：先检查结果，再测完整 generate 调用，最后保存 JSON。"""

import os

# 必须在导入 torch / step01 之前设置，避免 DummyModel 自动选择 GPU。
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.benchmark import Timer

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from step01 import generate


def check_correctness():
    """不在计时区域里做断言；有错误时，不生成正式耗时成绩。"""
    cases = [
        ([0], 10, [1, 2, 3, 4]),
        ([0], 2, [1, 2]),
        ([2], 10, [3, 4]),
        ([0], 0, []),
        ([0, 1], 1, [2]),
    ]
    for prompt, budget, expected in cases:
        original = prompt.copy()
        actual = generate(prompt, budget)
        assert isinstance(actual, list), "generate 应返回 list[int]"
        assert all(type(token) is int for token in actual), "输出元素应是 Python int"
        assert actual == expected, f"{original=}, {budget=}: {actual=}，应为 {expected}"
        assert prompt == original, "generate 修改了调用者的 prompt"


def measure():
    """返回 Measurement；median/iqr 已经是每次调用的秒数。"""
    timer = Timer(
        stmt="generate([0], 10)",
        globals={"generate": generate},
        num_threads=1,
        label="step01: CPU dummy model",
        description="4 new tokens, including EOS",
    )
    # 自动预热、选择每组重复次数，再累计测量至少约 1 秒。
    return timer.blocked_autorange(min_run_time=1.0)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    source = PROJECT / "step01.py"
    report = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "device": "cpu",
        "num_threads": 1,
        "workload": "generate([0], 10)",
        "expected_new_tokens": 4,
        "seed": 0,
        "min_run_time_s": 1.0,
        "boundary": "complete generate call; excludes imports, checks and printing",
    }
    try:
        check_correctness()
    except Exception as exc:
        report.update(status="correctness_failed", error=f"{type(exc).__name__}: {exc}")
        print("正确性检查失败，未进行性能测量：", report["error"])
        exit_code = 1
    else:
        result = measure()
        report.update(
            status="measured",
            median_us=result.median * 1_000_000,
            iqr_us=result.iqr * 1_000_000,
            number_per_run=result.number_per_run,
            raw_times_s=result.raw_times,
            has_warnings=result.has_warnings,
        )
        print(result)
        print(f"每次 generate 中位耗时：{report['median_us']:.3f} us")
        print(f"IQR：{report['iqr_us']:.3f} us")
        exit_code = 0

    # 每次独立运行生成新文件，不覆盖上一次基线。
    directory = PROJECT / "benchmarks" / "results"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = directory / f"step01_{stamp}_{os.getpid()}.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("记录已保存：", output)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
