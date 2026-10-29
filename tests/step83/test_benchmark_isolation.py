"""83-A：父进程编排与子进程失败传播（不初始化 CUDA）。"""

from __future__ import annotations

import sys
import textwrap

import pytest

from benchmarks.benchmark_runner import CaseFailure, run_case, run_matrix


def _worker(tmp_path):
    script = tmp_path / "child.py"
    script.write_text(textwrap.dedent("""
        import argparse, json
        from pathlib import Path

        parser = argparse.ArgumentParser()
        parser.add_argument("--case-json", required=True)
        parser.add_argument("--result-json", required=True)
        args = parser.parse_args()
        case = json.loads(Path(args.case_json).read_text())
        if case["mode"] == "fail":
            print("child says bad", file=__import__("sys").stderr)
            raise SystemExit(7)
        if case["mode"] == "crash":
            raise RuntimeError("boom")
        if case["mode"] == "missing":
            raise SystemExit(0)
        Path(args.result_json).write_text(json.dumps({
            "status": "ok",
            "result": {"echo": case["value"], "pid": __import__("os").getpid()},
        }))
    """))
    return [sys.executable, str(script)]


def test_run_case_gets_payload_from_a_fresh_process(tmp_path):
    worker = _worker(tmp_path)
    result = run_case(worker, {"mode": "ok", "value": 42})
    assert result["echo"] == 42
    assert result["pid"] != __import__("os").getpid()


def test_nonzero_exit_is_a_failure_not_zero_metrics(tmp_path):
    worker = _worker(tmp_path)
    with pytest.raises(CaseFailure) as raised:
        run_case(worker, {"mode": "fail", "value": 1})
    assert raised.value.returncode == 7
    assert "child says bad" in raised.value.stderr_tail


def test_exception_and_missing_result_are_failures(tmp_path):
    worker = _worker(tmp_path)
    with pytest.raises(CaseFailure, match="退出码 1"):
        run_case(worker, {"mode": "crash", "value": 1})
    with pytest.raises(CaseFailure, match="没有写结果文件"):
        run_case(worker, {"mode": "missing", "value": 1})


def test_timeout_is_a_failure(tmp_path):
    worker = [sys.executable, "-c", "import time; time.sleep(30)"]
    with pytest.raises(CaseFailure, match="超时"):
        run_case(worker, {"mode": "slow"}, timeout_s=0.2)


def test_matrix_keeps_failure_and_success_separate(tmp_path):
    worker = _worker(tmp_path)
    matrix = run_matrix(worker, [
        {"label": "a", "mode": "ok", "value": 1},
        {"label": "b", "mode": "fail", "value": 2},
        {"label": "c", "mode": "ok", "value": 3},
    ])
    assert matrix["ok"] is False
    assert matrix["labels"] == ["a", "b", "c"]
    assert [row["status"] for row in matrix["cases"]] == ["ok", "failed", "ok"]
    assert matrix["cases"][0]["result"]["echo"] == 1
    assert matrix["cases"][2]["result"]["echo"] == 3
    assert matrix["failures"][0]["case"]["label"] == "b"


def test_importing_benchmark_scripts_does_not_change_global_env():
    """父进程 import 基准脚本不能把 V2/UVA 开关写进全局环境。

    这条是实测踩过的坑：`check_step83_numerics` 顶层曾经 setdefault
    `VLLM_USE_V2_MODEL_RUNNER=1`，一次 pytest 进程里跑完它之后，后面的 V1 用例
    全被切到 V2，造成大面积 NotImplementedError。
    """
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    code = (
        "import os, sys; "
        f"sys.path.insert(0, {str(root)!r}); "
        f"sys.path.insert(0, {str(root / 'benchmarks')!r}); "
        "import benchmarks.check_step83_numerics; "
        "print(os.environ.get('VLLM_USE_V2_MODEL_RUNNER', ''))"
    )
    env = os.environ.copy()
    env.pop("VLLM_USE_V2_MODEL_RUNNER", None)
    completed = subprocess.run([sys.executable, "-c", code], text=True,
                               capture_output=True, env=env, check=True)
    assert completed.stdout.strip() == ""
