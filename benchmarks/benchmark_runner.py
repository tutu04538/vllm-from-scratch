"""83 关：基准的进程隔离边界。

**父进程只编排，子进程只跑一个配置。** 这不是性能优化，而是让测量结果可信：

    父进程：case 配置 -> 启动新解释器 -> 收结果 -> 子进程退出 -> 下一个 case
    子进程：建一个 engine -> 预热 -> 测量 -> 写结果 -> shutdown -> 退出

为什么不用 fork：父进程可能已经 import torch 并初始化 CUDA；在那种进程里 fork 出来的
子进程会继承 CUDA context 和分配器状态。这里统一用 `subprocess` 启动**新解释器**，并且
父进程不在启动 case 前构造 minivllm engine。

为什么结果走文件而不是 stdout：模型/测试可能打印日志；把结果写到独立 JSON 文件，
stdout 只作为诊断信息保存，不会因为一行日志污染指标。子进程失败（异常/OOM/超时）时父进程
必须拿到明确的失败，不允许用 0 tok/s 假装成功。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]


class CaseFailure(RuntimeError):
    """一个隔离 case 没有产出可用结果。"""

    def __init__(self, message: str, *, returncode: int | None = None,
                 stderr_tail: str = "", stdout_tail: str = "",
                 duration_s: float | None = None) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stderr_tail = stderr_tail
        self.stdout_tail = stdout_tail
        self.duration_s = duration_s


@dataclass
class CaseOutcome:
    case: dict[str, Any]
    status: str
    result: dict[str, Any] | None = None
    error: str | None = None
    duration_s: float = 0.0
    returncode: int | None = None
    stderr_tail: str = ""
    stdout_tail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "case": self.case,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "duration_s": self.duration_s,
            "returncode": self.returncode,
            "stderr_tail": self.stderr_tail,
            "stdout_tail": self.stdout_tail,
        }


def _tail(text: str, limit: int = 8000) -> str:
    return text[-limit:] if len(text) > limit else text


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    os.replace(temporary, path)


def run_case(worker_argv: Sequence[str], case: dict[str, Any], *,
             timeout_s: float | None = None,
             env: dict[str, str] | None = None,
             cwd: str | Path = ROOT) -> dict[str, Any]:
    """在全新解释器里跑一个 case，返回子进程写入的 `result`。

    失败统一抛 `CaseFailure`（含 returncode/stderr 尾部），由 `run_matrix` 变成记录，
    不会变成 0 值成功。
    """
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="minivllm-case-") as tmp:
        tmp_path = Path(tmp)
        case_path = tmp_path / "case.json"
        result_path = tmp_path / "result.json"
        _atomic_write_json(case_path, case)
        argv = [*worker_argv, "--case-json", str(case_path),
                "--result-json", str(result_path)]
        child_env = os.environ.copy()
        if env:
            child_env.update(env)
        # 子进程脚本可能不在仓库根目录；统一把仓库根放进 import 路径。
        pythonpath = child_env.get("PYTHONPATH")
        child_env["PYTHONPATH"] = (str(ROOT) if not pythonpath
                                   else str(ROOT) + os.pathsep + pythonpath)
        # minivllm 的 UVA 扩展用 load_inline 编译，需要解释器同目录的 ninja；用户
        # 可能没有把 conda env 的 bin 放进 PATH。这里只补工具路径，不换后端。
        python_bin = str(Path(sys.executable).resolve().parent)
        path = child_env.get("PATH", "")
        child_env["PATH"] = python_bin if not path else python_bin + os.pathsep + path
        try:
            completed = subprocess.run(
                argv, cwd=str(cwd), env=child_env, text=True,
                capture_output=True, timeout=timeout_s, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.perf_counter() - started
            stderr = exc.stderr or ""
            stdout = exc.stdout or ""
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors="replace")
            raise CaseFailure(
                f"case 超时（{timeout_s}s）：{argv}",
                returncode=None, stderr_tail=_tail(stderr),
                stdout_tail=_tail(stdout), duration_s=duration) from exc
        duration = time.perf_counter() - started
        stderr_tail = _tail(completed.stderr or "")
        stdout_tail = _tail(completed.stdout or "")
        envelope = None
        if result_path.is_file():
            try:
                envelope = json.loads(result_path.read_text())
            except json.JSONDecodeError:
                envelope = None
        if isinstance(envelope, dict) and envelope.get("status") == "failed":
            traceback_tail = _tail(str(envelope.get("traceback", "")))
            raise CaseFailure(
                f"case 失败：{envelope.get('error', 'unknown')}"
                f"（退出码 {completed.returncode}）",
                returncode=completed.returncode,
                stderr_tail=stderr_tail or traceback_tail,
                stdout_tail=stdout_tail, duration_s=duration)
        if completed.returncode != 0:
            raise CaseFailure(
                f"case 子进程退出码 {completed.returncode}：{argv}",
                returncode=completed.returncode, stderr_tail=stderr_tail,
                stdout_tail=stdout_tail, duration_s=duration)
        if not result_path.is_file():
            raise CaseFailure(
                f"case 子进程没有写结果文件（exit 0 不算成功）：{result_path}",
                returncode=completed.returncode, stderr_tail=stderr_tail,
                stdout_tail=stdout_tail, duration_s=duration)
        if envelope is None:
            raise CaseFailure(
                "case 结果不是合法 JSON",
                returncode=completed.returncode, stderr_tail=stderr_tail,
                stdout_tail=stdout_tail, duration_s=duration)
        if "status" not in envelope:
            raise CaseFailure(
                "case 结果缺少 status 字段",
                returncode=completed.returncode, stderr_tail=stderr_tail,
                stdout_tail=stdout_tail, duration_s=duration)
        result = envelope.get("result")
        if not isinstance(result, dict):
            raise CaseFailure(
                "case 成功结果必须是 JSON object",
                returncode=completed.returncode, stderr_tail=stderr_tail,
                stdout_tail=stdout_tail, duration_s=duration)
        result = dict(result)
        result.setdefault("_case_duration_s", duration)
        return result


def run_matrix(worker_argv: Sequence[str], cases: Sequence[dict[str, Any]], *,
               timeout_s: float | None = None,
               env: dict[str, str] | None = None,
               cwd: str | Path = ROOT) -> dict[str, Any]:
    """顺序跑多个 case；一个失败不会把另一个变成假成功。

    GPU case 必须串行（父进程不并发启动多个 GPU 子进程）。失败记录保留 case 配置、
    退出码与 stderr 尾部，调用方应把整体 `ok=False` 反映到进程退出码。
    """
    outcomes: list[CaseOutcome] = []
    for case in cases:
        label = case.get("label", case.get("kind", "case"))
        started = time.perf_counter()
        try:
            result = run_case(worker_argv, case, timeout_s=timeout_s,
                              env=env, cwd=cwd)
        except CaseFailure as exc:
            outcomes.append(CaseOutcome(
                case=case, status="failed", error=str(exc),
                duration_s=exc.duration_s or (time.perf_counter() - started),
                returncode=exc.returncode, stderr_tail=exc.stderr_tail,
                stdout_tail=exc.stdout_tail))
            continue
        outcomes.append(CaseOutcome(
            case=case, status="ok", result=result,
            duration_s=time.perf_counter() - started))
    records = [outcome.as_dict() for outcome in outcomes]
    failures = [record for record in records if record["status"] != "ok"]
    return {
        "ok": not failures,
        "cases": records,
        "failures": failures,
        "labels": [record["case"].get("label", record["case"].get("kind"))
                   for record in records],
    }


def case_worker_main(dispatch) -> int:
    """子进程统一入口：`--case-json` 进，`--result-json` 出。

    `dispatch(case) -> dict` 由具体 benchmark 提供；异常写成 `status=failed` 并返回非零。
    """
    import argparse
    import traceback

    parser = argparse.ArgumentParser()
    parser.add_argument("--case-json", required=True)
    parser.add_argument("--result-json", required=True)
    args = parser.parse_args()
    case_path = Path(args.case_json)
    result_path = Path(args.result_json)
    try:
        case = json.loads(case_path.read_text())
        result = dispatch(case)
        if not isinstance(result, dict):
            raise TypeError(f"dispatch 必须返回 dict，收到 {type(result).__name__}")
        envelope = {"status": "ok", "result": result}
        exit_code = 0
    except BaseException as exc:  # noqa: BLE001 —— 子进程边界要捕获一切并落盘
        envelope = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        exit_code = 1
    _atomic_write_json(result_path, envelope)
    return exit_code


__all__ = ["CaseFailure", "CaseOutcome", "ROOT", "case_worker_main", "run_case",
           "run_matrix"]
