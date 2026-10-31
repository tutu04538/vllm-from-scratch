"""88：正式脚本的 dry-run 不初始化 CUDA，且矩阵是 16+4。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_compare_script_dry_run_lists_exactly_20_cases():
    completed = subprocess.run(
        [sys.executable, str(ROOT / "benchmarks/compare_step88_execution.py"),
         "--dry-run"],
        cwd=ROOT, text=True, capture_output=True, check=True,
        env={**os.environ, "PYTHONPATH": str(ROOT)})
    payload = json.loads(completed.stdout)
    assert payload["ok"]
    assert payload["matrix"] == {
        "total": 20, "main": 16, "context": 4, "pairs": 10,
        "engines": ["minivllm", "vllm"], "batches": [1, 4],
        "ks": [0, 2], "graphs": [False, True],
    }
    assert len(payload["cases"]) == 20
    assert all(case["kind"] == "step88_perf" for case in payload["cases"])


def test_importing_compare_does_not_load_torch_or_set_worker_env():
    code = (
        "import os, sys; "
        f"sys.path.insert(0, {str(ROOT)!r}); "
        "import benchmarks.compare_step88_execution; "
        "print(int('torch' in sys.modules), int('vllm' in sys.modules), "
        "os.environ.get('VLLM_USE_V2_MODEL_RUNNER', ''), "
        "os.environ.get('VLLM_ALLOW_INSECURE_SERIALIZATION', ''))"
    )
    env = os.environ.copy()
    env.pop("VLLM_USE_V2_MODEL_RUNNER", None)
    env.pop("VLLM_ALLOW_INSECURE_SERIALIZATION", None)
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, text=True,
        capture_output=True, check=True, env=env)
    assert completed.stdout.strip().split() == ["0", "0"]
