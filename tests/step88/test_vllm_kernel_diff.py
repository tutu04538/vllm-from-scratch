"""88 后续：同输入 mini/vLLM 2D kernel 对照。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from benchmarks.check_step88_vllm_kernel_diff import (
    _kernel_meta,
    _prepare,
    _run,
    _vllm_run,
)
from benchmarks.check_step88_attention_micro import case_matrix

ROOT = Path(__file__).resolve().parents[2]


def test_kernel_diff_dry_run_lists_22_cases():
    completed = subprocess.run(
        [sys.executable,
         str(ROOT / "benchmarks/check_step88_vllm_kernel_diff.py"),
         "--dry-run"],
        cwd=ROOT, text=True, capture_output=True, check=True,
        env={**os.environ, "PYTHONPATH": str(ROOT)})
    payload = json.loads(completed.stdout)
    assert payload["ok"]
    assert len(payload["cases"]) == 22


@torch.no_grad()
def test_mini_and_vllm_2d_match_and_mini_no_spill():
    if not torch.cuda.is_available():
        return
    case = next(c for c in case_matrix() if c["label"] == "qlen3_ctx1024")
    prepared = _prepare(case)
    mini = _run(case, "2d", prepared).clone()
    vllm = _vllm_run(case, prepared).clone()
    torch.cuda.synchronize()
    assert torch.equal(mini, vllm), "同一 2D 输入下 mini/vLLM 输出不一致"

    from minivllm.attention.ops import triton_unified_attention_vllm as mini_module
    from vllm.v1.attention.ops import triton_unified_attention as vllm_module

    mini_meta = _kernel_meta(mini_module)
    vllm_meta = _kernel_meta(vllm_module)
    # Triton 缓存是进程级的，前面测试可能已编译别的 shape；这里按变体查找。
    # 2D 优化后 mini 应有一个无 spill 的变体（3D 路径仍可能保留旧变体）。
    assert any(item["n_regs"] <= 255 and item["n_spills"] == 0
               for item in mini_meta)
    # vLLM 的 2D hd256 变体应为 0 spill；hd512 变体可有少量 spill。
    assert any(item["n_spills"] == 0 for item in vllm_meta)
    del prepared
