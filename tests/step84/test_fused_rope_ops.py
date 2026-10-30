"""84-B：融合 Q/K 旋转的数值、布局、原地语义与 CUDA Graph。"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from minivllm.layers.ops.rotary_embedding import (apply_rotary_embedding,
                                                  apply_rotary_embedding_native)

from spec84_helpers import compact_cache, packed_qkv

ROOT = Path(__file__).resolve().parents[2]
CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="融合 RoPE 的 CUDA 路径需要 GPU")


def _cache(head_size: int, rotary_dim: int, proportional: bool,
           dtype: torch.dtype, max_position: int = 16) -> torch.Tensor:
    return compact_cache(
        head_size, rotary_dim, max_position,
        proportional=proportional, dtype=dtype, device="cuda")


def _tolerances(dtype: torch.dtype) -> dict:
    # 融合 kernel 内部用 FP32 算、写回时只舍入一次；Torch 参考按输入 dtype
    # 逐步舍入。BF16 下差一个中间 ulp 是预期，不是位置/配对错误。
    if dtype == torch.float32:
        return {"atol": 2e-6, "rtol": 2e-6}
    return {"atol": 0.08, "rtol": 0.02}


@pytest.mark.parametrize("head_size,rotary_dim,proportional", [
    (8, 8, False),
    (8, 4, False),
    (8, 2, True),
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_matches_native_qk_and_q_only(head_size, rotary_dim,
                                            proportional, dtype):
    torch.manual_seed(840)
    q_heads, k_heads = 3, 2
    q = torch.randn(4, q_heads * head_size, dtype=dtype, device="cuda")
    k = torch.randn(4, k_heads * head_size, dtype=dtype, device="cuda")
    positions = torch.tensor([3, 0, 15, 5], dtype=torch.int64, device="cuda")
    cache = _cache(head_size, rotary_dim, proportional, dtype)
    # proportional 的表覆盖完整 head_size；ops 的 rotary_dim 是“实际应用宽度”。
    apply_dim = head_size if proportional else rotary_dim

    expected_q, expected_k = apply_rotary_embedding_native(
        positions, q, k, cache, head_size, apply_dim, True)
    q_inplace, k_inplace = q.clone(), k.clone()
    got_q, got_k = apply_rotary_embedding(
        positions, q_inplace, k_inplace, cache, head_size, apply_dim, True)
    assert got_q is q_inplace and got_k is k_inplace
    assert got_q.data_ptr() == q_inplace.data_ptr()
    assert got_k.data_ptr() == k_inplace.data_ptr()
    torch.testing.assert_close(
        got_q.float(), expected_q.float(), **_tolerances(dtype))
    torch.testing.assert_close(
        got_k.float(), expected_k.float(), **_tolerances(dtype))

    # Q-only：key=None 时只旋转 Q，返回 (q, None)。
    q_only = q.clone()
    expected_q_only = apply_rotary_embedding_native(
        positions, q, None, cache, head_size, apply_dim, True)[0]
    out_q, out_k = apply_rotary_embedding(
        positions, q_only, None, cache, head_size, apply_dim, True)
    assert out_k is None
    assert out_q is q_only
    torch.testing.assert_close(
        out_q.float(), expected_q_only.float(), **_tolerances(dtype))


def test_fused_gptj_style_matches_native():
    head_size, rotary_dim = 8, 4
    dtype = torch.float32
    q = torch.randn(5, 2 * head_size, dtype=dtype, device="cuda")
    k = torch.randn(5, 3 * head_size, dtype=dtype, device="cuda")
    positions = torch.tensor([0, 2, 4, 6, 8], dtype=torch.int64, device="cuda")
    cache = _cache(head_size, rotary_dim, False, dtype, max_position=16)
    expected_q, expected_k = apply_rotary_embedding_native(
        positions, q, k, cache, head_size, rotary_dim, False)
    got_q, got_k = apply_rotary_embedding(
        positions, q, k, cache, head_size, rotary_dim, False)
    torch.testing.assert_close(got_q, expected_q, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(got_k, expected_k, atol=2e-6, rtol=2e-6)


def test_packed_qkv_slices_only_write_q_and_k():
    head_size = 8
    q_heads, kv_heads = 4, 2
    qkv, q, k, v = packed_qkv(
        5, q_heads, kv_heads, head_size, dtype=torch.float32, seed=11)
    positions = torch.tensor([0, 1, 3, 5, 7], dtype=torch.int64,
                             device="cuda")
    cache = _cache(head_size, head_size, False, torch.float32, 16)
    v_before = v.clone()
    q_expected, k_expected = apply_rotary_embedding_native(
        positions, q, k, cache, head_size, head_size, True)
    q_before = q.clone()
    apply_rotary_embedding(
        positions, q, k, cache, head_size, head_size, True)
    torch.testing.assert_close(q, q_expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(k, k_expected, atol=2e-6, rtol=2e-6)
    assert torch.equal(v, v_before)
    # Q/K 确实被修改了，不是“原样返回”。
    assert not torch.equal(q, q_before)


def test_column_strided_input_matches_native_without_contiguous_copy():
    head_size = 8
    q_heads, k_heads = 3, 2
    dtype = torch.float32
    q_storage = torch.randn(4, 2 * q_heads * head_size, dtype=dtype,
                            device="cuda")
    k_storage = torch.randn(4, 2 * k_heads * head_size, dtype=dtype,
                            device="cuda")
    q = q_storage[:, ::2]
    k = k_storage[:, ::2]
    assert q.stride(1) == 2 and k.stride(1) == 2
    positions = torch.tensor([0, 2, 4, 6], dtype=torch.int64, device="cuda")
    cache = _cache(head_size, head_size, False, dtype, 16)
    expected_q, expected_k = apply_rotary_embedding_native(
        positions, q, k, cache, head_size, head_size, True)
    got_q, got_k = apply_rotary_embedding(
        positions, q, k, cache, head_size, head_size, True)
    assert got_q.data_ptr() == q.data_ptr()
    assert got_k.data_ptr() == k.data_ptr()
    torch.testing.assert_close(got_q, expected_q, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(got_k, expected_k, atol=2e-6, rtol=2e-6)


def test_cuda_graph_capture_and_replay_reads_new_positions():
    head_size = 8
    dtype = torch.float32
    q_heads = 3
    cache = _cache(head_size, head_size, False, dtype, 32)
    static_src = torch.randn(4, q_heads * head_size, dtype=dtype,
                             device="cuda")
    static_q = torch.empty_like(static_src)
    static_positions = torch.tensor([1, 2, 3, 4], dtype=torch.int64,
                                    device="cuda")

    def run():
        static_q.copy_(static_src)
        apply_rotary_embedding(
            static_positions, static_q, None, cache,
            head_size, head_size, True)

    # warmup 必须在 capture 之前完成（Triton 首次 launch 会编译/分配）。
    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    def check(positions, source):
        static_src.copy_(source)
        static_positions.copy_(positions)
        graph.replay()
        torch.cuda.synchronize()
        expected = apply_rotary_embedding_native(
            positions, source, None, cache, head_size, head_size, True)[0]
        torch.testing.assert_close(static_q, expected, atol=2e-6, rtol=2e-6)

    check(torch.tensor([3, 4, 5, 6], dtype=torch.int64, device="cuda"),
          torch.randn_like(static_src))
    # 第二次 replay 用完全不同的位置和输入：不能读 capture 时的旧表/旧位置。
    check(torch.tensor([20, 21, 22, 23], dtype=torch.int64, device="cuda"),
          torch.randn_like(static_src))


def test_validation_rejects_bad_layouts_without_silent_copy():
    head_size = 8
    dtype = torch.float32
    q = torch.randn(3, 2 * head_size, dtype=dtype, device="cuda")
    cache = _cache(head_size, head_size, False, dtype, 16)
    positions = torch.tensor([0, 1, 2], dtype=torch.int64, device="cuda")

    with pytest.raises(ValueError, match="二维"):
        apply_rotary_embedding(positions, q.view(3, 2, head_size), None,
                               cache, head_size, head_size, True)
    with pytest.raises(ValueError, match="整除"):
        apply_rotary_embedding(positions, torch.randn(3, 15, device="cuda"),
                               None, cache, head_size, head_size, True)
    with pytest.raises(ValueError, match="数量"):
        apply_rotary_embedding(positions[:2], q, None, cache,
                               head_size, head_size, True)
    with pytest.raises(ValueError, match="dtype"):
        apply_rotary_embedding(positions, q.to(torch.bfloat16), None, cache,
                               head_size, head_size, True)
    with pytest.raises(ValueError, match="同设备"):
        apply_rotary_embedding(positions.cpu(), q, None, cache,
                               head_size, head_size, True)
    with pytest.raises(ValueError, match="token 行数"):
        apply_rotary_embedding(positions, q, q[:2], cache,
                               head_size, head_size, True)
    broadcast = torch.randn(1, 2 * head_size, device="cuda").expand(
        3, 2 * head_size)
    with pytest.raises(ValueError, match="正的 token/列 stride"):
        apply_rotary_embedding(positions, broadcast, None, cache,
                               head_size, head_size, True)
    with pytest.raises(ValueError, match="同一块起始地址"):
        apply_rotary_embedding(positions, q, q, cache,
                               head_size, head_size, True)


def test_cuda_out_of_range_is_a_device_error_in_isolated_process():
    """越界不能让 kernel 静默读相邻内存；设备端 assert 只在子进程里触发。"""
    code = r"""
import torch
from minivllm.layers.ops.rotary_embedding import apply_rotary_embedding
head = 8
q = torch.randn(2, 2 * head, device="cuda")
cache = torch.randn(4, head, device="cuda")
apply_rotary_embedding(
    torch.tensor([4, 0], device="cuda"), q, None, cache, head, head, True)
torch.cuda.synchronize()
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=env,
        text=True, capture_output=True, timeout=180)
    assert completed.returncode != 0
    assert "RoPE position out of range" in completed.stderr
    # 设备端 assert 会污染该进程的 CUDA context；所以必须在子进程里跑。
    assert "device-side assert" in completed.stderr
