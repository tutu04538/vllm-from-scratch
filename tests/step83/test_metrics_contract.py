"""83：指标口径的手算例子（不加载模型/GPU）。"""

from __future__ import annotations

import pytest

from benchmarks.spec_decode_metrics import (AcceptanceAggregate, RequestTiming,
                                             summarize_run)


def _summary(timings, wall=1.0):
    return summarize_run(timings=timings, acceptance=AcceptanceAggregate(),
                         wall_time_s=wall, engine_steps=1)


def test_burst_interval_and_tpot_are_different_metrics():
    """t=1s 交付 3 token，t=2s 交付 2 token：间隔 1s，TPOT=1/2s。"""
    timing = RequestTiming(
        "a", submitted_at=0.0, first_output_at=1.0,
        last_output_at=2.0, committed_tokens=5,
        output_events=[(1.0, 3), (2.0, 2)],
    )
    summary = _summary({"a": timing})
    assert summary["committed_output_tokens"] == 5
    assert summary["output_tokens_per_s"] == 5.0
    assert summary["inter_burst_latency_s"]["mean"] == pytest.approx(1.0)
    assert summary["tpot_s"]["mean"] == pytest.approx(0.5)
    assert summary["tpot_s"]["num_requests"] == 1
    assert summary["tpot_s"]["undefined_requests"] == 0


def test_single_burst_tpot_is_undefined_not_zero():
    timing = RequestTiming(
        "a", submitted_at=0.0, first_output_at=0.25,
        last_output_at=0.25, committed_tokens=3,
        output_events=[(0.25, 3)],
    )
    summary = _summary({"a": timing}, wall=0.5)
    assert summary["output_tokens_per_s"] == 6.0
    assert summary["inter_burst_latency_s"]["mean"] is None
    assert summary["tpot_s"]["mean"] is None
    assert summary["tpot_s"]["num_requests"] == 0
    assert summary["tpot_s"]["undefined_requests"] == 1


def test_committed_tokens_not_draft_tokens_drive_throughput():
    timing = RequestTiming("a", 0.0)
    timing.observe(0.1, 10)       # 输出 10 个 token 的 burst
    timing.observe(0.1, 10)       # 重复累计值不能重复计数
    summary = _summary({"a": timing}, wall=2.0)
    assert summary["committed_output_tokens"] == 10
    assert summary["output_tokens_per_s"] == 5.0
    assert summary["burst_tokens"]["max"] == 10


def test_percentile_does_not_reorder_input_and_empty_is_none():
    from benchmarks.spec_decode_metrics import median_range
    assert median_range([3.0, 1.0, 2.0]) == {
        "median": 2.0, "min": 1.0, "max": 3.0, "values": [3.0, 1.0, 2.0]}
    assert median_range([])["median"] is None
