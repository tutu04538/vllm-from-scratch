"""82：基准纯统计公式（小整数，不加载模型）。"""

from __future__ import annotations

import pytest

from benchmarks.spec_decode_metrics import (AcceptanceAggregate, RequestTiming,
                                             median_range, summarize_run)


class Stats:
    def __init__(self, drafts, draft_tokens, accepted, accepted_pos, draft_pos):
        self.num_drafts = drafts
        self.num_draft_tokens = draft_tokens
        self.num_accepted_tokens = accepted
        self.num_accepted_tokens_per_pos = accepted_pos
        self.num_draft_tokens_per_pos = draft_pos


def test_committed_tokens_use_increments_not_cumulative_lists():
    timing = RequestTiming("a", submitted_at=10.0)
    assert timing.observe(10.1, 1) == 1
    assert timing.observe(10.2, 3) == 2
    assert timing.observe(10.3, 3) == 0
    assert timing.committed_tokens == 3
    assert timing.first_output_at == 10.1
    assert timing.last_output_at == 10.2


def test_acceptance_formulas_and_zero_denominator():
    agg = AcceptanceAggregate()
    agg.observe(Stats(2, 4, 2, [1, 1, 0], [2, 1, 1]))
    assert (agg.num_drafts, agg.num_draft_tokens, agg.num_accepted_tokens) == (2, 4, 2)
    assert agg.acceptance_rate == 0.5
    assert agg.mean_acceptance_length == 2.0
    assert agg.position_acceptance() == [0.5, 0.5, 0.0]
    assert agg.position_acceptance_actual() == [0.5, 1.0, 0.0]

    empty = AcceptanceAggregate()
    assert empty.acceptance_rate is None
    assert empty.mean_acceptance_length is None
    assert empty.position_acceptance() == []


def test_summarize_run_and_median_range():
    timings = {
        "a": RequestTiming("a", 0.0, first_output_at=0.2, last_output_at=0.6,
                           committed_tokens=3, output_events=[(0.2, 1), (0.6, 2)]),
        "b": RequestTiming("b", 0.0, first_output_at=0.4, last_output_at=0.8,
                           committed_tokens=2, output_events=[(0.4, 2)]),
    }
    agg = AcceptanceAggregate()
    agg.observe(Stats(1, 2, 1, [1, 0], [1, 1]))
    summary = summarize_run(timings=timings, acceptance=agg,
                            wall_time_s=1.0, engine_steps=4, preemptions=1)
    assert summary["committed_output_tokens"] == 5
    assert summary["output_tokens_per_s"] == 5.0
    assert summary["ttft_s"]["mean"] == pytest.approx(0.3)
    assert summary["inter_burst_latency_s"]["mean"] == pytest.approx(0.4)
    # 83 口径：首 burst 之后两个 token，时间差 0.4s -> 0.2 s/token。
    assert summary["tpot_s"]["mean"] == pytest.approx(0.2)
    assert summary["burst_tokens"]["max"] == 2
    assert summary["acceptance"]["acceptance_rate"] == 0.5
    assert summary["preemptions"] == 1
    assert median_range([1, 2, 3]) == {"median": 2, "min": 1, "max": 3,
                                       "values": [1, 2, 3]}
    assert median_range([])["median"] is None
