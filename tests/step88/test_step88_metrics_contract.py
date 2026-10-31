"""88：指标公式、vLLM Prometheus 换算与汇总失败传播（不需要 GPU）。"""

from __future__ import annotations

from dataclasses import dataclass

from benchmarks.spec_decode_metrics import RequestTiming
from benchmarks.step88_common import (
    acceptance_from_spec_metrics,
    build_perf_summary,
    compare_output_batches,
    spec_metrics_delta,
    spec_metrics_snapshot,
    summarize_matrix_outcome,
    token_diff,
)


@dataclass
class _Counter:
    name: str
    value: int


@dataclass
class _Vector:
    name: str
    values: list[int]


def test_token_diff_handles_exact_prefix_and_length_mismatch():
    assert token_diff([1, 2, 3], [1, 2, 3])["exact"]
    diff = token_diff([1, 2, 3], [1, 9, 3])
    assert not diff["exact"]
    assert diff["first_diff"] == 1
    assert diff["common_prefix_len"] == 1
    prefix = token_diff([1, 2], [1, 2, 3])
    assert not prefix["exact"]
    assert prefix["first_diff"] is None
    assert prefix["common_prefix_len"] == 2
    batch = compare_output_batches([[1, 2], [3, 4]], [[1, 9], [3, 4]])
    assert not batch["exact"]
    assert batch["first_diff"] == 1
    assert [item["exact"] for item in batch["per_request"]] == [False, True]


def test_vllm_spec_metrics_are_normalized_by_delta():
    before = spec_metrics_snapshot([
        _Counter("vllm:spec_decode_num_drafts", 10),
        _Counter("vllm:spec_decode_num_draft_tokens", 20),
        _Counter("vllm:spec_decode_num_accepted_tokens", 8),
        _Vector("vllm:spec_decode_num_accepted_tokens_per_pos", [5, 3]),
    ])
    after = spec_metrics_snapshot([
        _Counter("vllm:spec_decode_num_drafts", 12),
        _Counter("vllm:spec_decode_num_draft_tokens", 24),
        _Counter("vllm:spec_decode_num_accepted_tokens", 10),
        _Vector("vllm:spec_decode_num_accepted_tokens_per_pos", [6, 4]),
    ])
    delta = spec_metrics_delta(after, before)
    assert delta["num_drafts"] == 2
    assert delta["num_draft_tokens"] == 4
    assert delta["num_accepted_tokens"] == 2
    assert delta["num_accepted_tokens_per_pos"] == [1, 1]
    acceptance = acceptance_from_spec_metrics(delta)
    assert acceptance["acceptance_rate"] == 0.5
    assert acceptance["mean_acceptance_length"] == 2.0
    assert acceptance["position_acceptance"] == [0.5, 0.5]
    assert acceptance["draft_per_pos_available"] is False


def test_tpot_uses_committed_tokens_after_first_burst():
    timing = RequestTiming(
        "r", submitted_at=0.0, first_output_at=1.0,
        last_output_at=2.0, committed_tokens=5,
        output_events=[(1.0, 3), (2.0, 2)],
    )
    from benchmarks.spec_decode_metrics import AcceptanceAggregate, summarize_run
    summary = summarize_run(
        timings={"r": timing}, acceptance=AcceptanceAggregate(),
        wall_time_s=1.0, engine_steps=2)
    assert summary["inter_burst_latency_s"]["mean"] == 1.0
    assert summary["tpot_s"]["mean"] == 0.5
    assert summary["output_tokens_per_s"] == 5.0


def _row(engine, batch, k, graph, tps, tokens):
    return {
        "status": "ok", "label": f"{engine}_B{batch}_K{k}",
        "engine": engine, "batch": batch, "k": k, "graph": graph,
        "workload": "short", "tokens": {
            f"r{i}": list(seq) for i, seq in enumerate(tokens)},
        "output_tokens_per_s": {"median": tps},
        "wall_time_s": {"median": 1.0},
        "acceptance": {
            "num_drafts": 1 if k else 0,
            "num_draft_tokens": k,
            "num_accepted_tokens": 1 if k else 0,
            "acceptance_rate": 0.5 if k else None,
        },
        "identity": {"ok": True},
    }


def test_perf_summary_keeps_pairs_and_failures_separate():
    rows = [
        _row("minivllm", 1, 0, False, 10.0, [[1, 2]]),
        _row("vllm", 1, 0, False, 20.0, [[1, 2]]),
        _row("minivllm", 1, 0, True, 11.0, [[1, 2]]),
        _row("vllm", 1, 0, True, 21.0, [[1, 2]]),
        _row("minivllm", 1, 2, False, 30.0, [[1, 2]]),
        _row("vllm", 1, 2, False, 15.0, [[1, 2]]),
        _row("minivllm", 1, 2, True, 31.0, [[1, 2]]),
        _row("vllm", 1, 2, True, 16.0, [[1, 2]]),
    ]
    failure = {"case": {"label": "vllm_B4_K2_graph"}, "status": "failed",
               "error": "boom"}
    summary = build_perf_summary(rows, [failure])
    assert not summary["ok"]
    assert summary["failures"] == [failure]
    assert {row["pair"] for row in summary["table"]} == {
        "B1_K0_eager", "B1_K0_graph", "B1_K2_eager", "B1_K2_graph"}
    b1k0 = next(row for row in summary["table"] if row["pair"] == "B1_K0_eager")
    assert b1k0["mini_over_vllm"] == 0.5
    assert b1k0["same_engine_outputs"] == {"minivllm": True, "vllm": True}
    assert summary["missing"]


def test_matrix_outcome_does_not_turn_failure_into_zero_metrics():
    matrix = {
        "ok": False,
        "cases": [
            {"case": {"label": "ok_case", "engine": "minivllm",
                      "batch": 1, "k": 0, "graph": False},
             "status": "ok", "result": _row("minivllm", 1, 0, False,
                                            10.0, [[1]])},
            {"case": {"label": "bad_case", "engine": "vllm"},
             "status": "failed", "result": None, "error": "OOM"},
        ],
        "failures": [{"case": {"label": "bad_case"}, "status": "failed",
                      "error": "OOM"}],
    }
    summary = summarize_matrix_outcome(matrix)
    assert not summary["ok"]
    assert len(summary["rows"]) == 1
    assert summary["case_status"][1]["status"] == "failed"
    assert "OOM" in summary["case_status"][1]["error"]
