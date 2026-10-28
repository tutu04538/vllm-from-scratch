"""81-B：投机统计的正式输出链（SchedulerStats → EngineCoreOutputs → LLMEngine）。"""

from __future__ import annotations

import pytest
import torch
from pathlib import Path
from minivllm import SamplingParams

from minivllm.metrics.stats import SchedulerStats
from minivllm.spec_decode.metrics import SpecDecodingLogging, SpecDecodingStats
from spec81_helpers import make_v1_engine, run_engine

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                                  reason="V1 投机拒绝采样需要 CUDA/Triton")


def test_stats_counts_and_position_arrays():
    stats = SpecDecodingStats.new(3)
    stats.observe_draft(3, 2)
    stats.observe_draft(1, 0)
    assert (stats.num_drafts, stats.num_draft_tokens,
            stats.num_accepted_tokens) == (2, 4, 2)
    assert stats.num_accepted_tokens_per_pos == [1, 1, 0]
    assert stats.num_draft_tokens_per_pos == [2, 1, 1]
    assert stats.num_accepted_tokens / stats.num_draft_tokens == 0.5
    assert 1 + stats.num_accepted_tokens / stats.num_drafts == 2


def test_logging_copies_snapshots_and_zero_denominator():
    logging = SpecDecodingLogging(log_fn=lambda *args: None)
    stats = SpecDecodingStats.new(2)
    stats.observe_draft(2, 1)
    logging.observe(stats)
    snapshot = list(logging.accepted_tokens_per_pos_lists[0])
    stats.num_accepted_tokens_per_pos[0] = 99
    assert logging.accepted_tokens_per_pos_lists[0] == snapshot

    result = logging.log()
    assert result["num_drafts"] == 1
    assert result["acceptance_rate"] == 0.5
    assert result["mean_acceptance_length"] == 2
    assert logging.num_drafts == []
    empty = logging.log()
    assert empty["acceptance_rate"] is None
    assert empty["mean_acceptance_length"] is None


@requires_cuda
def test_engine_exposes_scheduler_stats_and_logging():
    engine, _core = make_v1_engine(spec_k=3)
    try:
        outputs, stats, committed = run_engine(
            engine, [("a", [1, 2, 3, 4, 5, 6])], max_tokens=4)
    finally:
        engine.shutdown()
    assert outputs["a"]
    assert all(isinstance(entry, SchedulerStats) for entry in stats if entry)
    with_spec = [entry for entry in stats if entry and entry.spec_decoding_stats]
    assert with_spec, "真实投机跑完必须有 SchedulerStats.spec_decoding_stats"
    assert any(entry.spec_decoding_stats.num_accepted_tokens > 0
               for entry in with_spec)
    assert len(engine.spec_decoding_logging) == len(with_spec)
    result = engine.spec_decoding_logging.log()
    assert result["num_drafts"] == sum(
        entry.spec_decoding_stats.num_drafts for entry in with_spec)
    assert result["num_draft_tokens"] == sum(
        entry.spec_decoding_stats.num_draft_tokens for entry in with_spec)
    assert result["num_accepted_tokens"] == sum(
        entry.spec_decoding_stats.num_accepted_tokens for entry in with_spec)
    assert result["acceptance_rate"] == pytest.approx(
        result["num_accepted_tokens"] / result["num_draft_tokens"])
    assert sum(committed) == len(outputs["a"])


@requires_cuda
def test_scheduler_stats_survive_a_no_output_prefill_step():
    """中间 prefill 轮没有用户输出，但 SchedulerStats 仍必须送到前端。"""
    engine, _core = make_v1_engine(spec_k=2, budget=8)
    try:
        engine.add_request("a", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2],
                           SamplingParams(max_tokens=2, temperature=0.0,
                                          eos_token_id=999))
        out = engine.step()
        assert out == []
        assert engine.last_scheduler_stats is not None
        assert engine.last_scheduler_stats.num_running_reqs == 1
        assert engine.last_scheduler_stats.spec_decoding_stats is None
    finally:
        engine.shutdown()


def test_current_step81_package_hash_matches_results():
    import hashlib
    import json
    data = json.loads((Path(__file__).resolve().parents[2] / "docs/results.json").read_text())
    package = data["step81"]["closeout"]["results"]["package"]
    root = Path(__file__).resolve().parents[2] / "minivllm"
    names = sorted(root.rglob("*.py"), key=lambda p: str(p.relative_to(root)))
    digest = hashlib.sha256("".join(
        f"{p.relative_to(root)}\0{hashlib.sha256(p.read_bytes()).hexdigest()}\n"
        for p in names).encode()).hexdigest()
    assert package["sha256"] == digest
    assert package["python_files"] == len(names)
