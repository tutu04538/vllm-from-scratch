"""81-A：投机单卡主线的六类生命周期代表用例。"""

from __future__ import annotations

import pytest
import torch

from minivllm import SamplingParams
from minivllm.config import CacheConfig
from minivllm.core.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                              KVCacheGroup, SlidingWindowSpec)
from minivllm.core.kv_cache_manager import KVCacheManager
from minivllm.core.kv_cache_utils import (BlockHasher,
                                          make_block_hash_with_group_id)
from minivllm.request import Request
from spec81_helpers import make_v1_engine, make_v2_mtp_engine, run_engine

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                                  reason="81 生命周期集成用例需要 CUDA/Triton")


def _stats_of(stats):
    return [entry.spec_decoding_stats for entry in stats
            if entry is not None and entry.spec_decoding_stats is not None]


@requires_cuda
def test_rejection_truncation_keeps_verified_stats_but_commits_max_tokens():
    """K=3 已验证 3 枚，但 max_tokens=2：统计仍是 3，交付只有 2。"""
    engine, _core = make_v1_engine(spec_k=3)
    try:
        outputs, stats, committed = run_engine(
            engine, [("a", [1, 2, 3, 4, 5, 6])], max_tokens=2)
    finally:
        engine.shutdown()
    assert outputs["a"] == [7, 1]
    observed = _stats_of(stats)
    assert observed
    assert (observed[0].num_draft_tokens, observed[0].num_accepted_tokens) == (3, 3)
    assert sum(committed) == 2


@requires_cuda
@pytest.mark.parametrize("stop_mode", ["eos", "stop_token_ids"])
def test_stop_token_truncates_tail_without_changing_verified_stats(stop_mode):
    engine, _core = make_v1_engine(spec_k=3)
    params = dict(max_tokens=4, temperature=0.0, eos_token_id=999)
    if stop_mode == "eos":
        params["eos_token_id"] = 1
    else:
        params["stop_token_ids"] = [1]
    try:
        engine.add_request("a", [1, 2, 3, 4, 5, 6], SamplingParams(**params))
        outputs, stats = [], []
        for _ in range(20):
            if not engine.has_unfinished_requests():
                break
            outputs.extend((out.request_id, list(out.token_ids))
                           for out in engine.step())
            stats.append(engine.last_scheduler_stats)
    finally:
        engine.shutdown()
    assert outputs[-1] == ("a", [7, 1])       # 第二个 token 命中停止
    observed = _stats_of(stats)
    assert observed
    assert observed[0].num_accepted_tokens == 3   # 截断前验证口径
    assert observed[0].num_draft_tokens == 3


@requires_cuda
def test_partial_prefill_emits_nothing_and_then_decode_continues():
    """小 token budget：长 prompt 分块 prefill 时不能提前输出；另一请求正常 decode。"""
    engine, _core = make_v1_engine(spec_k=2, budget=8, max_num_seqs=2)
    try:
        engine.add_request("long", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2],
                           SamplingParams(max_tokens=2, temperature=0.0,
                                          eos_token_id=999))
        engine.add_request("short", [2, 3, 4, 5],
                           SamplingParams(max_tokens=2, temperature=0.0,
                                          eos_token_id=999))
        first = engine.step()
        assert first == [], "12-token prompt 在 budget=8 下第一步不可能完成 prefill"
        final = {}
        for _ in range(20):
            if not engine.has_unfinished_requests():
                break
            for out in engine.step():
                final[out.request_id] = list(out.token_ids)
    finally:
        engine.shutdown()
    assert set(final) == {"long", "short"}
    assert all(len(tokens) == 2 for tokens in final.values())


def _manager() -> KVCacheManager:
    cfg = KVCacheConfig(
        num_gpu_blocks=16,
        kv_cache_groups=[
            KVCacheGroup(FullAttentionSpec(4, 1, 8, torch.float32), ["g0"]),
            KVCacheGroup(SlidingWindowSpec(8, 1, 16, torch.float32,
                                           sliding_window=8), ["g1"]),
        ],
        enable_prefix_caching=True,
    )
    return KVCacheManager(cfg, max_model_len=64)


def _request(req_id: str, length: int = 16) -> Request:
    return Request(req_id, list(range(length)),
                   SamplingParams(max_tokens=4, eos_token_id=999),
                   block_hasher=BlockHasher(4))


def test_two_group_prefix_is_all_or_nothing():
    manager = _manager()
    first = _request("a")
    manager.allocate_slots(first, 16)
    manager.cache_blocks(first, 16)
    assert manager.get_computed_blocks(_request("b"))[1] == 8
    key = make_block_hash_with_group_id(first.block_hashes[1], 1)
    sliding = manager.coordinator.managers[1].get_blocks("a")[0]
    manager.block_pool.cached_block_hash_to_block.pop(key, sliding.block_id)
    hit, hit_length = manager.get_computed_blocks(_request("b"))
    assert hit_length == 0, "任一 group 不完整时不能按另一组前缀误命中"
    assert hit.get_block_ids() == ([], [])


@requires_cuda
def test_v2_preemption_recovers_to_cold_output():
    """真实 V2 tiny MTP：小块池触发抢占，恢复输出与冷算（大块池）逐 token 相同。"""
    requests = [("a", [1, 2, 3, 4, 5, 6, 7, 8]),
                ("b", [5, 6, 7, 8]),
                ("c", [9, 10, 1, 2, 3])]

    warm_engine, warm_core = make_v2_mtp_engine(
        spec_k=2, blocks=4, budget=16, max_num_seqs=3)
    try:
        warm, _, _ = run_engine(warm_engine, requests, max_tokens=6)
        warm_preemptions = warm_core.scheduler.num_preemptions
    finally:
        warm_engine.shutdown()
    cold_engine, cold_core = make_v2_mtp_engine(
        spec_k=2, blocks=64, budget=16, max_num_seqs=3)
    try:
        cold, _, _ = run_engine(cold_engine, requests, max_tokens=6)
        assert cold_core.scheduler.num_preemptions == 0
    finally:
        cold_engine.shutdown()
    assert warm_preemptions > 0, "blocks=4 没有真正发生抢占，用例失去意义"
    assert warm == cold


@requires_cuda
def test_cancel_then_reuse_same_request_id():
    engine, core = make_v2_mtp_engine(spec_k=2, blocks=32, budget=16, max_num_seqs=2)
    try:
        engine.add_request("A", [1, 2, 3, 4],
                           SamplingParams(max_tokens=4, temperature=0.0,
                                          eos_token_id=999))
        engine.step()
        engine.abort_request(["A"])
        for _ in range(5):
            if not engine.has_unfinished_requests():
                break
            engine.step()
        assert not engine.has_unfinished_requests()
        assert core.scheduler.requests == {}

        engine.add_request("A", [5, 6, 7, 8],
                           SamplingParams(max_tokens=3, temperature=0.0,
                                          eos_token_id=999))
        final = []
        for _ in range(20):
            if not engine.has_unfinished_requests():
                break
            final.extend(out.token_ids for out in engine.step())
    finally:
        engine.shutdown()
    assert final and final[-1] == [3, 6, 8], (
        "新请求不能继承被取消请求的旧草稿/输出")
    assert all(tokens[:1] == [3] for tokens in final)


@requires_cuda
def test_failure_stops_before_next_schedule():
    engine, core = make_v1_engine(spec_k=2)
    runner = core.model_executor.driver_worker.model_runner
    original = runner.execute_model

    def boom(*args, **kwargs):
        raise RuntimeError("injected execute failure")

    engine.add_request("a", [1, 2, 3, 4],
                       SamplingParams(max_tokens=2, temperature=0.0,
                                      eos_token_id=999))
    try:
        runner.execute_model = boom
        with pytest.raises(RuntimeError, match="injected execute failure"):
            engine.step()
        assert core.failure is not None
        steps_after_failure = core.scheduler.num_steps
        with pytest.raises(RuntimeError, match="已经失败"):
            engine.step()
        assert core.scheduler.num_steps == steps_after_failure, (
            "失败后下一次 step 必须在 schedule 之前拒绝")
    finally:
        runner.execute_model = original
        engine.shutdown()
