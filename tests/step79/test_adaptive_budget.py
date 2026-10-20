"""79 关：CPU 预算、GPU 分配、成本表与能力检查的单元测试。

这里的数字都是**手算可验**的：存活概率、top-k、成本表阶梯/插值、`argmax` 的 tie 规则
都直接写进断言，不依赖模型输出。
"""

from __future__ import annotations

import types

import numpy as np
import pytest
import torch

from adaptive_helpers import DEVICE, ManagerFixture

from minivllm.attention.backend import AttentionCGSupport
from minivllm.worker.gpu.async_utils import StepTimingSample
from minivllm.worker.gpu.attn_utils import AttentionCGSupportInfo
from minivllm.worker.gpu.spec_decode.adaptive_verification import (
    _assign_draft_token_budget,
    build_cost_tables_from_curves,
    maybe_create_adaptive_verification_manager,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="V2 的 DSpark/置信度缓冲需要 CUDA")


# ---------------- 成本表 ----------------


def test_cost_table_is_a_step_function_below_the_capture_limit():
    """捕获上限内按"补齐到下一个实测点"取成本，不能做平滑插值。"""
    draft, verify = build_cost_tables_from_curves(
        [(1, 1.0), (4, 4.0)], [(1, 2.0), (4, 5.0)], max_num_reqs=4,
        max_batch_tokens=4, cudagraph_limit=0,
    )
    assert draft.tolist() == [1.0, 1.0, 4.0, 4.0, 4.0]
    assert verify.tolist() == [2.0, 2.0, 5.0, 5.0, 5.0]


def test_cost_table_interpolates_only_above_the_capture_limit():
    """上限以上不做 pad，成本连续；跨过上限本身仍然不连续。"""
    _, verify = build_cost_tables_from_curves(
        [(1, 1.0)], [(1, 1.0), (4, 10.0), (8, 14.0), (12, 20.0)],
        max_num_reqs=4, max_batch_tokens=12, cudagraph_limit=4,
    )
    # 1、2 落在 (1,1) 后面、4 之前 → 取 10；4 也是 10。
    assert verify[:5].tolist() == [1.0, 1.0, 10.0, 10.0, 10.0]
    # 5..7 在"上限 4"和"第一个上限以上的点 8"之间：不能与 4 那点混合，取 14。
    assert verify[5:8].tolist() == [14.0, 14.0, 14.0]
    # 8..12 全在上限以上，按 (8,14)-(12,20) 插值。
    assert verify[8:].tolist() == [14.0, 15.5, 17.0, 18.5, 20.0]


def test_cost_table_tail_is_linear_not_snapped():
    """超出最后一个实测点时按最后一段斜率外推，不能一直卡在最后一个值。"""
    draft, _ = build_cost_tables_from_curves(
        [(1, 1.0), (4, 4.0)], [(1, 1.0)],
        max_num_reqs=8, max_batch_tokens=4,
    )
    assert draft[5:].tolist() == [5.0, 6.0, 7.0, 8.0]


def test_cost_table_forces_monotonic_nonnegative_values():
    """profiling 噪声不能让"多一行更便宜"；负成本裁到 0，verify 裁到 1e-6。"""
    draft, verify = build_cost_tables_from_curves(
        [(1, 5.0), (4, 2.0)], [(1, -1.0), (4, 0.0)],
        max_num_reqs=4, max_batch_tokens=4,
    )
    assert draft.tolist() == [5.0, 5.0, 5.0, 5.0, 5.0]
    assert (verify == 1e-6).all()


# ---------------- GPU 全局 top-k ----------------


@requires_cuda
def test_assign_budget_takes_globally_best_prefixes():
    """存活概率全局 top-k；同一请求内天然是连续前缀。"""
    conf = torch.tensor(
        [[0.9, 0.1, 0.1], [0.8, 0.7, 0.6]], dtype=torch.float32, device=DEVICE)
    # A: [0.9, .09, .009]；B: [.8, .56, .336]；top-3 = A0, B0, B1。
    capacities = torch.tensor([3, 3], dtype=torch.int32, device=DEVICE)
    _assign_draft_token_budget(conf, torch.tensor([0, 1], device=DEVICE),
                               capacities, 3, 3)
    assert capacities.tolist() == [1, 2]


@requires_cuda
def test_assign_budget_respects_per_request_capacities():
    """`capacities` 是"原有效候选数"：越界槽位 -inf，绝不能挤掉真实草稿。"""
    conf = torch.tensor(
        [[0.1, 0.1, 0.1], [0.9, 0.8, 0.7]], dtype=torch.float32, device=DEVICE)
    capacities = torch.tensor([1, 3], dtype=torch.int32, device=DEVICE)
    _assign_draft_token_budget(conf, torch.tensor([0, 1], device=DEVICE),
                               capacities, 2, 3)
    assert capacities.tolist() == [0, 2]


@requires_cuda
def test_assign_budget_handles_zero_confidence_and_ties():
    """全零置信度仍是合法输入：只要求总数/前缀合法，不自己发明 tie 规则。"""
    conf = torch.zeros((2, 3), dtype=torch.float32, device=DEVICE)
    capacities = torch.tensor([3, 3], dtype=torch.int32, device=DEVICE)
    _assign_draft_token_budget(conf, torch.tensor([0, 1], device=DEVICE),
                               capacities, 4, 3)
    got = capacities.tolist()
    assert sum(got) == 4
    assert all(0 <= c <= 3 for c in got)


@requires_cuda
def test_assign_budget_updates_in_place_with_capacity_dtype():
    conf = torch.full((1, 4), 0.5, dtype=torch.float32, device=DEVICE)
    capacities = torch.tensor([4], dtype=torch.int32, device=DEVICE)
    _assign_draft_token_budget(conf, torch.tensor([0], device=DEVICE),
                               capacities, 2, 4)
    assert capacities.dtype == torch.int32
    assert capacities.tolist() == [2]


# ---------------- CPU 预算 ----------------


@requires_cuda
def test_budget_full_when_extra_rows_are_free():
    """成本不随行数增加 → 期望接受数单调增 → 选满预算。"""
    fx = ManagerFixture()
    fx.manager.set_cost_curves(
        [(1, 1.0), (4, 1.0)], [(0, 1.0), (64, 1.0)])
    fx.set_stale({"a": [0.9, 0.8, 0.7], "b": [0.5, 0.4, 0.3]})
    fx.select()
    assert fx.manager._batch_budget[2] == 6


@requires_cuda
def test_budget_zero_when_extra_rows_are_expensive():
    """每多验一行成本陡增 → 只验 bonus 行。"""
    fx = ManagerFixture()
    fx.manager.set_cost_curves(
        [(1, 1.0), (4, 1.0)], [(n, 1000.0 * n) for n in range(0, 65)])
    fx.set_stale({"a": [0.9, 0.8, 0.7], "b": [0.5, 0.4, 0.3]})
    fx.select()
    assert fx.manager._batch_budget[2] == 0


@requires_cuda
def test_budget_intermediate_is_hand_computable():
    """2 条请求、成本在第 5 行前是平的 → 验证 3 枚。

    存活概率：A [0.9, 0.72, 0.504]，B [0.5, 0.2, 0.06]。
    b=0..3 的期望接受数 = 2 + [0, 0.9, 1.62, 2.124]；成本都是 draft(10)+verify(1)。
    所以 argmax 落在 b=3。
    """
    fx = ManagerFixture()
    fx.manager.set_cost_curves(
        [(1, 10.0), (4, 10.0)], [(0, 1.0), (5, 1.0), (64, 1000.0)])
    fx.set_stale({"a": [0.9, 0.8, 0.7], "b": [0.5, 0.4, 0.3]})
    assert fx.select() == 2 + 3  # 非草稿 1+1，预算 3
    assert fx.manager._batch_budget[2] == 3


@requires_cuda
def test_budget_tie_picks_the_smaller_budget():
    """所有 b 的收益/成本相同 → `np.argmax` 取第一个（b=0），不自创规则。"""
    fx = ManagerFixture()
    # 置信度全 0 → 期望接受数恒定；成本也恒定 → 所有 b 打平。
    fx.manager.set_cost_curves(
        [(1, 1.0), (4, 1.0)], [(0, 1.0), (64, 1.0)])
    fx.set_stale({"a": [0.0, 0.0, 0.0], "b": [0.0, 0.0, 0.0]})
    fx.select()
    assert fx.manager._batch_budget[2] == 0


@requires_cuda
def test_budget_is_capped_by_the_logit_chunk_limit():
    """预算必须能装进一个验证 chunk：`max_total_logits - B*bonus` 是硬上限。"""
    fx = ManagerFixture(k=4, max_total_logits=5)  # 2 reqs * 1 bonus -> 最多 3
    fx.manager.set_cost_curves(
        [(1, 1.0), (4, 1.0)], [(0, 1.0), (64, 1.0)])
    fx.set_stale({"a": [1.0] * 4, "b": [1.0] * 4})
    fx.select(tokens={"a": 5, "b": 5},
              drafts={"a": [1, 2, 3, 4], "b": [1, 2, 3, 4]})
    assert fx.manager._batch_budget[2] == 3


@requires_cuda
def test_prefill_request_gets_no_bonus_token_in_the_estimate():
    """中间 prefill 块不产 token；预算估计里的 bonus 只数真正会采样的请求。"""
    fx = ManagerFixture(prefill_len=10)
    # a 还在 prefill（computed=0 + 非草稿 1 < 10），b 已完成 prefill。
    fx.state.num_computed_tokens_np[fx.slot("a")] = 0
    fx.state.num_computed_tokens_np[fx.slot("b")] = 10
    fx.manager.set_cost_curves(
        [(1, 10.0), (4, 10.0)], [(0, 1.0), (64, 1.0)])
    fx.set_stale({"a": [0.0, 0.0, 0.0], "b": [0.0, 0.0, 0.0]})
    fx.select()
    # 置信度全 0 → 期望接受数只剩 bonus；a 还在 prefill，不能给它算 bonus。
    assert fx.manager.last_num_sampling_requests == 1
    assert fx.manager.last_scheduled_drafts == 6


# ---------------- compact_batch / reallocate_drafts ----------------


@requires_cuda
def test_compact_batch_full_budget_returns_original_layout():
    fx = ManagerFixture()
    fx.manager.set_cost_curves(
        [(1, 1.0), (4, 1.0)], [(0, 1.0), (64, 1.0)])
    fx.select()
    scheduled = np.array([4, 4], dtype=np.int32)
    cu = np.array([0, 4, 8], dtype=np.int32)
    got_tokens, got_cu = fx.manager.compact_batch(
        np.array([3, 3], dtype=np.int32), scheduled, cu)
    assert got_tokens is scheduled or np.array_equal(got_tokens, scheduled)
    assert got_cu is cu or np.array_equal(got_cu, cu)


@requires_cuda
def test_compact_batch_zero_budget_is_exact():
    """预算 0 时 CPU offsets 是精确值：每请求只有 bonus 行。"""
    fx = ManagerFixture()
    fx.manager._batch_budget = ({"a": 3, "b": 3}, {"a": 1, "b": 1}, 0)
    tokens, cu = fx.manager.compact_batch(
        np.array([3, 3], dtype=np.int32), np.array([4, 4], dtype=np.int32),
        np.array([0, 4, 8], dtype=np.int32))
    assert tokens.tolist() == [1, 1]
    assert cu.tolist() == [0, 1, 2]


@requires_cuda
def test_compact_batch_intermediate_is_even_cpu_upper_bound():
    """中间预算在 CPU 上均分；这是上界，不是 GPU 真实前缀。"""
    fx = ManagerFixture()
    fx.manager._batch_budget = ({"a": 3, "b": 3}, {"a": 1, "b": 1}, 3)
    tokens, cu = fx.manager.compact_batch(
        np.array([3, 3], dtype=np.int32), np.array([4, 4], dtype=np.int32),
        np.array([0, 4, 8], dtype=np.int32))
    assert tokens.tolist() == [3, 2]  # 非草稿 [1,1] + 均分 [2,1]
    assert cu.tolist() == [0, 4, 8]   # 原样保留，等 GPU 修正


@requires_cuda
def test_compact_batch_asserts_verification_requests_lead():
    """批量顺序由 `sort_batch_req_ids` 保证：有草稿的必须在前。"""
    fx = ManagerFixture()
    fx.manager._batch_budget = ({"a": 0, "b": 3}, {"a": 2, "b": 1}, 1)
    with pytest.raises(AssertionError):
        fx.manager.compact_batch(
            np.array([0, 3], dtype=np.int32), np.array([2, 4], dtype=np.int32),
            np.array([0, 2, 6], dtype=np.int32))


@requires_cuda
def test_reallocate_full_and_zero_budget_offsets():
    fx = ManagerFixture()
    fx.manager._batch_budget = ({"a": 3, "b": 1}, {"a": 1, "b": 1}, 4)
    fx.manager._confidence_probs[:] = 0.5
    cu, qsl, total = fx.manager.reallocate_drafts(["a", "b"], fx.indices())
    assert total == 4
    assert cu.tolist() == [0, 4, 6]       # caps [3,1] + bonus 1
    assert qsl[:3].tolist() == [0, 4, 6]  # caps + 非草稿 [1,1]
    assert fx.manager._batch_budget is None

    fx.manager._batch_budget = ({"a": 3, "b": 1}, {"a": 1, "b": 1}, 0)
    cu, qsl, total = fx.manager.reallocate_drafts(["a", "b"], fx.indices())
    assert total == 0
    assert cu.tolist() == [0, 1, 2]
    assert qsl[:3].tolist() == [0, 1, 2]
    assert fx.manager._batch_draft_capacity[:2].tolist() == [0, 0]


@requires_cuda
def test_reallocate_uses_gpu_current_confidence_not_cpu_stale():
    """CPU 预算来自 stale，GPU 前缀来自当前值：两份数据刻意不同。"""
    fx = ManagerFixture()
    # stale（CPU 预算用）说 A 差、B 好；当前 GPU 值反过来。
    fx.set_stale({"a": [0.1, 0.1, 0.1], "b": [0.9, 0.8, 0.7]})
    fx.manager._batch_budget = ({"a": 3, "b": 3}, {"a": 1, "b": 1}, 3)
    fx.manager._confidence_probs[fx.slot("a")] = torch.tensor(
        [0.9, 0.8, 0.7], device=DEVICE)
    fx.manager._confidence_probs[fx.slot("b")] = torch.tensor(
        [0.1, 0.1, 0.1], device=DEVICE)
    cu, _, total = fx.manager.reallocate_drafts(["a", "b"], fx.indices())
    assert total == 3
    # 当前值下 A 的前三枚全进，B 一枚不进：A 4 行 logits、B 只有 bonus。
    assert cu.tolist() == [0, 4, 5]


@requires_cuda
def test_batch_budget_is_consumed_exactly_once():
    fx = ManagerFixture()
    fx.manager.set_cost_curves(
        [(1, 1.0), (4, 1.0)], [(0, 1.0), (64, 1.0)])
    fx.select()
    fx.manager.reallocate_drafts(["a", "b"], fx.indices())
    assert fx.manager._batch_budget is None
    with pytest.raises(AssertionError):
        fx.manager.reallocate_drafts(["a", "b"], fx.indices())


# ---------------- 置信度双缓冲 ----------------


@requires_cuda
def test_record_confidences_keeps_two_generations_and_resets_new_requests():
    fx = ManagerFixture()
    fx.manager.set_cost_curves(
        [(1, 1.0), (4, 1.0)], [(n, 1000.0 * n) for n in range(0, 65)])
    idx = fx.indices()
    batch = types.SimpleNamespace(num_reqs=2, idx_mapping=idx)

    # 第一轮：写 A，但预算仍读初始化值 1.0。
    fx.manager.record_confidences(
        torch.zeros((2, 3), device=DEVICE), batch)
    fx.manager._copy_events[0].synchronize()
    assert fx.manager._stale_idx == 1
    used = [fx.slot(rid) for rid in fx.req_ids]
    assert np.all(fx.manager._stale_confidences[1].np[used] == 1.0)

    # 第二轮：写 B；预算现在读第一轮的 A（零置信度 → 预算 0）。
    fx.manager.record_confidences(
        torch.ones((2, 3), device=DEVICE), batch)
    fx.manager._copy_events[1].synchronize()
    assert fx.manager._stale_idx == 0
    assert np.all(fx.manager._stale_confidences[0].np[used] == 0.0)
    assert fx.manager._batch_budget is None
    fx.select()
    assert fx.manager._batch_budget[2] == 0

    # 新请求占用 slot 时，两代都重置成 1.0。
    slot = fx.slot("a")
    fx.manager.add_request(slot)
    assert np.all(fx.manager._stale_confidences[fx.manager._stale_idx].np[slot] == 1.0)
    assert torch.all(fx.manager._confidence_probs[slot] == 1.0)


# ---------------- profiling 曲线 ----------------


def test_batches_to_profile_covers_capture_and_tail_sizes():
    fx = ManagerFixture()
    got = list(fx.manager.batches_to_profile([4, 8]))
    assert [b["num_tokens"] for b in got] == [4] * 5 + [8] * 5 + [12] * 5 + [16] * 5 + [32] * 5 + [64] * 5
    assert all(b["context_len"] == 0 for b in got)
    assert fx.manager._cudagraph_limit == 8


def test_set_initial_cost_curves_uses_medians_and_prefers_full_graph_samples():
    fx = ManagerFixture()
    samples = [
        StepTimingSample(2.0, 10.0, 4, 2, True),
        StepTimingSample(4.0, 14.0, 4, 2, False),
        StepTimingSample(6.0, 30.0, 8, 2, True),
        StepTimingSample(8.0, 50.0, 8, 2, False),
    ]
    fx.manager.set_initial_cost_curves(samples)
    draft, verify = fx.manager.cost_curves
    # draft 只用 full 样本：num_reqs=2 的 [10,30] 中位数 20。
    assert draft == [(2, 20.0)]
    # verify 用全部样本：target=4 的 [2,4] → 3；target=8 的 [6,8] → 7。
    assert verify == [(4, 3.0), (8, 7.0)]


def test_set_initial_cost_curves_falls_back_when_no_full_graph_samples():
    fx = ManagerFixture()
    samples = [StepTimingSample(2.0, 10.0, 4, 2, False),
               StepTimingSample(4.0, 14.0, 4, 2, False)]
    fx.manager.set_initial_cost_curves(samples)
    assert fx.manager.cost_curves[0] == [(2, 12.0)]


def test_set_cost_curves_rejects_empty_curves():
    fx = ManagerFixture()
    with pytest.raises(RuntimeError, match="could not profile"):
        fx.manager.set_cost_curves([], [(1, 1.0)])


# ---------------- 能力检查 ----------------


class _FakeBackend:
    def __init__(self, name, mismatch):
        self.__name__ = name
        self._mismatch = mismatch

    def supports_device_cpu_query_lens_mismatch(self):
        return self._mismatch


@requires_cuda
def test_manager_rejects_backends_that_plan_from_cpu_query_lens():
    fx = ManagerFixture()
    bad = _FakeBackend("BAD", False)
    group = types.SimpleNamespace(backend=bad)
    with pytest.raises(ValueError, match="BAD"):
        maybe_create_adaptive_verification_manager(
            enable_adaptive_verification=True,
            attn_groups=[[group]],
            attn_cg_support=AttentionCGSupportInfo(
                AttentionCGSupport.ALWAYS, "BAD"),
            req_states=fx.state,
            query_start_loc=fx.query_start_loc,
            num_bonus_tokens=1,
            max_total_logits=64,
        )


@requires_cuda
def test_manager_requires_always_cudagraph_support_by_default():
    fx = ManagerFixture()
    good = _FakeBackend("GOOD", True)
    group = types.SimpleNamespace(backend=good)
    with pytest.raises(ValueError, match="ALWAYS"):
        maybe_create_adaptive_verification_manager(
            enable_adaptive_verification=True,
            attn_groups=[[group]],
            attn_cg_support=AttentionCGSupportInfo(
                AttentionCGSupport.UNIFORM_BATCH, "GOOD"),
            req_states=fx.state,
            query_start_loc=fx.query_start_loc,
            num_bonus_tokens=1,
            max_total_logits=64,
        )
    # 本仓库 target 图未接时 runner 传 False：语义能力满足即可建 manager。
    manager = maybe_create_adaptive_verification_manager(
        enable_adaptive_verification=True,
        attn_groups=[[group]],
        attn_cg_support=AttentionCGSupportInfo(
            AttentionCGSupport.UNIFORM_BATCH, "GOOD"),
        req_states=fx.state,
        query_start_loc=fx.query_start_loc,
        num_bonus_tokens=1,
        max_total_logits=64,
        require_varlen_cudagraph=False,
    )
    assert manager is not None


@requires_cuda
def test_manager_returns_none_when_disabled():
    fx = ManagerFixture()
    assert maybe_create_adaptive_verification_manager(
        enable_adaptive_verification=False,
        attn_groups=[],
        attn_cg_support=None,
        req_states=fx.state,
        query_start_loc=fx.query_start_loc,
        num_bonus_tokens=1,
        max_total_logits=64,
    ) is None


@requires_cuda
def test_reallocate_is_order_independent_via_req_id_dicts():
    """batch 行顺序可以重排：预算字典按 req_id 取，不能按数组下标猜请求。"""
    fx = ManagerFixture()
    fx.manager._batch_budget = ({"a": 3, "b": 3}, {"a": 1, "b": 1}, 3)
    fx.manager._confidence_probs[fx.slot("a")] = torch.tensor(
        [0.9, 0.8, 0.7], device=DEVICE)
    fx.manager._confidence_probs[fx.slot("b")] = torch.tensor(
        [0.1, 0.1, 0.1], device=DEVICE)
    idx = torch.tensor([fx.slot("b"), fx.slot("a")], dtype=torch.int32,
                       device=DEVICE)
    cu, qsl, total = fx.manager.reallocate_drafts(["b", "a"], idx)
    assert total == 3
    # 行顺序是 b, a：b 0 枚、a 3 枚；每请求一个 bonus。
    assert cu.tolist() == [0, 1, 5]
    assert qsl[:3].tolist() == [0, 1, 5]


def test_config_accepts_adaptive_for_dspark_and_rejects_other_methods():
    """配置期只放行 DSpark；其余方法仍按上游报错（执行期能力检查在 Runner）。"""
    from adaptive_helpers import model_configs
    from minivllm import SpeculativeConfig

    _, draft_config = model_configs(confidence_head=True)
    spec = SpeculativeConfig(
        method="dspark", num_speculative_tokens=3,
        draft_model_config=draft_config, enable_adaptive_verification=True)
    assert spec.enable_adaptive_verification is True
    with pytest.raises(ValueError, match="only supported with DSpark"):
        SpeculativeConfig(
            method="draft_model", num_speculative_tokens=2,
            draft_model_config=draft_config, enable_adaptive_verification=True)
