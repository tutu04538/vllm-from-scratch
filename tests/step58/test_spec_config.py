"""58：`SpeculativeConfig` 的派生量与非法配置（对应上游同名属性与容量校验）。"""

import pytest

from helpers import make_scheduler
from minivllm import ModelConfig, SpeculativeConfig


def draft_spec(k=3):
    return SpeculativeConfig(method="draft_model", num_speculative_tokens=k,
                             draft_model_config=ModelConfig(model="dummy", max_model_len=64))


def test_max_num_new_slots_for_drafting_draft_model():
    """普通自回归 draft 保留一个未切片的 token → 第一遍多要 1 行输入。"""
    assert draft_spec(3).max_num_new_slots_for_drafting == 1
    assert draft_spec(1).max_num_new_slots_for_drafting == 1


def test_max_num_new_slots_for_drafting_ngram():
    """ngram 不跑模型、不写 KV → 0。"""
    assert SpeculativeConfig(method="ngram",
                             num_speculative_tokens=3).max_num_new_slots_for_drafting == 0


def test_uses_draft_model():
    assert draft_spec().uses_draft_model() is True
    assert SpeculativeConfig(method="ngram").uses_draft_model() is False


def test_unknown_method_rejected():
    """没实现的 method 明确拒绝，不静默降级（EAGLE/MTP/PARD 在后面关卡）。"""
    with pytest.raises(ValueError, match="method"):
        SpeculativeConfig(method="eagle3", num_speculative_tokens=1)


def test_negative_num_speculative_tokens_rejected():
    with pytest.raises(ValueError, match="num_speculative_tokens"):
        SpeculativeConfig(method="ngram", num_speculative_tokens=-1)


def test_workspace_too_small_rejected_at_init():
    """M=1 装不下「1 个 target token + 1 行 draft 额外输入」→ 初始化就报错，不空转。"""
    with pytest.raises(ValueError, match="max_num_batched_tokens"):
        make_scheduler(max_num_batched_tokens=1, spec_method="draft_model")


def test_workspace_minimum_is_accepted():
    scheduler = make_scheduler(max_num_batched_tokens=2, spec_method="draft_model")
    assert scheduler.draft_slots == 1
    assert scheduler.max_num_batched_tokens == 2


def test_same_small_workspace_is_fine_without_extra_slots():
    """非投机与 ngram 都没有额外输入槽，小工作区不该被那条校验拦住。"""
    assert make_scheduler(max_num_batched_tokens=1, spec_method=None).draft_slots == 0
    assert make_scheduler(max_num_batched_tokens=1, spec_method="ngram",
                          k=3).draft_slots == 0


def test_two_budgets_share_the_same_capacity_source():
    """两份预算的起点都来自 max_num_batched_tokens（本关不加新配置项）。"""
    scheduler = make_scheduler(max_num_batched_tokens=8, spec_method="draft_model")
    assert scheduler.max_num_batched_tokens == 8
    assert scheduler.max_num_scheduled_tokens == 8
    assert scheduler.num_lookahead_tokens == 3       # KV lookahead = K，与 draft_slots 不同
