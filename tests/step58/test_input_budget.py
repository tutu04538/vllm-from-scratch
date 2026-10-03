"""58：双预算（token budget + input budget）的纯 Scheduler 单测（需求 §8.A）。"""

from helpers import add_request, make_scheduler
from minivllm import RequestStatus


def test_target_and_input_budget_together():
    """M=8、两条请求各要 4 行：排不出 target 8 / draft 10。"""
    scheduler = make_scheduler(max_num_batched_tokens=8, spec_method="draft_model")
    for req_id in ("A", "B"):
        add_request(scheduler, req_id, prompt_len=4)
    out = scheduler.schedule()
    plan = dict(out.num_scheduled_tokens)
    assert plan == {"A": 4, "B": 2}
    total = sum(plan.values())
    assert total + scheduler.draft_slots * len(plan) <= scheduler.max_num_batched_tokens
    assert scheduler.last_token_budget == 8 - total
    assert scheduler.last_input_budget == 8 - total - len(plan)


def test_waiting_obeys_input_budget_too():
    """先接纳的 A 吃光输入预算后，B 这一轮不进来（同一口径，不是只看 token 预算）。"""
    scheduler = make_scheduler(max_num_batched_tokens=8, spec_method="draft_model")
    add_request(scheduler, "A", prompt_len=8)
    add_request(scheduler, "B", prompt_len=4)
    out = scheduler.schedule()
    assert out.num_scheduled_tokens == {"A": 7}
    assert scheduler.last_input_budget == 0
    assert scheduler.last_token_budget == 1


def test_ngram_takes_no_extra_slots():
    scheduler = make_scheduler(max_num_batched_tokens=8, spec_method="ngram", k=3)
    assert scheduler.draft_slots == 0
    for req_id in ("A", "B"):
        add_request(scheduler, req_id, prompt_len=4)
    assert scheduler.schedule().num_scheduled_tokens == {"A": 4, "B": 4}


def test_guard_when_input_budget_is_not_bigger_than_draft_slots():
    """`input_budget <= draft_slots` 时本轮不排（上游 running/waiting 都有这条守卫）。"""
    scheduler = make_scheduler(max_num_batched_tokens=2, spec_method="draft_model")
    add_request(scheduler, "A", prompt_len=8)
    assert scheduler.schedule().num_scheduled_tokens == {"A": 1}


def test_non_spec_behaviour_unchanged():
    scheduler = make_scheduler(max_num_batched_tokens=8, spec_method=None)
    for req_id in ("A", "B"):
        add_request(scheduler, req_id, prompt_len=4)
    assert scheduler.schedule().num_scheduled_tokens == {"A": 4, "B": 4}


def test_preempting_scheduled_victim_restores_both_budgets():
    """victim 已在本轮计划里：撤销它的记录、两份预算都还回来、草稿计划一起删。"""
    scheduler = make_scheduler(max_num_batched_tokens=8, num_gpu_blocks=4,
                               policy="priority", spec_method="draft_model")
    a = add_request(scheduler, "A", prompt_len=2)
    scheduler.schedule()
    a.spec_token_ids = [7, 7, 7]                    # 模拟执行侧提回来的草稿
    b = add_request(scheduler, "B", prompt_len=4)
    scheduler.schedule()
    a.spec_token_ids = [7, 7, 7]
    b.spec_token_ids = [7, 7, 7]
    a.priority = 5                                  # A 成为优先级最低的 victim
    out = scheduler.schedule()
    assert out.num_scheduled_tokens == {"B": 4}
    assert a.status == RequestStatus.PREEMPTED and a.num_preemptions == 1
    assert scheduler.last_token_budget == 8 - 4
    assert scheduler.last_input_budget == 8 - 4 - 1
    assert set(out.scheduled_spec_decode_tokens) == {"B"}


def test_scheduled_spec_tokens_trimmed_by_budget():
    """预算不够时只采用草稿前缀（多余草稿丢掉，不留到下一轮）。"""
    scheduler = make_scheduler(max_num_batched_tokens=4, spec_method="draft_model")
    request = add_request(scheduler, "A", prompt_len=2)
    scheduler.schedule()
    request.spec_token_ids = [7, 8, 9, 10]
    out = scheduler.schedule()
    adopted = out.scheduled_spec_decode_tokens.get("A", [])
    assert adopted and len(adopted) < 4
    assert request.spec_token_ids == []


def test_budget_invariant_is_asserted():
    """计划必须同时满足两份预算：Σtarget ≤ 容量 且 Σ(target+draft_slots) ≤ 容量。"""
    scheduler = make_scheduler(max_num_batched_tokens=10, spec_method="draft_model")
    for req_id in ("A", "B", "C"):
        add_request(scheduler, req_id, prompt_len=6)
    out = scheduler.schedule()
    total = sum(out.num_scheduled_tokens.values())
    assert total <= 10
    assert total + scheduler.draft_slots * len(out.num_scheduled_tokens) <= 10
    assert scheduler.last_token_budget >= 0 and scheduler.last_input_budget >= 0
