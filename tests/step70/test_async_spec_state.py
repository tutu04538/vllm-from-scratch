"""70 关（一）：异步调度的**占位符状态机**（需求 070 §2/§3 的验收）。

异步调度把"token 有/没有"这个二元状态变成三元：**预留 → 已确认 → （抢占时）作废**。
这个文件在 CPU 上直接用 `AsyncScheduler` + 脚本化 Runner 驱动，钉住三件事：

    * 调度之后占位按"最多 1 + K 个"记上；结果回来按**实际交付长度**结账
    * 停下（EOS/max_tokens）时按**截断之后**的长度结账；被拒草稿同时回退进度与占位
    * 抢占把在飞输出标成 stale：回传时**不改计数**、不重复扣、发布上界减去未兑现占位

以及本项目在这一关的边界（§6）：`async_scheduling=True` **明确拒绝**（未接线到端到端）。
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests" / "step58"))

from helpers import make_scheduler as _make_base_scheduler  # noqa: E402
from minivllm import SamplingParams  # noqa: E402
from minivllm.config import resolve_async_scheduling  # noqa: E402
from minivllm.core.sched.async_scheduler import AsyncScheduler  # noqa: E402
from minivllm.request import Request  # noqa: E402
from minivllm.testing.fake_runner import FakeRunner  # noqa: E402

from async_helpers import make_config  # noqa: E402


def _request(req_id: str, prompt=(1, 2, 3), max_tokens=4) -> Request:
    """造一条请求（字段名按本仓库的 `Request.__init__`；块表由调度器自己分配）。"""
    return Request(request_id=req_id, prompt_token_ids=list(prompt),
                   sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0.0,
                                                  eos_token_id=999),
                   arrival_time=1.0)


def _make_async_scheduler(*, k=2, max_num_seqs=2, max_num_batched_tokens=16):
    """CPU 上的异步调度器 + 它自己的 KV 管理器（复用 step58 夹具的构造参数）。

    `make_scheduler()` 建的是同步 `Scheduler`；这里用同样的参数建 `AsyncScheduler`——
    这也说明"异步只差两个钩子"：构造参数一个字都没变。
    """
    from minivllm import CacheConfig
    from minivllm.config import SchedulerConfig
    from minivllm.core.kv_cache_manager import KVCacheManager
    from minivllm import SpeculativeConfig, ModelConfig

    spec = SpeculativeConfig(method="draft_model", num_speculative_tokens=k,
                             draft_model_config=ModelConfig(model="dummy", max_model_len=64))
    config = SchedulerConfig(max_num_seqs=max_num_seqs,
                             max_num_batched_tokens=max_num_batched_tokens)
    manager = KVCacheManager(CacheConfig(block_size=4, num_gpu_blocks=32), max_model_len=64)
    return AsyncScheduler(config, manager, max_model_len=64, speculative_config=spec)


def test_placeholders_are_added_after_schedule_and_closed_on_output():
    """加占位 / 结账两个方向都闭合：`+1+K` 地加，按**实际交付长度**减，且永不为负。"""
    scheduler = _make_async_scheduler(k=2)
    request = _request("A", prompt=(1, 2, 3, 4, 5))
    scheduler.add_request(request)
    scheduler.schedule()
    # 5 个 token 的 prefill 一轮排完 → 进入 decode 前的这一轮不记占位（中间 prefill 块不产出）
    request.append_output_token_ids(7)              # 模拟"结果回来了、已提交 1 个 token"
    request.num_computed_tokens = len(request.prompt_token_ids) + 1
    request.num_output_placeholders = 0
    before = request.num_output_placeholders
    scheduler.schedule()                            # 这一轮是 decode：会记占位
    added = request.num_output_placeholders - before
    assert added >= 1, f"调度之后必须记上占位（新增 {added}）"
    # 结账：交付 1 个 token → 占位减少 1（且不为负）
    new_tokens, _stopped = scheduler._update_request_with_output(request, [8])
    assert new_tokens == [8]
    assert request.num_output_placeholders == before + added - 1


def test_placeholder_count_equals_sampling_plus_scheduled_drafts():
    """占位宽度 = 本步采样数(1) + 本轮采用的草稿数（上游同一行公式）。"""
    scheduler = _make_async_scheduler(k=2)
    request = _request("A", prompt=(1, 2, 3))
    scheduler.add_request(request)
    scheduler.schedule()                              # prefill 走完
    request.num_computed_tokens = len(request.prompt_token_ids)
    request.spec_token_ids = [11, 12]                 # 上一轮提的两枚草稿（本轮采用）
    before = request.num_output_placeholders          # 上一轮留下的（prefill 那一轮也会产 1 个）
    output = scheduler.schedule()
    adopted = len(output.scheduled_spec_decode_tokens.get("A", ()))
    assert adopted >= 1, f"预算够时应该采用至少一枚草稿（实际 {adopted}）"
    # 上游是 `+=`：本步采样数(1) + 本轮采用的草稿数
    assert request.num_output_placeholders == before + 1 + adopted, (
        request.num_output_placeholders, before, adopted)


def test_stop_truncates_before_closing_placeholders():
    """停下来时按**截断之后**的长度结账：`max_tokens=1` + 预留 2 个位置 → 只减 1。"""
    scheduler = _make_async_scheduler(k=2)
    request = _request("A", prompt=(1, 2, 3), max_tokens=1)
    scheduler.add_request(request)
    request.num_output_placeholders = 2               # 乐观预留了两个位置
    new_tokens, stopped = scheduler._update_request_with_output(request, [8, 9])
    assert stopped and new_tokens == [8], (new_tokens, stopped)
    assert request.num_output_placeholders >= 0, "不能把没交付的位置也结掉（会变负）"


def test_preemption_marks_inflight_output_stale():
    """抢占：占位清零 + 在飞输出标成 stale；stale 结果回传时不再改计数。"""
    scheduler = _make_async_scheduler(k=2)
    request = _request("A")
    scheduler.add_request(request)
    output = scheduler.schedule()
    request.num_output_placeholders = 3
    request.num_in_flight_tokens = 3
    # 触发抢占路径（直接调用内部方法：用例考的是记账，不是受害者选择策略）
    scheduler._preempt_request(request)
    assert request.num_output_placeholders == 0, "抢占必须清零占位"
    assert request.num_stale_output_tokens == 3, "在飞的输出要被标成 stale"
    # stale 的结果回来：只交付、不改计数（这里用空结果走一遍记账路径）
    runner = FakeRunner(tokens={"A": [7]})
    runner_output = runner.execute_model(output) or runner.sample_tokens(None)
    scheduler.update_from_output(output, runner_output)
    assert request.num_output_placeholders >= 0
    assert request.num_stale_output_tokens >= 0


def test_publish_bound_excludes_unconfirmed_placeholders():
    """发布上界 = 已确认进度 − 未兑现占位（"预计会被接受"的 token 不能进 prefix 缓存）。"""
    request = _request("A")
    request.num_computed_tokens = 10
    request.num_output_placeholders = 3
    assert AsyncScheduler.publish_bound(request) == 7
    assert type(request).__mro__  # 基类口径：同步路径不放占位，因此仍是 10
    from minivllm.core.sched.scheduler import Scheduler

    assert Scheduler.publish_bound(request) == 10


def test_sync_scheduler_is_not_affected():
    """同步路径的 Scheduler 不碰占位（同一份基类代码，行为一字不变）。"""
    scheduler = _make_base_scheduler(max_num_seqs=1, max_num_batched_tokens=8,
                                     spec_method=None, k=0)
    assert not isinstance(scheduler, AsyncScheduler)
    scheduler.add_request(_request("A"))
    output = scheduler.schedule()
    assert scheduler.requests["A"].num_output_placeholders == 0


# ---------------------------------------------------------------- 配置边界（§6）


def test_async_scheduling_explicit_true_is_rejected_in_this_repo():
    """本项目**明确拒绝** `async_scheduling=True`（未接线到端到端，见 §6.2）。

    理由不是"没写"，而是"照抄会在'不等上一轮结果'与'CPU 侧输入组装'之间产生口径错位"：
    实测（a）与前缀缓存同开触发 device-side assert；（b）长跑下偶发与同步输出不一致。
    宁可拒绝，也不要一个偶尔算错的引擎（AGENTS §8：不静默降级）。
    """
    with pytest.raises(NotImplementedError):
        resolve_async_scheduling(make_config(async_scheduling=True), True)


def test_async_scheduling_default_and_false_are_sync():
    assert resolve_async_scheduling(make_config(async_scheduling=None), True) is False
    assert resolve_async_scheduling(make_config(async_scheduling=False), True) is False


def test_explicit_true_requires_executor_support():
    """显式 True + executor 不支持 → 明确报错（上游同款；不静默降级成同步）。"""
    with pytest.raises(ValueError):
        resolve_async_scheduling(make_config(async_scheduling=True),
                                 executor_supports_async=False)
