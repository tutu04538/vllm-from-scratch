"""70 关（二）：异步交付句柄的**缓冲生命周期**（需求 070 §2/§4）。

需求写明："可控 fake executor 用事件门闩延迟结果，证明 CPU 能推进且不会提前读取 buffer；
**不靠 `sleep(0.1)` 猜时序**"。本文件在 **CPU** 上做这件事（不依赖 CUDA，因此能进常规回归）：

    * 句柄**构造即发起拷贝**（不是等 `get_output()` 才拷——那样"异步"只剩名字）
    * 交付之前没人读结果；`get_output()` 才是**唯一**的读取点，且**幂等**（记账只做一次）
    * 交付之后立刻释放 device 张量引用（缓冲可复用）、清掉 Runner 上的欠账
    * 门闩挡住时 CPU 照样推进（调度不受"结果没回来"影响）

真正的端到端（异步引擎 + 真 GPU）在本仓库**未接线**（见 `docs/step70_alignment.md` §6）：
状态机与缓冲生命周期已按上游实现并被本文件的单测覆盖。
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from async_helpers import Latch  # noqa: E402
from minivllm.outputs import ModelRunnerOutput, SamplerOutput  # noqa: E402


class _FakeRunner:
    """句柄只依赖 Runner 的后半段（`_finish_async_output`）——用最小替身把它钉住。"""

    def __init__(self, latch: Latch | None = None) -> None:
        self.pending_draft_token_ids = None
        self.pending_draft_probs = None
        self.failure = None
        self.finished_calls = 0
        self.latch = latch

    # 句柄的"普通采样"分支会调它把 logprobs 摊到请求上（这里没有 logprobs）
    def _logprobs_by_request(self, tensors, sample_rows):
        return None

    def _parse_spec_sampler_output(self, state, sampler_output):   # "spec" 分支不用，占位
        raise AssertionError("本文件只测普通采样分支")

    def _finish_async_output(self, state, sampled, logprobs_by_req):
        if self.latch is not None:
            self.latch.wait()          # 模拟"结果要到交付边界才可用"
        self.finished_calls += 1
        return ModelRunnerOutput(req_ids=["A"], req_id_to_index={"A": 0},
                                 sampled_token_ids=list(sampled))


def _handle(runner, tokens=(7, 8), state=None):
    from minivllm.worker.gpu_model_runner import AsyncGPUModelRunnerOutput

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sampler_output = SamplerOutput(
        sampled_token_ids=torch.tensor([[7], [8]], dtype=torch.int32, device=device),
        logprobs_tensors=None)
    if state is None:
        from types import SimpleNamespace

        # 句柄的普通采样分支只需要 `state.sample_rows`（哪几行真的采到了 token）
        state = SimpleNamespace(sample_rows=[0, 1])
    handle = AsyncGPUModelRunnerOutput(runner=runner, state=state,
                                       sampler_output=sampler_output)
    runner._pending_async_output = handle
    return handle


def test_copy_starts_at_construction_not_at_delivery():
    """拷贝在构造句柄时就发出去（否则"异步"只是个名字）。"""
    runner = _FakeRunner()
    handle = _handle(runner)
    assert handle._started, "构造即发起 D2H"
    if torch.cuda.is_available():
        assert handle._event is not None and handle._tokens_cpu.is_pinned(), \
            "非阻塞拷贝必须落到 pinned 主机缓冲上"


def test_nothing_is_read_before_get_output():
    """交付之前不读结果：`_finish_async_output`（记账 + 提议）一次都没跑过。"""
    runner = _FakeRunner()
    handle = _handle(runner)
    assert runner.finished_calls == 0
    assert runner.pending_draft_token_ids is None


def test_get_output_is_idempotent_and_releases_buffers():
    """`get_output()` 幂等（记账只做一次）；交付后释放 device 引用并清掉欠账。"""
    runner = _FakeRunner()
    handle = _handle(runner)
    first = handle.get_output()
    second = handle.get_output()
    assert first is second, "第二次必须返回缓存（否则同一步会被结账两次）"
    assert runner.finished_calls == 1, "记账只允许发生一次"
    assert handle._sampler_output is None, "交付后要放开 device 张量（缓冲可复用）"
    assert runner._pending_async_output is None, "交付即清欠账"


def test_latch_defers_the_expensive_half_without_blocking_the_caller():
    """门闩挡住"昂贵的后半段"（记账 + 提议）时，调用方仍然可以继续做别的（CPU 推进）。

    这就是需求 §4 第一条要的证据：交付边界可以晚于执行边界——而且晚的那一段不影响
    "能不能继续推进"。这里用**门闩**而不是 `sleep`：放行前断言"没做"，放行后断言"做了"。
    """
    latch = Latch()
    runner = _FakeRunner(latch=latch)
    handle = _handle(runner)
    assert runner.finished_calls == 0, "门闩没放行 → 后半段不该发生"
    latch.release()
    output = handle.get_output()
    assert runner.finished_calls == 1
    assert output.sampled_token_ids == [[7], [8]]


def test_double_delivery_is_rejected_by_runner_state():
    """Runner 侧只有一个待交付句柄：新的结果会覆盖旧的之前，必须先把旧的结清。

    （`execute_model()` 的入口就调 `wait_for_pending_async_output()`；这里断言那个入口的
    语义：结清之后 `_pending_async_output` 为空，接着才能挂新的一份。）
    """
    runner = _FakeRunner()
    first = _handle(runner)
    assert runner._pending_async_output is first
    first.get_output()
    assert runner._pending_async_output is None
