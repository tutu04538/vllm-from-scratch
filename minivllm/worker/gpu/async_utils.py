"""V2 的异步结果交付（对应 vLLM `v1/worker/gpu/async_utils.py` 里的 `AsyncOutput`）。

为什么 V2 能异步而 V1 在本仓库不能：V2 的"结果的 D2H"只需要**采样输出张量**
（`sampled_token_ids [num_reqs, K+1]` 与 logprobs），它们与请求状态的更新（`post_update`）
是两件事——状态已经在 GPU 上按 slot scatter 落好了，所以 CPU 可以：

    1. 在**侧流**上启动 D2H（主流的 kernel 还在跑，不打断）；
    2. 先把句柄交给调度器（它可以在拷贝期间继续排下一轮）；
    3. 谁真的要值，谁 `get_output()` → 等 `copy_event` → 交 Python list。

V1 做不到这一点，是因为它的"请求状态"在 CPU 字典里、且下一轮输入要靠 CPU 拼——
结果没回来就不知道行号（见 70 关的差异账本）。

裁剪：上游这里还搬 `prompt_logprobs` / `num_nans_in_logits` / `sampling_masks` /
`routed_experts` / EP 故障检测 / 计时采集；本仓库没有这些通路（分别属于未接入的
prompt logprobs、未接指标、75 关的 synthetic 掩码、MoE），所以只留**本仓库真会交付的**
两份：`sampled_token_ids` 与 `logprobs`。
"""

import contextlib
import numpy as np
import torch

from ...outputs import AsyncModelRunnerOutput, ModelRunnerOutput
from ..gpu.sample.output import SamplerOutput
from dataclasses import dataclass, field
from collections.abc import Iterator


def async_copy_to_np(x: torch.Tensor) -> np.ndarray:
    return x.to("cpu", non_blocking=True).numpy()


@contextlib.contextmanager
def stream(to_stream: torch.cuda.Stream, from_stream: torch.cuda.Stream):
    """Lightweight version of torch.cuda.stream() context manager which
    avoids current_stream and device lookups.
    """
    try:
        torch.cuda.set_stream(to_stream)
        yield
    finally:
        torch.cuda.set_stream(from_stream)


@dataclass(frozen=True)
class StepTimingSample:
    """一个 dummy/真实 step 的两段 GPU 时间（79 关的成本曲线原料）。"""

    forward_ms: float
    drafter_ms: float
    num_target_tokens: int
    num_reqs: int
    full_cudagraph: bool


def _timing_event() -> torch.cuda.Event:
    return torch.cuda.Event(enable_timing=True)


@dataclass
class StepTimingEvents:
    forward_start: torch.cuda.Event = field(default_factory=_timing_event)
    forward_end: torch.cuda.Event = field(default_factory=_timing_event)
    drafter_start: torch.cuda.Event = field(default_factory=_timing_event)
    drafter_end: torch.cuda.Event = field(default_factory=_timing_event)


class StepTimingCollector:
    """Time every step run inside ``collect``; record calls no-op outside it.

    79 关从上游 `v1/worker/gpu/async_utils.py` 搬来的最小实现：每个 step 用独立 event，
    这样多个 step 可以背靠背排队，最后只在退出 `collect` 时同步一次。成本曲线按
    `num_target_tokens`（target 前向）与 `num_reqs`（草稿前向）分组，不能把不同执行模式的
    样本混进同一条曲线（上游注释同义）。
    """

    def __init__(self):
        self._collecting = False
        self._step: StepTimingEvents | None = None
        self._batch = (False, 0, 0)
        self._timed: list[tuple[StepTimingEvents, tuple[bool, int, int]]] = []

    @contextlib.contextmanager
    def collect(self) -> Iterator[list[StepTimingSample]]:
        samples: list[StepTimingSample] = []
        self._collecting = True
        try:
            yield samples
        finally:
            self._collecting = False
            timed, self._timed, self._step = self._timed, [], None
        if not timed:
            return
        # 同一 stream、发射顺序一致：最后一个 step 完成了，前面的也都完成了。
        timed[-1][0].drafter_end.synchronize()
        samples.extend(
            StepTimingSample(
                events.forward_start.elapsed_time(events.forward_end),
                events.drafter_start.elapsed_time(events.drafter_end),
                num_target_tokens,
                num_reqs,
                full_cudagraph,
            )
            for events, (full_cudagraph, num_target_tokens, num_reqs) in timed
        )

    def record_batch(self, input_batch, full_cudagraph: bool) -> None:
        """记录这一 step 的 batch 形状；不同执行模式不能共用一条成本曲线。"""
        self._batch = (full_cudagraph, input_batch.num_tokens, input_batch.num_reqs)

    def forward_start(self) -> None:
        if self._collecting:
            self._step = StepTimingEvents()
            self._step.forward_start.record()

    def forward_end(self) -> None:
        if self._step is not None:
            self._step.forward_end.record()

    def drafter_start(self) -> None:
        if self._step is not None:
            self._step.drafter_start.record()

    def drafter_end(self) -> None:
        """结束这一 step：只有走到这里的 step 才有草稿成本。"""
        if self._step is not None:
            self._step.drafter_end.record()
            self._timed.append((self._step, self._batch))
            self._step = None


class AsyncOutput(AsyncModelRunnerOutput):
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        sampler_output: SamplerOutput,
        num_sampled_tokens: torch.Tensor,
        main_stream: torch.cuda.Stream,
        copy_stream: torch.cuda.Stream,
    ):
        # NOTE(woosuk): We must retain references to the GPU tensors,
        # as the copy operations are performed on a different CUDA stream than
        # the one where the tensors were created.
        self.model_runner_output = model_runner_output
        self.sampler_output = sampler_output
        self.num_sampled_tokens = num_sampled_tokens
        # Blocking (sleep) event to avoid busy-polling the CUDA driver lock.
        self.copy_event = torch.cuda.Event(blocking=True)
        self._delivered = False

        with stream(copy_stream, main_stream):
            copy_stream.wait_stream(main_stream)

            self.sampled_token_ids = async_copy_to_np(sampler_output.sampled_token_ids)
            self.logprobs_tensors = None
            if sampler_output.logprobs_tensors is not None:
                self.logprobs_tensors = (
                    sampler_output.logprobs_tensors.to_cpu_nonblocking()
                )
            self.num_sampled_tokens_np = async_copy_to_np(num_sampled_tokens)
            self.copy_event.record(copy_stream)

    def get_output(self) -> ModelRunnerOutput:
        # 只允许交付一次：二次交付等于用一份可能已被复用的缓冲（70 关的句柄协议）。
        if self._delivered:
            raise RuntimeError("AsyncOutput 只能交付一次（缓冲在交付后即可被复用）")
        self._delivered = True
        self.copy_event.synchronize()

        # NOTE(woosuk): The following code is to ensure compatibility with
        # the existing model runner.
        sampled_token_ids: list[list[int]] = self.sampled_token_ids.tolist()
        num_sampled_tokens: list[int] = self.num_sampled_tokens_np.tolist()
        for token_ids, num_tokens in zip(sampled_token_ids, num_sampled_tokens):
            del token_ids[num_tokens:]
        self.model_runner_output.sampled_token_ids = sampled_token_ids

        if self.logprobs_tensors is not None:
            self.model_runner_output.logprobs = self.logprobs_tensors.tolists()

        return self.model_runner_output
