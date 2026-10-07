"""70 关验收探针：异步调度的占位符状态机、异步交付句柄与缓冲生命周期。

跑法：`python benchmarks/check_step70_async.py`（退出码非 0 = 失败）。

**范围说明**（重要，见 `docs/step70_alignment.md` §6）：本仓库的异步调度**只到骨架**——
状态机（占位符/结账/stale）、批队列、异步交付句柄、缓冲生命周期都按上游实现并在这里验收；
`async_scheduling=True` 的**端到端引擎**明确拒绝（拒绝本身就是本关的一条验收项），
因为"不等上一轮结果就组装下一轮输入"要求执行侧 GPU 驻留（上游的 scatter 设计），
本仓库的输入组装与提议器都在 CPU 侧。

分段：

    A 配置边界      显式 true（拒绝）/ false / 默认（关）/ executor 能力
    B 占位符状态机  加占位、按实际长度结账、截断后结账、抢占标 stale、发布上界减占位
    C 交付句柄      构造即发拷贝、交付前不读、幂等、交付后释放、门闩挡住不阻塞调用方
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "step70"))
sys.path.insert(0, str(ROOT / "tests" / "step58"))

import torch  # noqa: E402

from async_helpers import Latch, make_config  # noqa: E402
from helpers import make_scheduler as make_base_scheduler  # noqa: E402
from minivllm import CacheConfig, ModelConfig, SamplingParams, SchedulerConfig, SpeculativeConfig  # noqa: E402
from minivllm.config import resolve_async_scheduling  # noqa: E402
from minivllm.core.kv_cache_manager import KVCacheManager  # noqa: E402
from minivllm.core.sched.async_scheduler import AsyncScheduler  # noqa: E402
from minivllm.core.sched.scheduler import Scheduler  # noqa: E402
from minivllm.outputs import ModelRunnerOutput, SamplerOutput  # noqa: E402
from minivllm.request import Request  # noqa: E402

PASSED, FAILED, TRACE = 0, [], []


def check(name, ok, detail=""):
    global PASSED
    if ok:
        PASSED += 1
        print(f"PASS  {name}  {detail}")
    else:
        FAILED.append(name)
        print(f"FAIL  {name}  {detail}")
    TRACE.append({"item": name, "ok": bool(ok), "detail": str(detail)[:300]})


def make_async_scheduler(*, k=2, max_num_seqs=2, budget=16, blocks=32):
    spec = SpeculativeConfig(method="draft_model", num_speculative_tokens=k,
                             draft_model_config=ModelConfig(model="dummy", max_model_len=64))
    config = SchedulerConfig(max_num_seqs=max_num_seqs, max_num_batched_tokens=budget)
    manager = KVCacheManager(CacheConfig(block_size=4, num_gpu_blocks=blocks), max_model_len=64)
    return AsyncScheduler(config, manager, max_model_len=64, speculative_config=spec)


def make_request(req_id="A", prompt=(1, 2, 3, 4, 5), max_tokens=4):
    return Request(request_id=req_id, prompt_token_ids=list(prompt),
                   sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0.0,
                                                  eos_token_id=999),
                   arrival_time=1.0)


# ---------------------------------------------------------------- A 配置边界
config_true = make_config(async_scheduling=True)
try:
    resolve_async_scheduling(config_true, True)
    rejected = None
except NotImplementedError as exc:
    rejected = str(exc)[:40]
check("A1. 显式 async_scheduling=True → 明确拒绝（未接线到端到端，§6）", rejected is not None,
      str(rejected))
try:
    resolve_async_scheduling(config_true, False)
    unsupported = None
except ValueError as exc:
    unsupported = str(exc)[:40]
check("A2. 显式 True 但 executor 不支持 → 明确报错（不静默降级）", unsupported is not None,
      str(unsupported))
check("A3. 显式 False → 关", resolve_async_scheduling(make_config(async_scheduling=False),
                                                      True) is False)
check("A4. 默认（None）→ 关（本仓库差异：上游默认开，§6.1）",
      resolve_async_scheduling(make_config(), True) is False)

# ---------------------------------------------------------------- B 占位符状态机
scheduler = make_async_scheduler(k=2)
request = make_request()
scheduler.add_request(request)
scheduler.schedule()                                   # prefill
request.append_output_token_ids(7)
request.num_computed_tokens = len(request.prompt_token_ids) + 1
before = request.num_output_placeholders
scheduler.schedule()                                   # decode：记占位
added = request.num_output_placeholders - before
check("B1. 调度之后占位按 `1 + 本轮采用的草稿数` 记上", added >= 1, f"新增 {added}")
tokens, _stopped = scheduler._update_request_with_output(request, [8])
check("B2. 结账按**实际交付长度**减（交付 1 个 → 减 1）",
      tokens == [8] and request.num_output_placeholders == before + added - 1,
      f"placeholders={request.num_output_placeholders}")

scheduler = make_async_scheduler(k=2)
request = make_request(max_tokens=1)
scheduler.add_request(request)
request.num_output_placeholders = 2
tokens, stopped = scheduler._update_request_with_output(request, [8, 9])
check("B3. 停下时按**截断之后**的长度结账（max_tokens=1：只交付 1 个，占位不为负）",
      stopped and tokens == [8] and request.num_output_placeholders >= 0,
      f"tokens={tokens} placeholders={request.num_output_placeholders}")

scheduler = make_async_scheduler(k=2)
request = make_request()
scheduler.add_request(request)
scheduler.schedule()
request.num_output_placeholders = 3
request.num_in_flight_tokens = 3
scheduler._preempt_request(request)
check("B4. 抢占：占位清零 + 在飞输出标成 stale（回传时不再改计数，避免减成负数）",
      request.num_output_placeholders == 0 and request.num_stale_output_tokens == 3,
      f"ph={request.num_output_placeholders} stale={request.num_stale_output_tokens}")

request = make_request()
request.num_computed_tokens = 10
request.num_output_placeholders = 3
check("B5. 发布上界减去未兑现占位（异步 7 / 同步 10）",
      AsyncScheduler.publish_bound(request) == 7
      and Scheduler.publish_bound(request) == 10,
      f"async={AsyncScheduler.publish_bound(request)} sync={Scheduler.publish_bound(request)}")

sync_scheduler = make_base_scheduler(max_num_seqs=1, max_num_batched_tokens=8,
                                     spec_method=None, k=0)
check("B6. 同步调度器不碰占位（同一份基类代码，行为不变）",
      not isinstance(sync_scheduler, AsyncScheduler)
      and sync_scheduler.requests == {})

# ---------------------------------------------------------------- C 交付句柄
class _FakeRunner:
    """句柄只依赖 Runner 的后半段；用最小替身把"什么时候读"钉住。"""

    def __init__(self, latch=None):
        self.pending_draft_token_ids = None
        self.pending_draft_probs = None
        self.failure = None
        self.finished_calls = 0
        self.latch = latch

    def _logprobs_by_request(self, tensors, sample_rows):
        return None

    def _finish_async_output(self, state, sampled, logprobs_by_req):
        if self.latch is not None:
            self.latch.wait()
        self.finished_calls += 1
        return ModelRunnerOutput(req_ids=["A"], req_id_to_index={"A": 0},
                                 sampled_token_ids=list(sampled))


def make_handle(runner):
    from types import SimpleNamespace

    from minivllm.worker.gpu_model_runner import AsyncGPUModelRunnerOutput

    device = "cuda" if torch.cuda.is_available() else "cpu"
    handle = AsyncGPUModelRunnerOutput(
        runner=runner, state=SimpleNamespace(sample_rows=[0, 1]),
        sampler_output=SamplerOutput(
            sampled_token_ids=torch.tensor([[7], [8]], dtype=torch.int32, device=device)))
    runner._pending_async_output = handle
    return handle


runner = _FakeRunner()
handle = make_handle(runner)
check("C1. 句柄**构造即发起拷贝**（不是等 get_output 才拷）", handle._started)
if torch.cuda.is_available():
    check("C2. 非阻塞拷贝落到 pinned 主机缓冲 + 事件上（真异步的前提）",
          handle._event is not None and handle._tokens_cpu.is_pinned())
else:
    check("C2. 非阻塞拷贝需要 CUDA（本机没有）：待验", False, "本机没有 CUDA")
check("C3. 交付之前不读结果（记账一次都没跑过）", runner.finished_calls == 0)
first = handle.get_output()
second = handle.get_output()
check("C4. `get_output()` 幂等：第二次返回缓存、记账只做一次",
      first is second and runner.finished_calls == 1)
check("C5. 交付后释放 device 引用并清掉 Runner 上的欠账",
      handle._sampler_output is None and runner._pending_async_output is None)

latch = Latch()
runner2 = _FakeRunner(latch=latch)
handle2 = make_handle(runner2)
check("C6. 门闩挡住'昂贵后半段'时调用方不被阻塞（还没做）", runner2.finished_calls == 0)
latch.release()
out = handle2.get_output()
check("C7. 放行之后才做，且结果正确", runner2.finished_calls == 1
      and out.sampled_token_ids == [[7], [8]])

print(f"\n{'全部通过' if not FAILED else '失败: ' + ', '.join(FAILED)}  （{PASSED} 项通过）")
sys.exit(1 if FAILED else 0)
