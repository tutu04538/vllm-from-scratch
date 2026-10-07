"""step70 测试的共享夹具：造"可控时序"的引擎，而不是靠 sleep 猜。

需求 070 §4 的验收第一条就是"**可控 fake executor 用事件门闩延迟结果**，证明 CPU 能推进且
不会提前读取 buffer；不靠 `sleep(0.1)` 猜时序"。所以这里提供两样东西：

    make_config / make_engine   真的 tiny 引擎（可显式开异步），用于端到端一致性
    LatchExecutor               **可控 execuor**：`sample_tokens` 交出的结果被门闩挡住，
                                测试自己决定什么时候放行；同时记录"结果被读取了几次"

`LatchExecutor` 不做模型计算：它包住一个真的 Runner/Worker（或 FakeRunner），只把
"结果的交付时机"变成可控的——这正是异步调度要证明的那件事（交付边界可以晚于执行边界）。
"""

import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minivllm import (CacheConfig, DeviceConfig, LLMEngine, ModelConfig,  # noqa: E402
                      SamplingParams, SchedulerConfig, SpeculativeConfig, UniProcExecutor,
                      VllmConfig, Worker)
from minivllm.testing.tiny_models import tiny_qwen3_config, tiny_qwen3_dir  # noqa: E402

DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
TINY = tiny_qwen3_dir("tiny_gqa")
HF = tiny_qwen3_config("tiny_gqa")


def make_config(*, spec_k=None, async_scheduling=None, budget=32, blocks=32,
                max_model_len=64, max_num_seqs=2, prefix=False, method="draft_model",
                draft_dir=None, draft_config=None):
    spec = None
    if spec_k is not None:
        spec = SpeculativeConfig(
            method=method, num_speculative_tokens=spec_k,
            draft_model_config=ModelConfig(model=draft_dir or TINY, dtype="float32",
                                           max_model_len=max_model_len,
                                           hf_config=draft_config or HF))
    return VllmConfig(
        model_config=ModelConfig(model=TINY, dtype="float32", max_model_len=max_model_len,
                                 hf_config=HF),
        cache_config=CacheConfig(block_size=4, num_gpu_blocks=blocks,
                                 enable_prefix_caching=prefix),
        scheduler_config=SchedulerConfig(max_num_seqs=max_num_seqs,
                                         max_num_batched_tokens=budget,
                                         async_scheduling=async_scheduling),
        device_config=DeviceConfig(device=DEVICE), speculative_config=spec)


def make_engine(**kwargs):
    config = make_config(**kwargs)
    engine = LLMEngine(config, UniProcExecutor(config, Worker(config)))
    core = engine.engine_core.engine_core
    return engine, core, core.model_executor.driver_worker.model_runner


def run_to_end(engine, limit=400):
    final = {}
    for _ in range(limit):
        if not engine.has_unfinished_requests():
            break
        for out in engine.step():
            final[out.request_id] = list(out.token_ids)
    return final


def greedy(*, prompts=(("A", [1, 2, 3, 4, 5, 6]),), max_tokens=6, **kwargs):
    engine, _core, _runner = make_engine(**kwargs)
    for req_id, prompt in prompts:
        engine.add_request(req_id, list(prompt),
                           SamplingParams(max_tokens=max_tokens, temperature=0.0,
                                          eos_token_id=999))
    outputs = run_to_end(engine)
    engine.shutdown()
    return outputs


def drain_and_shutdown(engine, latch: "Latch | None" = None) -> None:
    """干净收尾：把在飞的异步结果**交清**再关引擎。

    为什么要专门做这件事：异步交付的拷贝在侧流上跑着，句柄没交出去就关引擎，那块 pinned
    缓冲/device 张量可能正好被下一轮的分配复用（同一个进程里还有别的用例在建引擎）——
    症状是后续用例莫名 device-side assert。测试收尾必须和生产者对齐，这也是"缓冲生命周期"
    这条要求在测试侧的落点。
    """
    if latch is not None:
        latch.release()
    core = getattr(engine.engine_core, "engine_core", None)
    if core is not None and getattr(core, "batch_queue", None):
        try:
            for _ in range(len(core.batch_queue) + 2):
                engine.engine_core.get_output()
                if not core.batch_queue:
                    break
        except Exception:                       # noqa: BLE001 —— 收尾阶段不掩盖原始断言
            pass
    engine.shutdown()


class Latch:
    """事件门闩：`wait()` 会阻塞直到 `release()`（或超时），用来把"结果什么时候可用"变成测试说了算。"""

    def __init__(self) -> None:
        self._event = threading.Event()
        self.timeout_hits = 0

    def release(self) -> None:
        self._event.set()

    def wait(self, timeout: float = 2.0) -> bool:
        ok = self._event.wait(timeout)
        if not ok:
            self.timeout_hits += 1
        return ok

    @property
    def released(self) -> bool:
        return self._event.is_set()


class LatchHandle:
    """被门闩挡住的异步结果句柄：真正的结果要等 `Latch` 放行才读得到。"""

    def __init__(self, latch: Latch, inner, stats: dict) -> None:
        self._latch = latch
        self._inner = inner
        self._stats = stats

    def get_output(self):
        self._stats["get_output_calls"] += 1
        if not self._latch.wait():
            # 超时不是错误：调用方（测试）就是要看"没放行时会怎样"
            self._stats["timeouts"] += 1
        if self._inner is None:
            raise RuntimeError("结果还没准备好（门闩没放行）")
        if hasattr(self._inner, "get_output"):
            return self._inner.get_output()
        return self._inner


class LatchExecutor:
    """包住真 executor：把 `sample_tokens(non_block=True)` 的结果用门闩挡起来。

    它只改"**什么时候**交出结果"，不改结果本身——所以"同步/异步输出一致"这类断言仍然有效，
    而"结果没回来时 CPU 已经推进了"这件事可以被精确观察（这是需求 §4 第一条要的证据）。
    """

    def __init__(self, inner, latch: Latch) -> None:
        self.inner = inner
        self.latch = latch
        self.stats = {"get_output_calls": 0, "timeouts": 0, "sample_calls": 0,
                      "execute_calls": 0}

    # ---- 转发（协议不变）----
    def get_cache_config(self):
        return self.inner.get_cache_config()

    def initialize_kv_cache(self, kv_cache_config):
        return self.inner.initialize_kv_cache(kv_cache_config)

    def supports_async_scheduling(self) -> bool:
        return True

    def take_draft_token_ids(self):
        return self.inner.take_draft_token_ids()

    def shutdown(self) -> None:
        return self.inner.shutdown()

    # ---- 被观察的两个入口 ----
    def execute_model(self, scheduler_output, non_block: bool = False):
        self.stats["execute_calls"] += 1
        if non_block:
            # 门闩**只挡采样结果的交付**，不挡执行本身（执行是 GPU 侧异步提交）
            self.inner.execute_model(scheduler_output, non_block=True)
            future = __import__("concurrent.futures", fromlist=["Future"]).Future()
            future.set_result(None)
            return future
        return self.inner.execute_model(scheduler_output, non_block=False)

    def sample_tokens(self, grammar_output, non_block: bool = False):
        self.stats["sample_calls"] += 1
        output = self.inner.sample_tokens(grammar_output, non_block=True)
        if hasattr(output, "result"):       # 内层 executor 也把非阻塞结果包成 Future
            output = output.result()
        handle = LatchHandle(self.latch, output, self.stats)
        if not non_block:
            return handle.get_output()
        future = __import__("concurrent.futures", fromlist=["Future"]).Future()
        future.set_result(handle)
        return future
