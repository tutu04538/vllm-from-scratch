"""UniProcExecutor：单进程执行部署（对应 vLLM `v1/executor/uniproc_executor.py`）。

Executor 回答的是"**交给哪种执行部署**"：本关只有单进程一种，所以它直接持有一个 Worker 并
转发。真实 vLLM 外面还有 `WorkerWrapperBase`（用来做 RPC / 动态 worker 初始化）——本关
**省略 wrapper**，因此**不能**把这里的直接调用说成"完整 RPC 实现"。

初始化顺序（195 §8，Executor 负责前两步）：

    init_device()                  → Worker.init_device()
    load_model()                   → Worker.load_model()
    然后把 Runner 报回的 KV 容量交给上层

`execute_model()` / `sample_tokens()` / `take_draft_token_ids()` 与 Worker 同名同参——
3 层同名的转发看着冗余，但换部署方式（多进程、远端）时改的只有这一层。
"""

from ..worker.worker import Worker


class UniProcExecutor:
    def __init__(self, vllm_config, worker: Worker | None = None) -> None:
        self.vllm_config = vllm_config
        # driver_worker 是"这个执行端"的入口。57A 允许注入（测试的 FakeRunner 挂在 Worker 上），
        # 生产路径由 57B 提供真实 Worker+ModelRunner。
        self.driver_worker = worker if worker is not None else Worker(vllm_config)
        self._init_executor()

    def _init_executor(self) -> None:
        self.driver_worker.init_device()
        self.driver_worker.load_model()

    # -------- 上层要的 --------

    def get_cache_config(self):
        """KV 池规格。57B 会由 Runner 实测/配置后报回；57A 直接用配置里的值。"""
        return self.vllm_config.cache_config

    def initialize_kv_cache(self, kv_cache_config) -> None:
        """把容量交给执行端绑定，然后**捕获 CUDA Graph**（69 关）。

        顺序不能反：图里录的是"注意力层访问自己那份物理 KV 缓存"的地址，缓存没绑定就捕获
        等于把图录到别的张量上——重放时读的是错的显存，而且不报错。上游同样在
        `initialize_from_config()` 之后才 `compile_or_warm_up_model()`。
        """
        self.driver_worker.initialize_from_config(kv_cache_config)
        self.driver_worker.compile_or_warm_up_model()

    def execute_model(self, scheduler_output):
        return self.driver_worker.execute_model(scheduler_output)

    def sample_tokens(self, grammar_output):
        return self.driver_worker.sample_tokens(grammar_output)

    def take_draft_token_ids(self):
        return self.driver_worker.take_draft_token_ids()

    def shutdown(self) -> None:
        self.driver_worker.shutdown()
