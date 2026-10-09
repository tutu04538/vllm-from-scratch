"""Worker：一个执行端的**运行环境**（对应 vLLM `v1/worker/gpu_worker.py`）。

它回答的问题是"这个执行端怎么把设备和模型准备好"，而不是"这一轮算什么"：

    init_device()                 把设备准备好（本关：只记下 device，不做显存 profiling）
    load_model()                  建 Runner、装模型（57B 起是真的装）
    initialize_from_config(kv)    按 KV 规格建物理缓存并绑定到 Attention 层
    execute_model(packet)         转发给 Runner
    sample_tokens(grammar)        转发给 Runner
    take_draft_token_ids()        转发给 Runner（57E）

**与 Executor / Runner 的分工**（195 §2）：换执行部署改 Executor、换设备准备改 Worker、
换 batch 打包方式改 Runner。这里看起来是薄转发，但责任边界是真的——不是三个空壳。

**两条路径**：

- **生产**：不传 `model_runner` → `load_model()` 里建 `GPUModelRunner` 并真的读权重。
  模型目录不存在、权重格式不支持都会在这里明确报错（不会静默退回假执行）。
- **测试**：注入 `model_runner`（如 `testing.FakeRunner`）→ `load_model()` **不碰模型**，
  设备与模型由注入者自己负责。57A 的全部用例走这条。
"""

from .gpu_model_runner import GPUModelRunner


class Worker:
    def __init__(self, vllm_config, model_runner=None) -> None:
        self.vllm_config = vllm_config
        self.device = vllm_config.device_config.device
        # 注入的 Runner（测试用）。None 表示走真实路径，在 load_model() 里建。
        self.model_runner = model_runner
        self.kv_cache_config = None

    # -------- 初始化 --------

    def init_device(self) -> None:
        """准备设备。本关只记下 device：显存 profiling、分布式、CUDA Graph 捕获都不在这里
        （57D 及以后）。"""
        return None

    def load_model(self) -> None:
        """建 Runner 并装模型。注入了 Runner（测试）时什么都不做。

        73 关：**两条执行路径由配置分派**（对应上游 `gpu_worker.py:424-435` 的同位置分派）——
        `VLLM_USE_V2_MODEL_RUNNER=1` 时建 `worker/gpu/model_runner.py::GPUModelRunner`（V2），
        否则建 V1 的 `worker/gpu_model_runner.py::GPUModelRunner`。两条路径共用
        Engine/Scheduler 协议，但**不互相调用对方的输入准备**（需求 073 §2）。
        """
        if self.model_runner is not None:
            return
        if self.vllm_config.use_v2_model_runner:
            # 支持矩阵在配置期已校验（`VllmConfig.validate_v2_model_runner`），这里只做分派。
            from .gpu.model_runner import GPUModelRunner as GPUModelRunnerV2

            self.model_runner = GPUModelRunnerV2(self.vllm_config, self.device)
        else:
            self.model_runner = GPUModelRunner(self.vllm_config, self.device)
        # 70 关：执行侧也要知道"这一轮走不走异步"（它决定草稿从哪来：协议里的值 vs 自己
        # 上一轮提的那份）。解析规则只有一份（`config.resolve_async_scheduling`），
        # 引擎侧问的是同一个问题、同一套输入，所以两边答案必然一致。
        from ..config import resolve_async_scheduling

        self.model_runner.async_scheduling = resolve_async_scheduling(
            self.vllm_config, self.supports_async_scheduling())
        self.model_runner.load_model()

    def initialize_from_config(self, kv_cache_config) -> None:
        """按上层定下的 KV 规格分配物理缓存并绑定到每个 Attention 层。

        容量（`num_gpu_blocks`）是**上层给的**：真实 vLLM 由 Runner 先做显存 profiling 定容、
        再回报给引擎；本关按配置写死，所以顺序反过来也不影响——但"容量由执行侧决定"这条
        设计意图保留在这里（`Executor.get_cache_config()` 问的就是 Worker 这边）。
        """
        self.kv_cache_config = kv_cache_config
        self.model_runner.initialize_kv_cache(kv_cache_config)

    def compile_or_warm_up_model(self) -> None:
        """捕获 CUDA Graph（69 关；对应上游 `Worker.compile_or_warm_up_model()`）。

        上游这里还做"编译 + 按 profile 结果定容"，本仓库没有编译路径，
        所以只剩捕获这一步：`GPUModelRunner.capture_model()` 按 `CudagraphDispatcher`
        列出的档位逐个热身 + 捕获。模式为 NONE（eager）时它是空操作。
        """
        if self.model_runner is None:
            return
        capture_model = getattr(self.model_runner, "capture_model", None)
        if capture_model is not None:
            capture_model()

    # -------- 执行 --------

    def supports_async_scheduling(self) -> bool:
        """70 关：这个执行端支不支持异步调度（上游 `Executor.supports_async_scheduling()`）。

        本仓库的执行端是**同进程直连**（没有 RPC 序列化），而"异步"在这里的含义不是"换线程"，
        而是"把结果的 D2H 放到侧流上、把等待推到交付边界"——这件事由 Runner 自己保证（它持有
        显存与 pinned 缓冲），所以本仓库支持。上游的多进程 executor 还要看它有没有异步 RPC。
        """
        return True

    def execute_model(self, scheduler_output, non_block: bool = False):
        if self.model_runner is None:
            raise RuntimeError("Worker 没有 model_runner：先 load_model()")
        return self.model_runner.execute_model(scheduler_output, non_block=non_block)

    def sample_tokens(self, grammar_output, non_block: bool = False):
        if self.model_runner is None:
            raise RuntimeError("Worker 没有 model_runner，无法采样（见 execute_model 的说明）")
        return self.model_runner.sample_tokens(grammar_output, non_block=non_block)

    def take_draft_token_ids(self):
        if self.model_runner is None:
            return None
        return self.model_runner.take_draft_token_ids()

    def shutdown(self) -> None:
        return None
