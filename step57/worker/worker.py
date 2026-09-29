"""Worker：一个执行端的**运行环境**（对应 vLLM `v1/worker/gpu_worker.py`）。

它回答的问题是"这个执行端怎么把设备和模型准备好"，而不是"这一轮算什么"：

    init_device()                 把设备准备好（本关：只记下 device，不做显存 profiling）
    load_model()                  把模型装上（57A：没有模型，见下）
    initialize_from_config(kv)    按 KV 规格建池并绑定（本关：把容量交给上层）
    execute_model(packet)         转发给 Runner
    sample_tokens(grammar)        转发给 Runner
    take_draft_token_ids()        转发给 Runner（57E）

**与 Executor / Runner 的分工**（195 §2）：换执行部署改 Executor、换设备准备改 Worker、
换 batch 打包方式改 Runner。现在它们是薄转发，但责任边界是真的——不是三个空壳。

**57A 的边界**：真实模型与 KV 分配属 57B，所以 `model_runner` 必须由外部给（测试注入
FakeRunner）。**生产入口不会自动退回 FakeRunner**：没给 runner 就在 `execute_model()` 里
明确报错，而不是悄悄用一个假的继续跑。
"""


class Worker:
    def __init__(self, vllm_config, model_runner=None) -> None:
        self.vllm_config = vllm_config
        self.device = vllm_config.device_config.device
        self.model_runner = model_runner
        self.kv_cache_config = None

    # -------- 初始化 --------

    def init_device(self) -> None:
        """准备设备。57A 不碰 CUDA（跑的是假执行），只记录；真实初始化（显存 profiling、
        分布式、CUDA Graph 捕获）是 57B/57D 的事。"""
        return None

    def load_model(self) -> None:
        """装载模型。57A 没有模型可装——Runner 自带它的"模型"（测试里的脚本化输出）。

        真实实现（57B）会在这里读权重、建 KV 池、把容量报回去；本关保持空实现，
        让"没有模型"这件事在代码里可见，而不是假装已经加载。
        """
        return None

    def initialize_from_config(self, kv_cache_config) -> None:
        """按上层定下的 KV 规格分配并绑定缓存（本关只是记下来）。"""
        self.kv_cache_config = kv_cache_config

    # -------- 执行 --------

    def execute_model(self, scheduler_output):
        if self.model_runner is None:
            raise RuntimeError(
                "57A 的 Worker 没有模型：`model_runner` 必须由外部注入（测试用 FakeRunner）。"
                "真实模型装载是 57B 的内容；这里明确报错，不会静默退回假执行。")
        return self.model_runner.execute_model(scheduler_output)

    def sample_tokens(self, grammar_output):
        if self.model_runner is None:
            raise RuntimeError("Worker 没有 model_runner，无法采样（见 execute_model 的说明）")
        return self.model_runner.sample_tokens(grammar_output)

    def take_draft_token_ids(self):
        if self.model_runner is None:
            return None
        return self.model_runner.take_draft_token_ids()

    def shutdown(self) -> None:
        return None
