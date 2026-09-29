"""配置：拆成几个小 dataclass，由 `VllmConfig` 聚合（对应 vLLM 的 `vllm/config/*`）。

step56 把二十多个参数平铺在 `Engine.__init__` 上，既看不出哪些是一类，也没法在不构造引擎的
情况下校验。这里按 vLLM 的分法切开：

    ModelConfig      模型与精度的"是什么"
    CacheConfig      KV 池的规格（块大小、块数）——**不再表示某条请求的缓存状态**
    SchedulerConfig  调度规格（并发数、token 预算、策略）
    DeviceConfig     跑在哪个设备
    SpeculativeConfig 投机方法与其参数

对应 vLLM：`vllm/config/model.py::ModelConfig`、`cache.py::CacheConfig`、
`scheduler.py::SchedulerConfig`、`device.py::DeviceConfig`、`speculative.py::SpeculativeConfig`。

**本关只保留被代码用到的字段**（vLLM 每个 Config 有几十个字段，大半是硬件/分布式分支）。
`num_gpu_blocks` 先手动配置：真实 vLLM 的显存 profiling 自动定容属于"明确延后"的部分。
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelConfig:
    """模型规格。57A 只用到 `max_model_len`（上下文上限，`check_stop` 要它），
    其余字段留给 57B 的真实加载（`model` / `dtype` / `hf_config`）。"""

    model: str
    dtype: str = "float32"
    max_model_len: int = 4096
    hf_config: dict | None = None
    eos_token_id: int | None = None

    def __post_init__(self):
        if self.max_model_len <= 0:
            raise ValueError(f"max_model_len 必须为正，收到 {self.max_model_len}")


@dataclass(frozen=True)
class CacheConfig:
    """KV 池规格。**块大小与总块数**——不是某条请求的 `length` / `block_table`。

    step56 用同一个 `CacheConfig` 既表示池子规格、又挂在请求上表示它的缓存进度，两件事混在
    一起；这里只留规格，请求的进度归 Scheduler（`num_computed_tokens`），物理块归
    `KVCacheManager`。
    """

    block_size: int = 16
    num_gpu_blocks: int = 8
    enable_prefix_caching: bool = False

    def __post_init__(self):
        if self.block_size <= 0:
            raise ValueError(f"block_size 必须为正，收到 {self.block_size}")
        if self.num_gpu_blocks <= 0:
            raise ValueError(f"num_gpu_blocks 必须为正，收到 {self.num_gpu_blocks}")
        if self.enable_prefix_caching:
            # 前缀缓存是 57C 的内容（块 hash、引用计数、命中查询）。现在打开它只会得到一个
            # "看起来开了、其实没做"的配置，明确拒绝。
            raise NotImplementedError(
                "57A 不支持 enable_prefix_caching（块 hash 与命中查询属 57C）；"
                "先按 False 配置")


@dataclass(frozen=True)
class SchedulerConfig:
    max_num_seqs: int = 8
    max_num_batched_tokens: int = 64
    policy: str = "fcfs"          # "fcfs" / "priority"

    def __post_init__(self):
        if self.max_num_seqs <= 0:
            raise ValueError(f"max_num_seqs 必须为正，收到 {self.max_num_seqs}")
        if self.max_num_batched_tokens <= 0:
            raise ValueError("max_num_batched_tokens 必须为正，收到 "
                             f"{self.max_num_batched_tokens}")
        if self.policy not in ("fcfs", "priority"):
            raise ValueError(f"policy 只能是 'fcfs' / 'priority'，收到 {self.policy!r}")


@dataclass(frozen=True)
class DeviceConfig:
    device: str = "cpu"


@dataclass(frozen=True)
class SpeculativeConfig:
    """投机配置（57E 才接进调度）。57A 只需要它存在、且默认 None。"""

    method: str = "ngram"
    num_speculative_tokens: int = 0
    draft_model_config: ModelConfig | None = None

    def __post_init__(self):
        if self.num_speculative_tokens < 0:
            raise ValueError("num_speculative_tokens 不能为负")


@dataclass(frozen=True)
class VllmConfig:
    model_config: ModelConfig
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)
    device_config: DeviceConfig = field(default_factory=DeviceConfig)
    speculative_config: SpeculativeConfig | None = None
