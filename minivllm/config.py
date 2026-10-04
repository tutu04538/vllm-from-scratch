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

import importlib.util
from dataclasses import dataclass, field


def has_arctic_inference() -> bool:
    """是否装了外部包 `arctic_inference`（上游 `vllm/utils/import_utils.py:542` 同名函数）。

    suffix decoding 的树与匹配是**上游依赖包的实现**（61 关明确要求"接入"而不是自研），
    所以这里只做"在不在"的判断，不在就由 `_resolve_suffix_decoding` 显式报错。
    """
    return importlib.util.find_spec("arctic_inference") is not None


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
        # 57C 起 `enable_prefix_caching=True` 是真的生效的：块 hash、引用计数、命中查询、
        # LRU 逐出都在 `core/{kv_cache_utils,block_pool,single_type_kv_cache_manager}.py`
        # 里（见 docs/step57c_kv_and_prefix.md）。


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
    """投机配置（57E 接进调度；58 增加输入槽位派生量；60 增加 ngram 匹配窗口；
    61 增加 suffix decoding 参数）。"""

    method: str = "ngram"
    num_speculative_tokens: int = 0
    draft_model_config: ModelConfig | None = None
    # 验证方式（上游 `SpeculativeConfig.rejection_sample_method`，59 关只接 standard）
    rejection_sample_method: str = "standard"
    # ngram 的匹配窗口（上游 `prompt_lookup_min/max`）：长度落在 [min, max] 的后缀 ngram
    # 才会被拿去匹配。**默认 5/5**（上游注释："arbitrarily chosen"），只在给了一个时对齐另一个。
    prompt_lookup_max: int | None = None
    prompt_lookup_min: int | None = None
    # suffix decoding（61 关；上游 `config/speculative.py:195-210`，注释也照抄）
    # 全局树与 prompt 树的**最大深度**：它同时限制"前缀匹配长度 + 推测长度"之和。
    suffix_decoding_max_tree_depth: int = 24
    # 全局树里最多缓存多少条请求，超了按 FIFO 淘汰。**0 = 关掉全局树**
    # （过去响应不再缓存，但每条请求自己的 prompt 树仍然用）。
    suffix_decoding_max_cached_requests: int = 10000
    # 推测长度相对前缀匹配长度的倍数上限：`max_spec_tokens = factor * match_len + offset`。
    suffix_decoding_max_spec_factor: float = 1.0
    # 只推测"按频次估计的概率" ≥ 该值的 token。
    suffix_decoding_min_token_prob: float = 0.1

    def __post_init__(self):
        if self.num_speculative_tokens < 0:
            raise ValueError("num_speculative_tokens 不能为负")
        if self.method not in ("ngram", "ngram_gpu", "draft_model", "suffix"):
            raise ValueError(
                f"本关只支持 method='ngram' / 'ngram_gpu' / 'draft_model' / 'suffix'，"
                f"收到 {self.method!r}"
                "（EAGLE/MTP/PARD 等按需求顺序在后续关卡实现）")
        if self.rejection_sample_method != "standard":
            raise ValueError(
                f"本关只支持 rejection_sample_method='standard'，收到 "
                f"{self.rejection_sample_method!r}：synthetic（合成接受率）与 block"
                f"（V2 块验证）按需求顺序在 75 关实现，不要用 standard 的结果冒充它们")
        if self.method in ("ngram", "ngram_gpu"):
            self._resolve_prompt_lookup()
        elif self.method == "suffix":
            self._resolve_suffix_decoding()

    def _resolve_prompt_lookup(self) -> None:
        """把 `prompt_lookup_min/max` 补全成上游那套（config/speculative.py:804-829）。

        规则逐条照抄：都没给 → `5/5`；只给一个 → 另一个取同一个值；最后校验 `min ≤ max`。
        `dataclass(frozen=True)` 里不能直接赋值，所以走 `object.__setattr__`——上游这里是普通
        赋值，效果一样（构造完之后读到的就是补全后的值）。
        """
        minimum, maximum = self.prompt_lookup_min, self.prompt_lookup_max
        if minimum is None and maximum is None:
            minimum = maximum = 5
        elif minimum is None:
            minimum = maximum
        elif maximum is None:
            maximum = minimum
        if minimum > maximum:
            raise ValueError(
                f"prompt_lookup_min={minimum} 不能大于 prompt_lookup_max={maximum}")
        object.__setattr__(self, "prompt_lookup_min", minimum)
        object.__setattr__(self, "prompt_lookup_max", maximum)

    def uses_draft_model(self) -> bool:
        """是否用独立的 draft 模型提议（上游同名方法）。"""
        return self.method == "draft_model"

    def _resolve_suffix_decoding(self) -> None:
        """suffix decoding 的缺包检查、默认值与取值校验（照抄上游
        `config/speculative.py:1146-1181::_validate_suffix_decoding`）。

        唯一的差异写在 `num_speculative_tokens` 那一条上：上游的字段是 `int | None`
        （`None` = 没设 → 取树深），本仓库是 `int`（`0` = 没设），所以把 `0` 当"没设"。
        显式传 `0` 想"一枚都不猜"时，直接别开 suffix 就行。
        """
        if not has_arctic_inference():
            raise ImportError(
                "suffix decoding 需要外部包 Arctic Inference（本关要求接入依赖实现，"
                "不自己写一棵后缀树）。上游钉的是 `pip install arctic-inference==0.1.1`；"
                "本机 torch 2.13.0 下 0.1.1 的构建依赖（torch==2.7.0）装不上，改装的 0.3.0 "
                "`suffix_decoding/cache.py` 与 0.1.1 逐行相同（安装命令与校验见 "
                "docs/step61_alignment.md §2）。")
        if self.num_speculative_tokens == 0:
            # 上游这里还打一条 warning（"Defaulted num_speculative_tokens to %s"）；
            # 本仓库不引 logger，把默认值写进 alignment 文档代替。
            object.__setattr__(self, "num_speculative_tokens",
                               self.suffix_decoding_max_tree_depth)
        if self.suffix_decoding_max_tree_depth < 1:
            raise ValueError(
                f"suffix_decoding_max_tree_depth="
                f"{self.suffix_decoding_max_tree_depth} must be >= 1")
        if self.suffix_decoding_max_cached_requests < 0:
            raise ValueError(
                f"suffix_decoding_max_cached_requests="
                f"{self.suffix_decoding_max_cached_requests} must be >= 0")
        if self.suffix_decoding_max_spec_factor < 0:
            raise ValueError(
                f"suffix_decoding_max_spec_factor="
                f"{self.suffix_decoding_max_spec_factor} must be >= 0")
        if not 0 <= self.suffix_decoding_min_token_prob <= 1:
            raise ValueError(
                f"suffix_decoding_min_token_prob="
                f"{self.suffix_decoding_min_token_prob} must be in [0, 1]")

    def use_ngram_gpu(self) -> bool:
        """是否用 GPU 版 ngram 提议者（上游同名方法）。"""
        return self.method == "ngram_gpu"

    @property
    def max_num_new_slots_for_drafting(self) -> int:
        """每条被调度的请求，draft 第一遍比 target query **多**要几个输入槽位。

        上游 `SpeculativeConfig.max_num_new_slots_for_drafting`（本机 0.28.0）的分支是
        按"用不用 draft 模型 / 是不是并行提议"分的：普通自回归 draft model → **1**，
        ngram / MTP → 0，P-EAGLE → K-1，DFlash / PARD → K。

        本关只实现**普通自回归 draft**：它保留一个未切片的 token 作为第一遍的最后一行
        （就是 target 本轮刚采出的那个），所以是 1；ngram 不跑模型、不写 KV，是 0。

        **不要和 `num_lookahead_tokens` 混**：那个是"额外保留几个 KV 位置"（=K），
        这个是"draft 输入工作区每请求多占几行"。
        """
        if self.uses_draft_model():
            return 1
        return 0


@dataclass(frozen=True)
class VllmConfig:
    model_config: ModelConfig
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)
    device_config: DeviceConfig = field(default_factory=DeviceConfig)
    speculative_config: SpeculativeConfig | None = None
