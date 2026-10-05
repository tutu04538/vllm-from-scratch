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
from dataclasses import dataclass, field, replace


def has_arctic_inference() -> bool:
    """是否装了外部包 `arctic_inference`（上游 `vllm/utils/import_utils.py:542` 同名函数）。

    suffix decoding 的树与匹配是**上游依赖包的实现**（61 关明确要求"接入"而不是自研），
    所以这里只做"在不在"的判断，不在就由 `_resolve_suffix_decoding` 显式报错。
    """
    return importlib.util.find_spec("arctic_inference") is not None


def extract_hidden_states_hf_config(target_hf_config: dict | None, cache_block_size: int,
                                    torch_dtype: str, **overrides) -> dict:
    """cache-only 模型的 hf 配置（对应上游
    `transformers_utils/configs/extract_hidden_states.py::ExtractHiddenStatesConfig`）。

    上游那个类做三件事，这里逐条对应：

    1. `combined = {**model_dict, **kwargs}` —— 先放 **target 的配置**，再用**用户给的 draft
       配置**覆盖（所以 `eagle_aux_hidden_state_layer_ids` 这类"要存哪几层"的参数来自 draft 侧）；
    2. 丢掉 base 的 `architectures`、强制成 `["ExtractHiddenStatesModel"]` —— 加载器按
       architectures 选类（本仓库走 `models/registry.py`），写别的名字会去建 target 模型；
    3. 另外两个字段是本仓库独有的：上游的 `CacheOnlyAttentionLayer` 从
       `get_current_vllm_config()` 读 `cache_config.block_size` 与模型 dtype，而本仓库的模型
       只吃一个 config dict，所以把 `cache_block_size` / `torch_dtype` 一并写进去（值就是上游
       会读到的同样两个值）。
    """
    combined = {**dict(target_hf_config or {}), **overrides}
    combined.pop("architectures", None)
    combined["architectures"] = ["ExtractHiddenStatesModel"]
    combined["cache_block_size"] = int(cache_block_size)
    combined["torch_dtype"] = str(torch_dtype)
    return combined


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
    61 增加 suffix decoding 参数；62 增加自定义 proposer 的接入与**方法推断**；
    64 增加 `extract_hidden_states`——一个**不做投机、只借 KV 缓存存特征**的方法）。"""

    # 上游 `SpeculativeConfig.method: str | None = None`：**不给**时由 `__post_init__` 推出来，
    # 这样"用哪种投机"只有一个判定点（62 关需求 §3.1：不让 CLI 判一次、Runner 再猜一次）。
    method: str | None = None
    # 上游注释：`method` 是给"非模型类提议者"用的新参数，而 `model` 用来放 draft 模型 /
    # EAGLE head / 额外权重；`method="custom_class"` 时它装的是**完整 module.Class 路径**。
    model: str | None = None
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

    @staticmethod
    def _is_custom_proposer_path(model: str | None) -> bool:
        """`model` 是不是"自定义提议者类的点号路径"（上游 `config/speculative.py:721-730`）。

        判定逐条照抄：`http(s)://` / `file://` 前缀不算（那是模型地址）；带 `/` 不算
        （`Qwen/Qwen3-0.6B` 这类 HF 模型名要走 draft_model）；必须至少两段且**每段都是
        合法标识符**（`module.Class`、`pkg.sub.Class` 都行，`my-module.Class` 不行）。
        """
        if model is None:
            return False
        if model.startswith(("http://", "https://", "file://")):
            return False
        if "/" in model:
            return False
        parts = model.split(".")
        return len(parts) >= 2 and all(part.isidentifier() for part in parts)

    def __post_init__(self):
        if self.num_speculative_tokens < 0:
            raise ValueError("num_speculative_tokens 不能为负")
        self._resolve_method()
        if self.method not in ("ngram", "ngram_gpu", "draft_model", "suffix", "custom_class",
                               "eagle", "eagle3", "extract_hidden_states"):
            raise ValueError(
                f"本关只支持 method='ngram' / 'ngram_gpu' / 'draft_model' / 'suffix' / "
                f"'custom_class' / 'eagle' / 'eagle3' / 'extract_hidden_states'，收到 "
                f"{self.method!r}"
                "（MTP/PARD/DFlash 等按需求顺序在后续关卡实现）")
        if self.rejection_sample_method != "standard":
            raise ValueError(
                f"本关只支持 rejection_sample_method='standard'，收到 "
                f"{self.rejection_sample_method!r}：synthetic（合成接受率）与 block"
                f"（V2 块验证）按需求顺序在 75 关实现，不要用 standard 的结果冒充它们")
        if self.method in ("ngram", "ngram_gpu"):
            self._resolve_prompt_lookup()
        elif self.method == "suffix":
            self._resolve_suffix_decoding()
        elif self.method == "custom_class":
            self._resolve_custom_class()
        elif self.use_eagle():
            self._resolve_eagle()
        elif self.uses_extract_hidden_states():
            self._resolve_extract_hidden_states()

    def _resolve_method(self) -> None:
        """`method` 没给时按上游规则推出来（`config/speculative.py:741-756`）。

        顺序也是照抄的：**先看 `model` 是不是自定义类的点号路径**（是 → `custom_class`），
        否则 `model` 是 `ngram`/`[ngram]` → `ngram`，其余一律 `draft_model`（连 `model` 都没给
        也算 draft_model，因为"没写方法"的默认语义是"给一个 draft 模型"）。

        这一步是本关的"分派边界"：**同一条事实只在这里判定一次**，Runner 只按
        `speculative_config.method` 分派，不再自己猜第二遍（否则 CLI 与 Runner 可能各判一套，
        出现"配置说 A、运行时走 B"的静默错）。
        """
        if self.method is None:
            if self._is_custom_proposer_path(self.model):
                object.__setattr__(self, "method", "custom_class")
            elif self.model in ("ngram", "[ngram]"):
                object.__setattr__(self, "method", "ngram")
            elif self.model and "eagle3" in self.model.lower():
                # 上游 `config/speculative.py:942-951`：从模型名认 EAGLE 系
                object.__setattr__(self, "method", "eagle3")
            elif self.model and "eagle-" in self.model.lower():
                object.__setattr__(self, "method", "eagle")
            else:
                object.__setattr__(self, "method", "draft_model")

    def _resolve_custom_class(self) -> None:
        """`method="custom_class"` 的取值校验（上游 `config/speculative.py:787-793`）。

        上游只在这里校验"`model` 不能为空"（**明确报错，不回退**）；点号路径、模块/类是否存在、
        能不能构造、`propose` 可不可调用，一律由 `create_custom_proposer()` 在**启动期**分类报错
        （见 `minivllm/spec_decode/custom_class_proposer.py`）。
        """
        if not self.model:
            raise ValueError(
                "method='custom_class' requires 'model' to contain the "
                "custom proposer module path (e.g. 'my_module.MyProposer').")

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

    def use_eagle(self) -> bool:
        """是否走 EAGLE 系提议者（EAGLE-1/2 与 EAGLE3 共用一套提议流程，只差模型类）。"""
        return self.method in ("eagle", "eagle3")

    def _resolve_eagle(self) -> None:
        """EAGLE 的取值校验：必须有 draft 模型目录、K > 0（上游同样要求）。

        **不支持 M-RoPE**（上游 `_raise_if_mrope`）：本仓库的 positions 是 1 维的，
        mrope 需要三路位置——这里明确报错而不是悄悄按 1 维跑。
        """
        if self.draft_model_config is None:
            raise ValueError("method='eagle'/'eagle3' 必须在 SpeculativeConfig 里给 "
                             "draft_model_config（EAGLE 的 draft 是独立权重）")
        if self.num_speculative_tokens is None or self.num_speculative_tokens <= 0:
            raise ValueError("EAGLE 的 num_speculative_tokens 必须 > 0（它决定自回归猜几枚）")
        if self.num_speculative_tokens > 32:
            raise ValueError(
                f"num_speculative_tokens={self.num_speculative_tokens} 太大："
                f"EAGLE 的自回归步数与它成正比，先支持到 32")

    def eagle3_use_aux_hidden_state(self) -> bool:
        """EAGLE3 是否吃**多个辅助层**的特征（EAGLE-1 吃最后一层，不吃 aux）。

        64 关：`extract_hidden_states` 也走同一条"target 顺带输出辅助层"的采集路径，但它
        **不是** EAGLE——没有 draft 模型、不吃这些特征去猜 token。所以这个判定仍然只认
        `eagle3`，采集开关由 `uses_extract_hidden_states()` 单独打开（Runner 里两个条件是或）。
        """
        return self.method == "eagle3"

    def uses_extract_hidden_states(self) -> bool:
        """是否走 cache-only 特征提取（上游同名方法，`config/speculative.py:1495`）。"""
        return self.method == "extract_hidden_states"

    def _resolve_extract_hidden_states(self) -> None:
        """`method="extract_hidden_states"` 的取值校验（上游 `config/speculative.py:850-874`）。

        上游在这个分支里做三件事：把 `model` 换成字面量 `"extract_hidden_states"`（它没有 draft
        模型目录，这个字段只是标记）、清掉 prompt lookup 参数、把 draft 配置换成"target 配置 +
        `ExtractHiddenStatesConfig`"。前两件这里照抄；第三件需要 **target 配置**（上游
        `SpeculativeConfig` 自己持有 `target_model_config` 字段，本仓库的配置拿不到），所以拆成
        `derive_extract_hidden_states_config()`，由同时持有两者的调用方（提议者）执行。

        **K 的约束**：上游是 `ExtractHiddenStatesProposer.__init__` 里的 `assert K == 1`
        （跑到建提议者时才炸）。本仓库提前到配置期报错——约束相同，报错更早、信息更清楚，
        差异记在 docs/step64_alignment.md §3。
        """
        object.__setattr__(self, "model", "extract_hidden_states")
        object.__setattr__(self, "prompt_lookup_max", 0)
        object.__setattr__(self, "prompt_lookup_min", 0)
        if self.num_speculative_tokens != 1:
            raise ValueError(
                f"method='extract_hidden_states' 只支持 num_speculative_tokens=1，收到 "
                f"{self.num_speculative_tokens}：这个方法不猜 token，每轮只借投机框架多跑"
                f"一行（target 自己采出的那一列）来缓存特征")
        if self.eagle_aux_hidden_state_layers() is None:
            # 上游原话：eagle_aux_hidden_state_layer_ids must be set in the draft model config
            raise ValueError(
                "method='extract_hidden_states' 必须在 draft 配置里给 "
                "eagle_aux_hidden_state_layer_ids（要存哪几层特征）：上游在提议者构造时检查，"
                "本仓库提前到配置期")

    def derive_extract_hidden_states_config(self, target_model_config: "ModelConfig",
                                            cache_config: "CacheConfig") -> "ModelConfig":
        """派生 cache-only 模型用的配置（上游 `config/speculative.py:861-873` 的等价物）。

        上游做的事：从**用户给的** draft 配置里取出 `hf_config`（`ExtractHiddenStatesConfig` 的
        覆盖项，`eagle_aux_hidden_state_layer_ids` 就在里面），再把它盖到 **target 的 hf 配置**上，
        于是"draft 模型目录"= target 的目录（cache-only 模型没有权重，加载器读到的权重被忽略）。

        本仓库的模型只吃一个 config dict（上游的 `CacheOnlyAttentionLayer` 从
        `get_current_vllm_config()` 取 `cache_config`），所以块大小与精度也一并写进 hf 配置，
        见 `extract_hidden_states_hf_config()`。
        """
        overrides = dict(self.draft_model_config.hf_config or {}) \
            if self.draft_model_config is not None else {}
        hf_config = extract_hidden_states_hf_config(
            target_model_config.hf_config, cache_block_size=cache_config.block_size,
            torch_dtype=str(target_model_config.dtype), **overrides)
        # `replace()` 会重跑 ModelConfig 的校验（target 那份已经过过一次，值不变）
        return replace(target_model_config, hf_config=hf_config)

    def eagle_aux_hidden_state_layers(self) -> tuple[int, ...] | None:
        """draft 配置里指定的辅助层编号（没有就返回 None，由 target 的默认值兜底）。

        上游 `_get_eagle3_aux_layers_from_config`：`eagle_aux_hidden_state_layer_ids` →
        `eagle_config.eagle_aux_hidden_state_layer_ids` → `dflash_config.target_layer_ids + 1`。
        """
        hf_config = (self.draft_model_config.hf_config or {}) if self.draft_model_config else {}
        layers = hf_config.get("eagle_aux_hidden_state_layer_ids")
        if not layers:
            eagle_config = hf_config.get("eagle_config") or {}
            layers = eagle_config.get("eagle_aux_hidden_state_layer_ids")
        return tuple(layers) if layers else None

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
        64 关的 `extract_hidden_states` 也是 0：它不跑 draft 模型，写的是 **target 本轮
        那些 query 行自己的槽位**（与上游的分支表一致：只有 `uses_draft_model()` 才是 1）。

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
