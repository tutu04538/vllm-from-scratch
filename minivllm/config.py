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

import enum
import importlib.util
from dataclasses import dataclass, field, replace


def has_arctic_inference() -> bool:
    """是否装了外部包 `arctic_inference`（上游 `vllm/utils/import_utils.py:542` 同名函数）。

    suffix decoding 的树与匹配是**上游依赖包的实现**（61 关明确要求"接入"而不是自研），
    所以这里只做"在不在"的判断，不在就由 `_resolve_suffix_decoding` 显式报错。
    """
    return importlib.util.find_spec("arctic_inference") is not None


def medusa_hf_config(draft_hf_config: dict | None, num_speculative_tokens: int) -> dict:
    """把 draft 的 hf 配置归一成 Medusa 的规格（66 关；对应上游
    `transformers_utils/configs/medusa.py::MedusaConfig` + `SpeculativeConfig.__post_init__`
    里对 Medusa 的两处改写）。

    上游在配置期做三件事，这里逐条对应：

    1. **旧 checkpoint 的 key 改名**：`MedusaConfig.from_pretrained()` 会把
       `medusa_num_heads`/`medusa_num_layers` 改写成 `num_heads`/`num_hidden_layers`。
       本机实测的真实旧文件 `FasterDecoding/medusa-vicuna-7b-v1.3/config.json` **只有**
       `{"medusa_num_heads": 2, "medusa_num_layers": 1, "base_model_name_or_path": ...}`——
       连 `model_type` / `vocab_size` / `hidden_size` / `architectures` 都没有，所以
       `AutoConfig` 认不出它是 Medusa（这就是上游要 `hf_overrides={"model_type": "medusa"}`
       的原因），而缺的字段全部落回 `MedusaConfig` 的默认值（vocab_size=32001、hidden_size=4096）。

    2. **model_type / architectures**：上游强制 `model_type="medusa"`；`MedusaConfig.__init__`
       在"配置里没有 architectures"时补 `["MedusaModel"]`（旧文件正好没有）。

    3. **K 对 head 数的约束**：上游有一段"draft 配置只要有 `num_lookahead_tokens` 属性，就把
       `num_speculative_tokens` 写进去"，而 `MedusaConfig.num_lookahead_tokens` 的 setter 是
       `self.num_heads = num_lookahead_tokens`——**所以 Medusa 的 head 数就是 K**，checkpoint
       自己声明的 `medusa_num_heads` 反而会被覆盖（实测那份 config 写 2、文件里其实有 5 个 head，
       上游按 K 建、多的丢掉）。本函数同样最后写 `num_heads = K`。

    本仓库的差异（记在 docs/step66_alignment.md §3）：上游的改名循环是"key 里同时含 num 与
    heads/layers 就改名"，于是 `num_attention_heads` / `num_key_value_heads` 也会被改名成
    `num_heads`（社区版 config 里两者都在）；因为它随后又被 K 覆盖，观测不到，本仓库只认
    `medusa_*` 前缀的字段，不做这个有副作用的宽匹配。
    """
    source = dict(draft_hf_config or {})
    # 1) 旧 checkpoint 的字段名 → MedusaConfig 的字段名（`MedusaConfig.from_pretrained` 同款）
    if "medusa_num_heads" in source:
        source["num_heads"] = source.pop("medusa_num_heads")
    if "medusa_num_layers" in source:
        source["num_hidden_layers"] = source.pop("medusa_num_layers")

    # 2) MedusaConfig.__init__ 的默认值（字段名与默认值逐个照抄）
    config = {
        "hidden_size": 4096,
        "vocab_size": 32001,
        "num_heads": 5,
        "num_hidden_layers": 1,
        "max_paths": 64,          # V1 的线性候选链用不到（论文的树才用），只保留字段
        "topk": 10,               # 同上
        "max_seq_len": int(2 ** 20),
    }
    config.update(source)         # checkpoint/用户给的字段覆盖默认值
    config["model_type"] = "medusa"
    # 3) 截断词表的默认值：不写就等于没截断（上游 `vocab_size if truncated_vocab_size is None`）
    config["truncated_vocab_size"] = config.get("truncated_vocab_size") or config["vocab_size"]
    # 4) architectures：`MedusaConfig` 只在"没有 architectures"时补 MedusaModel（旧文件正是如此）；
    #    写了别的（社区版常见：`["Qwen2ForCausalLM"]`，即基座的名字）**明确报错**而不是照抄——
    #    照抄的话加载器会去建基座模型，然后要么在权重名上炸、要么（上游）静默跑出一个
    #    "把 hidden 当 logits" 的错模型。改法写在报错信息里。
    architectures = config.get("architectures")
    if architectures not in (None, ["MedusaModel"]):
        raise ValueError(
            f"draft 配置里的 architectures={architectures!r} 不是 Medusa："
            f"method='medusa' 时本仓库只建 `MedusaModel`（上游 `MedusaConfig` 在配置里没有 "
            f"architectures 时补的就是它，旧 FasterDecoding checkpoint 正好没有这一项）。"
            f"社区版 checkpoint 常把基座的 architectures 抄进来，请把 draft 目录的 config.json "
            f"改成 \"architectures\": [\"MedusaModel\"] 再加载")
    config["architectures"] = ["MedusaModel"]
    # 5) **K 就是 head 数**（上游 `num_lookahead_tokens` → `num_heads` 的 setter）
    config["num_heads"] = int(num_speculative_tokens)
    return config


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
    # 66 关以前本仓库从不读 tokenizer（输入都是 token id），所以没有这个字段；67 关的 TLI 要建
    # "两套 tokenizer 的 token 级交集"，需要**分别**知道两边去哪读（上游 `ModelConfig.tokenizer`
    # 就是这个用途：默认与 `model` 同一个目录，异构词表时 target 用 target 的、draft 用 draft 的）。
    tokenizer: str | None = None
    # 68 关：`Sampler` 交付哪一套 logprobs（上游 `ModelConfig.logprobs_mode`，默认 raw_logprobs）。
    # 四种模式的区别见 sample/sampler.py；它决定的是"用户在 API 里拿到的数是原始分布还是
    # 被惩罚/温度/掩码改过的分布"，属于**引擎级**配置，不是逐请求参数。
    logprobs_mode: str = "raw_logprobs"
    # 上游 `ModelConfig.max_logprobs`（默认 20；-1 表示按词表全给）。68 关在请求期拿它校验
    # `SamplingParams.logprobs`：要 100 个却只留 20 个的话，静默截断会让用户以为拿到了全部。
    max_logprobs: int = 20
    # 69 关：`enforce_eager=True` 时**一律不做 CUDA Graph**（上游 `ModelConfig.enforce_eager`
    # 默认 False，但它同时决定 `_set_cudagraph_sizes` 走不走）。本仓库的解析规则见
    # `VllmConfig._resolve_cudagraph_config`：只有 CUDA + 没开 enforce_eager + 模式含 FULL
    # 才真的会建图。
    enforce_eager: bool = False

    def __post_init__(self):
        if self.max_model_len <= 0:
            raise ValueError(f"max_model_len 必须为正，收到 {self.max_model_len}")
        if self.logprobs_mode not in LOGPROBS_MODES:
            raise ValueError(
                f"logprobs_mode 只能是 {sorted(LOGPROBS_MODES)} 之一，收到 "
                f"{self.logprobs_mode!r}")
        if self.max_logprobs == 0 or self.max_logprobs < -1:
            raise ValueError(f"max_logprobs 只能是 -1 或正整数，收到 {self.max_logprobs}")

    @property
    def tokenizer_path(self) -> str:
        """去哪读 tokenizer（没单独指定就是模型目录，与上游默认一致）。"""
        return self.tokenizer or self.model

    def get_vocab_size(self) -> int:
        """词表大小（从 HF config 的 `vocab_size` 取；上游同名的取值口径）。

        结构化输出的掩码宽度、`logprobs=-1` 的"全词表"、TLI 的两边宽度都要它。
        没有 `hf_config` 时报错而不是猜一个值：宽度猜错不会报错，只会让掩码打到别的列上。
        """
        if not self.hf_config or "vocab_size" not in self.hf_config:
            raise ValueError(
                "模型配置里没有 vocab_size，无法确定词表宽度（结构化输出掩码 / logprobs=-1 "
                "都要它）。请先把 hf_config 读进来（模型目录里的 config.json）")
        return int(self.hf_config["vocab_size"])


#: 上游 `config/model.py` 的 `LogprobsMode`（四种模式）。前两种是"概率"、后两种是"logits"本身
#: （上游允许直接交付 logits，用于需要未归一化分数的场景）。
LOGPROBS_MODES = ("raw_logprobs", "processed_logprobs", "raw_logits", "processed_logits")
#: 上游 `PROCESSED_LOGPROBS_MODES`：需要"处理之后"的那两份的模式集合。
PROCESSED_LOGPROBS_MODES = ("processed_logprobs", "processed_logits")


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
    # 70 关：异步调度（CPU 不等 GPU 结果就排下一轮）。`None` = 按上游规则自动推断
    # （见 `resolve_async_scheduling`）；显式 True/False 就用用户给的值。
    async_scheduling: bool | None = None

    def __post_init__(self):
        if self.max_num_seqs <= 0:
            raise ValueError(f"max_num_seqs 必须为正，收到 {self.max_num_seqs}")
        if self.max_num_batched_tokens <= 0:
            raise ValueError("max_num_batched_tokens 必须为正，收到 "
                             f"{self.max_num_batched_tokens}")
        if self.policy not in ("fcfs", "priority"):
            raise ValueError(f"policy 只能是 'fcfs' / 'priority'，收到 {self.policy!r}")
        if self.async_scheduling is not None and not isinstance(self.async_scheduling, bool):
            raise ValueError(f"async_scheduling 只能是 True/False/None，收到 "
                             f"{self.async_scheduling!r}")


@dataclass(frozen=True)
class DeviceConfig:
    device: str = "cpu"


# 66 关：另一个"有配置枚举、但本机 V1 里没有实现"的方法（需求 066 §3）。
# 它不是别名、也不是"还没适配的家族"——上游 0.28.0 的注册表里那条是**注释掉的**，
# 所以配置层能认出它、执行层谁也建不起来。本仓库把这条事实写成明确的报错，
# 而不是自己写一套 MLP 实现去"骗过枚举"（见 tests/step66/test_mlp_support_boundary.py）。
MLP_SPECULATOR_MODEL_TYPE = "mlp_speculator"

# 上游 `MTPModelTypes`（`config/speculative.py:37-61`）：这一长串**别名**在配置期一律归一到
# `method="mtp"`（上游 L748-756：打个 deprecation 警告然后改写）。归一的理由：它们在引擎里
# 是**同一件事**——"MTP 权重就在 target checkpoint 里，用 target 的最后一层 hidden 迭代提议"，
# 差别只在模型结构（哪个家族的 decoder layer）与权重命名。别把每个别名做成一种独立算法。
MTP_MODEL_TYPES = (
    "deepseek_mtp",
    "dots3_note_mtp",
    "mimo_mtp",
    "mimo_v2_mtp",
    "glm4_moe_mtp",
    "glm4_moe_lite_mtp",
    "glm_ocr_mtp",
    "ernie_mtp",
    "nemotron_h_mtp",
    "exaone_moe_mtp",
    "exaone4_5_mtp",
    "qwen3_next_mtp",
    "qwen3_5_mtp",
    "longcat_flash_mtp",
    "bailing_hybrid_v3_mtp",
    "minimax_m3_mtp",
    "bailing_hybrid_mtp",
    "mtp",
    "kimi_k3_mtp",
    "pangu_ultra_moe_mtp",
    "step3p5_mtp",
    "hy_v3_mtp",
    "gemma4_mtp",
    "inkling_mtp",
)


@dataclass(frozen=True)
class SpeculativeConfig:
    """投机配置（57E 接进调度；58 增加输入槽位派生量；60 增加 ngram 匹配窗口；
    61 增加 suffix decoding 参数；62 增加自定义 proposer 的接入与**方法推断**；
    64 增加 `extract_hidden_states`——一个**不做投机、只借 KV 缓存存特征**的方法；
    65 增加 `mtp`——权重在 target checkpoint 里的原生多 token 预测层）。"""

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
    # 67 关（TLI，上游 `SpeculativeConfig.use_heterogeneous_vocab`）：允许 draft 与 target 用
    # **两套不同的 tokenizer**。开启后初始化会按 token 字符串建交集表，草稿 logits 只留交集里的列，
    # 历史/草稿 id 在两个空间之间搬运（见 `spec_decode/vocab_mapping.py`）。只支持 `draft_model`。
    use_heterogeneous_vocab: bool = False
    # 草稿怎么采（上游 `draft_sample_method`）：`"greedy"`（默认，草稿恒 argmax）或
    # `"probabilistic"`（在草稿自己的分布上采样并把 q 交给验证器）。
    # **TLI 目前只允许 greedy**（上游同款限制）：概率草稿要把 q 从 draft 空间搬到 target 空间，
    # 上游还没实现（代码里留着 TODO），需求 067 §3.5 明确要求不得自行放开。
    draft_sample_method: str = "greedy"
    # 71 关（动态投机长度，上游 `SpeculativeConfig.num_speculative_tokens_per_batch_size`）：
    # **闭区间**三元组表 `[(start, end, K), ...]`——批大小落在 [start, end] 内时本轮猜 K 枚。
    # 段间空隙与尾部沿用前一段/最后一段的 K，表里的 K 一律按 `num_speculative_tokens`（最大
    # 容量）裁剪；`None` = 关（固定用 `num_speculative_tokens`）。
    # 它是"用户给的表"，**不是自适应策略**（不按接受率调 K，需求 071 §1 明确禁止自创）。
    num_speculative_tokens_per_batch_size: list[tuple[int, int, int]] | None = None
    # 72 关（上游 `SpeculativeConfig.parallel_drafting`，`config/speculative.py:168`）：
    # 并行提议——一次 forward 同时给出 K 个位置的 hidden（PARD / P-EAGLE），而不是串行跑 K 次。
    # 它**要求按并行草稿训练的权重**：普通 EAGLE/draft 权重开它不会报错，只会让草稿质量变差
    # （上游注释原话："requires the speculative model be trained to support parallel drafting"）。
    # 输入布局随之变化：把"锚点 + K−1 个 mask token"排成一块（见 `max_num_new_slots_for_drafting`
    # 与 `spec_decode/utils.py::expand_parallel_draft_inputs`）。
    parallel_drafting: bool = False

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
        if self.method in MTP_MODEL_TYPES and self.method != "mtp":
            # 上游 `config/speculative.py:748-756`：别名一律归一到 "mtp"（那里还打一条
            # "method `x` is deprecated and replaced with mtp" 的 warning；本仓库不引 logger，
            # 把这条写进 docs/step65_alignment.md 的别名表）。
            object.__setattr__(self, "method", "mtp")
        if self.method == MLP_SPECULATOR_MODEL_TYPE:
            # 版本缺口（需求 066 §3）：本机 vLLM 0.28.0 里 `mlp_speculator` **只有配置枚举**，
            # 没有可用的模型类、也没有 Runner 分支。三条实测证据（tests/step66 里各有断言）：
            #   1. `ModelRegistry._try_inspect_model_cls("MLPSpeculatorPreTrainedModel")` → None，
            #      因为 `registry.py:687-689` 那一行是注释掉的（`# Temporarily disabled.`）；
            #   2. `model_executor/models/mlp_speculator.py` 里 `MLPSpeculator` **没有 forward**，
            #      只剩 `__init__` / `load_weights`（V0 时代的遗留件）；
            #   3. `GPUModelRunner` 的提议者分派里没有 `mlp_speculator` 分支。
            # 所以这里明确报错，**不自己写一套 MLP 实现**去骗过枚举（那就不再是"对齐 vLLM"了）。
            raise NotImplementedError(
                "method='mlp_speculator' 是本机 vllm==0.28.0 的**版本缺口**："
                "配置枚举存在（MLPSpeculatorConfig 能解析），但模型类没有注册、Runner 也没有分派，"
                "上游自己在 registry.py 里把这一行注释成 'Temporarily disabled'。"
                "本仓库保持不支持（不写自创实现冒充对齐）。若日后要实现，必须先钉一个**真正支持"
                "它**的上游提交，单独增补需求；不得悄悄换参考版本。"
                "（需求 066 §3；证据与最小启动测试见 tests/step66/test_mlp_support_boundary.py）")
        if self.method not in ("ngram", "ngram_gpu", "draft_model", "suffix", "custom_class",
                               "eagle", "eagle3", "extract_hidden_states", "mtp", "medusa"):
            raise ValueError(
                f"本关只支持 method='ngram' / 'ngram_gpu' / 'draft_model' / 'suffix' / "
                f"'custom_class' / 'eagle' / 'eagle3' / 'extract_hidden_states' / 'mtp' / "
                f"'medusa'，收到 "
                f"{self.method!r}"
                "（PARD/DFlash/DSpark 等按需求顺序在后续关卡实现）")
        if self.draft_sample_method not in ("greedy", "probabilistic"):
            raise ValueError(
                f"draft_sample_method 只能是 'greedy' 或 'probabilistic'，收到 "
                f"{self.draft_sample_method!r}（上游 `DraftSampleMethod` 同款）")
        if self.use_heterogeneous_vocab and self.method != "draft_model":
            # 上游 `config/speculative.py:1387-1390` 的原话
            raise ValueError(
                "use_heterogeneous_vocab only works with method='draft_model'"
                f"（收到 method={self.method!r}）：分词空间不同是「拿另一个训练好的模型当 draft」"
                f"才会遇到的事；EAGLE/MTP 这类与 target 共享词表的 draft 不需要它")
        if self.use_heterogeneous_vocab and self.draft_sample_method != "greedy":
            # 上游 `config/speculative.py:1392-1396` 的原话 + 本仓库的说明
            raise ValueError(
                "use_heterogeneous_vocab currently only supports greedy draft sampling. "
                f"收到 draft_sample_method={self.draft_sample_method!r}：概率草稿的 q 是 draft 空间的"
                f"分布，要无损验证必须先把它搬到 target 空间（上游留了 TODO，尚未实现），"
                f"本仓库照抄这条边界，不自行放开（需求 067 §3.5）")
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
        elif self.method == "mtp":
            # **必须排在 use_eagle() 之前**：上游的 `use_eagle()` 把 mtp 也算进"吃 target hidden
            # 的 EAGLE 系"（L1477-1481），但 MTP 的 draft 配置是从 target 派生的、不需要用户给
            # draft_model_config，所以校验规则与 EAGLE 不同（见 _resolve_mtp）。
            self._resolve_mtp()
        elif self.method == "medusa":
            # 66 关：Medusa 的 draft 配置是**用户给的**（一个只有 head 权重的小目录），
            # 处理顺序与上游一致：先归一 hf 配置（旧 checkpoint 的 key 改名 / model_type /
            # architectures / K→head 数），"与 target 对齐词表"那一步需要 target 配置，
            # 由 derive_medusa_draft_config() 在提议者构造时补（见该方法的说明）。
            self._resolve_medusa()
        elif self.use_eagle():
            self._resolve_eagle()
        elif self.uses_extract_hidden_states():
            self._resolve_extract_hidden_states()
        # 71 关：动态投机长度**放在最后**——它要读 `method` 与最终的 `num_speculative_tokens`，
        # 而这两者可能被上面的分支改写（别名归一、ngram 的 K 缺省、extract 的 K=1 校验…）。
        # 上游是懒校验（在 `Scheduler.__init__` 建查找表时才炸），本仓库把配置错误提前到这里：
        # 症状相同，但启动期就报，而不是排到第一轮调度才炸（差异记 docs/step71_alignment.md §6）。
        self._resolve_dynamic_sd()
        # 72 关：并行提议的取值校验（上游只在 docstring 里写"只与 EAGLE 和 draft model 兼容"，
        # 本仓库按"不静默降级"的约定在配置期明确拒绝其它方法）。
        self._resolve_parallel_drafting()

    @staticmethod
    def _draft_model_type(draft_model_config) -> str | None:
        """draft 配置里的 `model_type`（没有 draft 配置时报 None）。

        上游是从 `draft_model_config.hf_config.model_type` 认方法的（`config/speculative.py:958-961`）；
        本仓库的 `SpeculativeConfig` 直到 66 关才有 draft 配置可用，所以在 `_resolve_method()`
        里补上这条**只看配置、不猜**的判定。
        """
        if draft_model_config is None:
            return None
        hf_config = draft_model_config.hf_config or {}
        return hf_config.get("model_type")

    def _resolve_method(self) -> None:
        """`method` 没给时按上游规则推出来（`config/speculative.py:741-756`）。

        顺序也是照抄的：**先看 `model` 是不是自定义类的点号路径**（是 → `custom_class`），
        否则 `model` 是 `ngram`/`[ngram]` → `ngram`，其余一律 `draft_model`（连 `model` 都没给
        也算 draft_model，因为"没写方法"的默认语义是"给一个 draft 模型"）。

        这一步是本关的"分派边界"：**同一条事实只在这里判定一次**，Runner 只按
        `speculative_config.method` 分派，不再自己猜第二遍（否则 CLI 与 Runner 可能各判一套，
        出现"配置说 A、运行时走 B"的静默错）。

        66 关补的两条（上游同位置的 `hf_config.model_type` 分支）：
        `model_type == "medusa"` → `medusa`；`model_type == "mlp_speculator"` →
        `mlp_speculator`（随后在 `__post_init__` 里作为**版本缺口**明确报错）。
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
            elif (draft_model_type := self._draft_model_type(self.draft_model_config)) \
                    in ("medusa", MLP_SPECULATOR_MODEL_TYPE):
                # 上游 `:958-961`：draft 配置自己声明了 model_type 时按它认（旧 FasterDecoding
                # checkpoint 没有 model_type，所以那条路要求用户显式给 method="medusa"）
                object.__setattr__(self, "method", draft_model_type)
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
        """是否走 EAGLE 系提议者（EAGLE-1/2、EAGLE3 与 **MTP** 共用一套提议流程）。

        上游 `config/speculative.py:1477-1481` 逐字如此（那里的注释也写明："这个方法是
        '用 target hidden 做投机'的统称"）：`("eagle", "eagle3", "mtp", "dflash", "dspark")`。
        本仓库实现了前三者。

        **后果**（调度侧依赖它）：`num_lookahead_tokens` = K（它们都要往 target query 之外写
        K 个位置的 KV）、Runner 建 `EagleProposer`、`capture_aux_hidden_states` 之外还要给
        MTP 留 target 的最后一层 hidden。
        """
        return self.method in ("eagle", "eagle3", "mtp")

    def uses_mtp(self) -> bool:
        """是否为原生 MTP（权重在 target checkpoint 里）。"""
        return self.method == "mtp"

    def use_multi_module_mtp(self) -> bool:
        """是否用到**多个** MTP 模块（上游 `config/speculative.py:1501-1507`）。

        `min(num_nextn_predict_layers, K) > 1` 就是多模块。通用 V1 提议路径里
        `spec_step_idx` 恒为 0（只有 step3p5 的专用提议者会递进），所以多模块的调度/状态行为
        属 **80 关**；本关只把判定做出来并在文档里标出边界。
        """
        if self.method != "mtp" or self.draft_model_config is None:
            return False
        num_mtp_layers = int(self.draft_model_config.hf_config.get(
            "num_nextn_predict_layers", 1) or 1)
        return min(num_mtp_layers, self.num_speculative_tokens) > 1

    def _resolve_mtp(self) -> None:
        """`method="mtp"` 的校验（上游 `config/speculative.py:759-772` 与 L1046-1084 的 K 约束）。

        上游在这个分支里做三件事：要求 `target_model_config` 存在、把 `model` 换成 target 的
        模型目录（**MTP 权重就在 target checkpoint 里**，没有第二个模型目录）、对齐量化。
        本仓库的 `SpeculativeConfig` 拿不到 target 配置（上游有这个字段），所以"派生 draft 配置"
        这一步做成 `derive_mtp_draft_config()`，由同时持有两者的提议者调用——与 64 关同一个做法。

        K 的约束：上游"没给 K 就取 `n_predict`"、"K > n_predict 时必须能被 n_predict 整除
        （模块复用）"两条都需要 target 配置里的 `num_nextn_predict_layers`，所以也在
        `derive_mtp_draft_config()` 里判；这里只挡住"根本没给 K"。
        """
        if self.num_speculative_tokens <= 0:
            raise ValueError(
                "MTP 的 num_speculative_tokens 必须 > 0：上游在没给时会取 target 的 "
                "num_nextn_predict_layers 当默认值，本仓库的配置拿不到 target（见 "
                "derive_mtp_draft_config），所以必须显式给")
        object.__setattr__(self, "prompt_lookup_max", 0)
        object.__setattr__(self, "prompt_lookup_min", 0)

    def derive_mtp_draft_config(self, target_model_config: "ModelConfig") -> "ModelConfig":
        """从 **target 配置**派生 MTP 的 draft 配置（上游 `hf_config_override` + `ModelConfig(...)`）。

        上游做的事（`config/speculative.py:342-687` 的 `hf_config_override` + L897-924 的
        `ModelConfig(model=self.target_model_config.model, hf_overrides=...)`）：

            model_type     →  "<家族>_mtp"（按 target 的 model_type/architectures 查表）
            n_predict      →  num_nextn_predict_layers
            architectures  →  ["<家族>MTPModel"]（让加载器去建 MTP 模型而不是 target 模型）
            model          →  target 的模型目录（同一个 checkpoint 文件）

        本仓库的差异（逐条记在 docs/step65_alignment.md §3）：**架构名用我们自己的
        `Qwen3MTPModel`**——本仓库的 MTP 层是稠密 Qwen3 解码层，而 Qwen3-Next 的 MTP 用的是
        混合注意力（GatedDeltaNet），注册成上游名字会让真 checkpoint 静默跑错结构。
        `num_speculative_tokens` 与 `n_predict` 的整除关系也在这里判（上游在配置期判，
        理由同上：本仓库要等拿到 target 配置）。
        """
        target_hf = dict(target_model_config.hf_config or {})
        n_predict = int(target_hf.get("num_nextn_predict_layers") or 0)
        if n_predict <= 0:
            raise ValueError(
                f"MTP 的权重在 target checkpoint 里，但 target 配置里没有 "
                f"num_nextn_predict_layers（收到 {target_hf.get('num_nextn_predict_layers')!r}）："
                f"没有这个字段就无法知道 target 后面附了几层 MTP，无法加载")
        num_drafts = self.num_speculative_tokens
        if num_drafts > n_predict and num_drafts % n_predict != 0:
            # 上游原话：Ensure divisibility for MTP module reuse.
            raise ValueError(
                f"num_speculative_tokens:{num_drafts} 必须能被 n_predict={n_predict} 整除"
                f"（MTP 模块复用：每枚草稿都要落到某一个 MTP 层上）")
        hf_config = {**target_hf,
                     "n_predict": n_predict,
                     "architectures": ["Qwen3MTPModel"]}
        if self.parallel_drafting:
            # 72 关：MTP 的 draft 配置是**从 target 派生**的（没有用户给的 draft 目录），
            # 所以并行开关要在这里补进 hf 配置，模型才会注册 `mask_hidden` buffer
            hf_config["parallel_drafting"] = True
        return replace(target_model_config, hf_config=hf_config)

    def _resolve_medusa(self) -> None:
        """`method="medusa"` 的配置归一与校验（上游 `config/speculative.py:883-935`）。

        上游在 `__post_init__` 的这个分支里做三件事：

        1. 显式给了 `method="medusa"` 时强制 `hf_overrides={"model_type": "medusa"}`
           （旧 FasterDecoding checkpoint 的 config.json 里根本没有 model_type，
           `AutoConfig` 会认成基座模型）；
        2. 建 draft 的 `ModelConfig`（目录 = 用户给的 `self.model`）；
        3. **与 target 对齐词表**：`draft_hf.vocab_size != target_vocab` 时把
           `vocab_size` / `truncated_vocab_size` 都改成 target 的（旧 config 里缺 vocab_size，
           `MedusaConfig` 的默认值是 32001，而 lm_head 的真实宽度是 target 的词表）。

        本仓库的差异：`SpeculativeConfig` 拿不到 target 配置（与 64/65 关同样的情况），
        所以第 3 件拆成 `derive_medusa_draft_config()`，由同时持有两者的 Runner 调用；
        第 1、2 件在这里做（第 2 件在本仓库就是"用户直接给 `draft_model_config`"）。
        """
        if self.draft_model_config is None:
            raise ValueError(
                "method='medusa' 必须在 SpeculativeConfig 里给 draft_model_config"
                "（Medusa 的 head 是独立权重的小目录；上游的 `model=` 就是它）")
        if self.num_speculative_tokens <= 0:
            # 上游："A speculative model was provided, but `num_speculative_tokens` was not
            # provided"——而且对 Medusa 来说 K **就是 head 数**（见 medusa_hf_config），
            # 所以它不能是 0，也没有"从 checkpoint 猜"的余地。
            raise ValueError(
                "method='medusa' 必须给 num_speculative_tokens > 0："
                "它同时就是 Medusa 的 head 数（上游把 draft 配置的 num_heads 改写成 K）")
        object.__setattr__(self, "prompt_lookup_max", 0)
        object.__setattr__(self, "prompt_lookup_min", 0)
        normalized = medusa_hf_config(self.draft_model_config.hf_config,
                                      self.num_speculative_tokens)
        object.__setattr__(self, "draft_model_config",
                           replace(self.draft_model_config, hf_config=normalized))

    def derive_medusa_draft_config(self, target_model_config: "ModelConfig") -> "ModelConfig":
        """补上 Medusa 配置里"只有 target 才能回答"的那部分（上游同位置的最后一段）：

            target_vocab = target.hf_config.vocab_size
            if draft_hf.vocab_size != target_vocab:
                draft_hf.vocab_size = target_vocab
                draft_hf.truncated_vocab_size = target_vocab

        为什么必要：旧 checkpoint 的 config.json 里**没有 vocab_size**，于是落到
        `MedusaConfig` 的默认值 32001，而 `lm_heads.{i}.weight` 的真实宽度是 target 的词表
        （151936）；对齐之后形状才谈得上匹配。**注意**上游是"不等就把两个都改成 target 的"，
        所以显式声明的 `truncated_vocab_size` 在这个前提下也会被冲掉（要保留截断词表，
        就得让配置里的 vocab_size 与 target 一致，例如社区版 config 抄了基座的 vocab_size）。

        另外这里多做一条上游没有的检查：**draft 的 hidden_size 必须等于 target 的**
        （Medusa head 吃的就是 target 的 hidden）。上游靠第一次前向的形状报错，本仓库提前到
        配置期——差异记在 docs/step66_alignment.md §3。
        """
        draft_hf = dict(self.draft_model_config.hf_config or {})
        target_hf = dict(target_model_config.hf_config or {})
        target_vocab = target_hf.get("vocab_size")
        if target_vocab is None:
            raise ValueError("target 配置里没有 vocab_size：无法给 Medusa 的 lm_head 定宽度")
        if draft_hf.get("vocab_size") != target_vocab:
            draft_hf["vocab_size"] = target_vocab
            draft_hf["truncated_vocab_size"] = target_vocab
        if int(draft_hf["hidden_size"]) != int(target_hf.get("hidden_size", 0)):
            raise ValueError(
                f"Medusa 的 hidden_size={draft_hf['hidden_size']} 与 target 的 "
                f"hidden_size={target_hf.get('hidden_size')} 不一致：head 吃的就是 target 的 "
                f"hidden，对不上时上游会在第一次前向炸形状错，本仓库在配置期直接拒绝")
        return replace(self.draft_model_config, hf_config=draft_hf)

    def uses_medusa(self) -> bool:
        """是否走 Medusa 多头提议（上游 Runner 的分派判据就是 `method == "medusa"`）。"""
        return self.method == "medusa"

    #: 72 关：支持并行提议的方法（上游 docstring："Only compatible with EAGLE and draft model
    #: methods"）。EAGLE 系与 MTP 都吃 target hidden（`pass_hidden_states_to_model=True`，左移布局），
    #: draft_model 走不左移布局 —— 这正是 P-EAGLE 与 PARD 两条输入协议。
    _PARALLEL_DRAFTING_METHODS = ("eagle", "eagle3", "mtp", "draft_model")

    def _resolve_parallel_drafting(self) -> None:
        """并行提议的取值校验（72 关）。

        上游 `parallel_drafting` 只在 docstring 里声明"只与 EAGLE 和 draft model 兼容"，
        代码里没有一条检查；不兼容的组合（ngram/suffix/medusa/extract/custom_class）会在
        `SpecDecodeBaseProposer` 里被 `needs_extra_input_slots` 那条分支静默忽略——
        也就是"配置写了、实际没生效"。本仓库按约定在配置期明确拒绝。

        K 的约束也在这里：并行提议一次要采 K 行，K=1 时没有 mask 行（P-EAGLE 的净增槽位是 0，
        仍要能跑），K=0 与串行路径一样不在配置期拦（`num_speculative_tokens=0` 时提议者根本
        不建）。
        """
        if not self.parallel_drafting:
            return
        if self.method not in self._PARALLEL_DRAFTING_METHODS:
            raise ValueError(
                f"parallel_drafting=True（并行提议）只支持 "
                f"{self._PARALLEL_DRAFTING_METHODS}，收到 method={self.method!r}："
                f"这条输入协议要『一次 forward 出 K 个位置』的权重，其余方法的提议者不消费它的"
                f"mask/槽位（上游没有这条校验，写了会被静默忽略）。DFlash / DSpark 的并行协议不同，"
                f"按需求顺序在 76/77 关实现（需求 072 §5 明确要求不能拿同一套 mask 代替）")
        if self.num_speculative_tokens <= 0:
            raise ValueError(
                f"parallel_drafting=True 需要 num_speculative_tokens > 0，收到 "
                f"{self.num_speculative_tokens}：一次并行 forward 要采 1 个锚点 +（K−1）个 mask")
        # 把开关**写进 draft 的 hf 配置**：本仓库的模型只吃一个 config dict（64 关起就是这个
        # 约定），而上游模型是从 live `vllm_config.speculative_config` 读这个开关来决定要不要
        # 注册 `mask_hidden` buffer。不注入的话，一份**合法的 P-EAGLE 权重**会因为模型没建
        # buffer 而加载失败（权重里的 `mask_hidden` 无处可去）。
        # 只有吃 target hidden 的 EAGLE 系需要它（PARD 是普通 LM，没有 mask hidden）。
        if self.use_eagle() and self.draft_model_config is not None:
            draft_hf = dict(self.draft_model_config.hf_config or {})
            if not draft_hf.get("parallel_drafting"):
                draft_hf["parallel_drafting"] = True
                object.__setattr__(self, "draft_model_config",
                                   replace(self.draft_model_config, hf_config=draft_hf))

    #: 71 关：**能真正按轮改 K** 的方法。判据不是"名字看起来行"，而是上游代码里提议者
    #: 确实把 `num_spec_tokens_to_schedule` 当参数用、且支持变宽/零宽返回：
    #:   * `llm_base_proposer.propose(num_speculative_tokens, ...)`：可变宽，且 K=0 时
    #:     **先跑完第一遍**（同步 draft KV）再返回 `[B, 0]`——draft_model / eagle / eagle3 / mtp
    #:   * `ngram_proposer.propose(num_speculative_tokens, ...)`：`assert K <= self.k`，可变宽
    #: 其余方法的提议者都写死了"K == 配置值"的断言（ngram_gpu / suffix / medusa /
    #: extract_hidden_states），custom_class 则根本收不到 K（上游调用形态里没有这个参数），
    #: 所以对它们开动态表**上游会断言失败或静默不生效**。本仓库按约定在配置期明确拒绝，
    #: **不去删上游的断言**（需求 071 §3.6 点名要求）。
    _DYNAMIC_SD_METHODS = ("draft_model", "eagle", "eagle3", "mtp", "ngram")

    def uses_dynamic_speculative_decoding(self) -> bool:
        """要不要按批大小动态选 K（上游同名方法，`config/speculative.py:1489-1490`）。

        判据只有一个：`num_speculative_tokens_per_batch_size` 是不是 None。调度器、图模式降级、
        DP 回退三处都读它，**不各自再判一遍**（否则会出现"配置说开、图说关"的分叉）。
        """
        return self.num_speculative_tokens_per_batch_size is not None

    def _resolve_dynamic_sd(self) -> None:
        """动态投机长度的配置校验（71 关；上游在这个字段上是**懒校验**）。

        三件事，顺序不能反：

        1. **表本身的规则**全部交给上游同名函数 `validate_and_normalize_dynamic_sd_schedule`
           （首段从 1 开始、区间正数/不重叠、K≥0、逐项 int 转换）——不在这里重写一遍规则，
           免得两处口径分叉。校验结果（排序 + 转 int + 转 tuple）写回字段：它是这个配置的
           **唯一权威表示**，后面 `build_dynamic_sd_schedule_lookup` 与测试都读它。
        2. **容量**：`num_speculative_tokens` 是最大 K（工作区、KV lookahead、掩码缓冲都按它建），
           表里的 K 只是被它裁剪，**不能反向放大容量**。所以它必须 > 0
           （上游 `build_dynamic_sd_schedule_lookup` 里同一句 `vllm_num_speculative_tokens <= 0`
           检查；本仓库提前到配置期）。
        3. **方法边界**：只允许 `_DYNAMIC_SD_METHODS`，其余明确报错。
        """
        schedule = self.num_speculative_tokens_per_batch_size
        if schedule is None:
            return
        from .spec_decode.dynamic.utils import (
            validate_and_normalize_dynamic_sd_schedule)

        normalized = validate_and_normalize_dynamic_sd_schedule(schedule)
        object.__setattr__(self, "num_speculative_tokens_per_batch_size", normalized)
        if self.num_speculative_tokens <= 0:
            raise ValueError(
                f"开 num_speculative_tokens_per_batch_size 时 num_speculative_tokens"
                f"（最大 K，工作区/图/掩码都按它建）必须 > 0，收到 "
                f"{self.num_speculative_tokens}：表里的 K 只会被它**裁剪**，"
                f"不能反向放大容量（上游 build_dynamic_sd_schedule_lookup 同款检查）")
        if self.method not in self._DYNAMIC_SD_METHODS:
            raise ValueError(
                f"num_speculative_tokens_per_batch_size（动态投机长度）不支持 "
                f"method={self.method!r}：只有 {self._DYNAMIC_SD_METHODS} 的提议者能按轮改 K。"
                f"上游对其余方法是固定 K 的硬断言（ngram_gpu `assert num_speculative_tokens "
                f"== self.k`、suffix 与 medusa `assert num_speculative_tokens == "
                f"self.num_speculative_tokens`、extract_hidden_states 恒 K=1），"
                f"而 custom_class 的调用形态里根本不传 K"
                f"（动态表对它静默无效）。本仓库不删这些断言、也不静默忽略配置，"
                f"在配置期直接拒绝（需求 071 §3.6）")

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

        逐条照抄上游 `SpeculativeConfig.max_num_new_slots_for_drafting`（本机 0.28.0 的
        `config/speculative.py:1421-1459`）：

        ==================== ============= ======== ==================
        算法                  method       并行      额外槽位
        ==================== ============= ======== ==================
        EAGLE3               eagle3       否       0
        P-EAGLE              eagle3       是       K − 1
        DFlash               dflash       是       K        （76 关）
        DSpark               dspark       是       K − 1    （77 关）
        MTP                  mtp          否       0
        N-gram               ngram        否       0
        Draft model          draft_model  否       1
        PARD                 draft_model  是       K
        ==================== ============= ======== ==================

        **为什么 P-EAGLE 是 K−1、PARD 是 K**：并行提议一次要采样的行是"1 个锚点 + (K−1) 个
        mask"= K 行（`extra_slots_per_request = K`）；而"比 target query 多占几行"取决于那一行
        能不能复用——EAGLE 左移会把 target 块的最后一行**改放锚点**（位置/槽位不变，不新增行），
        所以净增 K−1；PARD 不左移，target 那一行原样当内容行、锚点与 mask 全部接在后面，净增 K。

        64 关的 `extract_hidden_states` 也是 0：它不跑 draft 模型，写的是 **target 本轮那些
        query 行自己的槽位**（与上游的分支表一致：只有 `uses_draft_model()` 才是 1）。

        **不要和 `num_lookahead_tokens` 混**：那个是"额外保留几个 KV 位置"（=K），
        这个是"draft 输入工作区每请求多占几行"。
        """
        if self.uses_dflash():
            # DFlash 用 1 个 bonus query + K 个 mask query（72 关不实现 DFlash，属 76 关）
            return self.num_speculative_tokens
        if self.parallel_drafting:
            if self.uses_draft_model():
                # PARD 不左移：K 个 query 位置全部要新槽位
                return self.num_speculative_tokens
            # 复用 target 已有的那一行，只有 mask query 要新槽位
            return self.num_speculative_tokens - 1
        if self.uses_draft_model():
            # 普通自回归 draft 保留一个未切片的 token（target 本轮刚采出的那个）
            return 1
        return 0

    def uses_dflash(self) -> bool:
        """DFlash（76 关）。本仓库的配置期校验**不允许** `method="dflash"`，所以这里恒为 False；
        留着它是为了让 `max_num_new_slots_for_drafting()` 与上游的分支表逐条对应
        （76 关接进来时只需放开配置校验，槽位口径已经写对）。"""
        return self.method == "dflash"


@dataclass(frozen=True)
class StructuredOutputsConfig:
    """结构化输出的**引擎级**配置（对应 vLLM `config/structured_outputs.py` 的本关子集）。

    为什么是引擎级而不是请求级：后端是多进程执行时每张卡都要有同一套编译环境，上游因此
    **不支持**请求级选后端（它只在 `auto` 时按请求内容挑 backend）。68 关照抄这条边界。
    """

    backend: str = "auto"           # "auto" / "xgrammar"（其余在请求期明确拒绝）
    disable_any_whitespace: bool = False

    def __post_init__(self):
        if self.backend not in ("auto", "xgrammar", "guidance", "outlines",
                                "lm-format-enforcer"):
            raise ValueError(f"未知的 structured_outputs.backend={self.backend!r}")
        if self.disable_any_whitespace and self.backend not in ("xgrammar", "guidance",
                                                                "auto"):
            raise ValueError(
                "disable_any_whitespace 只对 xgrammar / guidance 后端有意义"
                "（上游同款校验）")


class CompilationMode(enum.IntEnum):
    """模型怎么被"编译"（对应上游 `config/compilation.py::CompilationMode`）。

    本仓库**只实现 NONE**：另外三种都要 torch.compile 参与（上游的 `VLLM_COMPILE` 还要
    算子拆分 + 自定义 pass）。把它们照抄进来是为了让"不支持"这件事有一个**共同的名字**：
    用户写 `mode="vllm_compile"` 时得到的是"本仓库未实现该模式"的明确报错，
    而不是"这个字段被忽略、悄悄跑 eager"（那会让人以为编译生效了）。
    """

    NONE = 0
    """纯 eager：模型按写好的 PyTorch 代码逐步执行，不做图编译。"""
    STOCK_TORCH_COMPILE = 1
    """标准 `torch.compile`。"""
    DYNAMO_TRACE_ONCE = 2
    """只做一次 Dynamo trace，避免重复编译。"""
    VLLM_COMPILE = 3
    """上游自定义后端：算子拆分 + 分段编译 + 自定义 pass。"""


class CUDAGraphMode(enum.Enum):
    """CUDA Graph 的模式（逐字对应上游 `config/compilation.py::CUDAGraphMode`）。

    这个枚举同时承担两个角色，**不能混**：

    1. **配置值**（可以有"分段"取值）：`FULL_DECODE_ONLY = (FULL, NONE)` 读作
       "decode 批用 FULL 图，混合 prefill/decode 批不用图"；
       `FULL_AND_PIECEWISE = (FULL, PIECEWISE)` 读作"decode 用 FULL，混合批用 PIECEWISE"。
    2. **运行模式**（只能是具体的那三个）：`NONE` / `PIECEWISE` / `FULL`——图包装器实际
       比对的就是它，`valid_runtime_modes()` 给的就是这三个。

    `decode_mode()` / `mixed_mode()` 把那对取值拆开；`separate_routine()` 说明"两种批走
    不同套路"，`has_mode()` 说明"整体上含不含某个具体模式"。
    """

    NONE = 0
    PIECEWISE = 1
    FULL = 2
    FULL_DECODE_ONLY = (FULL, NONE)
    FULL_AND_PIECEWISE = (FULL, PIECEWISE)

    def decode_mode(self) -> "CUDAGraphMode":
        """decode（统一 1+K 行）批用哪个具体模式。"""
        return CUDAGraphMode(self.value[0]) if self.separate_routine() else self

    def mixed_mode(self) -> "CUDAGraphMode":
        """混合 prefill/decode 批用哪个具体模式。"""
        return CUDAGraphMode(self.value[1]) if self.separate_routine() else self

    def has_mode(self, mode: "CUDAGraphMode") -> bool:
        if mode.separate_routine():
            raise ValueError(f"has_mode() 只接受具体运行模式，收到分段取值 {mode}")
        if self.separate_routine():
            return mode.value in self.value
        return self == mode

    def requires_piecewise_compilation(self) -> bool:
        return self.has_mode(CUDAGraphMode.PIECEWISE)

    def max_cudagraph_mode(self) -> "CUDAGraphMode":
        return CUDAGraphMode(max(self.value)) if self.separate_routine() else self

    def has_full_cudagraphs(self) -> bool:
        return self.max_cudagraph_mode() == CUDAGraphMode.FULL

    def has_piecewise_cudagraphs(self) -> bool:
        return self.requires_piecewise_compilation()

    def separate_routine(self) -> bool:
        return isinstance(self.value, tuple)

    @classmethod
    def valid_runtime_modes(cls) -> frozenset["CUDAGraphMode"]:
        """可以作为**运行模式**出现的三个（图包装器只认这三个）。"""
        return frozenset({cls.NONE, cls.PIECEWISE, cls.FULL})

    def is_valid_runtime_mode(self) -> bool:
        return self in CUDAGraphMode.valid_runtime_modes()

    def __str__(self) -> str:
        return self.name

    def __bool__(self) -> bool:
        # 上游同款：`if cudagraph_mode:` 的语义是"不是 NONE"
        return self != CUDAGraphMode.NONE


#: `CompilationMode` 字符串别名 → 枚举。上游是 pydantic 的 Literal，本仓库收这几个写法。
_COMPILATION_MODE_ALIASES = {
    "none": CompilationMode.NONE,
    "stock_torch_compile": CompilationMode.STOCK_TORCH_COMPILE,
    "dynamo_trace_once": CompilationMode.DYNAMO_TRACE_ONCE,
    "vllm_compile": CompilationMode.VLLM_COMPILE,
}
#: `CUDAGraphMode` 字符串别名 → 枚举（含上游那对分段取值）。
_CUDAGRAPH_MODE_ALIASES = {
    "none": CUDAGraphMode.NONE,
    "piecewise": CUDAGraphMode.PIECEWISE,
    "full": CUDAGraphMode.FULL,
    "full_decode_only": CUDAGraphMode.FULL_DECODE_ONLY,
    "full_and_piecewise": CUDAGraphMode.FULL_AND_PIECEWISE,
}

#: "分段图的切分点"。上游 `CompilationConfig._attention_ops` 填的是**编译期**要拆出来的
#: 注意力算子名（`vllm::unified_attention_with_output` 等），`splitting_ops_contain_attention()`
#: 靠比较这份名单来回答"切分点是不是注意力"。本仓库没有编译期算子名可拆，切分点是模型结构里
#: 的 `Attention` 边界，所以这里放的是**同一个语义的名字**：图里不含注意力核心，只含它前后的段。
_ATTENTION_SPLIT_OPS = ["minivllm::attention_core"]


@dataclass
class CompilationConfig:
    """编译与 CUDA Graph 的引擎级配置（对应上游 `config/compilation.py` 的子集）。

    **为什么不加"本仓库自己的旋钮"**（AGENTS §8）：能对齐的字段就照抄上游的名字与默认值，
    对不上的能力（piecewise / torch.compile / LoRA 特化）用**明确的报错**表达，
    而不是加一个"打开就能用"的开关——那会变成"看起来支持"。

    字段：
        mode                     编译模式。本仓库只接受 NONE，其余在构造期报错。
        cudagraph_mode           图模式。None = 交给 `VllmConfig` 按能力解析。
        cudagraph_num_of_warmups 捕获前的热身次数（上游默认 0；热身的用处是把 lazy
                                 init/workspace 分配从"被录进图"里排除掉）。
        cudagraph_capture_sizes  要捕获的**档位**列表；None = 按上游的默认档位生成。
        max_cudagraph_capture_size  档位上限（None = 按 max_num_seqs*(1+K)*2 与 512 取小）。
        compile_sizes            上游给"编译若干固定 shape"用；本仓库没有编译，只保留字段，
                                 非空即报错（不接受"配了但没生效"）。
    """

    mode: CompilationMode | str = CompilationMode.NONE
    cudagraph_mode: CUDAGraphMode | str | None = None
    cudagraph_num_of_warmups: int = 0
    cudagraph_capture_sizes: list[int] | None = None
    max_cudagraph_capture_size: int | None = None
    compile_sizes: list[int] | None = None
    # 上游用它把注意力算子从被编译的图里拆出来（piecewise 的前提）。本仓库不编译，
    # 保留空列表只为让"为什么不支持 piecewise"这句话有据可依。
    splitting_ops: list[str] | None = None
    # 让"编译模式 NONE + 图模式 FULL"这条组合在文档里有一个显式名字：图与编译是两件事，
    # 可以只做后者（上游 `cudagraph_mode=FULL, mode=NONE` 也是合法组合）。
    enforce_eager: bool | None = None

    def __post_init__(self):
        self.mode = self._as_compilation_mode(self.mode)
        if self.cudagraph_mode is not None:
            self.cudagraph_mode = self._as_cudagraph_mode(self.cudagraph_mode)
        if self.mode != CompilationMode.NONE:
            raise NotImplementedError(
                f"compilation_config.mode={self.mode.name} 在本仓库未实现：torch.compile "
                f"这条路要算子拆分/自定义后端，而本仓库的注意力是逐请求的 Torch 循环"
                f"（Python 层有同步与动态形状），编译它只会到处 graph break。"
                f"用 mode='none' + cudagraph_mode='full'（图只减少 kernel 启动开销，"
                f"不改变算子实现，与编译是两件独立的事）。上游编号见 vllm/config/compilation.py")
        if self.compile_sizes:
            raise NotImplementedError(
                "compile_sizes 需要编译路径（本仓库 mode 只能是 NONE）：配了却不生效的字段"
                "比报错更危险")
        if self.splitting_ops is not None:
            raise NotImplementedError(
                f"splitting_ops 在本仓库不可配置（收到 {self.splitting_ops!r}）：上游靠它"
                f"指定编译期切分点，而本仓库的切分点由模型结构决定（每层注意力边界），"
                f"配置它不会有任何效果——配了不生效比报错更危险")
        # 切分点是**模型结构**决定的常量（不是配置出来的）：每层拆成"注意力前段 / 注意力核心 /
        # 后段"，图只录前/后两段。上游在 `mode=VLLM_COMPILE` 时把这里填成注意力算子名，
        # 我们填的是同一语义的名字（`_ATTENTION_SPLIT_OPS`），供降级规则里的
        # `splitting_ops_contain_attention()` 使用——它恒为 True，因为我们的切分点就在注意力处。
        self.splitting_ops = list(_ATTENTION_SPLIT_OPS)

    def splitting_ops_contain_attention(self) -> bool:
        """切分点里有没有注意力（上游同名方法）。

        上游比较的是"用户配的算子名 vs 各注意力后端的算子名"；本仓库的切分点由模型结构决定
        （`_ATTENTION_SPLIT_OPS`），所以这个答案是常量 True——它在降级规则里的作用不变：
        "切分点在注意力处 = 分段图可行"（上游在这一点上降 PIECEWISE，否则降 NONE/纯 decode）。
        """
        return any(op in _ATTENTION_SPLIT_OPS for op in (self.splitting_ops or []))

    @staticmethod
    def _as_compilation_mode(value) -> CompilationMode:
        if isinstance(value, CompilationMode):
            return value
        if isinstance(value, str) and value in _COMPILATION_MODE_ALIASES:
            return _COMPILATION_MODE_ALIASES[value]
        raise ValueError(f"未知的 compilation mode={value!r}；"
                         f"可选 {sorted(_COMPILATION_MODE_ALIASES)}")

    @staticmethod
    def _as_cudagraph_mode(value) -> CUDAGraphMode:
        if isinstance(value, CUDAGraphMode):
            return value
        if isinstance(value, str) and value in _CUDAGRAPH_MODE_ALIASES:
            return _CUDAGRAPH_MODE_ALIASES[value]
        raise ValueError(f"未知的 cudagraph_mode={value!r}；"
                         f"可选 {sorted(_CUDAGRAPH_MODE_ALIASES)}")

    def adjust_cudagraph_sizes_for_spec_decode(self, uniform_decode_query_len: int) -> None:
        """把档位表**向上取整**到 `1+K` 的倍数（对应上游 `config/compilation.py:1519`）。

        为什么必须做：统一 decode 图要求"每请求恰好 1+K 行"，所以能当图键的总行数只能是
        `1+K` 的倍数。上游默认档位表里有 1/2/4/8/... 这些非倍数档位，K>0 时它们既建不出键、
        也会让 `_create_padded_batch_descriptor()` 的整除断言失败（上游 issue #28207 就是这个：
        投机 + CUDA Graph 在 K 不是 2 的幂减一时直接崩）。

        规则逐行对齐上游：
            每个档位向上取整到 `q` 的倍数，超过 `max_cudagraph_capture_size` 的丢掉；
            一个都不剩且 `q <= max` → 用 `[q]`；
            还是不剩（q 比上限还大）→ 明确报错（不能静默变回"没有图"）。
        本仓库没有 sequence parallelism，所以没有上游"再取 tp 的倍数"那一步。
        """
        multiple_of = uniform_decode_query_len
        if not self.cudagraph_capture_sizes or multiple_of <= 1:
            return
        if self.max_cudagraph_capture_size is None:
            raise RuntimeError("取整档位前必须先定下 max_cudagraph_capture_size")
        round_up = lambda size: -(-size // multiple_of) * multiple_of   # noqa: E731
        rounded_sizes = sorted({round_up(size) for size in self.cudagraph_capture_sizes
                                if round_up(size) <= self.max_cudagraph_capture_size})
        if not rounded_sizes and multiple_of <= self.max_cudagraph_capture_size:
            rounded_sizes = [multiple_of]
        if not rounded_sizes:
            raise ValueError(
                f"按 1+K={multiple_of} 取整之后没有任何合法档位（上限 "
                f"{self.max_cudagraph_capture_size}）：请调小 num_speculative_tokens "
                f"或调大 max_cudagraph_capture_size / max_num_batched_tokens")
        self.max_cudagraph_capture_size = rounded_sizes[-1]
        self.cudagraph_capture_sizes = rounded_sizes

    def post_init_cudagraph_sizes(self) -> None:
        """档位表的收尾校验（上游 `post_init_cudagraph_sizes`）：

        升序、最大档位必须等于 `max_cudagraph_capture_size`。少了这条，`_bs_to_padded_graph_size`
        的"最近档位"映射会在尾部对不上（大于最大档位却仍被判成"该走图"）。
        """
        if self.cudagraph_capture_sizes:
            self.cudagraph_capture_sizes = sorted(self.cudagraph_capture_sizes)
            if self.cudagraph_capture_sizes[-1] != self.max_cudagraph_capture_size:
                raise ValueError(
                    f"cudagraph_capture_sizes 的最大值 "
                    f"{self.cudagraph_capture_sizes[-1]} 与 max_cudagraph_capture_size "
                    f"{self.max_cudagraph_capture_size} 不一致")


#: 上游允许开异步调度的投机方法白名单（`config/vllm.py:1194-1203` 的那串判断）。
#: Eagle 系（eagle / eagle3 / mtp 各别名）、GPU ngram、draft_model；其余方法（CPU ngram、
#: suffix、medusa、extract_hidden_states、用户插件）自动**关**异步——本仓库照抄这份白名单，
#: 不自己放宽也不自己解释原因（上游源码里只给了 warning，没有写理由）。
_ASYNC_SPEC_METHODS = frozenset({
    "eagle", "eagle3", "mtp", "draft_model", "ngram_gpu", "dspark",
    *MTP_MODEL_TYPES,
})


def resolve_async_scheduling(vllm_config: "VllmConfig",
                             executor_supports_async: bool) -> bool:
    """把 `SchedulerConfig.async_scheduling` 解析成最终布尔值（对应上游 `VllmConfig.__post_init__`
    里那段 `async_scheduling is None` 的推断，`config/vllm.py:1185-1234`）。

    与上游的差异只有一处、而且是**结构差异**：上游在 `VllmConfig.__post_init__` 里做，那里它能
    从 `executor_backend` 字符串查到 executor 类并问 `supports_async_scheduling()`；本仓库的
    `VllmConfig` 不持有 executor（executor 是 `EngineCore` 的构造参数），所以这部分由
    `EngineCore.__init__` 调用本函数完成，规则逐条照抄：

        显式 True   → 用；但 executor 不支持就**报错**（上游同款：`raise ValueError`）
        显式 False  → 用
        None        → 默认开；除非 executor 不支持、或投机方法不在白名单里（打 warning 后关）
    """
    requested = vllm_config.scheduler_config.async_scheduling
    if requested is True:
        if not executor_supports_async:
            raise ValueError(
                "显式要求 async_scheduling=True，但当前 executor 不支持异步调度"
                "（上游同样直接报错，不静默降级）")
        # ⚠️ 本项目**未接线的组合**（70 关只做到骨架，证据见 `docs/step70_alignment.md` §6）：
        # 异步要求"下一轮输入不必等上一轮结果"——上游靠执行侧把上一轮采样的 token 与草稿留在
        # GPU 侧、直接 scatter 进输入缓冲（`prev_sampled_token_ids`），而本仓库的输入组装
        # （`_prepare_inputs`）与全部提议器都是 CPU 驱动的。照抄的结果是"行起点/草稿值的口径
        # 与调度器的乐观计划错位"：实测（a）与前缀缓存同开会算错进度并触发 device-side assert；
        # （b）长跑下偶发与同步输出不一致。宁可拒绝，也不要一个偶尔给出错结果的引擎。
        raise NotImplementedError(
            "async_scheduling=True 在本仓库尚未接线到端到端：70 关交付了状态机骨架"
            "（占位符/批队列/异步交付/缓冲生命周期，单测覆盖并实测通过），但执行侧的输入组装"
            "仍是 CPU 驱动的，无法在「不等上一轮结果」的前提下组装下一轮输入。"
            "要放开需要把输入组装搬到 GPU 侧（上游的 prev_sampled_token_ids scatter），"
            "属 74/75 关那条路。默认（async_scheduling=None 或 False）走同步路径，一切照旧。")
    # `None`（自动推断）：**本仓库的默认是关**，与上游默认开不同，理由写在
    # `docs/step70_alignment.md` §6.1：上游的 runner 能把"上一轮采样的 token 与草稿"留在
    # GPU 侧直接 scatter 进下一轮输入（所以整条链都不需要 D2H），而本仓库的输入组装
    # （`_prepare_inputs`）与全部提议器都是 CPU 驱动的——异步在这里必须先把上一轮的结果
    # 结清才能组装下一轮输入。状态机、占位符、批队列、异步拷贝都已按上游实现并验证，
    # 但"默认开"要等执行侧真正 GPU 驻留（74/75 关那条路）才算兑现。
    if not executor_supports_async:
        return False
    speculative_config = vllm_config.speculative_config
    if speculative_config is not None \
            and speculative_config.method not in _ASYNC_SPEC_METHODS:
        import warnings

        warnings.warn(
            f"async scheduling 与 {speculative_config.method} 投机不兼容（上游同款白名单），"
            f"显式开启时也会按上游规则关掉", stacklevel=2)
        return False
    return False


@dataclass(frozen=True)
class VllmConfig:
    model_config: ModelConfig
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)
    device_config: DeviceConfig = field(default_factory=DeviceConfig)
    speculative_config: SpeculativeConfig | None = None
    structured_outputs_config: StructuredOutputsConfig = field(
        default_factory=StructuredOutputsConfig)
    # 69 关：编译/图配置（上游 `VllmConfig.compilation_config`）。默认值本身是 None 语义：
    # `cudagraph_mode=None` 表示"让引擎按能力解析"，解析结果写回这个可变对象。
    compilation_config: CompilationConfig = field(default_factory=CompilationConfig)

    def __post_init__(self):
        self._resolve_cudagraph_config()

    def _resolve_cudagraph_config(self) -> None:
        """把 `cudagraph_mode=None` 解析成本仓库**真的支持**的模式，并算出图档位表。

        对应上游 `VllmConfig._set_cudagraph_sizes()`（`config/vllm.py:1878-2043`）的两件事：
        决定要不要建图、决定 `cudagraph_capture_sizes` / `max_cudagraph_capture_size`。

        本仓库的解析规则（每一条都有理由，写在 `docs/step69_alignment.md` §2）：

            enforce_eager=True            → NONE（上游同款开关）
            设备不是 CUDA                 → NONE（CPU 上没有图可捕获）
            用户显式给了 NONE             → NONE
            其它（None / FULL / 含 FULL） → FULL_AND_PIECEWISE（= 上游 V1 的默认）

        粗解析之后还有 71 关的两条**配置期改写**（见下面两个 `_maybe_*` 方法）：动态投机
        长度在 DP>1 下被关掉、在含 full graph 时把模式降成 PIECEWISE。两条都发生在这里，
        所以外部读到的模式/档位表已经是"最终会执行的那个"。

        这里只做"**粗解析**"：把 None 变成默认值、把 CUDA 不可用变成 NONE。真正"这个模式在
        这份配置下能不能成立"要等**注意力后端就绪**才能回答（能力档位决定 FULL 能不能用于
        混合批），那一步在 `GPUModelRunner.initialize_cudagraph_capture()` 里调用
        `resolve_cudagraph_mode_and_sizes()` ——与上游的分工一致
        （上游 `_set_cudagraph_sizes()` 管档位，`_check_and_update_cudagraph_mode()` 管降级）。
        """
        compilation_config = self.compilation_config
        requested = compilation_config.cudagraph_mode
        device = self.device_config.device
        if self.model_config.enforce_eager or str(device) != "cuda":
            resolved = CUDAGraphMode.NONE
        elif requested is None:
            resolved = CUDAGraphMode.FULL_AND_PIECEWISE
        else:
            resolved = requested
        compilation_config.cudagraph_mode = resolved
        # 71 关：动态投机长度的两条**配置期改写**，位置与上游一致
        # （上游 `config/vllm.py:1414-1415`，都在 `_set_cudagraph_sizes()` 之前）：
        # 1) DP>1 直接关掉动态表（各 rank 选的 K 可能不同 → 分歧/死锁）；
        # 2) V1 的 full graph 降级成 PIECEWISE（每轮 1+K 的形状都在变，全图冻结不了形状）。
        # 顺序不能反：DP 关掉表之后，"是不是动态投机"就已经是 False，第 2 条自然不再触发。
        self._maybe_disable_dynamic_sd_for_data_parallel()
        self._maybe_override_dynamic_sd_cudagraph_mode()
        if resolved == CUDAGraphMode.NONE:
            compilation_config.max_cudagraph_capture_size = 0
            compilation_config.cudagraph_capture_sizes = []
        else:
            self._set_cudagraph_sizes()
        compilation_config.post_init_cudagraph_sizes()

    def _maybe_disable_dynamic_sd_for_data_parallel(self) -> None:
        """DP>1 时关掉动态投机长度（对应上游 `VllmConfig._maybe_disable_dynamic_sd_for_data_parallel`，
        `config/vllm.py:929-946`）。

        为什么 DP 与动态表不能共存：K 由**每个 rank 自己的**本轮批大小决定，而 DP 的各 rank
        数据不同 → 选的 K 不同 → 提议/验证长度的形状不一致 → 集合通信里就是分歧与死锁
        （上游注释原话："causing DP divergence and deadlocks"）。所以上游不是"警告后继续"，
        而是**把这个配置项清掉**、退回固定 K，并留一条 warning。

        本仓库的差异（记在 docs/step71_alignment.md §6）：本项目是**单进程单卡**，没有
        `ParallelConfig` 这一轴，所以 `parallel_config` 不存在时按 `data_parallel_size=1` 处理
        （这条判断因此在本仓库的正常路径上不可达）。规则本身照抄，测试用一个带
        `data_parallel_size` 的替身对象把它钉住——**不是**"写了不跑"的死代码。
        """
        speculative_config = self.speculative_config
        if (speculative_config is None
                or not speculative_config.uses_dynamic_speculative_decoding()):
            return
        data_parallel_size = getattr(
            getattr(self, "parallel_config", None), "data_parallel_size", 1)
        if data_parallel_size <= 1:
            return

        import logging

        logging.getLogger(__name__).warning(
            "动态投机长度不支持 data parallel（DP=%d）：各 rank 可能选出不同的 K，"
            "导致 DP 分歧与死锁。已清空 num_speculative_tokens_per_batch_size，"
            "回退到固定 num_speculative_tokens=%d（上游同款处理）",
            data_parallel_size, speculative_config.num_speculative_tokens)
        # 配置是 frozen dataclass，只能这样清（上游那里是普通赋值，效果相同）
        object.__setattr__(speculative_config,
                           "num_speculative_tokens_per_batch_size", None)

    def _maybe_override_dynamic_sd_cudagraph_mode(self) -> None:
        """动态投机长度 + full graph → 降级成 PIECEWISE（对应上游
        `VllmConfig._maybe_override_dynamic_sd_cudagraph_mode`，`config/vllm.py:910-927`）。

        为什么必须降：full CUDA graph 的形状（请求数、每请求行数）在捕获时就冻结了，而动态
        投机长度**每轮都可能换 K** → 1+K 也在变。上游 V1 的处理是"不用全图，改用分段图"
        （分段图只锁 token 数、注意力留在图外，见 69b）。V2 runner 支持在动态 K 下capture
        多组 query 长度，上游因此放行了 `use_v2_model_runner`；**本仓库没有 V2 runner 这一轴**
        （属 73/74 关），所以那半个条件恒为真——差异记在 docs/step71_alignment.md §6。

        只改模式、不改档位表：`_set_cudagraph_sizes()` 在这一步之后才跑，所以投机的
        "档位取整到 1+K 的倍数"也不会再发生（那一步本来就只在 decode FULL 时做）。
        """
        speculative_config = self.speculative_config
        if (speculative_config is None
                or not speculative_config.uses_dynamic_speculative_decoding()):
            return
        compilation_config = self.compilation_config
        if not compilation_config.cudagraph_mode.has_full_cudagraphs():
            return

        import logging

        logging.getLogger(__name__).warning(
            "动态投机长度会逐轮改变验证长度，full CUDA graph 冻结不了这个形状："
            "把 cudagraph_mode 从 %s 降级为 PIECEWISE（上游同款；"
            "要保留全图需要 V2 model runner，本仓库没有这一轴）",
            compilation_config.cudagraph_mode.name)
        compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE

    def resolve_cudagraph_mode_and_sizes(self, min_cg_support, min_cg_attn_backend: str,
                                         uniform_decode_query_len: int = 1) -> CUDAGraphMode:
        """按**注意力后端的能力**决定最终模式（对应上游 `CompilationConfig.
        resolve_cudagraph_mode_and_sizes()`，`config/compilation.py:1369-1470`）。

        为什么必须有这一步：`FULL` 的含义是"混合 prefill/decode 批也录全图"，而那要求注意力
        后端支持任意批（`AttentionCGSupport.ALWAYS`）。本仓库的 Torch 后端只支持"每请求行数
        相同"的批（`UNIFORM_BATCH`），所以请求 `full` 必须被**降级**到
        `FULL_AND_PIECEWISE`（decode 全图 + 混合批分段图），而不是照字面执行、然后在捕获时炸。

        三条检查，逐条与上游同义（上游还有 SP/inductor 分支，本仓库没有对应能力，故略）：

            1. `mixed_mode() == FULL` 但能力不是 ALWAYS → 降到 FULL_AND_PIECEWISE
               （切分点含注意力时）/ FULL_DECODE_ONLY（不含时，本仓库恒为前者）
            2. `decode_mode() == FULL` 但能力是 NEVER   → 降到 PIECEWISE，或（切分点不含注意力
               时）降到 NONE
            3. 投机（`1+K > 1`）且能力低于 UNIFORM_BATCH → 同样降级
            最后再确认一次：降级完还要求 FULL 而能力是 NEVER → 明确报错（不静默变 eager）

        降级一律**带 warning 且改的是 `compilation_config.cudagraph_mode`**（调用方与图包装器
        读的都是它），所以"实际用了什么模式"在配置对象上是可查的，不是只在日志里。
        """
        from .attention.backend import AttentionCGSupport

        compilation_config = self.compilation_config
        cudagraph_mode = compilation_config.cudagraph_mode
        if cudagraph_mode is None or cudagraph_mode == CUDAGraphMode.NONE:
            return CUDAGraphMode.NONE

        def _warn(message: str) -> None:
            import logging

            logging.getLogger(__name__).warning(message)

        # 1) 混合批要全图 → 必须 ALWAYS
        if (cudagraph_mode.mixed_mode() == CUDAGraphMode.FULL
                and min_cg_support != AttentionCGSupport.ALWAYS):
            msg = (f"CUDAGraphMode.{cudagraph_mode.name} 不被 {min_cg_attn_backend} 后端支持"
                   f"（能力档位 {min_cg_support.name}）：它要求把**混合 prefill/decode 批**"
                   f"也录进一张图，而这个后端的图内路径只认'每请求行数相同'的批")
            if compilation_config.splitting_ops_contain_attention():
                msg += "；降级为 cudagraph_mode=FULL_AND_PIECEWISE（混合批走分段图）"
                cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
            else:
                msg += "；降级为 cudagraph_mode=FULL_DECODE_ONLY（混合批回退 eager）"
                cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
            _warn(msg)

        # 2) decode 要全图但后端完全不能进图
        if (cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
                and min_cg_support == AttentionCGSupport.NEVER):
            msg = (f"CUDAGraphMode.{cudagraph_mode.name} 不被 {min_cg_attn_backend} 后端支持"
                   f"（能力档位 NEVER）")
            if compilation_config.splitting_ops_contain_attention():
                msg += "；降级为 cudagraph_mode=PIECEWISE（注意力留在图外）"
                cudagraph_mode = CUDAGraphMode.PIECEWISE
            else:
                msg += "；降级为 cudagraph_mode=NONE（没有可用的分段图）"
                cudagraph_mode = CUDAGraphMode.NONE
            _warn(msg)

        # 3) 投机：图内注意力要求"每请求 1+K 行"，低于 UNIFORM_BATCH 的后端做不到
        if (cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
                and uniform_decode_query_len > 1
                and min_cg_support.value < AttentionCGSupport.UNIFORM_BATCH.value):
            msg = (f"投机解码（每请求 {uniform_decode_query_len} 行）下 "
                   f"CUDAGraphMode.{cudagraph_mode.name} 不被 {min_cg_attn_backend} 支持"
                   f"（能力档位 {min_cg_support.name}）")
            if compilation_config.splitting_ops_contain_attention():
                msg += "；降级为 cudagraph_mode=PIECEWISE"
                cudagraph_mode = CUDAGraphMode.PIECEWISE
            else:
                msg += "；降级为 cudagraph_mode=NONE"
                cudagraph_mode = CUDAGraphMode.NONE
            _warn(msg)

        # 降级完仍要求全图、而后端完全不能进图 → 明确报错（不静默变 eager）
        if (cudagraph_mode.has_full_cudagraphs()
                and min_cg_support == AttentionCGSupport.NEVER):
            raise ValueError(
                f"cudagraph_mode={cudagraph_mode.name} 要求全图，但 {min_cg_attn_backend} "
                f"后端的能力档位是 NEVER：请显式改成 cudagraph_mode='piecewise'")

        compilation_config.cudagraph_mode = cudagraph_mode
        # 模式变了（尤其 NONE ↔ 非 NONE）就要重算档位表：NONE 下档位表必须是空的，
        # 从 NONE 降级过来的路径不会再跑 `_set_cudagraph_sizes()`。
        if cudagraph_mode == CUDAGraphMode.NONE:
            compilation_config.max_cudagraph_capture_size = 0
            compilation_config.cudagraph_capture_sizes = []
        elif not compilation_config.cudagraph_capture_sizes:
            self._set_cudagraph_sizes()
        compilation_config.post_init_cudagraph_sizes()
        return cudagraph_mode

    def _set_cudagraph_sizes(self) -> None:
        """图档位表（上游 `VllmConfig._set_cudagraph_sizes` 的等价实现，去掉 SP/LoRA 分支）。

        上游默认档位是 `[1, 2, 4] + range(8, 256, 8) + range(256, max+1, 16)`——小 batch 密、
        大 batch 疏。每个真实形状都会被**补齐到最近的档位**（见 `CudagraphDispatcher`），
        所以档位越密、padding 越少，但捕获的图越多（捕获耗时 + 显存）。

        与上游一致的两条边界：
          - `max_cudagraph_capture_size` 不超过 `max_num_batched_tokens`（图不可能比输入预算还大）；
          - `max_num_batched_tokens` 本身若在范围内就额外补一个档位（否则"满批"永远命中不了图）。
        """
        compilation_config = self.compilation_config
        max_cudagraph_capture_size = compilation_config.max_cudagraph_capture_size
        if max_cudagraph_capture_size is None:
            # 上游：`min(max_num_seqs * (1+K) * 2, 512)`（数据中心 Blackwell 是 1024），
            # 再被 max_num_batched_tokens 夹住。
            decode_query_len = 1 + self.num_speculative_tokens
            max_cudagraph_capture_size = min(
                self.scheduler_config.max_num_seqs * decode_query_len * 2, 512)
        max_num_tokens = self.scheduler_config.max_num_batched_tokens
        max_cudagraph_capture_size = min(max_num_tokens, max_cudagraph_capture_size)
        if max_cudagraph_capture_size < 1:
            raise ValueError(
                f"max_cudagraph_capture_size 解析成了 {max_cudagraph_capture_size}："
                f"要么 max_num_batched_tokens 太小，要么显式配置不合理")

        if compilation_config.cudagraph_capture_sizes is not None:
            sizes = sorted({int(size) for size in compilation_config.cudagraph_capture_sizes
                            if int(size) <= max_num_tokens})
            if not sizes:
                raise ValueError(
                    "用户给的 cudagraph_capture_sizes 里没有一个不超过 "
                    f"max_num_batched_tokens={max_num_tokens}：这些图永远不可能被命中")
        else:
            sizes = [size for size in (1, 2, 4) if size <= max_cudagraph_capture_size]
            if max_cudagraph_capture_size >= 8:
                sizes += list(range(8, min(max_cudagraph_capture_size + 1, 256), 8))
            if max_cudagraph_capture_size >= 256:
                sizes += list(range(256, max_cudagraph_capture_size + 1, 16))
            if max_num_tokens <= max_cudagraph_capture_size and max_num_tokens not in sizes:
                sizes.append(max_num_tokens)
            sizes = sorted(set(sizes))

        valid_max_size = sizes[-1] if sizes else 0
        if (compilation_config.max_cudagraph_capture_size is not None
                and compilation_config.max_cudagraph_capture_size != valid_max_size):
            if compilation_config.cudagraph_capture_sizes is not None:
                raise ValueError(
                    f"显式给的 max_cudagraph_capture_size="
                    f"{compilation_config.max_cudagraph_capture_size} 与 "
                    f"cudagraph_capture_sizes 的最大值 {valid_max_size} 不一致")
        compilation_config.max_cudagraph_capture_size = valid_max_size
        compilation_config.cudagraph_capture_sizes = sizes
        # 投机：decode 走 FULL 图时必须把档位取整到 1+K 的倍数（上游同款，
        # 见 `adjust_cudagraph_sizes_for_spec_decode` 的说明）
        if (compilation_config.cudagraph_mode is not None
                and compilation_config.cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
                and self.num_speculative_tokens > 0):
            compilation_config.adjust_cudagraph_sizes_for_spec_decode(
                1 + self.num_speculative_tokens)

    @property
    def num_speculative_tokens(self) -> int:
        """本轮最多几枚草稿（上游 `VllmConfig.num_speculative_tokens`）。

        结构化输出要用它算掩码缓冲的大小：`max_num_seqs * (1 + K)` 行；69 关的图档位、
        `uniform_decode_query_len = 1 + K` 也要它。
        """
        return (self.speculative_config.num_speculative_tokens
                if self.speculative_config is not None else 0)
