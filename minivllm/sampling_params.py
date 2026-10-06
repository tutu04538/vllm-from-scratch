"""请求级采样参数（对应 vLLM 的 `vllm/sampling_params.py`，这里只留子集）。

三条约定：

1. **属于请求，不属于引擎**：同一个引擎里的两条请求可以有不同的 temperature / top_k / seed。
2. **不持有 `torch.Generator`**：参数是"可复制、可校验的配置"；随机流是请求的运行期状态，
   归执行侧（57E 的采样路径）管。step56 把 generator 塞在 `SamplingState` 里跟着请求走，
   那会让配置对象变成活对象，不能当快照传。
3. **`max_tokens` 是"要生成多少"，不是"总上下文多长"**——总长度上限在 `ModelConfig.max_model_len`。
   `check_stop` 两个都要看。

### 68 关新增的三类字段

- **logprobs**（`logprobs`）：`None` = 不要；`k>0` = 每个位置给 top-k + 实际采到的那个；
  `-1` = 全词表。四种"原始/处理后"模式的开关在 `ModelConfig.logprobs_mode`（引擎级，
  上游同款）。
- **min_p**：低于"最大概率 × min_p"的 token 全屏蔽。上游把它做成一个"不改变 argmax"的
  logits processor（`MinPLogitsProcessor`），本项目没有插件注册表，按**同一位置与同一算式**
  内联在采样器里（见 `sample/sampler.py`）。
- **structured_outputs**：语法约束（JSON schema / regex / choice / EBNF / structural tag）。
  编译与推进在 `structured_output/`，本文件只负责"参数长什么样、什么时候拒绝"。

### 明确**没有**接入的能力：收下参数就报错，不许静默忽略

上游这些字段本项目一个都没实现（三态矩阵里的"本项目尚未接入"）：`allowed_token_ids`（白名单）、
`bad_words`、`logit_bias`、`thinking_token_budget`（思考预算）、`logprob_token_ids`
（指定 token 的 logprobs）、`prompt_logprobs`。068 §3.6 的要求是"不能把缺口改成静默忽略参数"，
所以它们在这里**有字段但一律当场报错**——这样用户至少能看见"这个参数本引擎不认"，
而不是以为它生效了。
"""

from dataclasses import dataclass, field


@dataclass
class StructuredOutputsParams:
    """结构化输出的**请求级**参数（对应 vLLM `vllm/sampling_params.py` 的同名 dataclass）。

    八个字段里只有一个是"真正用哪种约束"，其余是后端的编译选项；一次只能指定一种约束
    （上游同款校验：多给或都不给都报错）。
    """

    json: str | dict | None = None
    regex: str | None = None
    choice: list[str] | None = None
    grammar: str | None = None
    json_object: bool | None = None
    structural_tag: str | None = None
    disable_any_whitespace: bool = False
    disable_additional_properties: bool = False

    #: 请求期校验挑出来的后端（上游 `_backend`：只由 Processor 写）。
    _backend: str | None = field(default=None, init=False)
    #: 这份 `_backend` 是不是 `auto` 自动挑的（上游同名字段：参数对象被复用时靠它区分
    #: "用户显式指定"与"上一轮自动挑的"）。
    _backend_was_auto: bool = field(default=False, init=False)

    def __post_init__(self):
        count = sum([
            self.json is not None,
            self.regex is not None,
            self.choice is not None,
            self.grammar is not None,
            self.json_object is not None,
            self.structural_tag is not None,
        ])
        if count > 1:
            raise ValueError(
                f"结构化输出一次只能给一种约束，收到 {count} 种：{self.__dict__}")
        if count < 1:
            raise ValueError(
                f"structured_outputs 给了、但里面一种约束都没有：{self.__dict__}")

    def all_constraints_none(self) -> bool:
        return all(getattr(self, name) is None for name in
                   ("json", "regex", "choice", "grammar", "json_object",
                    "structural_tag"))


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    # 68 关：min_p（上游 `SamplingParams.min_p`）。0 = 不筛。
    min_p: float = 0.0
    seed: int | None = None

    # 生成长度：至少 min_tokens、最多 max_tokens（`check_stop` 按已生成数判断）
    max_tokens: int = 16
    min_tokens: int = 0

    # 停止条件
    eos_token_id: int | None = None
    stop_token_ids: list[int] = field(default_factory=list)
    ignore_eos: bool = False

    # 惩罚项（57E 接采样时才会真正参与计算，这里先作为配置存在）
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    # 68 关：logprobs 要几个（None = 不要；-1 = 全词表）
    logprobs: int | None = None

    # 68 关：结构化输出（语法约束）。None = 不受约束。
    structured_outputs: StructuredOutputsParams | None = None

    # ---- 本项目尚未接入的能力：字段在、但设了就报错（068 §3.6）----
    allowed_token_ids: list[int] | None = None
    bad_words: list[str] | None = None
    logit_bias: dict[int, float] | None = None
    thinking_token_budget: int | None = None
    logprob_token_ids: list[int] | None = None
    prompt_logprobs: int | None = None

    def __post_init__(self):
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens 必须为正，收到 {self.max_tokens}")
        if self.min_tokens < 0:
            raise ValueError(f"min_tokens 不能为负，收到 {self.min_tokens}")
        if self.min_tokens > self.max_tokens:
            raise ValueError(f"min_tokens({self.min_tokens}) 不能大于 max_tokens({self.max_tokens})")
        if self.temperature < 0.0:
            raise ValueError(f"temperature 不能为负，收到 {self.temperature}")
        if self.top_k < -1:
            raise ValueError(f"top_k 只能是 -1（不筛）或非负整数，收到 {self.top_k}")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p 必须落在 (0, 1]，收到 {self.top_p}")
        if not 0.0 <= self.min_p <= 1.0:
            raise ValueError(f"min_p 必须落在 [0, 1]，收到 {self.min_p}")
        if self.repetition_penalty <= 0.0:
            raise ValueError(f"repetition_penalty 必须为正，收到 {self.repetition_penalty}")
        if self.logprobs is not None and self.logprobs != -1 and self.logprobs < 1:
            raise ValueError(
                f"logprobs 只能是 None（不要）、-1（全词表）或正整数（top-k + 选中那个），"
                f"收到 {self.logprobs}")

        # ---- 未接入能力的显式拒绝（每一条都点名"上游有、本项目没有"）----
        if self.allowed_token_ids is not None:
            raise NotImplementedError(
                "allowed_token_ids（白名单）本项目尚未接入：上游在 "
                "apply_logits_processors 里用 allowed_token_ids_mask 屏蔽词表，本仓库还没有"
                "这张掩码（068 §3.1/§3.6 的三态矩阵里属「本项目尚未接入」）。"
                "不静默忽略这个参数——请删掉它，或等它进实现")
        if self.bad_words:
            raise NotImplementedError(
                "bad_words 本项目尚未接入：上游走 apply_bad_words_with_drafts（投机时要按"
                "草稿前缀逐行判），本仓库没有这条路径（068 §3.1）")
        if self.logit_bias:
            raise NotImplementedError(
                "logit_bias 本项目尚未接入：上游用 LogitBiasLogitsProcessor，本仓库没有"
                "logits processor 插件框架（068 §3.1）")
        if self.thinking_token_budget is not None:
            raise NotImplementedError(
                "thinking_token_budget 本项目尚未接入：上游用 thinking_budget_state_holder "
                "在采样前改 logits（投机时还要按草稿展开），本仓库没有思考模式（068 §3.1）")
        if self.logprob_token_ids:
            raise NotImplementedError(
                "logprob_token_ids（只交付指定 token 的 logprobs）本项目尚未接入：上游用 "
                "Sampler.gather_specific_token_logprobs，本仓库只交付 top-k + 选中 token"
                "（068 §3.5 的三态矩阵）")
        if self.prompt_logprobs is not None:
            raise NotImplementedError(
                "prompt_logprobs 本项目尚未接入：上游要为 prefill 的每一行留 logprobs 并跨块"
                "累积，本仓库只做**生成**位置的 logprobs（068 §3.5）")

    @property
    def is_greedy(self) -> bool:
        """温度低于 1e-5 就算贪心（与采样侧的阈值一致，见 sample/metadata.py）。"""
        return self.temperature < 1e-5

    @property
    def all_stop_token_ids(self) -> set[int]:
        """这条请求的**全部**停止 token：eos（除非 ignore_eos）+ 显式 stop token。

        采样侧用它做 min_tokens 的屏蔽（"还没生成够就不许吐停止 token"），
        Scheduler 侧 `check_stop` 用它判断结束——同一个集合，两处用法不同（199 §2）。
        68 关结构化输出编译 grammar 时也传它（上游同款）：FSM 结束之后才允许吐这些 token。
        """
        stop_ids = set(self.stop_token_ids)
        if self.eos_token_id is not None and not self.ignore_eos:
            stop_ids.add(self.eos_token_id)
        return stop_ids

    @property
    def num_logprobs(self) -> int | None:
        """每个生成位置要交付几个 logprobs（None = 这条请求不要）。

        上游同名属性还要考虑 `logprob_token_ids` 的长度；本项目没接那个字段，
        所以就是 `logprobs` 本身。
        """
        return self.logprobs
