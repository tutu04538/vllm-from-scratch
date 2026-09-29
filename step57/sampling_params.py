"""请求级采样参数（对应 vLLM 的 `vllm/sampling_params.py`，这里只留子集）。

三条约定：

1. **属于请求，不属于引擎**：同一个引擎里的两条请求可以有不同的 temperature / top_k / seed。
2. **不持有 `torch.Generator`**：参数是"可复制、可校验的配置"；随机流是请求的运行期状态，
   归执行侧（57E 的采样路径）管。step56 把 generator 塞在 `SamplingState` 里跟着请求走，
   那会让配置对象变成活对象，不能当快照传。
3. **`max_tokens` 是"要生成多少"，不是"总上下文多长"**——总长度上限在 `ModelConfig.max_model_len`。
   `check_stop` 两个都要看。
"""

from dataclasses import dataclass, field


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
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

    def __post_init__(self):
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens 必须为正，收到 {self.max_tokens}")
        if self.min_tokens < 0:
            raise ValueError(f"min_tokens 不能为负，收到 {self.min_tokens}")
        if self.min_tokens > self.max_tokens:
            raise ValueError(f"min_tokens({self.min_tokens}) 不能大于 max_tokens({self.max_tokens})")
        if self.temperature < 0.0:
            raise ValueError(f"temperature 不能为负，收到 {self.temperature}")

    @property
    def is_greedy(self) -> bool:
        """温度 0 即贪心。57A 不用它采样，但 `check_stop` 之外的调用方会问。"""
        return self.temperature == 0.0
