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
        if self.top_k < -1:
            raise ValueError(f"top_k 只能是 -1（不筛）或非负整数，收到 {self.top_k}")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p 必须落在 (0, 1]，收到 {self.top_p}")
        if self.repetition_penalty <= 0.0:
            raise ValueError(f"repetition_penalty 必须为正，收到 {self.repetition_penalty}")

    @property
    def is_greedy(self) -> bool:
        """温度低于 1e-5 就算贪心（与采样侧的阈值一致，见 sample/metadata.py）。"""
        return self.temperature < 1e-5

    @property
    def all_stop_token_ids(self) -> set[int]:
        """这条请求的**全部**停止 token：eos（除非 ignore_eos）+ 显式 stop token。

        采样侧用它做 min_tokens 的屏蔽（"还没生成够就不许吐停止 token"），
        Scheduler 侧 `check_stop` 用它判断结束——同一个集合，两处用法不同（199 §2）。
        """
        stop_ids = set(self.stop_token_ids)
        if self.eos_token_id is not None and not self.ignore_eos:
            stop_ids.add(self.eos_token_id)
        return stop_ids
