"""`SamplingMetadata`：**按行组织**的采样参数（对应 vLLM `v1/sample/metadata.py`）。

为什么要有这么一层：`Sampler` 只认"logits 的第 i 行"和"第 i 行的采样配置"，它**不接 Request**。
把请求的采样参数、历史、随机源翻译成按行对齐的东西，是 Runner 的事。这一层就是那个翻译结果。

三条约定：

1. **行对齐 compact 行号**：`logits` 的第 i 行 ↔ 元数据的第 i 项 ↔ `rows[i]` 那个 batch 行。
   本关只对**要采样的行**建元数据（未 ready 的请求连 logits 都没有），所以元数据里的行号是
   紧凑下标，不是 batch 行号——映射在 Runner 里显式重建（`sample_rows`）。
2. **能省则省**：全是贪心时 `temperature=None`（采样器连温度都不除，避免除以 0）；
   没人要 top-k/top-p/惩罚时就传 `None` / `no_penalties=True`。这与 vLLM 一致
   （它就是为了不把没用的东西拷上 GPU）。
3. **`no_penalties` 是"整批都没人用惩罚"**，不是"这一行不用"。逐行的惩罚值仍然是张量
   （不用惩罚的行是 1.0/0.0 这种无操作值），这样惩罚算子可以一次批处理。

**本关省略的字段**（vLLM 有、57D 不需要）：logprobs 相关（`max_num_logprobs`、
`logprob_token_ids`）、`allowed_token_ids_mask`（白名单）、`bad_words_token_ids`、
`spec_token_ids`（57E）、`logitsprocs` 插件框架、thinking budget。
"""

from dataclasses import dataclass, field

import torch

# 温度低于它就算贪心（vLLM 同款阈值：0 与 1e-5 之间的温度没有实际意义，
# 但直接拿它做除数会放大数值噪声）
SAMPLING_EPS = 1e-5


@dataclass
class SamplingMetadata:
    # 全是贪心时是 None（采样器不做温度缩放）；否则是 [num_rows] 的 fp32
    temperature: torch.Tensor | None
    all_greedy: bool
    all_random: bool
    top_k: torch.Tensor | None                 # [num_rows] int64；None = 不筛
    top_p: torch.Tensor | None                 # [num_rows] fp32；None = 不筛
    # 紧凑行号 → 该请求自己的 generator（没设 seed 的行不在字典里 → 用全局 RNG）
    generators: dict[int, torch.Generator] = field(default_factory=dict)
    no_penalties: bool = True
    # 惩罚的"历史"：prompt 与**已提交**的 output。逐行一个 list
    # （不共享请求对象：元数据里只有采样要用的部分）
    prompt_token_ids: list[list[int]] = field(default_factory=list)
    output_token_ids: list[list[int]] = field(default_factory=list)
    presence_penalties: torch.Tensor | None = None
    frequency_penalties: torch.Tensor | None = None
    repetition_penalties: torch.Tensor | None = None
    # min_tokens：还没生成够的行，要把它的停止 token 屏蔽掉（"暂不允许采到什么"）
    min_tokens: list[int] = field(default_factory=list)
    stop_token_ids: list[list[int]] = field(default_factory=list)

    @property
    def num_rows(self) -> int:
        if self.temperature is not None:
            return int(self.temperature.shape[0])
        for tensor in (self.top_k, self.top_p, self.presence_penalties):
            if tensor is not None:
                return int(tensor.shape[0])
        return len(self.min_tokens)

    @classmethod
    def from_input_batch(cls, input_batch, rows: list[int],
                         device=None) -> "SamplingMetadata":
        """把 batch 的若干行翻译成采样元数据。`rows` 是 batch 行号，顺序就是 logits 的行序。

        `device` 必须与 **logits 所在设备**一致（采样器要把温度/惩罚直接作用在 logits 上，
        两个设备会当场报错）。Runner 传自己的设备；缺省按输入缓冲的设备（CPU 用例）。

        取"整批都没人用某功能"的判断是**逐行扫一遍**（vLLM 在增删请求时维护
        `all_greedy`/`no_top_p` 这类增量标志位）。本关批量小，每轮扫一遍更简单，也不会漏。
        """
        params = [input_batch.sampling_params[row] for row in rows]
        if not params:
            return cls(temperature=torch.empty(0), all_greedy=True, all_random=False)

        all_greedy = all(parameter.temperature < SAMPLING_EPS for parameter in params)
        all_random = all(parameter.temperature >= SAMPLING_EPS for parameter in params)
        no_top_k = all(parameter.top_k in (-1, 0) for parameter in params)
        no_top_p = all(parameter.top_p >= 1.0 for parameter in params)
        no_penalties = all(parameter.repetition_penalty == 1.0
                           and parameter.presence_penalty == 0.0
                           and parameter.frequency_penalty == 0.0 for parameter in params)

        if device is None:
            device = input_batch.temperature_cpu.device
        rows_tensor = torch.tensor(rows, dtype=torch.int64)
        return cls(
            temperature=(None if all_greedy
                         else input_batch.temperature_cpu[rows_tensor].to(device, torch.float32)),
            all_greedy=all_greedy,
            all_random=all_random,
            top_k=None if no_top_k else input_batch.top_k_cpu[rows_tensor].to(device),
            top_p=None if no_top_p else input_batch.top_p_cpu[rows_tensor].to(device, torch.float32),
            generators={index: input_batch.generators[row]
                        for index, row in enumerate(rows)
                        if input_batch.generators[row] is not None},
            no_penalties=no_penalties,
            prompt_token_ids=[input_batch.prompt_token_ids(row) for row in rows],
            # 与请求镜像**共享同一个 list**：采样前看到的是"已提交"的历史
            output_token_ids=[input_batch.req_output_token_ids[row] for row in rows],
            presence_penalties=None if no_penalties
            else input_batch.presence_penalties_cpu[rows_tensor].to(device, torch.float32),
            frequency_penalties=None if no_penalties
            else input_batch.frequency_penalties_cpu[rows_tensor].to(device, torch.float32),
            repetition_penalties=None if no_penalties
            else input_batch.repetition_penalties_cpu[rows_tensor].to(device, torch.float32),
            min_tokens=[parameter.min_tokens for parameter in params],
            stop_token_ids=[sorted(parameter.all_stop_token_ids) for parameter in params],
        )
