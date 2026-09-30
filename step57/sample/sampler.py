"""最小采样器：贪心 + 温度随机（对应 vLLM `v1/sample/sampler.py` 的极小化版本）。

**57B 的边界**：这里只够跑通"模型 → token"。惩罚项、top_k/top_p、种子派生、概率校验、
把温度编成 GPU 张量做一次批采样，都属于 57D（`test_sampler.py`）。所以：
配置里出现本关没实现的采样参数时**明确报错**，不静默忽略——静默忽略会得到一个"看起来
在按 top_p 采样、其实是纯多项分布"的结果。

**随机流跟请求走，不跟行号走**：随机源是请求自己的 `torch.Generator`（由 Runner 从
`CachedRequestState` 里按行取出来传进来，见 `gpu_model_runner._bookkeeping_sync`）。
`seed=None` 时用全局 RNG（与 vLLM 的 `SamplingType.RANDOM` 一致）：**不可复现**，除非调用方
自己先 `torch.manual_seed`。
"""

import torch


class Sampler:
    def sample(self, logits: torch.Tensor, sampling_params: list,
               generators: list[torch.Generator | None]) -> list[list[int]]:
        """`logits` 的第 i 行用 `sampling_params[i]` 采样。返回每行一个 token 的列表。

        逐行采样（不是一次批采样）是本关的简化：批采样要把 top_k/top_p 编成张量、用排序+累积
        概率做掩码，那是 57D。这里行数 = 本轮要采样的请求数，本身也不多。
        """
        if len(sampling_params) != logits.shape[0]:
            raise ValueError(f"采样参数有 {len(sampling_params)} 份，logits 有 "
                             f"{logits.shape[0]} 行：两者必须逐行对应")
        sampled: list[list[int]] = []
        for row, params in enumerate(sampling_params):
            self._check_supported(params)
            row_logits = logits[row]
            if params.temperature == 0.0:
                # 贪心：不加温度、不做随机。argmax 的并列取最小下标（torch 的约定）
                token = int(torch.argmax(row_logits))
            else:
                probs = torch.softmax(row_logits.float() / params.temperature, dim=-1)
                generator = generators[row] if generators[row] is not None else None
                token = int(torch.multinomial(probs, num_samples=1, generator=generator))
            sampled.append([token])
        return sampled

    @staticmethod
    def _check_supported(params) -> None:
        """本关支持的采样：`temperature`（0=贪心，>0=按温度随机）。其余明确报错。"""
        unsupported = []
        if params.top_k not in (-1, 0):
            unsupported.append(f"top_k={params.top_k}")
        if params.top_p != 1.0:
            unsupported.append(f"top_p={params.top_p}")
        if params.repetition_penalty != 1.0:
            unsupported.append(f"repetition_penalty={params.repetition_penalty}")
        if params.presence_penalty != 0.0:
            unsupported.append(f"presence_penalty={params.presence_penalty}")
        if params.frequency_penalty != 0.0:
            unsupported.append(f"frequency_penalty={params.frequency_penalty}")
        if unsupported:
            raise NotImplementedError(
                f"57B 的最小采样器不支持 {unsupported}；惩罚项与 top_k/top_p 属 57D"
                f"（见 sample/sampler.py 的模块说明）")
