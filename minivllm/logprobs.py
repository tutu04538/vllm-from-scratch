"""用户可见的 logprobs 容器（对应 vLLM 顶层 `vllm/logprobs.py` 的 list 那一支）。

一条请求的 `logprobs` 是**逐生成位置**的一串字典：

    [ {token_id: Logprob(...), ...},   # 第 1 个生成位置：选中 token + top-k
      {token_id: Logprob(...), ...},   # 第 2 个生成位置 …
      ... ]

每个位置里那个**实际被采样的 token** 一定在（它在 `Sampler.gather_logprobs` 里被放在第 0 列），
这就是"只交付实际生成的 token"的含义（068 §3.5）：被拒绝的草稿、被截断的尾巴都不会出现在这里。

`rank` 是"这个 token 在这份分布里排第几"（1 = 概率最大）；`decoded_token` 是它解码出来的
字符串（需要 tokenizer，没有 tokenizer 时是 None）。上游把它做成 dataclass 是因为它要
被 OpenAI 兼容层序列化。

**与上游的差异**（docs/step68_alignment.md 记账）：上游还有一支 `FlatLogprobs`
（把四个字段摊平成平行 list，省对象数），由 `SamplingParams.flat_logprobs` 选择。
本仓库只实现 list 这一支（语义相同、代码少一半），也没有 `flat_logprobs` 这个开关——
没有开关就没有"收下参数却按另一种结构返回"这种静默差异。
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass
class Logprob:
    """一个候选 token 的 logprob 信息（上游同名 dataclass，字段逐个对应）。"""

    logprob: float
    rank: int | None = None
    decoded_token: str | None = None


#: 一个位置上"选中 token + top-k"的字典：`token_id -> Logprob`
LogprobsOnePosition = dict[int, Logprob]
#: 一条请求的生成侧 logprobs（逐位置）
SampleLogprobs = list[LogprobsOnePosition]


def create_sample_logprobs():
    """建一个空的生成侧 logprobs 容器（上游同名函数不接收 flat_logprobs 的那一支）。"""
    return []


def append_logprobs_for_next_position(request_logprobs: SampleLogprobs,
                                      token_ids: list[int], logprobs: list[float],
                                      decoded_tokens: Iterable[str | None], rank: int,
                                      num_logprobs: int) -> None:
    """把"一个生成位置"的 logprobs 追加进容器（上游同名函数的核心逻辑）。

    两个容易写错的点，都按上游来：

    1. **选中 token 在前**：`gather_logprobs` 把它的 logprob 放在第 0 列，所以这里
       `token_ids[0]` 就是实际生成的 token；`rank` 是它在整份分布里的名次。
    2. **top-k 的名次是 1..k**：`num_logprobs=-1` 表示"全词表都给"，此时名次依次排下去。
       重复的 token（选中那个同时也在 top-k 里）由 dict 覆盖，效果等于只留一份 —— 上游
       注释里专门说了"插两次与插一次相同"。
    """
    if num_logprobs == -1:
        num_logprobs = len(logprobs)
    topk_ranks = range(1, num_logprobs + 1)
    ranks = itertools.chain((rank,), topk_ranks)

    request_logprobs.append({
        token_id: Logprob(logprob=logprob, rank=entry_rank, decoded_token=token)
        for token_id, logprob, entry_rank, token in
        zip(token_ids, logprobs, ranks, decoded_tokens)
    })
