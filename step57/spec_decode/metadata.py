"""`SpecDecodeMetadata`：投机验证要用的**全部索引**（对应 vLLM `v1/spec_decode/metadata.py`
与 runner 里的 `_calc_spec_decode_metadata`）。

投机一轮的形状（K = 本轮采用的草稿数）：

```text
请求的 query 行（本轮 forward 真正算的行）：
    [ b ][ d1 ][ d2 ] ... [ dK ]        K+1 行
      │     │      │         │
      │     └──────┴─────────┴─→ 草稿 token（它们**是输入**，所以验证时要的元素就在输入缓冲里）
      └─ b = 上一个已提交但**还没算过 KV** 的 token（就是普通 decode 里那一行）
```

为什么是 K+1 行而不是 K 行：验证 d1 需要"b 位置上的 logits"，验证 d2 需要"d1 位置上的 logits"……
最后一个位置 dK 的 logits 用来采 **bonus token**（全接受时白送一个）。所以：

    target_logits_indices   前 K 行的 logits（验证草稿用）
    bonus_logits_indices    第 K 行的 logits（采 bonus）
    logits_indices          两者合起来，去 forward 的 hidden 里取行

**两个坐标系**（这是本关最容易搞错的地方）：

    logits_indices          指向**扁平化的 forward 行**（P+B 个）
    target/bonus indices    指向**取完之后的那个 [P+B, V] logits 张量**

所以 `target_logits_indices` 的前缀和用的是 `cu_num_sampled_tokens`（每个请求占 K+1 行），
不是 forward 的行号。一份具体的算例（K=[3,0,2,0,1]，直接照抄 vLLM 的注释）：

```text
cu_num_scheduled_tokens: [  4, 104, 107, 207, 209]
num_draft_tokens:        [  3,   0,   2,   0,   1]
cu_num_draft_tokens:     [  3,   3,   5,   5,   6]
cu_num_sampled_tokens:   [  4,   5,   8,   9,  11]
logits_indices:          [0,1,2,3, 103, 104,105,106, 206, 207,208]
target_logits_indices:   [0,1,2, 5,6, 9]
bonus_logits_indices:    [3, 4, 7,8, 10]
```

K=0 的请求（第 2、4 条）：query 只有 1 行（就是 b），它的 logits 直接当 bonus 行用——
**普通 decode 就是这个特例**，所以投机打开时解码路径不需要另写一套。

**本关的差异**：vLLM 从 GPU 输入缓冲里取草稿 token（`input_ids[logits_indices][target+1]`，因为
草稿本来就是输入）；本关直接从 Scheduler 发下来的 `scheduled_spec_decode_tokens` 里取，少一次
依赖执行顺序的取数。

本关不做：`num_lookahead_tokens`（EAGLE/MTP 那种"提议者就是 target 模型自己"的预留）、
padded 输出矩阵（vLLM 的 `SamplerOutput` 会 padding 到 `max_spec_len+1`，本关返回 ragged 的行）。
"""

from dataclasses import dataclass, field

import torch


def _repeat_arange(bases: list[int], counts: list[int]) -> torch.Tensor:
    """`repeat(bases, counts) + arange`：把"每个请求一段连续下标"摊平。

    例：bases=[0,103,104], counts=[4,1,3] → [0,1,2,3, 103, 104,105,106]
    """
    indices: list[int] = []
    for base, count in zip(bases, counts):
        indices.extend(range(base, base + count))
    return torch.tensor(indices, dtype=torch.int64)


@dataclass
class SpecDecodeMetadata:
    draft_token_ids: torch.Tensor          # [P]   扁平的草稿 token
    num_draft_tokens: list[int]            # [B]   每条请求采用了几枚草稿（K_i）
    cu_num_draft_tokens: torch.Tensor      # [B]   K 的末端累积和（**不带开头的 0**）
    cu_num_sampled_tokens: torch.Tensor    # [B]   K_i + 1 的末端累积和
    target_logits_indices: torch.Tensor    # [P]   验证行（指向取完的 logits 张量）
    bonus_logits_indices: torch.Tensor     # [B]   bonus 行（同上）
    logits_indices: torch.Tensor           # [P+B] forward 里要取 logits 的行
    # 行序对应的请求 ID（vLLM 不带这个字段——它按批行号对齐；本关要按请求对齐 q，
    # 因为"上一轮提的草稿"是按请求存的，见 runner._align_draft_probs）
    req_ids: list[str] = field(default_factory=list)
    max_spec_len: int = field(init=False)

    def __post_init__(self) -> None:
        self.max_spec_len = max(self.num_draft_tokens) if self.num_draft_tokens else 0

    @property
    def num_draft_tokens_total(self) -> int:
        return int(self.draft_token_ids.shape[0])

    @property
    def batch_size(self) -> int:
        return len(self.num_draft_tokens)

    @classmethod
    def from_scheduled(cls, scheduled_spec_decode_tokens: dict[str, list[int]],
                       num_scheduled_tokens: dict[str, int],
                       req_ids: list[str]) -> "SpecDecodeMetadata":
        """按**批的行序**（`req_ids`）打包。没有草稿的请求也要出现（K=0），
        否则 `bonus_logits_indices` 的下标会和转发行的位置对不上。
        """
        num_draft_tokens = [len(scheduled_spec_decode_tokens.get(req_id, []))
                            for req_id in req_ids]
        num_scheduled = [num_scheduled_tokens[req_id] for req_id in req_ids]
        for req_id, num_draft, num_sched in zip(req_ids, num_draft_tokens, num_scheduled):
            # 两种合法形状：
            #   K = 0（普通 decode / 中间 prefill 块）→ 排多少都行，取**最后一行**的 logits
            #   K > 0（本轮采用草稿）→ 必须恰好 K+1 行：b 一行 + K 枚草稿
            # 第二种为什么必须恰好：草稿是 query 的**尾部**，多出来的行说明调度侧把它当
            # prefill 块排了（那是不该发生的：prefill 块的草稿要被丢掉，见 Scheduler）
            if num_draft > 0 and num_sched != num_draft + 1:
                raise ValueError(
                    f"{req_id!r} 本轮排了 {num_sched} 个 token，却带了 {num_draft} 枚草稿："
                    f"带草稿的请求 query 必须恰好是 K+1 行（b + K 枚草稿）。"
                    f"不一致说明调度侧算错了草稿的采用数")

        num_sampled_tokens = [count + 1 for count in num_draft_tokens]
        cu_num_sampled = [sum(num_sampled_tokens[:index + 1])
                          for index in range(len(req_ids))]
        cu_num_draft = [sum(num_draft_tokens[:index + 1])
                        for index in range(len(req_ids))]
        cu_num_scheduled = [sum(num_scheduled[:index + 1])
                            for index in range(len(req_ids))]

        # logits 行：每请求取它 query 块的**最后 num_sampled 行**（K+1 行）
        logits_bases = [cu_num_scheduled[index] - num_sampled_tokens[index]
                        for index in range(len(req_ids))]
        logits_indices = _repeat_arange(logits_bases, num_sampled_tokens)
        # target/bonus 行：取完之后的紧凑空间。前 K 行验证草稿，最后一行的 logits 采 bonus
        compact_bases = [0] + cu_num_sampled[:-1]
        target_logits_indices = _repeat_arange(compact_bases, num_draft_tokens)
        bonus_logits_indices = torch.tensor(
            [cu_num_sampled[index] - 1 for index in range(len(req_ids))], dtype=torch.int64)

        draft_token_ids: list[int] = []
        for req_id in req_ids:
            draft_token_ids.extend(scheduled_spec_decode_tokens.get(req_id, []))

        return cls(
            draft_token_ids=torch.tensor(draft_token_ids, dtype=torch.int64),
            num_draft_tokens=num_draft_tokens,
            cu_num_draft_tokens=torch.tensor(cu_num_draft, dtype=torch.int64),
            cu_num_sampled_tokens=torch.tensor(cu_num_sampled, dtype=torch.int64),
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            logits_indices=logits_indices,
            req_ids=list(req_ids))

    def req_draft_slice(self, index: int) -> tuple[int, int]:
        """第 `index` 条请求的草稿在扁平草稿里的区间 `[start, end)`。"""
        start = int(self.cu_num_draft_tokens[index]) - self.num_draft_tokens[index]
        return start, int(self.cu_num_draft_tokens[index])
