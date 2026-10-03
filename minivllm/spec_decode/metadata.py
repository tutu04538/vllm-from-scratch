"""`SpecDecodeMetadata`：一轮投机验证要用的**全部索引**（对应 vLLM `v1/spec_decode/metadata.py`）。

字段与上游逐个对应（59 关把旧版多带的 `req_ids` 去掉了：请求 → 行的对应关系是 **Runner** 的
事，元数据只描述"扁平的 logits 行怎么切"，见 059 §2）。

一轮的形状（K = 该请求本轮采用的草稿数）：

```text
请求的 query 行（本轮 forward 真正算的行）：
    [ b ][ d1 ][ d2 ] ... [ dK ]        K+1 行
      │     │      │         │
      │     └──────┴─────────┴─→ 草稿 token（它们**是输入**，见下）
      └─ b = 上一个已提交但**还没算过 KV** 的 token（就是普通 decode 里那一行）
```

为什么是 K+1 行而不是 K 行：验证 d1 要用"b 位置上的 logits"，验证 d2 要用"d1 位置上的 logits"……
最后一个位置 dK 的 logits 用来采 **bonus token**（全接受时白送一个）。于是：

    logits_indices          从 forward 的 hidden 里要取的那 K+1 行（每请求一段连续行）
    target_logits_indices   取完之后的前 K 行（验证草稿用）
    bonus_logits_indices    取完之后的第 K 行（采 bonus）

**两个坐标系**（这是本关最容易搞错的地方）：

    logits_indices          指向**扁平化的 forward 行**（P+B 个）
    target/bonus indices    指向**取完之后的那个 [P+B, V] logits 张量**

所以 `target_logits_indices` 的前缀和用的是 `cu_num_sampled_tokens`（每请求占 K+1 行），
不是 forward 的行号。上游 `_calc_spec_decode_metadata` 的算例（Runner 里逐行对应）：

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

**草稿 token 的来源**：上游从 GPU 输入缓冲里取
（`draft_token_ids = input_ids.gpu[logits_indices][target_logits_indices + 1]`——草稿本来就是
输入行）。本机还没有常驻的 GPU 输入缓冲（那是 69 关 CUDA Graph 的事），所以在 Runner 里对
**同一份 CPU 输入数组**做同样的索引（`_calc_spec_decode_metadata`），值逐行一致。

**索引的 dtype/设备**（059 §2）：`cu_num_draft_tokens` 是 GPU 上的 **int32**、且**不带开头的 0**
（`[3,3,5,5,6]` 而不是 `[0,3,3,5,5,6]`）；`cu_num_sampled_tokens`、`logits_indices`、
`target_logits_indices`、`bonus_logits_indices` 同理；`draft_token_ids` 是 `[num_tokens]` int32。
内核按"前一行的累积值"取区间，多一个开头的 0 会让所有区间整体错位。

**上游有、本关没有的**：`make_dummy()`（profiling/哑输入用，69 关 CUDA Graph 才需要）。
"""

from dataclasses import dataclass

import torch


@dataclass
class SpecDecodeMetadata:
    # [num_tokens]       扁平的草稿 token（GPU int32）
    draft_token_ids: torch.Tensor
    # [batch_size]       每条请求采用了几枚草稿（K_i）
    num_draft_tokens: list[int]
    # [batch_size]       K 的末端累积和（**不带开头的 0**）
    cu_num_draft_tokens: torch.Tensor
    # [batch_size]       K_i + 1 的末端累积和（query 行数）
    cu_num_sampled_tokens: torch.Tensor
    # [num_tokens]       验证行（指向取完之后的紧凑 logits 张量）
    target_logits_indices: torch.Tensor
    # [batch_size]       bonus 行（同上）
    bonus_logits_indices: torch.Tensor
    # [num_tokens + batch_size]  forward 里要取 logits 的行
    logits_indices: torch.Tensor

    def __post_init__(self):
        # 上游逐字一致：`num_draft_tokens` 里至少要有一条请求（空批不该走到验证）
        self.max_spec_len = max(self.num_draft_tokens)
