"""57E 验收（对应需求里的 `test_spec_metadata.py`）：投机验证的索引数学。

基准算例（照抄 vLLM `_calc_spec_decode_metadata` 的注释）：

```text
cu_num_scheduled_tokens: [  4, 104, 107, 207, 209]
num_draft_tokens:        [  3,   0,   2,   0,   1]
cu_num_draft_tokens:     [  3,   3,   5,   5,   6]
cu_num_sampled_tokens:   [  4,   5,   8,   9,  11]
logits_indices:          [0,1,2,3, 103, 104,105,106, 206, 207,208]
target_logits_indices:   [0,1,2, 5,6, 9]
bonus_logits_indices:    [3, 4, 7,8, 10]
```

两个坐标系是重点：`logits_indices` 指 forward 的行，`target/bonus_logits_indices` 指**取完之后**
的紧凑 logits 张量。

**59 关的改动**（059 §2/§3.1）：构造从 `SpecDecodeMetadata.from_scheduled` 搬到了 Runner 的
`_calc_spec_decode_metadata`（上游就在这里），字段去掉 `req_ids`（请求 → 行是 Runner 的事），
`cu_num_*` 变成 GPU int32。所以本脚本改成：

  - 用一份"只带索引算法所需属性"的 Runner 替身调那个方法（不建模型、不碰 KV）；
  - 输入行按上游 `[b][d1..dK]` 布局摆，草稿由 `input_ids[logits_indices][target+1]` 取；
  - 原来"请求的行数缺失 → KeyError"那条换成新 API 的等价失败模式：**输入行与协议不一致**。

期望值一个字没改。
"""

import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, "/home/user/proj/vllm-from-scratch")

from minivllm.spec_decode.metadata import SpecDecodeMetadata   # noqa: F401（导入即验证模块在）
from minivllm.worker.gpu_model_runner import GPUModelRunner

FAIL = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def first_line(error):
    return error.splitlines()[0] if error else "没有报错"


def stub_runner(req_ids):
    """只带 `_calc_spec_decode_metadata` 需要的属性（device / arange 缓冲 / 批的行序）。"""
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.device = "cuda" if torch.cuda.is_available() else "cpu"
    runner._arange_np = np.arange(4096, dtype=np.int64)
    runner._arange_scratch = np.empty(4096, dtype=np.int64)
    runner.input_batch = SimpleNamespace(
        req_ids=list(req_ids),
        req_id_to_index={req_id: index for index, req_id in enumerate(req_ids)})
    return runner


def input_rows(blocks, cu_scheduled):
    """`blocks[i] = (b_token, [drafts...])`，按块尾对齐摆进扁平输入行。"""
    flat = [0] * int(cu_scheduled[-1])
    for (backup, drafts), end in zip(blocks, cu_scheduled):
        end = int(end)
        flat[end - len(drafts) - 1:end] = [backup] + list(drafts)
    return torch.tensor(flat, dtype=torch.int64)


def make(blocks, num_scheduled, req_ids, protocol=None):
    runner = stub_runner(req_ids)
    cu_scheduled = np.cumsum(np.array(num_scheduled, dtype=np.int32))
    drafts = {req_id: list(draft) for req_id, (_, draft) in zip(req_ids, blocks) if draft}
    return runner._calc_spec_decode_metadata(
        np.array([len(draft) for _, draft in blocks], dtype=np.int32), cu_scheduled,
        input_rows(blocks, cu_scheduled), drafts if protocol is None else protocol)


def batch_size(meta):
    return len(meta.num_draft_tokens)


# ------------------------------------------------ 1. vLLM 的算例逐值对照

req_ids = ["r0", "r1", "r2", "r3", "r4"]
blocks = [(100, [10, 11, 12]), (200, []), (101, [20, 21]), (201, []), (102, [30])]
num_scheduled = [4, 100, 3, 100, 2]
meta = make(blocks, num_scheduled, req_ids)

check("1. num_draft_tokens 与 cu_num_draft_tokens（末端累积和，不带开头的 0）",
      meta.num_draft_tokens == [3, 0, 2, 0, 1]
      and meta.cu_num_draft_tokens.tolist() == [3, 3, 5, 5, 6],
      f"{meta.num_draft_tokens} / {meta.cu_num_draft_tokens.tolist()}")
check("1. cu_num_sampled_tokens = 每项 K_i + 1 的累积和",
      meta.cu_num_sampled_tokens.tolist() == [4, 5, 8, 9, 11],
      str(meta.cu_num_sampled_tokens.tolist()))
check("1. logits_indices：每请求取它 query 块**最后 K+1 行**（forward 行号）",
      meta.logits_indices.tolist() == [0, 1, 2, 3, 103, 104, 105, 106, 206, 207, 208],
      str(meta.logits_indices.tolist()))
check("1. target_logits_indices：验证行（指向**取完之后**的紧凑 logits）",
      meta.target_logits_indices.tolist() == [0, 1, 2, 5, 6, 9],
      str(meta.target_logits_indices.tolist()))
check("1. bonus_logits_indices：每请求一行（K=0 的请求，它那唯一一行就是 bonus 行）",
      meta.bonus_logits_indices.tolist() == [3, 4, 7, 8, 10],
      str(meta.bonus_logits_indices.tolist()))
check("1. max_spec_len = max(K)，P = sum(K)",
      meta.max_spec_len == 3 and meta.draft_token_ids.shape[0] == 6,
      f"max_spec_len={meta.max_spec_len}、P={meta.draft_token_ids.shape[0]}")
check("1. 两个坐标系的长度关系：target+bonus 恰好覆盖 [0, P+B)",
      sorted(meta.target_logits_indices.tolist() + meta.bonus_logits_indices.tolist())
      == list(range(meta.draft_token_ids.shape[0] + batch_size(meta))),
      f"target={meta.target_logits_indices.tolist()}、bonus={meta.bonus_logits_indices.tolist()}")
check("1. 草稿 token 来自输入行的下一行（上游 `input_ids[logits_indices][target+1]`）",
      meta.draft_token_ids.tolist() == [10, 11, 12, 20, 21, 30],
      str(meta.draft_token_ids.tolist()))

# ------------------------------------------------ 2. 退化与边界

only_k0 = make([(1, []), (2, [])], [1, 1], ["a", "b"])
check("2. 全批 K=0：退化成普通解码（每请求一行，那行既是 forward 行也是 bonus 行）",
      only_k0.logits_indices.tolist() == [0, 1]
      and only_k0.target_logits_indices.tolist() == []
      and only_k0.bonus_logits_indices.tolist() == [0, 1]
      and only_k0.max_spec_len == 0,
      f"logits={only_k0.logits_indices.tolist()}、bonus={only_k0.bonus_logits_indices.tolist()}")

ragged = make([(7, [7, 8, 9]), (8, [1])], [4, 2], ["a", "b"])
check("2. ragged（K=[3,1]）：行数与索引都对得上",
      ragged.logits_indices.tolist() == [0, 1, 2, 3, 4, 5]
      and ragged.target_logits_indices.tolist() == [0, 1, 2, 4]
      and ragged.bonus_logits_indices.tolist() == [3, 5],
      f"logits={ragged.logits_indices.tolist()}、target={ragged.target_logits_indices.tolist()}")

prefix = make([(7, [7])], [2], ["a"])       # 预算只够 1 枚草稿（K=1，排 2 行）
check("2. 草稿被预算截短（只剩 1 枚、排 2 行）：K+1 == 行数，索引自洽",
      prefix.logits_indices.tolist() == [0, 1]
      and prefix.target_logits_indices.tolist() == [0]
      and prefix.bonus_logits_indices.tolist() == [1],
      f"logits={prefix.logits_indices.tolist()}")

# ------------------------------------------------ 3. 不合法输入要被挡住

try:
    make([(7, [7, 8])], [5], ["a"])
    error = None
except ValueError as exc:
    error = str(exc)
check("3. 草稿数与排的行数对不上（K=2 却排了 5 行）→ 报错，不静默算错索引",
      error is not None and "恰好是 K+1 行" in error, first_line(error))

try:
    make([(7, [7, 8])], [3], ["a"], protocol={"a": [7, 9]})     # 输入行里是 8，协议里是 9
    error = None
except RuntimeError as exc:
    error = str(exc)
check("3. 输入行与调度协议不一致 → 报错（草稿取错来源会静默验证别的 token）",
      error is not None and "scheduled_spec_decode_tokens" in error, first_line(error))

print()
print(f"{'全部通过' if not FAIL else '失败: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
