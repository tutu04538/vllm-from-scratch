"""57E 验收（对应需求里的 `test_spec_metadata.py`）：投机验证的索引数学。

`SpecDecodeMetadata` 把"这一轮 forward 的哪些行要 logits、验证行是哪几行、bonus 行是哪几行"
算清楚。这一步错了症状很隐蔽：采样结果会**配错请求**（把 A 的 logits 当成 B 的验证行），
所以用例直接对着 vLLM 源码注释里的算例逐值比。

基准算例（照抄 `_calc_spec_decode_metadata` 的注释）：

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
"""

import sys

sys.path.insert(0, "/home/user/proj/vllm-from-scratch")

from step57.spec_decode.metadata import SpecDecodeMetadata

FAIL = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def first_line(error):
    return error.splitlines()[0] if error else "没有报错"


def make(drafts: dict, num_scheduled: dict, req_ids: list):
    return SpecDecodeMetadata.from_scheduled(drafts, num_scheduled, req_ids)


# ------------------------------------------------ 1. vLLM 的算例逐值对照

req_ids = ["r0", "r1", "r2", "r3", "r4"]
drafts = {"r0": [10, 11, 12], "r2": [20, 21], "r4": [30]}
num_scheduled = {"r0": 4, "r1": 100, "r2": 3, "r3": 100, "r4": 2}
meta = make(drafts, num_scheduled, req_ids)

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
      meta.max_spec_len == 3 and meta.num_draft_tokens_total == 6,
      f"max_spec_len={meta.max_spec_len}、P={meta.num_draft_tokens_total}")
check("1. 两个坐标系的长度关系：target+bonsus 恰好覆盖 [0, P+B)",
      sorted(meta.target_logits_indices.tolist() + meta.bonus_logits_indices.tolist())
      == list(range(meta.num_draft_tokens_total + meta.batch_size)),
      f"target={meta.target_logits_indices.tolist()}、bonus={meta.bonus_logits_indices.tolist()}")
check("1. 草稿 token 按请求块摊平（r2 的两枚在 [3,5)）",
      meta.req_draft_slice(2) == (3, 5)
      and meta.draft_token_ids[3:5].tolist() == [20, 21],
      f"r2 区间={meta.req_draft_slice(2)}、草稿={meta.draft_token_ids.tolist()}")

# ------------------------------------------------ 2. 退化与边界

only_k0 = make({}, {"a": 1, "b": 1}, ["a", "b"])
check("2. 全批 K=0：退化成普通解码（每请求一行，那行既是 forward 行也是 bonus 行）",
      only_k0.logits_indices.tolist() == [0, 1]
      and only_k0.target_logits_indices.tolist() == []
      and only_k0.bonus_logits_indices.tolist() == [0, 1]
      and only_k0.max_spec_len == 0,
      f"logits={only_k0.logits_indices.tolist()}、bonus={only_k0.bonus_logits_indices.tolist()}")

ragged = make({"a": [7, 8, 9], "b": [1]}, {"a": 4, "b": 2}, ["a", "b"])
check("2. ragged（K=[3,1]）：行数与索引都对得上",
      ragged.logits_indices.tolist() == [0, 1, 2, 3, 4, 5]
      and ragged.target_logits_indices.tolist() == [0, 1, 2, 4]
      and ragged.bonus_logits_indices.tolist() == [3, 5],
      f"logits={ragged.logits_indices.tolist()}、target={ragged.target_logits_indices.tolist()}")

prefix = make({"a": [7]}, {"a": 2}, ["a"])       # 预算只够 1 枚草稿（K=1，排 2 行）
check("2. 草稿被预算截短（只剩 1 枚、排 2 行）：K+1 == 行数，索引自洽",
      prefix.logits_indices.tolist() == [0, 1]
      and prefix.target_logits_indices.tolist() == [0]
      and prefix.bonus_logits_indices.tolist() == [1],
      f"logits={prefix.logits_indices.tolist()}")

# ------------------------------------------------ 3. 不合法输入要被挡住

try:
    make({"a": [7, 8]}, {"a": 5}, ["a"])
    error = None
except ValueError as exc:
    error = str(exc)
check("3. 草稿数与排的行数对不上（K=2 却排了 5 行）→ 报错，不静默算错索引",
      error is not None and "恰好是 K+1 行" in error, first_line(error))

try:
    make({"a": [7, 8], "b": []}, {"a": 3}, ["a", "b"])          # b 的行数没给
    error = None
except KeyError as exc:
    error = str(exc)
check("3. 请求的行数缺失 → 报错（批的行序必须完整，否则 bonus 下标会整体错位）",
      error is not None and "b" in error, first_line(error))

print()
print(f"{'全部通过' if not FAIL else '失败: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
