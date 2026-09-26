# step53：批量投机验证与抢占恢复

从 `step52/` 复制。`step52/` 原样保留，未改。

## 结论

把投机接进 continuous batching：**同一步里同时有不同 K 的投机请求、无草稿的普通
decode 和中间 prefill**，池子紧张时先缩草稿、不够才抢占，抢完还能恢复继续投机。

本关的核心是**行映射的两个坐标系**：

```text
原始输入行号   本轮全部输入 token 拼成一维后的下标（中间 prefill 也占位置）
筛选后偏移     模型返回的 logits 只含选中的行，行内下标从 0 开始
```

每个要采样的计划项都记下自己的 `num_sample_rows` 与 `sample_offset`，`_sample()`
整批一次 argmax、一次 `.tolist()`，再按区间切 Python 列表——不逐请求做 GPU 回传。

三条规则：

1. **提交顺序严格按 picked**。第五十二关「先 plain 后 drafts」的写法在批量下会打乱
   本轮事件顺序，改成一条循环走到底。
2. **真实 token 优先，草稿只吃余量**。每个 ready 请求先留 1 个真实 token，剩下的额度
   才给草稿和 prefill；预算不够所有 ready 时按 running 顺序保留靠前的（以前这里
   是一句 assert，把可配置的组合当成不可能发生的事）。
3. **先缩草稿、再抢占**。可选草稿不能成为额外抢占别人的理由；缩到 K=0 自然回到
   普通 1-token 路径，不需要另开分支。

顺带修了第五十二关验收记下的洞：`prompt_ids` 是迭代器时会被消费两次，入队后变成
空序列。`_check_request_ids()` 现在**返回**物化后的 list，建请求用它。

## 明确限制

`speculative_mode="ngram"` 仍只支持：`fcfs`、关前缀缓存、Torch attention、
关 CUDA Graph、请求贪心且无惩罚项。**`max_num_seqs` 不再受限**（本关放开的）。

## 怎么证明它是对的

| 检查 | 结果 |
|---|---|
| `benchmarks/check_step53_batch.py`：行映射 + 混批 + 预算/容量压力 + 恢复 | **35 项全通过** |
| `benchmarks/check_step53_engine.py`：多请求等价性 + 第五十二关整套用例 | **51 项全通过** |
| `benchmarks/diff_step52_step53.py`：投机关闭时与 step52 逐步对照 | **88 项全通过**（未放宽任何字段） |
| 随机压测 600 组多请求投机 | 0 崩溃 0 活锁，0 个 0-token 计划项，引用归零 |
| 随机对照 400 组（投机 vs 普通贪心逐 request_id） | **0 组不同**（其中 40 轮真提了草稿） |
| CUDA FP32 / BF16 多请求对照 | 逐 request_id 相同 |

复现：

```bash
python benchmarks/check_step53_batch.py
python benchmarks/check_step53_engine.py
python benchmarks/diff_step52_step53.py
```

完整说明（含一个 4 步例子的逐请求表格）见
[`docs/step53_batched_speculative.md`](../docs/step53_batched_speculative.md)。

**本关不承诺加速**：批量下每条请求每步输出数不同，跨请求完成顺序也会变——
只保证「每条请求自己的 token 序列不变」。
