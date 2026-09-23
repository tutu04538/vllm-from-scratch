# step44：重计算式抢占与恢复（第一阶段）

从 `step43/` 复制。`step43/` 原样保留，未改。

## 结论

新增一个 keyword-only 开关，把「准入时按最坏情况承诺未来所有块」换成**允许容量超卖**：
准入只判断这条请求单独跑是否可行，块在真正排到 token 时才补；补不到就从 `running` 尾部
选犠牲者、释放它的 KV，被抢占者稍后重算历史再继续。

```python
Engine(..., *, on_token=None, preemption_mode=None)    # 默认，第四十三关行为
Engine(..., preemption_mode="recompute")               # 本关新路径
```

`"recompute"` 要求显式 `enable_prefix_caching=False`，两个一起开在构造阶段报 `ValueError`。
新参数一律加在 `*` 之后的所有旧参数之后——第四十三关就是因为把它插在旧参数中间导致位置调用错位。

## 最核心的一处：请求历史要重新定义

```python
all_token_ids        = prompt_ids + output_ids
num_uncomputed_tokens = len(all_token_ids) - cache.length
```

`cache.length`（已进模型并写入 KV 的 token 数）是「已计算长度」的唯一真相。旧代码的
`max(len(prompt_ids) - cache.length, 0)` 里那个 `max(..., 0)` 恰好把「刚采样、还没进模型」的
最后一个 output token 藏了起来——只重算 prompt 的话，恢复时会在 prompt 末尾又采样一次。

中间重算 chunk **不采样、不重放 `on_token`**；只有本轮算到当前 `all_token_ids` 末尾才采样。
`output_ids`、惩罚计数、每请求的随机 `Generator` 都保留，重算不消耗随机数。

## 实测

CPU / FP32 确定性小模型，每请求输出与「独占引擎运行」的参考**逐 token 完全一致**：

| 场景 | 抢占次数 | 重算 token 数 | 结果 |
|---|---:|---:|---|
| 容量充足（3 条，64 块） | 0 | 0 | 与承诺式基线逐 token 相同 |
| 4 块池子跑两条各 8 输出 | 1 | 7 | B 被抢占一次、重放 7 个 token 后追平 |
| 5 块池子跑 4 条（连续释放尾部犠牲者） | 2 | 16 | 4 条全部完成且逐条一致 |

A–E 五条请求一起到达时（`max_num_seqs=3`、4 块池子），首次抢占后 `waiting` 仍是 `[B, C, D, E]`，
完成顺序 `A, B, C, D, E`——被抢占者不会排到比它更晚到达的请求后面。

固定 seed 随机采样、惩罚计数在抢占 + 重算后同样与独占运行一致；
`on_token` 拼接等于最终 `output_ids`，旧 token 不重复通知；结束后 running/waiting 均空、
块活动引用全 0、承诺额度归零。

承诺账本**每一步**都查（池级与每请求都为 0、块引用不为负），不只看结束值——
结束时「减去负数」恰好回到 0，只看结束状态发现不了运行中的错账。

复现：`benchmarks/check_step44_uncomputed.py`（历史重放语义 + legacy 等价回归）、
`benchmarks/check_step44_preemption.py`（抢占红线 + 逐步账本 + 全局 FCFS）。

## 遗留

本阶段只做重算式抢占：没有 swap/CPU offload、优先级、取消、防抖，也没有 prefix-aware
recovery（`"recompute"` 强制关闭 prefix cache，恢复时从 0 重算）。抢占不承诺总吞吐提升——
它让更多请求能进入并调整谁先拿资源，但重算会浪费计算。

完整说明见 [`docs/step44_recompute_preemption.md`](../docs/step44_recompute_preemption.md)。
