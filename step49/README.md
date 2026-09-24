# step49：增量维护空闲 KV 块

从 `step48/` 复制。`step48/` 原样保留，未改。**改动只有 `cache.py` 一个文件。**

## 结论

剖析测到：32 请求 / 8192 块的调度负载里 `schedule()` 累计 45 ms，其中 43 ms 花在补块——
因为「真正空闲」是每次**临时扫全池**算出来的，而多数时候只想拿 1 块。池子越大这份无用功越大。

改成在 `KVCachePool` 里**增量维护一个小根堆**：

```python
self.free_heap = list(range(num_kv_blocks)); heapq.heapify(self.free_heap)
```

- 计划阶段只读：`_peek_smallest_free(k)` 取编号最小的 k 个，**不弹堆**（计划失败不得有副作用）。
  `k == 1` 读堆顶；`k > 1` 用一个 O(k) 候选堆沿孩子走——不排序、不扫池。
- 提交阶段才 `_pop_free_blocks(k)`。
- 释放时**只有引用数 1→0 且该块不带 prefix hash**才放回。
  prefix 缓存块是「闲置缓存」，**不是**「真正空闲」，不进堆。
- 最小堆保证「编号最小的先被选中」，与旧实现顺序一致。

闲置缓存的 LRU 扫描**原样保留**——那是下一种瓶颈，不在这一关混着改。

## 效果（3 次独立运行，原始数据在 `benchmarks/results/`）

`_plan_block_growth(seq, 1)` 中位耗时（μs）：

| 池子块数 | 占用形态 | step48 | step49 |
|---|---:|---:|---:|
| 256 | 真正空闲 | 5.15–5.25 | **0.22** |
| 4096 | 真正空闲 | 105.12–107.69 | **0.21–0.22** |
| 16384 | 真正空闲 | 425.24–437.14 | **0.29–0.31** |
| 16384 | 闲置 prefix 缓存 | 1410.43–1445.55 | 971.88–1010.73 |

**真正空闲路径从「随池子线性增长」变成常数**。`idle_cached` 仍线性（走的是下一关的 LRU 扫描）。

32 请求 / 8192 块 / prefix 关：`schedule()` 累计从 **42.2–42.4 ms** 降到 **1.8–3.5 ms**；
池子 1024 → 8192 的劣化从 **6.6 倍**降到 **1.0–1.9 倍**。`_reserve_blocks()` 从 40 ms 降到 0.5–0.8 ms。

**这不是端到端吞吐结论**——剖析已说明调度 CPU 开销在当前小模型里还不是端到端主瓶颈。

## 怎么证明行为没变

| 检查 | 结果 |
|---|---|
| `benchmarks/diff_step48_step49.py`：逐步对照，**含物理 `block_table` 逐请求一致** | **88 项全通过**（11 场景 × 2 seed） |
| `benchmarks/check_step49_free_heap.py`：不变量 + 无副作用 | **24 项全通过** |
| 每次**状态转换操作后**校验 `set(free_heap) == 真实空闲集合`、无重复 | 全部成立 |
| step47/45/46 自查、step44 三不变量、CPU 压测 120、GPU 三路径、18 个既有回归 | 全过 |

复现：`benchmarks/compare_step48_step49_free_alloc.py`（性能）、
`benchmarks/diff_step48_step49.py`（行为）、`benchmarks/check_step49_free_heap.py`（不变量）。

完整说明见 [`docs/step49_free_block_heap.md`](../docs/step49_free_block_heap.md)。
