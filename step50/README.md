# step50：闲置 prefix 缓存的 LRU 索引

从 `step49/` 复制。`step49/` 原样保留，未改。**改动只有 `cache.py` 一个文件。**

## 结论

上一关验收测到的瓶颈：**池子全被闲置 prefix 缓存占满时，拿 1 块仍要扫描并排序整个池子**，
8192 块时 `ensure_blocks()` p50 约 500 μs，且随池子线性增长。

改成**仿本机 vLLM 的单条双向链表**（用户指定方向：不用堆，按 vLLM 来实现）：

```python
# block_pool.py::free_blocks 的原样语义
无 hash 的「真正空闲」 -> 前插到队首（优先复用）
带 hash 的「闲置缓存」 -> 后插到队尾（最后才淘汰）
分配从队首取；取到带 hash 的就地清掉缓存条目再用
```

**位置即优先级**：这个顺序保证了「先用真正空闲的块，而不是先去淘汰缓存」。
我做诊断时试过把无 hash 的块也改成后插，立刻出现 `input_ids`/`plan`/`active_refs`
的真实行为差异——位置不是装饰。

删除靠块自带的前后指针（`block_next`/`block_prev` 数组 + 两个哨兵槽位），**O(1)**，
不需要位置索引，也不用堆。计划阶段只读（`_peek_allocatable` 沿 next 走 k 步），
失败时链表、hash、引用计数一个字节都不动。

## 效果（3 次独立运行，原始数据在 `benchmarks/results/`）

| 池子 | step49 | step50 | 改善 |
|---|---:|---:|---:|
| 1024 块 | 57.82 μs | **0.67 μs** | 86× |
| 8192 块 | 491.12 μs | **0.66 μs** | **744×** |

池子扩大 8 倍：step49 慢 **8.5 倍**，step50 **平坦（1.0 倍）**。
验收要求「≥4 倍改善、增长 ≤3 倍」，两项都远超。

## 怎么证明功能正常

| 检查 | 结果 |
|---|---|
| `benchmarks/diff_step49_step50.py`：逐步对照（计划、队列、输出、计数、结束态） | **88 项全通过**（11 场景 × 2 seed） |
| `benchmarks/check_step50_idle_index.py`：不变量 + 无副作用 | **27 项全通过** |
| 每次**状态转换操作后**校验集合等式、无重复、双向指针、计数 | 全部成立 |
| step47/46/45 自查、step44 三不变量、CPU 压测 120、GPU 三路径、18 个既有回归 | 全过 |

**物理块编号刻意不比较**（用户明确放开：只测功能）。已知且解释得了的两处差异：
物理块编号（位置语义不同）、`cached_blocks_kept`（淘汰选择不同，实测 4 vs 3）。
其余——计划、队列、输出、抢占次数、重算/复用计数、承诺归零、引用归零——全部逐项相同。

复现：`benchmarks/profile_step50_warm_prefix.py`（性能）、`benchmarks/diff_step49_step50.py`（行为）、
`benchmarks/check_step50_idle_index.py`（不变量）。

完整说明见 [`docs/step50_idle_lru_index.md`](../docs/step50_idle_lru_index.md)。
