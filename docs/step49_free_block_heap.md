# step49：增量维护空闲 KV 块

- 对应代码：`step49/`（新增，从 `step48/` 复制，入口改名 `step49.py`）
- 包摘要 SHA256：`eb30958b9f2ea3b5…`（15 个 .py / 3302 行，验收方 `source_digest()` 口径）
- 基线：`step48/`，指纹 `015d673290ed2adc…`（15 个 .py / 3228 行），原样保留未改
- **改动只有 `cache.py` 一个文件**（+79 / −5）+ 入口改名；`scheduler.py`、`engine.py` 一行未动

## 0. 需求大概

第四十八关后的剖析（`169_第四十八关后_调度与KV基准剖析`）测到：32 请求 / 8192 块的 CPU 负载里
`schedule()` 累计 **45.09 ms**，其中 `_reserve_blocks()` **43.35 ms**；`cProfile` 显示
`_plan_block_growth()` 内部的 `_free_block_indices()` 累计约 **40 ms**。

原因是「真正空闲」由 `block_usage[b] == 0 and b not in block_to_hash` **临时扫全池算出来**，
而多数时候只想拿 **1 块**。池子越大，这份无用功越大：

| 只测 `_plan_block_growth(seq, 1)` | 256 块 | 4096 块 | 16384 块 |
|---|---:|---:|---:|
| 全部真正空闲 | 5.32 μs | 106.93 | 454.80 |
| 全部是闲置 prefix 缓存 | 16.16 | 361.63 | 1480.09 |

本关只修这一条热路径：**把「真正空闲」增量维护成一个小根堆**。

## 1. 实现

### 1.1 小根堆

```python
self.free_heap = list(range(self.num_kv_blocks))
heapq.heapify(self.free_heap)
```

为什么是**编号最小**：旧实现每次按物理块编号从小到大选，保持同样顺序才能与 `step48`
逐步对照（验收里逐请求比较 `block_table`）。最小堆天然给出同样的顺序。

### 1.2 只读的 `_peek_smallest_free(k)`

计划阶段**不能弹堆**——`_plan_block_growth()` 失败时不得改变池状态。

```python
if k == 1:
    return [self.free_heap[0]]          # 直接读堆顶，O(1)
```

`k > 1` 用一个 **O(k) 的候选堆**沿孩子往下走：弹出候选后把它的两个孩子压进候选堆，
取出前 k 小，原堆一动不动。**不用 `sorted(free_heap)`，也不用 `heapq.nsmallest(1, ...)`**
——两者都会扫全池，正是这一关要消掉的开销。

### 1.3 提交阶段才弹

```python
def _commit_block_growth(self, seq, plan):
    self._pop_free_blocks(plan.num_free_blocks)   # 按编号升序弹出，与 peek 顺序一致
    for b in plan.evict_block_ids: self._evict_block(b)
    for b in plan.new_block_ids:   self.block_usage[b] = 1; ...
```

`BlockGrowthPlan` 因此多了一个 `num_free_blocks`：`new_block_ids` 里**来自空闲堆的前几个**。
空闲块在前、淘汰块在后，前缀正好就是要弹出的那些。

### 1.4 只在状态转换点更新

| 时机 | 对堆做什么 |
|---|---|
| 初始化 | 装入 `0..num_kv_blocks-1` |
| `_commit_block_growth()` | 弹出本次真正用掉的 k 个 |
| `deallocate_block()` | **引用数从 1 变 0 且不带 prefix hash** 才放回 |
| prefix 发布 | 不动——块还被请求持有，本来就不在堆里 |
| 命中借用 | 不动——被借用的块带 hash，本来就不在堆里 |
| 淘汰（`_evict_block`） | 不动——见下 |

**淘汰为什么不动堆**：`_evict_block()` 只从 `_commit_block_growth()` 里被调用，
而被淘汰的闲置块在同一次提交里**立刻被复用**（`new_block_ids` 就包含它们），
所以「淘汰后变成真正空闲」从来不是一个可观测状态。这一点有断言兜着：
测试里 `_evict_block` 若在提交之外被调用就判失败。

### 1.5 `_available_blocks()` 改用堆

```python
return (len(self.free_heap) + len(self._evictable_block_indices(exclude=exclude))
        - self.promised_blocks)
```

`_free_block_indices()` **保留为全池扫描**，但只给两个用途：报错信息，
以及**作为校验堆的独立基准**——不能拿堆自己校验自己。正常补块路径不再调用它。

闲置缓存的 LRU 扫描/淘汰**原样保留**：那是下一种瓶颈，需求明确说不要在同一次修改里混入。

## 2. 不变量

```python
set(pool.free_heap) == {b for b in range(pool.num_kv_blocks)
                        if pool.block_usage[b] == 0 and b not in pool.block_to_hash}
len(pool.free_heap) == len(set(pool.free_heap))     # 不重复放回
```

**校验口径要说清楚**：堆和 `block_usage` 是两个结构，任何**中间微操作**都不自洽
（先标使用再弹堆、或先弹堆再标使用，中间那一刻都不满足等式）。所以不变量成立的范围是
**每个状态转换操作完成之后**：`allocate_block` / `ensure_blocks` / `deallocate_block` /
`publish_computed_blocks` / `_commit_admission` / `_commit_block_growth`。
`_evict_block` 作为提交的内部步骤被排除，但用得上面那条断言兜住。

## 3. 验证

### 3.1 不变量与无副作用（`benchmarks/check_step49_free_heap.py`，24 项全通过）

- 需求给的小例子：池子 4 块，`0` 被使用、`1` 是闲置 prefix 缓存、`2/3` 真正空闲
  → 下一次应拿 **2**；计划阶段 `2` 仍在堆里，提交后才离开；释放后回到堆。
- 从 step48 移植的四条失败路径（不可行 / 暂时不够 / 补块失败 / 计划阶段不淘汰），
  **快照里加了 `free_heap`**，全部「一个字节都不改」。
- 8 组完整负载（fcfs/priority × prefix 开/关 × 容量压力 × 大池子 × 2 seed），
  **每一次状态转换操作后**校验集合等式与无重复，全部成立。

### 3.2 逐步对照（`benchmarks/diff_step48_step49.py`，88 项全通过）

同 seed、同请求、同到达时刻，逐步比较：队列顺序、本轮计划、输入 token、
**每个请求的物理 `block_table`**、完成输出、每请求计数、承诺总额、块引用与双向 hash。

覆盖 11 个场景 × 2 seed：fcfs/priority、同批与动态到达、容量压力、prefix 开/关、
共享前缀、`budget=1`、`max_num_seqs=1`、承诺式。
**物理块编号逐请求一致**——这正是「保持编号最小优先」要保证的。

### 3.3 定点性能对比（`benchmarks/compare_step48_step49_free_alloc.py`）

负载形状与 `benchmarks/profile_step48_scheduler_kv.py` 一致；3 次独立运行，
原始数据在 `benchmarks/results/step49_vs_step48_free_alloc_*_r{1,2,3}.json`。

`_plan_block_growth(seq, 1)` 中位耗时（μs，三次运行的极差）：

| 池子块数 | 占用形态 | step48 | step49 |
|---|---:|---:|---:|
| 256 | 真正空闲 | 5.15 – 5.25 | **0.22** |
| 4096 | 真正空闲 | 105.12 – 107.69 | **0.21 – 0.22** |
| 16384 | 真正空闲 | 425.24 – 437.14 | **0.29 – 0.31** |
| 256 | 闲置 prefix 缓存 | 15.97 – 16.26 | 11.34 – 12.02 |
| 4096 | 闲置 prefix 缓存 | 351.30 – 356.35 | 237.52 – 247.57 |
| 16384 | 闲置 prefix 缓存 | 1410.43 – 1445.55 | 971.88 – 1010.73 |

**真正空闲路径已经从「随池子线性增长」变成常数**（0.3 μs 与池子大小无关）。
`idle_cached` 仍随池子线性增长（16384 块 1002 μs），只比 step48 快约 30%
——因为它走的是**下一关**要处理的 LRU 扫描，本关没动。

需求指定的 32 请求 / 8192 块 / prefix 关负载，`schedule()` 累计耗时（ms）：

| 负载 | step48 | step49 |
|---|---:|---:|
| 32 请求 / 8192 块 / prefix 关 | 42.17 – 42.43 | **1.81 – 3.47** |
| 32 请求 / 1024 块 / prefix 关 | 6.17 – 6.31 | 1.69 – 1.91 |
| 32 请求 / 8192 块 / prefix 开 | 42.70 – 44.66 | 2.21 – 2.62 |

关键不是「快了多少倍」，而是**池子从 1024 涨到 8192 时几乎不再变慢**：
step48 是 6.6 倍，step49 是 1.0–1.9 倍（残差来自仍在线性扫描的 LRU 索引）。
`_reserve_blocks()` 从 40 ms 降到 0.5–0.8 ms。

**这不是端到端吞吐结论**。剖析已说明：在当前这个 Python/Torch 小模型里，
调度 CPU 开销还不是端到端主瓶颈（8192 块池时 `schedule()` 只占端到端 0.46%）。

### 3.4 其余

| 项 | 结果 |
|---|---|
| step47 / step45 / step46 的自查（新引擎换成 step49） | 42 / 38 / 44 项全过 |
| step44 的三个不变量（指向 step49） | 3 / 3 |
| CPU 随机压测 120 组 | PASS |
| GPU：CUDA/Torch、Triton eager、Triton Graph | 各 14 步有界完成，引用归零 |
| 既有 18 个回归脚本 | 全过（副本里同步了 step48 的 `block_hash` → `hash_to_block` 改名） |

## 4. 接口变化与遗留

### 4.1 接口变化

**新增**：`KVCachePool.free_heap`、`_peek_smallest_free(k)`、`_pop_free_blocks(k)`、
`_push_free_block(b)`；`BlockGrowthPlan` 多一个 `num_free_blocks`。

**语义变化**：`_free_block_indices()` 从「补块路径的第一步」变成「报错信息 + 校验基准」，
仍返回按编号升序的**全池扫描**结果。

**未改**：`Engine` / `from_model_dir` 参数、`step()` 返回格式、调度策略、
抢占与阻塞者规则、prefix hash 与 LRU 规则、token 历史表示、模型代码。
`_available_blocks()` 的语义不变（数值也一致，因为堆与扫描等价）。

### 4.2 遗留

1. **闲置 prefix 缓存的 LRU 扫描仍是 O(池子大小)**（16384 块 1002 μs）。
   需求明确说这是下一种瓶颈，本关不混入。
2. **`_plan_tokens()` 里长输出的历史复制**（剖析里的第二项发现）未处理。
3. **不做端到端吞吐断言**：本关只消除一个明确的热路径缺陷。
4. 物理块**编号本身**与 step48 一致是刻意保持的（方便逐步对照）；
   如果将来不再需要对照，可以放开选择顺序以获得更好的局部性。
