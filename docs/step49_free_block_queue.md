# step49：增量维护空闲 KV 块（空闲队列）

- 对应代码：`step49/`（新增，从 `step48/` 复制，入口改名 `step49.py`）
- 包摘要 SHA256：`f1cb800c11789e8b…`（15 个 .py / 3298 行，验收方 `source_digest()` 口径）
- 基线：`step48/`，指纹 `015d673290ed2adc…`（15 个 .py / 3228 行），原样保留未改
- **改动只有 `cache.py` 一个文件**（+75 / −5）+ 入口改名；`scheduler.py`、`engine.py` 一行未动

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

本关只修这一条热路径：**把「真正空闲」增量维护成一个队列**。

> **用队列而不是小根堆。** 初版用的是最小堆（每次挑编号最小的空闲块，好与 `step48`
> 逐步对照）；后来明确放开「空闲块顺序不必与前面 step 一致」，就换成了 `deque`：
> 更简单（`popleft` / `append` 都是 O(1)，`k>1` 的只读预览用 `islice` 而不是候选堆），
> 而且顺序与本机 vLLM 的 `FreeKVCacheBlockQueue` 一致——它初始按块编号排、
> 释放的块追加到队尾。实测两者性能没有可测差别（见 §3.3）。

## 1. 实现

### 1.1 空闲队列

```python
self.free_queue = deque(range(self.num_kv_blocks))
```

初始按块编号升序；`deallocate_block()` 释放回来的块 `append` 到**队尾**。
这正好是本机 vLLM `FreeKVCacheBlockQueue` 的顺序语义
（它不用内置 `deque`，只是因为需要 O(1) 删除队列中间的块，本实现不需要那个能力）。

### 1.2 只读的 `_peek_free_blocks(k)`

计划阶段**不能真的取走**——`_plan_block_growth()` 失败时不得改变池状态。

```python
return list(itertools.islice(self.free_queue, k))   # 只读前 k 个，O(k)
```

不用 `list(self.free_queue)[:k]`：那会复制整个队列，池子大时又是 O(池子大小)。

### 1.3 提交阶段才取

```python
def _commit_block_growth(self, seq, plan):
    self._pop_free_blocks(plan.num_free_blocks)   # 从队首取走，与 peek 顺序一致
    for b in plan.evict_block_ids: self._evict_block(b)
    for b in plan.new_block_ids:   self.block_usage[b] = 1; ...
```

`BlockGrowthPlan` 因此多了一个 `num_free_blocks`：`new_block_ids` 里**来自空闲队列的前几个**。
空闲块在前、淘汰块在后，前缀正好就是要取走的那些。

### 1.4 只在状态转换点更新

| 时机 | 对堆做什么 |
|---|---|
| 初始化 | 装入 `0..num_kv_blocks-1` |
| `_commit_block_growth()` | 从队首取走本次真正用掉的 k 个 |
| `deallocate_block()` | **引用数从 1 变 0 且不带 prefix hash** 才 `append` 到队尾 |
| prefix 发布 | 不动——块还被请求持有，本来就不在队列里 |
| 命中借用 | 不动——被借用的块带 hash，本来就不在队列里 |
| 淘汰（`_evict_block`） | 不动——见下 |

**淘汰为什么不动队列**：`_evict_block()` 只从 `_commit_block_growth()` 里被调用，
而被淘汰的闲置块在同一次提交里**立刻被复用**（`new_block_ids` 就包含它们），
所以「淘汰后变成真正空闲」从来不是一个可观测状态。这一点有断言兜着：
测试里 `_evict_block` 若在提交之外被调用就判失败。

### 1.5 `_available_blocks()` 改用队列长度

```python
return (len(self.free_queue) + len(self._evictable_block_indices(exclude=exclude))
        - self.promised_blocks)
```

`_free_block_indices()` **保留为全池扫描**，但只给两个用途：报错信息，
以及**作为校验队列的独立基准**——不能拿队列自己校验自己。正常补块路径不再调用它。

闲置缓存的 LRU 扫描/淘汰**原样保留**：那是下一种瓶颈，需求明确说不要在同一次修改里混入。

## 2. 不变量

```python
set(pool.free_queue) == {b for b in range(pool.num_kv_blocks)
                         if pool.block_usage[b] == 0 and b not in pool.block_to_hash}
len(pool.free_queue) == len(set(pool.free_queue))     # 不重复放回
```

不变量**与顺序无关**（集合等式），所以从最小堆换成队列后它原样成立。

**校验口径要说清楚**：队列和 `block_usage` 是两个结构，任何**中间微操作**都不自洽
（先标使用再弹堆、或先弹堆再标使用，中间那一刻都不满足等式）。所以不变量成立的范围是
**每个状态转换操作完成之后**：`allocate_block` / `ensure_blocks` / `deallocate_block` /
`publish_computed_blocks` / `_commit_admission` / `_commit_block_growth`。
`_evict_block` 作为提交的内部步骤被排除，但用得上面那条断言兜住。

## 3. 验证

### 3.1 不变量与无副作用（`benchmarks/check_step49_free_queue.py`，24 项全通过）

- 小例子：池子 4 块，`0` 被使用、`1` 是闲置 prefix 缓存、`2/3` 真正空闲
  → 下一次拿到的必须是**真正空闲块、不是闲置缓存块**；计划阶段它仍在队列里，
  提交后才离开；释放后回到队尾。（换成队列后不再断言「必须拿编号最小的 2」。）
- 从 step48 移植的四条失败路径（不可行 / 暂时不够 / 补块失败 / 计划阶段不淘汰），
  **快照里加了 `free_queue`**，全部「一个字节都不改」。
- 8 组完整负载（fcfs/priority × prefix 开/关 × 容量压力 × 大池子 × 2 seed），
  **每一次状态转换操作后**校验集合等式与无重复，全部成立。

### 3.2 逐步对照（`benchmarks/diff_step48_step49.py`，88 项全通过）

同 seed、同请求、同到达时刻，逐步比较：调度队列顺序、本轮计划、输入 token、
完成输出、每请求计数、承诺总额、活动引用**总数**、块引用与双向 hash。

覆盖 11 个场景 × 2 seed：fcfs/priority、同批与动态到达、容量压力、prefix 开/关、
共享前缀、`budget=1`、`max_num_seqs=1`、承诺式。

两处**刻意不再比较**，都是换队列的直接后果：

- **物理块编号**：用哪一块不影响正确性——同一逻辑位置无论落在哪个物理块，
  KV 内容与输出都一样。「同一请求不会读到错误 KV」由既有回归里的 KV 数值对照覆盖。
- **逐下标的 `block_usage`**：两边用的物理块不同，按下标的引用计数必然不同；
  有意义的是活动引用**总数**（每个请求占几块由 `cache.length` 决定），这个仍然逐个比对。

### 3.3 定点性能对比（`benchmarks/compare_step48_step49_free_alloc.py`）

负载形状与 `benchmarks/profile_step48_scheduler_kv.py` 一致；3 次独立运行，
原始数据在 `benchmarks/results/step49_vs_step48_free_alloc_*_r{1,2,3}.json`。

`_plan_block_growth(seq, 1)` 中位耗时（μs，三次运行的极差）：

| 池子块数 | 占用形态 | step48 | step49（队列） | 初版（最小堆） |
|---|---:|---:|---:|---:|
| 256 | 真正空闲 | 5.16 – 5.36 | **0.29** | 0.22 |
| 4096 | 真正空闲 | 102.24 – 106.93 | **0.28 – 0.29** | 0.21 – 0.22 |
| 16384 | 真正空闲 | 420.69 – 433.81 | **0.42 – 0.44** | 0.29 – 0.31 |
| 256 | 闲置 prefix 缓存 | 15.86 – 16.16 | 11.66 – 12.21 | 11.34 – 12.02 |
| 4096 | 闲置 prefix 缓存 | 342.70 – 382.37 | 239.58 – 250.45 | 237.52 – 247.57 |
| 16384 | 闲置 prefix 缓存 | 1419.59 – 1457.40 | 982.63 – 1013.51 | 971.88 – 1010.73 |

队列版比最小堆版慢约 0.1 μs（0.42 vs 0.30，16384 块）——**在噪声量级**
（`deque.popleft` 要走 C 层的 block 机制，而 `heappop` 是 14 次整数比较）。
两者都远低于 step48 的 420 μs，选哪个都不影响结论。

**真正空闲路径已经从「随池子线性增长」变成常数**（0.3 μs 与池子大小无关）。
`idle_cached` 仍随池子线性增长（16384 块 1002 μs），只比 step48 快约 30%
——因为它走的是**下一关**要处理的 LRU 扫描，本关没动。

需求指定的 32 请求 / 8192 块 / prefix 关负载，`schedule()` 累计耗时（ms）：

| 负载 | step48 | step49 |
|---|---:|---:|
| 32 请求 / 8192 块 / prefix 关 | 40.91 – 42.13 | **1.76 – 3.22** |
| 32 请求 / 1024 块 / prefix 关 | 6.04 – 6.50 | 1.79 – 1.86 |
| 32 请求 / 8192 块 / prefix 开 | 44.62 – 45.42 | 2.14 – 2.53 |

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

**新增**：`KVCachePool.free_queue`、`_peek_free_blocks(k)`、`_pop_free_blocks(k)`、
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
4. **物理块编号与 step48 不同**：空闲队列是先进先出的，不再保证「每次挑编号最小的」。
   这是刻意的。若将来想要「复用编号小的块」带来的局部性，再换回堆即可——
   不变量与顺序无关，换回来不影响正确性。
