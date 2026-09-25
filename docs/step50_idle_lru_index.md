# step50：闲置 prefix 缓存的 LRU 索引

- 对应代码：`step50/`（新增，从 `step49/` 复制，入口改名 `step50.py`）
- 包摘要 SHA256：`a89b86a4698464f6…`（15 个 .py / 3344 行，验收方 `source_digest()` 口径）
- 基线：`step49/`，指纹 `f1cb800c11789e8b…`（15 个 .py / 3298 行），原样保留未改
- **改动只有 `cache.py` 一个文件**（+145 / −86）+ 入口改名；`scheduler.py` 一行未动

## 0. 需求大概

第四十九关的验收测到下一个瓶颈：**池子全被闲置 prefix 缓存占满时，拿 1 块仍要扫描并排序整个池子**。
8192 块的真实暖池流程里 `ensure_blocks()` p50 约 **500 μs**，且随池子线性增长。

根因是 `_evictable_block_indices()`：每次淘汰都 `[i for i in range(N) if ...]` 再 `sort()`，
而多数时候只需要 1 块。

## 1. 做法：仿本机 vLLM 的单条链表

> **这一版是用户指定的方向：不用堆，按 vLLM 的实现来。** 第一版我写的是
> 「`free_queue`（deque）+ `idle_heap`（最小堆 + `idle_pos` 位置索引）」两套结构，
> 用它把 `(block_last_used, b)` 精确排序；改成 vLLM 的做法后两套结构合并成一条链表。

看本机 vLLM 0.28 的实现：

```python
# vllm/v1/core/block_pool.py::free_blocks
for block in ordered_blocks:
    block.ref_cnt -= 1
    if block.ref_cnt == 0:
        if block.block_hash is None:  blocks_to_evict_first.append(block)
        else:                         blocks_to_evict_last.append(block)
self.free_block_queue.prepend_n(blocks_to_evict_first)   # 无 hash -> 队首
self.free_block_queue.append_n(blocks_to_evict_last)     # 带 hash -> 队尾
```

`FreeKVCacheBlockQueue` 是一条**双向链表**，块自己带 `prev_free_block` / `next_free_block`，
所以 `remove(block)` 是 O(1)，不需要位置索引。它用假头/假尾哨兵避免边界分支。

本实现照搬这套：

| | 位置 | 含义 |
|---|---|---|
| 无 hash 的「真正空闲」 | **队首**（前插） | 下一次分配最先拿到，优先复用 |
| 带 hash 的「闲置缓存」 | **队尾**（后插） | 真正空闲的用完了才轮到它，被取走时就地清 hash |

**位置即优先级**——这一条很关键。我做过一次诊断，把无 hash 的块也改成后插（想贴近
step49 的 FIFO），结果出现了 `input_ids` / `plan` / `active_refs` 的**真实行为差异**：
那样会让分配器先去淘汰缓存、而不是先用真正空闲的块。位置不是装饰。

链表用两个**哨兵槽位**（下标 `N` / `N+1`）避免首尾分支，`block_next[b] == -1` 表示不在链上。

## 2. 状态转换点

| 操作 | 对链表做什么 |
|---|---|
| 初始化 | 全部块逆序前插 → 最终按编号升序 |
| `_commit_block_growth` | 从队首取走计划选中的 k 个；取到带 hash 的就地清掉缓存条目 |
| `deallocate_block` | 引用降到 0：无 hash 前插、带 hash 后插 |
| `_commit_admission` | 借用命中的闲置块时先摘掉（O(1)）；命中已被别的请求持有的块不在链上 |
| `publish_computed_blocks` | 不动——发布的块还被请求持有，本来就不在链上 |

`_plan_block_growth()` 仍然**只读**：`_peek_allocatable(k)` 沿 next 指针走 k 步。
失败时链表、hash、引用计数一个字节都不动。

`_available_blocks(exclude)` 直接用 `num_allocatable`，只从里面扣掉 `exclude` 中确实在链上的块。

## 3. 不变量

```python
set(链上的块) == {b | block_usage[b] == 0}
链内无重复；num_allocatable == 链长；prev/next 双向一致
```

`_allocatable_block_indices()` 保留为**全池扫描**，只给报错信息与测试当独立基准
（不能拿链表自己校验自己）。`_free_block_indices()` / `_evictable_block_indices()`
同样保留扫描版，正常路径不再调用。

校验口径与上一关相同：堆/链表与 `block_usage` 是两个结构，**中间微操作**本就不自洽，
所以不变量成立的范围是**每个状态转换操作完成后**。`_evict_block` 只是提交的内部步骤
（只删 hash 关联），被豁免——但测试里有断言兜住：它在提交之外被调用就判失败。

## 4. 验证

### 4.1 定点性能（`benchmarks/profile_step50_warm_prefix.py`，3 次独立运行）

复用第四十九关验收用过的真实暖池负载（池子全被闲置缓存占满，再发 300 个必然缺块的请求）：

| 池子 | step49 `ensure_blocks` p50 | step50 | 改善 |
|---|---:|---:|---:|
| 1024 块 | 59.06 μs | **0.69 μs** | 86× |
| 8192 块 | 498.94 μs | **0.68 μs** | **734×** |

池子扩大 8 倍：step49 慢 **8.4 倍**，step50 **完全平坦（1.0 倍）**。

验收要求是「至少 4 倍改善、跨池大小增长不超过 3 倍」——两项都远超。
原始数据：`benchmarks/results/step50_warm_prefix_profile.json`。

### 4.2 逐步对照（`benchmarks/diff_step49_step50.py`，88 项全通过）

同 seed、同请求、同到达时刻，逐步比较调度队列、本轮计划、输入 token、完成输出、
每请求计数、承诺总额、活动引用总数、结束态。11 场景 × 2 seed。

**物理块编号刻意不比较**（用户明确放开：只测功能）。两处已知且解释得了的差异：

- **物理块编号**：vLLM 的位置语义（空闲块前插）决定了选择顺序与 step49 的 FIFO 不同。
  用哪一块不影响正确性——同一逻辑位置无论落在哪个物理块，KV 内容与输出都一样；
  「不会读到错误 KV」由既有回归里的 KV 数值对照覆盖。
- **`cached_blocks_kept`**：淘汰选择不同，最后留在缓存里的块数也不同（实测 4 vs 3）。
  这不是功能差异，比对时已排除。

其余全部逐项相同：计划、队列、输出、抢占次数、重算/复用计数、承诺归零、引用归零。

### 4.3 不变量与无副作用（`benchmarks/check_step50_idle_index.py`，27 项全通过）

- 每次状态转换操作后校验集合等式、无重复、双向指针、计数；
- 需求给的小例子：带 hash 的块一直在队尾一侧，计划阶段不动链表，提交后才离开；
- 命中借用的块先离开链表，释放后回到链上；
- 四条失败路径（不可行 / 暂时不够 / 补块失败 / 计划阶段不动链表）**一个字节都不改**，
  快照里含链表与计数。

### 4.4 提交阶段不需要按数量重走链表

`_commit_block_growth()` 原本写的是：

```python
self._pop_allocatable(len(plan.new_block_ids))     # 按数量从队首再走一遍
for block_idx in plan.new_block_ids: ...
```

但 `plan.new_block_ids` **就是**队首前 k 个（计划就是从队首 peek 的，计划与提交之间
没有任何东西改链表），按数量重新走一遍是重复劳动，还多了一层「数量必须与队首一致」
的隐式耦合。改成逐个按块自己的前后指针摘掉，O(1)，`_pop_allocatable` 随之删除。

### 4.5 自查脚本的两处修正

- `profile_step50_warm_prefix.py` 里对 step50 的断言原本被 `hasattr(pool, "free_queue")`
  挡掉（step50 没有这个结构），等于没跑。改成按 `block_next` 判断，并写清两关的差别：
  step49 的 `free_queue` 只装真正空闲块 → 应当为空；step50 的一条链装**全部**引用为 0 的块
  → 链长应为整池、而「真正空闲」为 0。修好后断言真正执行并通过。
- `cache.py` 里 `_evictable_block_indices()` 出现了**两个定义**（我编辑时留下的）：
  一个是新写的未排序版，一个是 step49 留下的按 `block_last_used` 排序版。Python 用后者，
  所以行为侥幸没变，但那是隐患。删掉重复的那个，只留报错信息需要的那份。

### 4.6 其余

| 项 | 结果 |
|---|---|
| step47 / step46 / step45 的自查（新引擎换成 step50） | 42 / 44 / 38 项全过 |
| step44 的三个不变量（指向 step50） | 3 / 3 |
| CPU 随机压测 120 组 | PASS |
| GPU：CUDA/Torch、Triton eager、Triton Graph | 各 14 步有界完成，引用归零 |
| 既有 18 个回归脚本 | 全过（副本里同步了 `block_hash` → `hash_to_block` 改名） |

## 5. 接口变化与遗留

### 5.1 接口变化

**新增**：`KVCachePool.block_next` / `block_prev` / `num_allocatable` / `_SENTINEL_HEAD` /
`_SENTINEL_TAIL`；`_list_prepend` / `_list_append` / `_list_remove` / `_in_alloc_list` /
`_peek_allocatable` / `_release_block` / `_allocatable_block_indices` /
`_evict_hash_if_cached`。

**移除**：`free_queue`（deque）、`_peek_free_blocks` / `_pop_free_blocks` / `_push_free_block`；
`_pop_allocatable`（见下：提交时按块自己的指针摘掉，不需要按数量从队首重走一遍）；
`BlockGrowthPlan` 只剩 `new_block_ids` 一个字段（原来的 `evict_block_ids` / `num_free_blocks`
在单一链表里没有意义了）。

**未改**：`Engine` / `from_model_dir` 参数、`step()` 返回格式、调度与优先级、抢占与阻塞者规则、
prefix hash 算法与命中规则、token 历史表示、模型与采样。

### 5.2 遗留

1. **淘汰顺序由「释放先后」决定**，不是 step49 的精确 `(block_last_used, b)`。
   想严格按最后使用时间排序就得回到堆（或别的有序结构）——用户已明确放开这一条。
2. **`_evictable_block_indices()` 仍是全池扫描**，但只在报错信息与测试里用；正常路径不碰。
3. **不改长输出的历史复制**（剖析里的第二项发现），也还没转投机解码。
4. 性能是**定点**结论（池子全为闲置缓存时的 `ensure_blocks`），不是端到端吞吐。
