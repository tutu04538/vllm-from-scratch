# step44：重计算式抢占与恢复（第一阶段）

- 对应代码：`step44/`（新增，从 `step43/` 复制，入口改名 `step44.py`）
- 包摘要 SHA256：`9626e95dcba34f8f…`（14 个 .py / 2886 行，验收方 `source_digest()` 口径；§8 修正后的最终值）
- 基线：`step43/`，指纹 `febd8c3663a061f6…`（14 个 .py / 2729 行），原样保留未改
- **改动文件只有 4 个 + 入口改名**，逐个列出：

| 文件 | 改动 |
|---|---|
| `cache.py`（+61 / −9） | ① `SequenceConfig` 新增 `all_token_ids`、`num_uncomputed_tokens`，`prefill_len` 改为它的别名（旧公式的 `max(..., 0)` 去掉）；② 新增 `is_ready_for_next_token`；③ 新增 `num_preemptions` / `recomputed_tokens` / `high_water` 三个计数；④ `KVCachePool.__init__` 新增 `over_subscribe=False`；⑤ `allocate_block()` 超卖时跳过可用量检查、不写承诺额度；⑥ `ensure_blocks()` 超卖时容量不足改为**无副作用**返回 `False`（承诺式仍 `raise RuntimeError`），且「消耗承诺」只在承诺式路径发生、淘汰闲置块时不再传多余的 `exclude`（见 §8.5） |
| `scheduler.py`（+114 / −34） | ① `__init__` 新增 `preemption_mode`、`num_preemptions`、`_allocated_this_step`，并拒绝 `max_num_batched_tokens <= 0`；② `schedule()` 的预留判定改用 `is_ready_for_next_token`，decode / prefill 两条分支合成「从 `all_token_ids[cache.length]` 续算」，补块阶段加抢占循环；③ 新增 `_make_room()`（尾部选犠牲者）；④ 新增 `_preempt()`；⑤ `post_step()` 判停守卫改为 `not seq.output_ids`，并在开头记重算量；⑥ 零进展守卫去掉 admission 一项（`_admitted_this_step` 随之删除）；⑦ `_preempt()` 改为插 waiting 队首以保全局 FCFS |
| `engine.py`（+31 / −6） | ① 新增 `PREEMPTION_MODES` 与 `_check_preemption_mode()`；② `Engine.__init__` / `Engine.from_model_dir` 尾部新增 keyword-only `preemption_mode=None`，两个入口都在构造阶段校验；③ `_init_runtime` 透传该值给 `KVCachePool(over_subscribe=...)` 与 `Scheduler(...)` |
| `__init__.py`（+1 / −1） | 包 docstring 与模块说明改成第 44 关，无逻辑改动 |
| `step44.py` | 由 `step43/step43.py` 改名而来，只有包名引用变化 |

**未改的 9 个文件**：`attention.py`、`model.py`、`norm.py`、`rope.py`、`sampler.py`、`sampling.py`、
`formats/__init__.py`、`formats/native.py`、`formats/qwen3.py`
—— 模型 forward、attention / norm / rope kernel、采样算法、Graph key、外部格式适配
和数值路径一个字节没动（需求 §10：「不改 attention、RoPE、norm、Linear、Graph 的数值路径」）。

## 0. 需求大概

一条请求还没完成时，可以**暂时释放它占的 KV 块**，但保留 prompt、已生成的 token 和采样状态；
稍后从历史重算 KV，再从原来的输出之后继续生成。这就是抢占（preemption），区别于「取消」。

需求先解释了为什么现在不会自然发生抢占：`allocate_block()` 在准入时按**最坏情况**把未来所有块
都承诺掉了。池子 8 块时，A 承诺 6 块、B 需要 4 块，于是 B 永远进不来——即使 A 当前只用了 2 块。
所以本关的根本改变是**在 recompute 模式下允许容量超卖**：准入只判断「这条请求单独跑是否可行」，
不锁未来块；真的不够时，调度器选犠牲者释放它的 KV。

代价是被抢占者稍后要重算。

本阶段明确不做：swap/CPU offload、优先级、取消、多线程、异步在途批次。

## 1. 接口变化

```python
Engine(..., *, on_token=None, preemption_mode=None)          # keyword-only，排在最后
Engine.from_model_dir(..., *, on_token=None, preemption_mode=None)
```

- `preemption_mode=None`：完全保留第四十三关的承诺式记账、prefix cache 与输出行为。
- `preemption_mode="recompute"`：启用本关新路径。
- 只接受这两个值，其他值在**构造阶段**报 `ValueError`。
- `"recompute"` 要求显式 `enable_prefix_caching=False`；两个一起开在构造阶段报 `ValueError`，
  **不静默关掉其中一个**（重算会丢弃并重建 KV，与共享前缀块的引用计数语义冲突）。

`preemption_mode` 放在 `*` 之后、`on_token` 之后——第四十三关首次验收就是因为把新参数插在旧参数
中间、让旧的位置调用整体错位（`Engine(..., None, False)` 里的 `False` 从 `enable_prefix_caching`
变成了 `on_token`）。所以这次新增参数一律加在尾部。

## 2. 核心改动：重新定义「请求历史」

这是本关最重要的一处，也是 §16 要求先做的一步。

生成 `y1` 之后，KV 里是 `[p0, p1, p2, y0]`——`y1` 是刚采样出来的、还没进模型的下一个输入。
被抢占后如果只重算 prompt，引擎会在 prompt 末尾又采样一次，结果就错了。

```python
@property
def all_token_ids(self):            # prompt_ids + output_ids
@property
def num_uncomputed_tokens(self):    # len(prompt_ids) + len(output_ids) - cache.length
```

四条守住的语义：

1. **`cache.length` 是「已计算长度」的唯一真相**——已经进入模型并写入 KV 的 token 数。
   没有另开 `num_computed_tokens`，`num_uncomputed_tokens` 只算差值，不会互相矛盾。
2. 未计算数不再只看 prompt。旧代码是 `max(len(prompt_ids) - cache.length, 0)`，那个 `max(..., 0)`
   恰好把「刚采样、还没进模型」的最后一个 output token 藏了起来。
3. 中间重算 chunk **不采样、不重放 `on_token`**：`can_sample = (num_scheduled_tokens == num_uncomputed_tokens)`，
   只有本轮算到当前 `all_token_ids` 末尾才允许采样。
4. `output_ids`、`SamplingState.generated_counts`、随机 `Generator`/offset 全部保留；
   重算历史不消耗随机数（`SamplingState` 的 `torch.Generator` 是每请求一个，抢占根本不碰它）。

### 顺带把 decode / prefill 两条分支合成了一条

本轮要算的永远是 `all_token_ids[cache.length : cache.length + n]`。旧代码里 decode 分支写的
`input_ids = [output_ids[-1]]`，在新定义下正好等于 `all_token_ids[cache.length]`（decode 时
`cache.length == len(prompt)+len(output)-1`）。所以抢占后的重放不需要另写一条路径——
它天然就走「从 cache.length 处续算」。

两处判定的等价改写（`prefill_len` 保留为 `num_uncomputed_tokens` 的别名）：

| 位置 | 旧 | 新 |
|---|---|---|
| 预留 decode 额度 | `prefill_len == 0` | `is_ready_for_next_token`（未计算数为 1 且 prompt 已算完） |
| `post_step` 判停守卫 | `prefill_len > 0: continue` | `not seq.output_ids: continue` |

第二条值得说明：旧守卫的真实含义是「还没有可判停的 output token」，而不是「prompt 没算完」。
在旧代码可达的状态里 `prefill_len > 0` 与 `output_ids == []` 完全等价；换成后者既保持旧行为，
又不会把「重算还没追上」的请求误判成可以采样。

## 3. recompute 模式的 KV 分配

- `KVCachePool(over_subscribe=True)`：`allocate_block()` 仍然拒绝「单独给它整个池子也跑不完」的请求
  （`InfeasibleRequest`，走既有明确失败路径），但**不写** `seq.promised_blocks` / 池级 `promised_blocks`。
  刚准入的请求可以是空块表（`cache.length == 0`）。
- `ensure_blocks()` 容量不足时返回 `False`，**且没有副作用**：不部分追加 block table、不改引用计数、
  不改 `cache.length`。（承诺式模式下仍然保留 `RuntimeError`——那时它是记账被破坏的信号，
  而且承诺保证了这个分支不可达。）
- **补块成功时的「消耗承诺」只发生在承诺式路径**：`seq.promised_blocks -= extra` /
  `self.promised_blocks -= extra` 都放在 `if not self.over_subscribe` 里。超卖模式准入时根本没加过承诺，
  从 0 减 `extra` 会把账本减成负数——详见 §8.1。
- **具体抢占谁由 Scheduler 决定**，`KVCachePool` 不认识请求优先级。

## 4. 抢占策略：FCFS + 从 running 尾部选犠牲者

调度仍从前往后处理请求，分两步：先按 token 预算排出计划，再逐条补块。

补块失败时 `_make_room()` 从 **running 尾部**找犠牲者：

```python
for victim in reversed(running):
    if victim is seq:
        return False                              # 当前请求自己就是最后候选
    if victim not in self.running:  continue      # 本轮已被抢占过
    if id(victim) in self._allocated_this_step:
        return False                              # 再往前就会回滚已记账的计划
    self._preempt(victim)
    if self.kv_cache_pool.ensure_blocks(seq, num_tokens):  return True
```

**为什么要 `_allocated_this_step`**：规划和补块是两个阶段。只按 running 顺序判断「是否已安排」
会把**已经补过块**的请求当成候选释放掉，而它的块和 token 预算都已经记账。第一次实现就踩了这个坑
（抢占 B 之后，B 的计划还在 `scheduled_items` 里，补块时撞上 `block_table is None`）。
现在两道防线：补块循环开头丢弃已被抢占者的计划，犠牲者候选排除已补块的请求。

**不允许抢占已补块的请求**：本轮的 token 预算、`scheduled_items`、刚分配的块都已记账，
回滚它们要同时回滚计划和预算——不是永远不做，而是第一阶段先让计划保持单调。

被抢占者回到 `waiting`：从 running 尾部依次取 C、B，**每条都插到队首**，合起来正好是
`[B, C] + 原有的 waiting`：

```text
waiting = [D, E]
取 C -> [C, D, E]
取 B -> [B, C, D, E]
```

这些请求比 D/E 更早到达，就不能排到它们后面。同一次 `schedule()` 不会再准入它们
——准入在选犠牲者之前就已经做完了。

## 5. `_preempt()` 做了什么

```python
seq.high_water = max(seq.high_water, seq.cache.length)   # 记住抢占前算到哪儿
self.running.remove(seq)
self.kv_cache_pool.deallocate_block(seq)
seq.cache = CacheConfig()        # 物理 KV 进度归零
seq.num_preemptions += 1;  self.num_preemptions += 1
self.waiting.insert(0, seq)      # 全局 FCFS，见 §4
```

抢占不是完成、也不是失败，所以：**不调用** `on_finished` / `on_token`，**不清空** `output_ids`，
**不重置**惩罚计数和随机数发生器，不记为错误。释放后不为了「看起来干净」去清零整个 KV Tensor——
块重新分配后会覆盖有效位置。

## 6. 可观测性

- 每请求 `num_preemptions`、`recomputed_tokens`、`high_water`（放在 `SequenceConfig`）。
- 全局 `scheduler.num_preemptions`。
- **实际重算 token 数**：本轮真正算的 `[start, end)` 与旧高水位 `high_water` 的重叠长度，
  不是用高水位猜的上限。

这里也踩了一个坑：一开始在**规划时**记账，但计划可能随后在补块阶段被丢弃（没跑就没有重算），
于是同一个区间被反复计入——B 实际只重算了 7 个 token，账面却是 36。
现在记账挪到 `post_step()`，只统计真的进过模型的计划：

```python
end = seq.cache.length
start = end - item["num_scheduled_tokens"]     # 跑到这里的 item 都真的算过
seq.recomputed_tokens += max(0, min(end, seq.high_water) - start)
seq.high_water = max(seq.high_water, end)
```

## 7. 验证

`benchmarks/check_step44_uncomputed.py`（§16 第 2 步的证明，26 项全通过）：

| 检查 | 结果 |
|---|---|
| 独占运行拿到 4 个 token | PASS `[37, 36, 46, 3]` |
| 重放 `prompt + 前 k 个 output` 恰好只多出 1 个**新** token（k=0..3） | PASS 四次全对 |
| 换一个输出非重复的模型重放，同样只多 1 个 | PASS |
| 分块 prefill 中间块：`cache.length` 只推进到预算、无 output | PASS 8 / `[]` / 剩 4 |
| prompt 算完后 `all_token_ids == prompt + output`，未计算数为 1 | PASS |
| `is_ready_for_next_token` 认得出这个状态 | PASS |
| **legacy 等价**：step43 vs step44(`None`)，5 组 block_size/容量 × prefix cache 开关 | PASS 10/10 |
| legacy 下 `on_token` 事件序列一致 | PASS |

`benchmarks/check_step44_preemption.py`（§10 红线，全部通过）：

| 检查 | 结果 |
|---|---|
| 容量充足：0 次抢占，输出与承诺式基线逐 token 相同 | PASS |
| 4 块池子跑两条各 8 输出的请求：确实发生抢占 | PASS 1 次（B 重算 7 token） |
| 每条请求最终输出与**它独占引擎运行**完全一致（CPU/FP32 精确参考） | PASS A/B 各 8 个 token 逐位相同 |
| 实际重算 token 数 > 0 | PASS 两条 7 / 四条 16 |
| 固定 seed 随机采样：被抢占 + 重算后仍与独占运行一致 | PASS |
| 惩罚计数（repetition/presence/frequency）不被重置 | PASS |
| `on_token` 拼接 == 最终 `output_ids`，`output_index` 连续；旧 token 不重复通知 | PASS |
| 单独不可行的请求仍明确失败（不靠抢占无限重试），同批可行请求照常完成 | PASS |
| 4 条请求在 5 块池子里全部完成，连续释放多个尾部犠牲者 | PASS 2 次抢占 / 重算 16 token |
| 完成后 `running`/`waiting` 均空、块活动引用全 0、承诺额度归零 | PASS |
| 零输出预算请求：无 token 通知、立即完成、池子干净 | PASS |
| **逐步**账本：每一步池级/每请求承诺额度都是 0、块引用不为负（两条 / 四条 / FCFS 场景） | PASS 9 项 |
| 全局 FCFS：A–E 一起到达，首次抢占后 `waiting == [B, C, D, E]` | PASS |
| 全局 FCFS：完成顺序 `A, B, C, D, E` | PASS |

### 验收方的不变量专项（首次验收 0 / 3 → 现在 3 / 3）

`verify_step44_invariants.py`：

| 用例 | 首次验收 | 现在 |
|---|---|---|
| `recompute_promise_counters_stay_zero` | FAIL | PASS |
| `preempted_requests_keep_global_fcfs_order` | FAIL | PASS |
| `admission_is_not_zero_progress` | FAIL | PASS |

另有 `fuzz_step44_cpu.py`：120 组随机 CPU 负载全部有界完成、每请求输出与独占引擎一致。

### 回归

验收方第四十三关回归脚本（`verify_step43_*.py`）复制后把路径指向 `step44/`：

| 脚本 | 结果 | 脚本 | 结果 |
|---|---|---|---|
| contract | 96 / 96 | precision | 23 / 23 |
| io_contract | 46 / 46 | qwen3 | 21 / 21 |
| external | 53 / 53 | capacity | 17 / 17 |
| merge | 41 / 41 | numerics | 12 / 12 |
| selection | 32 / 32 | slots | 10 / 10 |
| features | 29 / 29 | stride | 9 / 9 |
| norm | 56 / 56 | real_attention | 56 / 56 |
| sampling | 通过 | tiles | 通过 |
| structure | 通过 | rope | 33 / 34（**既有失败**） |

rope 唯一失败项是 `越界位置被拒绝 没有报错`，与 step39/41/43 完全相同（见 §8.6），本关未动 rope。

## 8. 首次验收后的修正：三个状态不变量

首次验收（`156_第四十四关首次验收_先修三个状态不变量`）判定 **暂不通过**，三个运行时不变量没守住。
抢占、历史重放和三条 GPU 路径本身是对的，只修这三处。

### 8.1 recompute 的承诺账本变成了负数

实测前四步池级账本：`-1 → -3 → -4 → -4`。

`ensure_blocks()` 末尾无条件执行了：

```python
seq.promised_blocks -= extra
self.promised_blocks -= extra
```

承诺式模式里这是「已承诺额度换成真实块」；但 recompute 模式准入时**根本没有增加承诺**，
从 0 减 `extra` 当然变负。现有测试没发现，是因为请求释放时又「减去负数」，最后恰好回到 0
——**结束值正确，不代表运行中账本正确**。

修法：把这两行放进 `if not self.over_subscribe`，让「消耗承诺」只在承诺式路径发生。
没有用 `max(value, 0)` 把错账夹成 0——那只是把症状藏起来。

### 8.2 被抢占的旧请求排到了新请求后面

原始 FCFS 次序 `A, B, C, D, E`，首次压力下 B/C 被抢占，正确队列应是
`waiting = [B, C, D, E]`，实际却是 `[D, E, B, C]`，完成顺序也变成 `A, D, C, B, E`。

原因是我写的 `insert(len(waiting) - n, seq)` 只保住了**同一轮几个犠牲者之间**的相对顺序，
却让它们整体落在原有 waiting 的队尾。修法是 `insert(0, seq)`：从 running 尾部依次取 C、B，
每条插队首，组合起来自然就是全局 FCFS（见 §4）。没有另写排序器。

### 8.3 准入被误当成了「计算进展」

零进展规则是「本轮至少真正算一个 token，或明确完成/拒绝一条」。守卫里的
`and not self._admitted_this_step` 让「只从 waiting 移到 running、一个 token 也没算」的空轮次混了过去，
例如 `max_num_batched_tokens=0` 时第一次 `step()` 静默返回、第二次才报错。

修法两条：
1. 守卫去掉 `_admitted_this_step`（该计数器随之删除，已无人读）；
2. `Scheduler.__init__` 直接拒绝 `max_num_batched_tokens <= 0`。

需求给了「构造时拒绝」和「第一个空轮次就报错」两个选项，两个都做了——语义上前者更早暴露配置错误，
后者是守卫本身该有的强度。

### 8.4 自查脚本也补上了逐步检查

`benchmarks/check_step44_preemption.py` 现在每一步都采一次账本快照
（池级承诺、每请求承诺、块引用是否为负），而不只看结束状态——**正是「结束值正确」骗过了我自己**。
另外加了与验收方同参数的 FCFS 用例：A–E 一起到达、`max_num_seqs=3`、4 块池子，
断言首次抢占后 `waiting == [B, C, D, E]`，且完成顺序为 `A, B, C, D, E`。

### 8.5 顺带删掉一个永不生效的参数

`ensure_blocks()` 淘汰闲置缓存时写的是 `_evictable_block_indices(exclude=set(seq.cache.block_table))`。
这个 `exclude` 是多余的：`seq.cache.block_table` 里的块 `block_usage >= 1`，
而 `_evictable_block_indices` 的第一项筛选就是 `block_usage[i] == 0`，本来就选不中它们。

同一个参数在**准入**那边是必需的，两处不能一起删：

```python
# allocate_block：命中的前缀块此刻 block_usage 还是 0、又挂在 block_to_hash 里，
# 看起来就是「可淘汰的闲置缓存」，但它们马上要被这条请求借走。
# 不排掉，_available_blocks() 会多算 len(matched_blocks) 个可用块（need 那边已经抵扣过），
# 准入因此偏松——这是第 40 关修的那个 bug。
if not self.over_subscribe and need > self._available_blocks(exclude=set(matched_blocks)):
```

实测两个调用点：48 组负载里 `ensure_blocks` 侧非空 exclude 调用 68 次、**真正排除过块的 0 次**；
`allocate_block` 侧在前缀命中场景下 4 次调用里 1 次真的排除了（`exclude=[0,1]`，
带 exclude 得 `[]`、不带得 `[0,1]`）——少了它就会把马上就要借的块算成可用。

删掉后自查、不变量 3/3、CPU 压测 120、GPU 三路径全部不变。**无行为改动。**

## 9. 遗留

1. **只做了重算式抢占**，没有 swap/CPU offload、优先级抢占、取消。
2. **prefix-aware recovery 没做**：`"recompute"` 强制关闭 prefix cache，恢复时从 0 重算，
   不能先命中可用前缀少算一点。这是需求 §15 列的下一步。
3. **没有防抖**：同一请求可能被反复抢占、刚重算一段又被抢。本阶段只保证可观测
   （`num_preemptions` / `recomputed_tokens`），不做冷却或公平性调整。
4. **不承诺总吞吐提升**。抢占让更多请求能进入并调整谁先拿资源，但重算会浪费计算；
   FCFS 且同优先级时对延迟的改善也可能很小。性能测量按需求 §12 由验收方分三层做。
5. **零进展守卫未改**：每次 `step()` 仍要求至少算 1 个 token 或明确结束/拒绝一条请求；
   「只是把 A 放到 waiting、把 B 拿到 running」不算进展。找不到犠牲者时本轮不排该请求，
   若整轮什么都排不出来仍会明确报错，不静默旋转。
6. **rope 的既有失败**（33/34）与本关无关：检查脚本要求越界 position 抛异常，
   而第四十关的需求 §7 明确说不要加这个检查、只记录后果。两者冲突，维持原样未改。
