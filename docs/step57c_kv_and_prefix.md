# step57C：真实 KV 控制面、前缀缓存与抢占恢复

- 对应代码：`step57/core/{kv_cache_utils,block_pool,single_type_kv_cache_manager,kv_cache_coordinator,kv_cache_manager}.py`
  （KV 这条链重写），`core/sched/{scheduler,request_queue}.py`（抢占与 trace），
  `request.py`（hash 挂载点），`config.py`（前缀缓存开关生效），`step57.py`（`--scheduler-trace`）
- 包摘要 SHA256：`82df12a5293c31d2…`（53 个 .py / 5459 行；口径 = 包内 `*.py` 按相对路径排序，
  每个文件取自身 sha256，拼成 `名字\0哈希\n` 再取 sha256）
- 验收脚本：`benchmarks/check_step57_{block_pool,prefix_cache,preemption}.py`
  （对应需求里点名的 `test_block_pool.py` / `test_prefix_cache.py` / `test_preemption.py`）
- 可打印轨迹：`Scheduler.trace` + `Scheduler.format_trace()`，示例入口
  `python step57/step57.py --scheduler-trace "同样的问题" "同样的问题"`
- 参考实现：本机 `vllm 0.28.0`（`v1/core/{block_pool,kv_cache_utils,single_type_kv_cache_manager,
  kv_cache_coordinator,kv_cache_manager}.py`、`v1/core/sched/scheduler.py`）

## 0. 需求大概

197（KV 控制面与物理存储）+ 200 §57C：把 57A 的**临时块分配器**换成正式管理链，接上
**前缀缓存、共享/逐出、priority、重算式抢占、abort**。交付三个测试与一个可打印的
`scheduler_trace`。判定点：计划撤销返还预算、原子失败、最后请求清理、恢复整表替换、
共享块不覆写、hash 发布边界。**"旧特殊公平策略不是通过条件"**——本关用 vLLM 的主线
（先 running 后 waiting），不搬 step4x 那套全局轮转/`resume_blocker`。

## 1. 改动内容（五层链，一层一个问题）

| 文件 | 对应 vLLM | 它回答的问题 |
|---|---|---|
| `kv_cache_utils.py` | `v1/core/kv_cache_utils.py` | 块的元数据长什么样；`ref_cnt == 0` 的块怎么排成 O(1) 摘除的双向链表；**块 hash 怎么算**（链式、稳定、带 salt） |
| `block_pool.py` | `v1/core/block_pool.py` | 分配/引用/释放/发布四个动作的**顺序**；同 hash 多个物理块怎么登记；容量口径 |
| `single_type_kv_cache_manager.py` | `v1/core/single_type_kv_cache_manager.py` | 请求 → 逻辑块；"还差几个块"的核算（含零引用命中块）；发布到哪；命中怎么查 |
| `kv_cache_coordinator.py` | `v1/core/kv_cache_coordinator.py` | **KV group** 这一层的语义（本关只有一个 group，所以是退化实现但不是空壳） |
| `kv_cache_manager.py` | `v1/core/kv_cache_manager.py` | 面向请求的接口：`get_computed_blocks` / `allocate_slots` / `cache_blocks` / `free` / `get_blocks` |
| `core/sched/scheduler.py` | `v1/core/sched/scheduler.py` | 什么时候**抢**（分配失败）、抢谁（victim）、怎么**撤计划**、什么时候**发布**、什么时候放 waiting 进来 |
| `core/sched/request_queue.py` | 同上（内部工具） | 加 `request_ids()`：priority 队列是堆，trace 要按出队顺序列出等待者 |

## 2. 设计要点

### 2.1 三步闭环：分配 → 引用 → 释放 → 发布

    分配 get_new_blocks  取队头；**先清掉它自己身上的 hash 登记**（马上要覆写）；ref_cnt = 1
    命中 touch           0 → 1 时**从空闲队列摘出**（不能再被分走）；ref_cnt += 1
    释放 free_blocks      ref_cnt -= 1；到 0 才回队列：无 hash → **队头**（复用优先）、
                         有 hash → **队尾**（LRU 淘汰）
    发布 cache_full_blocks  给**完整块**打 hash 并登记（此后别人能命中）

顺序写反会怎样，我在代码里都写了注释，其中最隐蔽的是"先分配再清登记"：那样新请求会读到
上一份已经被覆写的 KV，而 `set_block_hash` 的断言（一个块只有一个 hash）根本来不及报错。

**引用计数的语义**：数的是"有多少条活请求在用"。到 0 **不等于**内容作废——带 hash 的块仍是
可复用缓存，直到被分配出去。所以"结束之后池子里还剩一堆零引用带 hash 的块"不是泄漏（197 §7）。

### 2.2 hash 计算与发布是两件事（197 §4）

    算 hash      确定 token 一凑满块就算（`BlockHasher` 增量、链式）；**半块不算**
    发布 hash    只有"完整块 + KV 确实写完"才登记进索引

本关的显式教学差异：**发布发生在结果处理之后**（`update_from_output` 里调
`cache_blocks(request, num_computed_tokens)`），而不是 vLLM 那样在 `allocate_slots` 里顺手发布。
理由是安全性判据最直白——本轮的 KV 在 forward 之后才成立。代价是放弃了一些更早的命中机会
（性能差异，不是正确性差异）。发布上限 `floor(min(num_computed_tokens, num_tokens) / block_size)`
两头都夹住了：`num_computed_tokens` 可能超过"已提交历史"（本轮算完、采样还没回来），
不夹就会把不存在的 token 算进块里。

abort 走 `_free_request(publish=False)`：取消的请求不把自己的半截历史发布出去。

### 2.3 命中粒度：留一个 token 给 logits

`max_cache_hit_length = request.num_tokens - 1`（vLLM 同款）。全命中意味着"这段 token 一个都
不用算"，可**下一个 token 的 logits 恰恰来自最后一个位置的 hidden**——凭空拿不到。按块粒度，
实际会保留一整个块：8 个 token、块大小 4 时只命中 4 个 token，而不是 8 个。

命中块被 `touch` 之后就不再是"空闲可用"的了，所以容量核算必须把它算进去：

    num_blocks_to_allocate = max(需要 - (命中 + 已有), 0) + 命中里 ref_cnt == 0 的块数

漏掉后面那一项，就会出现"检查时说够、分配时不够"——本关的用例专门构造了这个边界
（2 个空闲块、其中 1 个是零引用命中块、新请求要 3 个块），断言 `allocate_slots` 返回 `None`
且状态一个字节不变，而不是先通过再在分配时抛错。

### 2.4 抢占：victim、撤计划、退预算

    running 循环里分配失败 → 挑 victim → 从 running 摘掉 → **撤销它本轮的记录**
    （num_scheduled_tokens / 新增块 / spec 计划）并把预算退回来 → free 它的块
    → status = PREEMPTED、num_computed_tokens = 0 → 放回 waiting **队首** → 重试分配

victim 选择：FCFS 取 `running[-1]`；priority 取 `max(running, key=(priority, arrival_time))`
（数值大 = 优先级低 = 先让路）。**FCFS 下 victim 永远是队尾**，所以"victim 已经在本轮计划里"
这条路径只有 priority 会走到（victim 可能排在当前请求前面，这时还要把循环下标回退一格）——
用例分别覆盖了两条路径。

被抢占请求保留：prompt、已提交输出、优先级、采样配置、**块 hash 链**（它是 token 历史的函数，
不随抢占改变）。恢复时 `get_computed_blocks` 可能命中仍留在池子里的完整块，所以不必从位置 0
全部重算。

### 2.5 本轮发生过抢占就不接纳 waiting（196 §5.6）

否则刚释放的块立刻被新请求吃掉，被抢占的请求永远排不回来。这条在 trace 里很显眼：
`preempted` 非空的那一轮，`scheduled` 里只有 running，`waiting` 还排着队。

### 2.6 恢复的协议语义：整表替换

`CachedRequestData.new_block_ids` 对**普通续跑**是"新增块"（执行侧追加），对
`resumed_req_ids` 里的请求是"**整张**表"（执行侧替换）。判别靠集合本身，不靠 `None`/空列表。
执行侧那一侧的实现与用例在 57B 就位（`gpu_model_runner._update_states` 的 resumed 分支），
本关只是**开始真的产生 resumed 请求**。

### 2.7 scheduler_trace

`Scheduler.trace` 每轮记一条：`scheduled` / `hits` / `preempted` / `running` / `waiting` /
`free` / `cached`，`format_trace()` 排成表。它是"不运行模型也能看懂调度器"的交付物，也是本关
用例的断言来源（比如"本轮因抢占不接纳 waiting"）。

## 3. 与 vLLM 的差异账本

| 差异 | 原因 / 后续 |
|---|---|
| **发布时机在结果之后**（vLLM 在 `allocate_slots` 里发布） | 197 §4 允许的显式教学差异：本关不实现"尚在本轮执行中的块被复用"，那条要求发布与执行顺序严格配合 |
| 没有 `null_block`（vLLM 会从空闲队列拿走一个） | 没有滑窗/CoW，`num_gpu_blocks` 个块**全部可用**；用例里明确记了这条容量口径差异 |
| `get_computed_blocks` 只返回两项（vLLM 还返回 `shared_prefix_boundary`） | 那是稀疏保留的滑窗/混合模型才需要的交界点，本关单 group 全注意力用不到；197 要求记录这一点 |
| 只有 `FullAttentionManager` 一个具体类，没有抽象基类 | 196/197 明确说不要为不存在的混合模型先写抽象；要加滑窗时再提基类 |
| 没有 partial-tail / fine-grained hash / CoW | 只做完整块路径：一个块要么没 hash，要么是"整块算完且已发布" |
| 没有 KV 事件（`BlockStored`/`BlockRemoved`）、metrics 收集器 | 那是给外部网关/监控用的 |
| `preempted_req_ids` 不进协议包 | 执行侧不需要它（恢复语义全靠 `resumed_req_ids` + 整表），vLLM 有它是给 connector/事件用的 |
| `Scheduler.trace` 是本关的教学扩展 | vLLM 用日志与 metrics；这里做成结构化记录，测试与示例共用 |
| 没有 `reset_prefix_cache` 的调度器入口 | 池子上的方法实现了（测试用），引擎级入口（切换权重时失效缓存）不在本关范围 |

## 4. 验证

| 脚本 | 项数 | 覆盖 |
|---|---:|---|
| `check_step57_block_pool.py` | 28 | 空闲队列不变量（链表 vs 计数器自洽、摘除非成员报错）、分配/touch/释放三个动作、非缓存块进队头 vs 带 hash 进队尾、同 hash 多块不误删、发布只覆盖新增段且幂等、两次失败原子性（状态指纹逐字节一致）、无 null block 的容量口径、结束后的引用对账 |
| `check_step57_prefix_cache.py` | 25 | hash 增量/链式（改头变全链、只改尾不动头）/稳定/salt/未满不算，发布边界（半块不进、上限夹 min(computed,num_tokens)）、命中留一个 token、首次接纳即命中、共享块 ref_cnt=2 且新 token 全部落在新块、容量核算含零引用命中块、逐出后不再命中、**开关缓存输出逐 token 一致**、abort 不发布 |
| `check_step57_preemption.py` | 23 | FCFS/priority 两种 victim 选择、抢占动作（进度归零/状态/计数/队首/保留历史与配置）、计划撤销与预算退回（priority 专属路径）、抢占轮不接纳 waiting、恢复走 resumed + 整表替换 + 输出不丢、恢复命中缓存块、**真模型小池子端到端与不抢占时逐 token 一致**、引用账目收尾、`format_trace()` 可打印 |

九个脚本全部通过（共 225 项）。前缀缓存的正确性证据里最硬的一条是"开/关输出一致"：
同一批请求跑两遍，生成的 token 逐 token 相同——缓存只该改变速度。

真实模型演示（本机 Qwen3-1.7B，bf16，CUDA；19 个 prompt token 的同一句话问两遍）：

```text
step scheduled                    hits           preempted    running            waiting        free cached
   1 {'q0': 19}                                               ['q0']             ['q1']           10      0
   2 {'q0': 1, 'q1': 3}           {'q1': 16}                  ['q0', 'q1']       []                9      1
```

第 2 轮 q1 命中 16 个 token（= 一个块），只补算 3 个；两条请求的文本输出完全相同。

**没有跑**性能测试（本关只做功能与不变量验证）：块池与命中查询都是纯 CPU 记账，
`--scheduler-trace` 里的 `free/cached` 是状态快照，不是性能指标。

## 5. 接口变化与遗留

- `KVCacheBlocks.blocks` 从"块号列表"变成 `list[KVCacheBlock]`；`get_block_ids()` 仍然产出
  新 list of int（跨执行边界的快照不变），"没有新增块"仍是 `None`。
- `KVCacheManager(cache_config, max_model_len=None)`：不传 `max_model_len` 表示**不做上下文
  夹取**（只有纯分配用例该这么用）；`EngineCore` 传真实值。
- 新增可导入名字：`step57.core.kv_cache_utils.{KVCacheBlock, FreeKVCacheBlockQueue,
  BlockHashToBlockMap, BlockHasher, hash_block_tokens, init_none_hash}`、
  `step57.core.block_pool.BlockPool`、`step57.core.single_type_kv_cache_manager.FullAttentionManager`、
  `step57.core.kv_cache_coordinator.UnitaryKVCacheCoordinator`。
- `CacheConfig.enable_prefix_caching=True` 从"构造时报错"改为**真的生效**（块 hash + 引用计数 +
  LRU 逐出）。默认仍是 `False`；示例入口默认打开，用 `--no-prefix-caching` 关闭。
- `RequestQueue` 增加 `request_ids()`（trace 用）；`Request` 增加 `attach_block_hasher()`。
- **遗留**：完整采样与惩罚（57D）、投机（57E）；字符串 stop；KV 事件的对外接口；
  `reset_prefix_cache` 的引擎级入口；多 KV group（本关所有地方都写死第 0 组）。
