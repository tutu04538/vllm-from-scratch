# step48：KV 缓存与调度的行为不变重构

- 对应代码：`step48/`（新增，从 `step47/` 复制，入口改名 `step48.py`）
- 包摘要 SHA256：`9e684ac762d264aa…`（**15** 个 .py / 3228 行，验收方 `source_digest()` 口径；比 step47 多一个 `request.py`）
- 基线：`step47/`，指纹 `3884f4f43a7de238…`（14 个 .py / 3130 行），原样保留未改
- **本关是重构关：不加功能、不改行为**。验收标准就是「证明只是重构」

## 0. 需求大概

第四十九关要做投机解码，它要求「一次计划多个 token，验证后可能撤回部分 KV」。
现在就加，会同时改动三处混杂的分支，很难判断错误来自新功能还是旧状态管理。所以先重构。

原来两处容易看乱：

- `cache.py` 391 行：请求状态（`SequenceConfig`）、物理 KV Tensor、块引用、容量承诺、
  prefix hash/LRU、slot mapping 挤在同一文件；
- `scheduler.py::schedule()` 约 120 行：零预算完成、准入/名额抢占、token budget、
  KV 补块/容量抢占、零进展检查连成一个长函数；`post_step()` 又把缓存发布、重算统计、
  完成回收混在一起。

目标边界：

```text
请求状态：我已知/已计算了什么？        request.py
Scheduler：这一轮谁运行、各运行几个 token？  scheduler.py
KVCachePool：块在哪里、谁引用、哪些完整块可复用？  cache.py
Engine/Model：执行 forward、采样并提交真实输出   engine.py
```

## 1. 改动清单

| 文件 | 改动 |
|---|---|
| `request.py`（**新增**，86 行） | `CacheConfig` + `SequenceConfig` 从 `cache.py` 搬出，成为「一个请求自己的状态」 |
| `cache.py`（391 → 368 行） | 只留 `KVCachePool`；顶部 `from .request import ...` **重导出**旧名字；准入与补块拆成「计划 / 提交」两个阶段 |
| `scheduler.py`（410 → 445 行） | `schedule()` 拆成 5 个阶段方法；`post_step()` 拆成 3 个；`_resume_blocked()` 的实验记录移到文档 |
| `engine.py`、`model.py` 等 11 个文件 | **未改** |
| `step48.py` | 由 `step47/step47.py` 改名而来 |

依赖方向 `request → cache → scheduler → engine`，无环。`request.py` 不 import 包内任何东西。

**`from step48.cache import SequenceConfig, CacheConfig` 仍然可用**（重导出，实测是同一个类对象）；
`step48.request` 也是同一个对象。`__init__.py` 的导出没变。

## 2. 阶段划分

### 2.1 `KVCachePool`：先计划、后提交

```python
def allocate_block(self, seq):
    plan = self._plan_admission(seq)      # 只读；None = 暂时不够，抛 InfeasibleRequest = 永远不行
    if plan is None:
        return False
    self._commit_admission(seq, plan)     # 从这里才改引用与计数
    return True
```

`AdmissionPlan` 只装 `matched_block_ids` / `matched_hashes` / `required_new_blocks`。
`_plan_admission()` 保持原有顺序：算最坏逻辑块数 → 查 `all_token_ids` 的完整块命中
（最多 `len-1` 个 token，因为没有缓存 logits）→ 可行性检查（**用最坏逻辑块数**，
命中块也占物理块）。查询**不改 LRU、不加引用**。

补块同理拆成 `_plan_block_growth()` / `_commit_block_growth()`，中间用 `BlockGrowthPlan`
传递「淘汰哪些闲置块、新占哪些块」。**计划阶段绝不为了「试一试」先淘汰**——
第一版实现把这句写在注释里，现在它由函数边界保证。

按 §6 的建议把局部变量改直白：`need` → `required_new_blocks`，
`extra` → `num_missing_blocks`，`running` 快照 → `running_snapshot`。
跨模块使用的字段（`block_usage`、`promised_blocks`、`cache.length`）一律不改名。

### 2.2 `Scheduler.schedule()`：五段

```python
def schedule(self):
    self._begin_step()                   # 清空上一轮计划与完成记录
    self._finish_zero_budget_waiting()   # 只处理 max_new_tokens == 0
    self._admit_waiting()                # 名额、阻塞者、KV 准入
    planned_items = self._plan_tokens()  # 草稿：只决定谁跑几个 token
    self.scheduled_items = self._reserve_blocks(planned_items)   # 再补物理块
    self._check_progress()
    return self.scheduled_items
```

两个关键顺序都保留：**先计划、再补块**（token 预算在补块之前定），以及
**先 forward、再发布**（见 §2.3）。

`_reserve_blocks()` 用「提交清单」代替原来遍历 `list(self.scheduled_items)` 反复
`remove(item)`：

```python
for item in planned_items:
    seq = item["request"]
    if seq not in self.running:  continue      # 已被本轮选成犠牲者，计划作废
    if ensure_blocks(...):       committed.append(item); continue
    if self._make_room(...):     committed.append(item)
```

这样「实际会 forward 的是谁」一眼可见。语义与原来逐字节相同：被牺牲的请求跳过、
额度不转给别人，当前请求自己是最后候选时不排它。

### 2.3 `Scheduler.post_step()`：三段

```python
def post_step(self):
    self._publish_computed_blocks()      # 必须在 forward 之后
    self._update_recompute_metrics()     # 只统计真的跑过的计划
    self._finish_completed_requests()    # 回调、释放引用、移出 running
```

发布时机仍是**模型 forward 之后、完成释放之前**：`cache.length` 已是本轮真正写进 KV 的
进度，而采样 append 发生在它之前，所以刚采样、还没进模型的 token 绝不会被登记。
重算量仍只按实际执行的 `scheduled_items` 统计，草稿计划和被撤销的 item 不算。

## 3. 验证：证明只是重构

### 3.1 逐步对照（`benchmarks/diff_step47_step48.py`，72 项全通过）

同 seed、同请求、同到达时刻，逐步比较 `running` / `waiting` 顺序、本轮计划的
（请求、`num_scheduled_tokens`、`can_sample`）、实际输入 token、`step_done`、
每请求 `cache.length`；结束时比较 `num_preemptions`、`recomputed_tokens`、
`reused_tokens`、`high_water`、承诺总额、块引用归零、双向 hash 一致。

覆盖 9 个场景 × 2 个 seed：fcfs/priority、同批与动态到达、容量压力、
prefix 开/关、`budget=1`、`max_num_seqs=1`、高优先级后到。

**逐步计划与队列完全一致，每请求计数完全一致。**

### 3.2 KV 计划/提交边界（`benchmarks/check_step48_kv_plan_commit.py`，21 项全通过）

失败路径**一个字节都不改**（比较块引用、双向 hash、块表、`cache.length`、
两级承诺计数、`reused_tokens`、LRU）：

| 场景 | 结果 |
|---|---|
| 单请求不可行 → `InfeasibleRequest` | 抛异常，池与请求状态未变 |
| 暂时不够（承诺式）→ `_plan_admission` 返回 `None` | `allocate_block` 返回 `False`，无副作用 |
| 本轮补块失败（超卖）→ `ensure_blocks` 返回 `False` | 没有部分淘汰、没有部分分配 |
| 计划阶段挑出的可淘汰块 | 与 hash 登记的闲置块一致，**此时一个块都还没被淘汰** |
| 提交阶段 | 淘汰的 hash 双向索引同步消失，新块引用为 1 并进块表 |
| 正命中 | 查询阶段不改引用、不改 LRU、`reused_tokens` 仍为 0；**提交后才加** |
| 承诺式记账错误 | 仍在计划阶段抛出原提示，且没有部分淘汰 |

### 3.3 其余

| 项 | 结果 |
|---|---|
| `check_step47_priority.py`（指向 step48） | 全部通过 |
| `check_step45_blocker.py` / `check_step46_resume_prefix.py`（新引擎换成 step48） | 38 / 44 项全过 |
| step44 的三个不变量（指向 step48） | 3 / 3 |
| CPU 随机压测 120 组 | PASS |
| GPU：CUDA/Torch、Triton eager、Triton Graph | 各 14 步有界完成，引用归零 |
| 既有 18 个回归脚本 | 见下 |

**本关没有新的性能假设，不做性能测试**（需求 §9.4）。

## 4. 变量归属表

需求 §6 要求的表（实现按此检查过一遍）：

| 变量 | 谁持有 | 准确含义 | 只在何时修改 |
|---|---|---|---|
| `seq.cache.length` | 请求 | **已经写入 KV** 的 token 数；唯一的计算进度 | 模型准备输入时前进；命中恢复/抢占时设置 |
| `seq.cache.block_table` | 请求 | 逻辑块编号 → 物理块编号 | 准入提交、补块提交、抢占/完成重置 |
| `seq.block_hashes` | 请求 | 从块 0 起连续的 hash 链 | 命中准入、发布完整块、抢占清空 |
| `pool.block_usage[b]` | KV 池 | 物理块 `b` 的**活动请求引用数**，不是 LRU 热度 | 准入提交、补块提交、释放 |
| `pool.block_hash[h]` / `block_to_hash[b]` | KV 池 | 已登记完整块的双向索引 | 发布、淘汰（**同一次操作增删**） |
| `pool.promised_blocks` / `seq.promised_blocks` | 池 / 请求 | 未兑现的承诺数 | 承诺式准入提交、补块提交、释放 |
| `seq.high_water` | 请求 | 曾经真正计算到的最高位置 | forward 之后、抢占前 |
| `seq.recomputed_tokens` | 请求 | 再次进入模型的旧历史 token 数 | **真的 forward 之后**，不按计划计 |
| `seq.reused_tokens` | 请求 | 准入**成功借到**的完整块折算 token 数 | 准入提交后，不在查询阶段 |
| `scheduler.running` / `waiting` | 调度器 | 活动 / 等待请求，priority 下按排序键有序 | 准入、抢占、完成 |
| `scheduler.scheduled_items` | 调度器 | **本轮最终获准执行**的计划，不是草稿 | `schedule()` 结束前定稿，`post_step()` 读取 |

三个长度概念没有混用：`len(all_token_ids)`（已知）≠ `cache.length`（已计算）
≠ `len(block_hashes) * block_size`（已发布完整块的上界）。

## 5. 接口变化与遗留

### 5.1 接口变化

**新增**：`step48/request.py`（`CacheConfig`、`SequenceConfig`）、
`cache.py` 内的 `AdmissionPlan` / `BlockGrowthPlan` 与四个私有方法。

**未改**：`Engine(...)` / `Engine.from_model_dir(...)` 的参数与校验、`step()` 返回格式、
`on_finished` / `on_token`、`scheduled_items` 仍是字典、`KVCachePool` 仍是单类、
`cache.length` / `block_usage` / `promised_blocks` 的名字。
`from step48.cache import SequenceConfig` 照旧可用。

### 5.2 遗留

1. **不实现投机解码**，也不为了「看起来优雅」引入继承结构或五六个 Manager。
2. **未处理「不可行高优先级请求先造成一次多余抢占」**这个已记录的边角——需求 §3 明说
   不要把重构与行为修复混在一起。
3. **`scheduled_items` 仍是字典**，需求 §3 明说不要求改成类；本关只是把「谁能改、何时改」
   写成 `_plan_tokens()` 产出草稿、`_reserve_blocks()` 定稿。
4. **不做 CPU swap、取消请求、新调度策略。**
5. 性能未测，本关没有新的性能假设。
