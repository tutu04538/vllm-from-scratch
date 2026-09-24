# step47：优先级调度与抢占

- 对应代码：`step47/`（新增，从 `step46/` 复制，入口改名 `step47.py`）
- 包摘要 SHA256：`3884f4f43a7de238…`（14 个 .py / 3130 行，验收方 `source_digest()` 口径）
- 基线：`step46/`，指纹 `1cfbcb68c5de4869…`（14 个 .py / 2987 行），原样保留未改
- **改动文件只有 4 个 + 入口改名**：

| 文件 | 改动 |
|---|---|
| `cache.py`（+10 / −1） | `SequenceConfig` 新增 `priority`、`arrival_order` 与 `sort_key = (priority, arrival_order)` |
| `scheduler.py`（+168 / −52） | ① `__init__` 新增 `scheduling_policy`、`_arrival_counter`、`num_priority_preemptions`；② `add_request` 校验 `priority` 并写排序键；③ 新增 `_ordered_insert()` / `_enqueue()`；④ 准入循环加名额抢占；⑤ 新增 `_budget_groups()` 与分层 token budget；⑥ `_preempt()` 加 `priority_preemption` 与按策略回队；⑦ 删掉第四十四关留下的 `_allocated_this_step` 与那处永不触发的候选检查（见 §2.2） |
| `engine.py`（+23 / −5） | 两个入口新增 keyword-only `scheduling_policy="fcfs"`；新增 `_check_scheduling_policy()` |
| `__init__.py`（+1 / −1） | 包 docstring 改成第 47 关 |
| `step47.py` | 由 `step46/step46.py` 改名而来，只有包名引用变化 |

**未改的 10 个文件**：`attention.py`、`model.py`、`norm.py`、`rope.py`、`sampler.py`、`sampling.py`、
`formats/` 三个 —— 模型 forward、kernel、采样、KV 池的分配/淘汰、外部格式适配一个字节没动。

## 0. 需求大概

高优先级请求晚到时，为什么不能一直排在低优先级长请求后面？本关把优先级做进**三种资源**：
运行名额、KV 物理块、本轮 token budget。语义对齐本机 vLLM 的 `priority` 模式：
**数字越小越优先，同级按到达时间**。

`fcfs` 是默认值，忽略 `priority`，调度顺序与第四十六关完全一致；`priority` 只支持
`preemption_mode="recompute"`（承诺式容量策略不支持强制让位）。

## 1. 稳定排序键

```python
@property
def sort_key(self):
    return (self.priority, self.arrival_order)
```

`arrival_order` 是调度器发的**内部递增整数**，`add_request` 时发一次，抢占与恢复都不改它。
不用外部 `request_id` 当键——引擎没有建立「ID 唯一」的约束（第四十五关的教训）。

`priority` 必须是整数且 `bool` 不算：`isinstance(priority, bool) or not isinstance(priority, int)`
在 `add_request` 里直接拒绝。

## 2. 三种资源

### 2.1 运行名额

准入循环从「只在有空名额时看队首」改成：

```python
while self.waiting:
    next_seq = self.waiting[0]
    if len(self.running) >= self.max_num_seqs:
        if not self._preempt_for_slot(next_seq):
            break
    ...
```

`_preempt_for_slot()` 只在 `priority` 模式下、且存在**严格更低优先级**（`v.priority > seq.priority`）
的 running 请求时生效，选排序键最大的那条（优先级最低、同级里最晚到达）。**同级不得互相顶替**，
否则同级请求会互相插队。`fcfs` 下它永远返回 `False`，准入循环与第四十六关逐字节一致。

这发生在**本轮计划与补块之前**（需求 §3 的事务边界），所以被顶掉的请求还没拿到本轮的 token
预算和物理块，不需要回滚任何东西。

### 2.2 KV 物理块

犠牲者候选从 `reversed(running)` 换成 `_victims_after()`：

```python
# priority：排序键严格更靠后，且本轮 forward 还没跑
later = [v for v in running if v.sort_key > seq.sort_key]
later.sort(key=lambda v: v.sort_key, reverse=True)
# fcfs：就是 running 尾部的未安排请求（第四十四关语义）
```

**同级但到达更晚的请求必须能让位**——否则两条同级请求把池子占满时会谁也动不了。
更靠前的排序键一律不碰。

#### 顺带删掉一个永不生效的分支

第四十四关在 `_make_room()` 里还有一道防线：

```python
if id(victim) in self._allocated_this_step:
    return False        # 排在它前面的都补过块了，再往前找就会回滚已记账的计划
```

本关确认它**永不触发**，于是连同 `_allocated_this_step` 一起删掉。理由是顺序不变量：

- 补块循环按 `scheduled_items` 的顺序依次调用 `_make_room()`，而 `scheduled_items`
  是按 `running` 的顺序排出来的（priority 下 `_budget_groups()` 按排序键升序）；
- `_victims_after()` 只返回排序键**严格更靠后**的请求；
- 两者方向刚好相反，所以候选犠牲者的 item 一定还没被补块循环处理过。

实测 180 组负载（两种 policy × prefix 开/关 × 池子 4~12 块 × 并发 2~4 × 3 个 seed）：
`_make_room` 调用 984 次、候选犠牲者 572 个，该分支命中 **0 次**。

删掉它还更安全：万一不变量被破坏，`_make_room` 会照常尝试抢占，而被抢占者的计划项由补块循环
开头那句 `if seq not in self.running` 丢弃——代价只是那一轮少算一条，而不是像 `return False`
那样让当前请求直接放弃本轮。

### 2.3 本轮 token budget

原来的规则是「ready（只差一个 token）的请求各留 1 个，其余给 prefill」，但不分优先级，
于是 `max_num_batched_tokens=1` 时一个低优先级的 decode 会把唯一额度永远占住。

改成**按优先级分层**分配，组内保持原规则：

```python
remaining_budget = self.max_num_batched_tokens
for group in self._budget_groups():        # priority: 按优先级升序分组
    ready_req = [r for r in group if r.is_ready_for_next_token]
    ...
    prefill_token_budget = remaining_budget - len(ready_req)
    for seq in group: ...
    remaining_budget = prefill_token_budget  # 剩余额度交给下一层
```

实测对比（`max_num_batched_tokens=1`，L 在 decode，H 后到需要 prefill）：

| 策略 | H 首次进模型的步数 |
|---|---:|
| `priority` | **1** |
| `fcfs` | 6（被 L 的 decode 饿住） |

`fcfs` 模式下只有一个组 = 整个 `running`，算法与第四十六关逐字节等价（原来的断言也保留）。

## 3. 回队与计数

`_preempt(seq, blocker, priority_preemption=False)` 多了一个标记：名额被高优先级顶掉的
单独计进 `num_priority_preemptions`，与普通容量压力区分开。

回队按策略分两种，但语义都是「按优先级和原到达序重排」：

```python
if self.scheduling_policy == "priority":
    self._ordered_insert(self.waiting, seq)   # 按 (priority, arrival_order) 插回原位
else:
    self.waiting.insert(0, seq)               # 第四十四关的全局 FCFS
```

两个列表在 `priority` 模式下都按排序键维护，所以 `waiting[0]` / `running` 的顺序**就是**
调度顺序；`fcfs` 下它们仍是到达序，与之前完全一致。

阻塞者规则（第四十五关）原样保留：被阻塞的队首仍然让整个队列等，低优先级不能越过等待中的
高优先级。

## 4. 验证（只做功能正确性）

`benchmarks/check_step47_priority.py`，**45 项全部通过**：

| 需求 §4 条目 | 检查 | 结果 |
|---|---|---|
| 1 | `max_num_seqs=1`：L 已生成 token，H 后到顶掉它并**先完成**（完成序 `H, L`） | PASS |
| 1 | 计进 `num_priority_preemptions` | PASS 1 |
| 1 | H/L 输出与独占运行一致；`on_token` 不重放旧 token、索引连续 | PASS |
| 1 | `arrival_order` 不因抢占改变 | PASS |
| 2 | `budget=1`：priority 下 H 第 1 步拿到额度，fcfs 下被饿住 6 步 | PASS |
| 3 | 同级不互相顶替名额 | PASS |
| 3 | 排在前面运行的高优先级不被更低优先级顶替 | PASS |
| 3 | 三档：后到的高优先级 `p0` 先完成，同级按到达序 `p1 → p1b` | PASS |
| 3 | 同级容量犠牲者：两条同级请求都能有界完成，先到的先完成 | PASS |
| 3 | 阻塞关系不让低优先级 C 越过等待中的 B | PASS 违规 0 |
| 4 | KV 压力（prefix 开/关）：有界完成、输出正确、块引用与承诺归零、hash 双向一致 | PASS |
| 5 | `fcfs` 与 step46 在相同负载下输出与完成顺序一致（prefix 开/关） | PASS |
| 5 | 非法 `scheduling_policy` / `priority`（bool、浮点、字符串）及时报错 | PASS |
| 5 | `priority` + `preemption_mode=None` 在构造期报错 | PASS |
| 5 | 新参数是 keyword-only 且排在最后，默认 `fcfs` | PASS |
| 1 | `from_model_dir` 同样支持 `priority`，非法 policy / 组合在构造期报错 | PASS |

### 回归

| 项 | 结果 |
|---|---|
| 既有 18 个脚本（指向 `step47/`） | 见下 |
| `fuzz_step45_cpu.py`（指向 step47） | 120 PASS |
| `smoke_step45_gpu.py`（指向 step47） | CUDA/Torch、Triton eager、Triton Graph 三条路径 14 步有界完成 |
| `priority` 模式 GPU 有界冒烟（需求 §4.5） | 三条路径各 14 步完成、`priority` 生效（完成序 `H, L`）、块引用归零 |
| step44 的三个不变量（指向 step47） | 3 / 3 |
| step45 的阻塞者自查、step46 的前缀恢复自查（新引擎换成 step47） | 全部通过（44 项） |

**本关不做性能测试**（按 `02_每次更新后的性能测量约定`）。

## 5. 接口变化与遗留

### 5.1 接口变化

**新增**：`Engine(..., *, scheduling_policy="fcfs")`、`Engine.from_model_dir(...)` 同名参数；
请求字典可选 `priority`（默认 `0`）；`SequenceConfig.priority` / `.arrival_order` / `.sort_key`；
`Scheduler.num_priority_preemptions`。

**约束**：`scheduling_policy="priority"` 要求 `preemption_mode="recompute"`，否则构造期 `ValueError`。

**未改**：`step()` 返回格式、`on_finished` / `on_token` 语义、`preemption_mode` 的取值与校验、
全部模型 / kernel 接口。`fcfs`（默认）下调度顺序与输出与第四十六关一致。

### 5.2 遗留

1. **不做 aging / 防抖**：严格优先级可能让低优先级等待很久，需求明确说本关不加。
   一条低优先级请求可能被反复抢占（`num_preemptions` 会累积），只做可观测。
2. **不做 priority preemption 之外的新策略**：没有时间片、没有公平队列、没有随机犠牲者。
3. **不抢占已经拿到本轮块的请求**（需求 §3 明说不要求走这条更难的路径）。
4. **不做 swap/offload、取消请求、异步在途批次。**
5. **性能未测**。优先级会不会降低总吞吐、低优先级被拖慢多少，等专题收尾统一测。
6. **`priority` 与 `preemption_mode=None` 的组合暂不支持**，构造期明确报错而不是静默降级。
