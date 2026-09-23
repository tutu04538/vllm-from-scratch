# step45：阻塞者感知的恢复准入

- 对应代码：`step45/`（新增，从 `step44/` 复制，入口改名 `step45.py`）
- 包摘要 SHA256：`8e9bb0ccfb0cc798…`（14 个 .py / 2929 行，验收方 `source_digest()` 口径）
- 基线：`step44/`，指纹 `9626e95dcba34f8f…`（14 个 .py / 2886 行），原样保留未改
- **改动文件只有 3 个 + 入口改名**：

| 文件 | 改动 |
|---|---|
| `cache.py`（+4 / −0） | `SequenceConfig` 新增 `resume_blocker = None`：是谁迫使本请求让出 KV。保存**请求对象引用**，不是 `request_id` 字符串 |
| `scheduler.py`（+41 / −2） | ① `__init__` 新增 `num_blocked_admissions`；② 准入循环里对被阻塞的队首跳过并 `break`；③ 准入成功时清掉 `resume_blocker`；④ 新增 `_resume_blocked()`（判断 + 失效清理）；⑤ `_preempt()` 加 `blocker` 参数并记录；⑥ `_make_room()` 把「当前要块的请求」传下去 |
| `__init__.py`（+1 / −1） | 包 docstring 改成第 45 关 |
| `step45.py` | 由 `step44/step44.py` 改名而来，只有包名引用变化 |

**未改的 10 个文件**：`engine.py`、`attention.py`、`model.py`、`norm.py`、`rope.py`、`sampler.py`、
`sampling.py`、`formats/` 三个 —— 模型 forward、kernel、采样、KV 池的分配/淘汰逻辑、
外部格式适配都没动。`Engine` 也没有新参数：这套行为是 `preemption_mode="recompute"` 的一部分。

## 0. 需求大概

第四十四关已经会正确重算，但**恢复策略太急**：刚被 A 抢占的 B，下一步可能又被准入；
A 还没完成，内存再次紧张，于是 B 又被 A 抢占。固定 5 请求、4 块池子的例子里全部完成，
却发生 **17 次抢占**，其中 C/D/E 分别被抢占 5/6/5 次。

**「输出正确」不等于「调度合理」。**

本关只加一条清楚、可解释的策略：**记录是谁迫使你让路；在它结束之前，不要急着恢复你。**

这是一种保守的教学策略：可能减少并发，不声称 vLLM 采用同样的规则，也不保证总吞吐更高。

## 1. 策略

```text
A 需要新 KV 块
  → 抢占 B，为 A 腾块
  → B 回到 waiting，并记住阻塞者是 A
```

之后准入 waiting 队首 B 时：A 仍未完成 → B 继续等，不重建 KV、不占 `running` 名额；
A 已完成 → B 按原有 FCFS 规则重新准入，从旧 `prompt + output` 恢复。

## 2. 实现

### 2.1 记录阻塞者

`_make_room()` → `_preempt()` 的调用链上把「当前是谁要块」传下去：

```python
# _make_room：seq 就是那个补不到块的请求
self._preempt(victim, blocker=seq)

# _preempt
seq.resume_blocker = blocker
```

若它后来又被另一条更早的请求抢占，这里**直接覆盖**——阻塞者始终是「这次真正使它让路的请求」，
不需要额外的历史。

### 2.2 准入时判断

```python
while len(self.running) < self.max_num_seqs and self.waiting:
    next_seq = self.waiting[0]
    if self._resume_blocked(next_seq):
        self.num_blocked_admissions += 1
        break
    ...
```

`_resume_blocked()` 既判断也清理：

```python
blocker = seq.resume_blocker
if blocker is None: return False
if blocker in self.running or blocker in self.waiting: return True
seq.resume_blocker = None      # 阻塞者已结束或失败，关系作废
return False
```

这里 **`break` 而不是 `continue`**：被阻塞的队首同样不能让更晚到达的请求越过它，
即使 `running` 还有空名额（需求 §2「一个容易漏掉的细节」）。准入本来就只看 `waiting[0]`，
所以 `break` 与既有 FCFS 语义一致。

### 2.3 为什么保存对象引用

需求明确要求：**不要只比较外部 `request_id` 字符串**，因为当前引擎没有建立「ID 唯一」的强约束。
所以 `resume_blocker` 保存 `SequenceConfig` 对象，判活也是对象成员判断（`in`，默认按身份比较）。

### 2.4 `waiting` 那一半为什么保留

判活写的是 `blocker in self.running or blocker in self.waiting`，而不是只看 `running`。
需求的条件是「**A 仍未完成**」——被抢占的 A 依然未完成（历史还在、还会回来）。

实测 73 组负载、654 次「继续等」的判定，**全部由 `running` 那半触发，「只在 waiting」0 次**：
阻塞者抢占时插的是 waiting 队首，必然排在它所阻塞的请求前面，而被阻塞的请求又只可能是队首，
两者不可能同时成立。

仍然保留，因为这一半兜的是一条**全局顺序不变量**，而且失效方式更响：

| | 不变量被破坏时 |
|---|---|
| 保留 `waiting` 那一半 | 被阻塞的队首不让任何人准入，running 排空后**零进展守卫明确报错**（守卫刻意不把 `num_blocked_admissions` 算作进展） |
| 删掉它 | B 在阻塞者未完成时就恢复——正是本关要消灭的现象，而且**悄无声息** |

对比第四十四关删掉的那个 `exclude`：那处由 `block_usage > 0` 这个**局部**不变量保证，
删掉严格安全；这里靠的是全局调度顺序，代价不对等。

### 2.5 为什么不会死锁

阻塞者只可能处于两种状态：

- **在 `running`** → 它不受阻塞规则约束，会持续算 token 直到完成 → 关系失效。
- **在 `waiting`** → 它一定在被阻塞者**前面**：抢占时 `_preempt()` 把受害者插到 waiting 队首，
  阻塞者当时在 `running`，后来若被更早的请求抢占，也是插到队首。所以队列最前面的那个
  「被阻塞者」，它的阻塞者必在 `running`。

因此链条一定终止于一条正在推进的 `running` 请求，不会出现「running 空、队首却被挡住」。
真出现这种状态，第四十四关的零进展守卫会明确报错，而不是静默旋转。

### 2.6 关系失效

阻塞者完成（从 `running` 移除）或被明确拒绝（`_fail` 先从 `waiting` 弹出再结束），
`_resume_blocked()` 下次一看就不在队列里，立刻把 `resume_blocker` 置 `None`。
被阻塞者重新准入时也会清掉——**关系已经用上了，不再保留引用**。

## 3. 可观测性

- 每请求 `seq.resume_blocker`：当前迫使它等待的请求对象（`None` 表示没有）。
- 全局 `scheduler.num_blocked_admissions`：因阻塞者未结束而跳过准入的次数。

都是**有界**的状态，不保存每轮历史日志。

## 4. 验证（只做功能正确性）

性能测试按约定（`02_每次更新后的性能测量约定`）不在本关进行。

`benchmarks/check_step45_blocker.py`，**38 项全部通过**：

| 需求 §4 条目 | 检查 | 结果 |
|---|---|---|
| 1 | A/B：两条都完成、确实抢占、确实发生阻塞跳过 | PASS（抢占 1 次、阻塞跳过 4 次） |
| 1 | A/B：**被 A 挡住之后、A 完成之前，B 不会回到 running** | PASS（第 4 步被挡、A 第 8 步结束，违规 0） |
| 1 | A/B：B 等待期间 `resume_blocker` 指向 A | PASS 共 5 步 |
| 1 | A/B：输出与独占运行一致、`on_token` 不重放旧 token | PASS |
| 2 | A–E：全部有界完成、完成序仍为 `A,B,C,D,E` | PASS |
| 2 | A–E：抢占次数 **4 < step44 的 17**（同一配置） | PASS |
| 2 | A–E：每条输出与独占运行一致 | PASS |
| 2 | A–E：同一轮多个犠牲者都以**同一个请求对象**为阻塞者，回队顺序 B 先于 C | PASS |
| 3 | 动态到达：B 被挡期间加入 F，**F 没有越过 B** | PASS 违规 0 |
| 3 | 动态到达：A 完成后按 FCFS 恢复，顺序 `A,B,F` | PASS |
| 4 | 容量充足：不抢占、阻塞跳过次数为 0、输出与 step44 一致 | PASS |
| 5 | legacy：输出与 step44 一致（prefix cache 开/关）、承诺账本与 step43 一致 | PASS |
| 6 | 固定 seed 随机采样 / 惩罚计数：被阻塞 + 重算后仍与独占运行一致 | PASS |
| 6 | 分块 prefill、单请求无残留、阻塞者不可行时不留下死锁 | PASS |
| 6 | 72 组随机配置（12 seed × 3 池子 × 2 并发）：有界完成、输出正确、无残留 | PASS |

### 抢占次数对照（同一配置、同一权重）

| 版本 | 抢占次数 | 阻塞跳过准入 | 完成序 |
|---|---:|---:|---|
| step44 | 17 | — | `A,B,C,D,E` |
| step45 | **4** | 20 | `A,B,C,D,E` |

功能正确性与全局 FCFS 都没变，反复抢占显著减少。

## 5. 接口变化与遗留

### 5.1 接口变化

**新增**：`SequenceConfig.resume_blocker`、`Scheduler.num_blocked_admissions`。
**`_preempt()` 新增 `blocker=None` 参数**（内部方法）。

**未改**：`Engine(...)` / `Engine.from_model_dir(...)` 的参数、`step()` 返回格式、
`on_finished` / `on_token` 语义、`preemption_mode` 的取值与校验、全部模型/kernel 接口。

### 5.2 遗留

1. **不是 vLLM 的策略**。vLLM 0.28 的 FCFS 路径没有这套「阻塞者」规则；这是本关为了
   先验证「能否显著减少无效反复抢占」而引入的保守教学策略，可能减少并发。
2. **没有防抖/冷却**：阻塞者结束后 B 立刻恢复；若之后又有更早的请求要块，B 可能再次被抢占。
   本关只保证「不被同一个阻塞者反复准入—抢占」。
3. **不做 priority preemption / prefix-aware recovery**：等本关状态转换与边界验收通过后再定。
4. **不做 swap/offload、取消请求、异步在途批次**。
5. **性能未测**（按 `02_每次更新后的性能测量约定`，本关不跑性能矩阵）。
   抢占次数减少是功能指标，不是吞吐结论。
