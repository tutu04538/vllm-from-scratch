# step48：KV 缓存与调度的行为不变重构

从 `step47/` 复制。`step47/` 原样保留，未改。

## 结论

**这一关只是重构，不加功能、不改行为。** 目的是给下一关的投机解码腾出干净边界——
那个功能要求「一次计划多个 token，验证后可能撤回部分 KV」，现在就加会同时改动三处混杂的分支。

```text
请求状态：我已知/已计算了什么？                 request.py
Scheduler：这一轮谁运行、各运行几个 token？      scheduler.py
KVCachePool：块在哪里、谁引用、哪些完整块可复用？ cache.py
Engine/Model：执行 forward、采样并提交真实输出   engine.py
```

依赖方向 `request → cache → scheduler → engine`，无环。

## 三处改动

**1. `request.py`（新增）** — `CacheConfig` + `SequenceConfig` 搬出 `cache.py`。
`from step48.cache import SequenceConfig` 照旧可用（重导出，实测是同一个类对象），
搬文件本身不破坏任何旧使用者。

**2. `KVCachePool` 先计划、后提交**

```python
plan = self._plan_admission(seq)     # 只读；None = 暂时不够，抛 InfeasibleRequest = 永远不行
if plan is None: return False
self._commit_admission(seq, plan)    # 从这里才改引用与计数
```

补块同理。**计划阶段绝不为了「试一试」先淘汰**——现在由函数边界保证，不再只是注释。

**3. `Scheduler` 拆成阶段**

```python
schedule(): _begin_step → _finish_zero_budget_waiting → _admit_waiting
            → _plan_tokens（草稿）→ _reserve_blocks（定稿）→ _check_progress
post_step(): _publish_computed_blocks → _update_recompute_metrics → _finish_completed_requests
```

两个关键顺序都保留：**先计划、再补块**；**先 forward、再发布**。

## 怎么证明只是重构

| 检查 | 结果 |
|---|---|
| `benchmarks/diff_step47_step48.py`：同 seed/请求/到达时刻**逐步对照** | **72 项全通过** |
| — 逐步比较队列顺序、本轮计划、输入 token、每请求计数、承诺与块引用 | 完全一致 |
| `benchmarks/check_step48_kv_plan_commit.py`：失败路径**无副作用** | **21 项全通过** |
| step47 自查、step45/46 对照自查、step44 三个不变量 | 全过 |
| CPU 随机压测 120 组 / GPU 三条路径 | PASS |
| 既有 18 个回归脚本 | 全过 |

覆盖 fcfs/priority、prefix 开/关、容量压力、正/零命中、动态到达、`budget=1`、
`max_num_seqs=1`、随机采样与惩罚项。

复现：`benchmarks/diff_step47_step48.py`、`benchmarks/check_step48_kv_plan_commit.py`。

**本关没有新的性能假设，不做性能测试。**

完整说明见 [`docs/step48_refactor_kv_scheduler.md`](../docs/step48_refactor_kv_scheduler.md)。
