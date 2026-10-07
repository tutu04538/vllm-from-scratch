# 第 70 关对齐记录：异步调度占位符与 GPU 结果回传

需求：`投机解码完整需求/070_异步调度占位符与GPU结果回传.md`。上游基线：本机 `vllm==0.28.0`。
本关状态：**状态机与流水线骨架构架完成并自检；端到端（`async_scheduling=True` 跑真引擎）
按"未接线"明确拒绝**——拒绝本身是一条验收项，理由与实测证据见 §6。

---

## 1. 交付物

| 文件 | 内容 |
|---|---|
| `minivllm/core/sched/async_scheduler.py`（新） | `AsyncScheduler(Scheduler)`：只覆写两个钩子（`_update_after_schedule` 加占位、`_update_request_with_output` 结账）+ `publish_bound` |
| `minivllm/engine/core.py`（改） | `step_with_batch_queue()`（有界批队列、非阻塞执行/采样、交付边界取结果、结构化输出的 deferred 采样）、`post_step` 异步下不取草稿 |
| `minivllm/worker/gpu_model_runner.py`（改） | `AsyncGPUModelRunnerOutput`（侧流非阻塞 D2H + pinned 缓冲 + 事件 + 幂等 `get_output()`）、`wait_for_pending_async_output()`、`sample_tokens(non_block=...)`、失败态 |
| `minivllm/executor/uniproc_executor.py`、`worker/worker.py`（改） | `non_block` 三层同参、`supports_async_scheduling()`、Future 包装 |
| `minivllm/config.py`、`request.py`、`core/sched/scheduler.py`（改） | `SchedulerConfig.async_scheduling` + `resolve_async_scheduling()`；`Request.num_output_placeholders / num_stale_output_tokens / num_in_flight_tokens`；占位修正过的调度公式（`num_new_tokens` 加占位、`num_scheduled_spec_tokens` 减占位）、stale 抵扣、发布上界、抢占清零、`num_output_tokens + 占位` 上报、停止请求也要从 waiting 摘掉 |
| `tests/step70/test_async_spec_state.py` | 9 项：占位加/结账、公式、截断、抢占 stale、发布上界、同步不受影响、配置边界 |
| `tests/step70/test_async_buffer_lifetime.py` | 5 项：构造即发拷贝、交付前不读、幂等、释放、门闩不阻塞调用方 |
| `benchmarks/check_step70_async.py` | **17 项**脚本式验收（A 配置 / B 状态机 / C 交付句柄） |
| `docs/results.json → step70` | 命令、设备、哈希与真实执行轨迹 |

包摘要（94 个 `*.py`）：见 `docs/results.json` → `step70.results.package.sha256`。

---

## 2. 路径映射与三态矩阵

| 本项目 | 上游 | 状态 |
|---|---|---|
| `AsyncScheduler._update_after_schedule` | `v1/core/sched/async_scheduler.py:17-45` | 一致（占位 `+= num_sampled_tokens_per_step + cur_num_spec_tokens`、`spec_token_ids` 设为占位列表、prefill 块跳过）；少了 `pending_structured_output_tokens` 与 PP 的 `next_decode_eligible_step` |
| `AsyncScheduler._update_request_with_output` | 同文件 `:47-70` | 一致（`is_stale` 不扣占位 + 断言非负；`cache_blocks` 用减过占位的上界——本仓库把它放进 `publish_bound` 一处） |
| `EngineCore.step_with_batch_queue()` | `v1/engine/core.py:624-738` | 一致（有界队列、先排队再取结果、deferred 结构化输出）；少了 DP/PP 协调、`_process_aborts_queue`（本仓库 abort 直接走 `finish_requests`，`update_from_output` 会跳过已结束请求） |
| `AsyncGPUModelRunnerOutput` | `v1/worker/gpu_model_runner.py:291-397` | 一致（侧流非阻塞 D2H、pinned 缓冲、blocking 事件、`get_output()` 等事件）；差异：我们的采样结果解析/记账也在 `get_output()` 里（上游在 GPU 侧记账），因此多了"幂等交付"这条保护 |
| `Request.num_output_placeholders` / `num_stale_output_tokens` / `num_in_flight_tokens` | `v1/request.py` | 一致（含抢占 `num_stale_output_tokens = num_in_flight_tokens`） |
| 调度公式（`num_new_tokens` / `num_scheduled_spec_tokens` / 上报 `num_output_tokens`） | `scheduler.py:556-565, 705-716, 1513-1519` | 逐行对齐（占位参与三处算术） |
| —— 执行侧"行起点"校正 | 上游 `spec_decode/utils.py::update_num_computed_tokens_for_batch_change`（GPU 侧） | **本仓库用 CPU 镜像做**（`num_tokens_no_spec - 1`），因为输入组装在 CPU 上；这也是端到端未接线的原因（§6） |

**三态矩阵**：

| 能力 | 上游 | 本项目 |
|---|---|---|
| 占位符状态机（预留/确认/stale）+ 批队列 + 异步交付 | 支持 | **支持**（单测 + 探针 17 项覆盖） |
| `async_scheduling=None` 默认推断 | 兼容时默认**开** | **默认关**（差异 §6.1） |
| `async_scheduling=True` 端到端（非投机） | 支持 | **明确拒绝**（差异 §6.2） |
| `async_scheduling=True` + 投机 | 支持（草稿留在 worker） | **明确拒绝**（同上） |
| `async_scheduling=True` + 前缀缓存 | 支持 | **明确拒绝**（§6.3） |
| 结构化输出 + 异步的 deferred 采样路径 | 支持 | 代码已就位（`deferred_scheduler_output`），随端到端一起未启用 |

---

## 3. 关键机制

### 3.1 为什么要有占位符

异步调度的收益是"CPU 不等 GPU 结果就排下一轮"，代价是"排下一轮时不知道上一轮产出了什么"。
好处要拿到、又不能猜错，于是引入**预留**：

```
同步：排(t) → 跑(t) → 采(t) → 等结果 → 提交(t) → 排(t+1)
异步：排(t) → 跑(t) → 采(t) → 入队 → 排(t+1) → 跑(t+1) → … → 队列满时回头提交(t)
                                  ↑ 此刻不知道 t 产出了什么，先按最多 1+K 个占位
```

占位参与三处算术（都是上游的原式）：

```
num_new_tokens        = num_tokens_with_spec + num_output_placeholders - num_computed_tokens
num_scheduled_spec    = num_new_tokens + num_computed_tokens - num_tokens - num_output_placeholders
上报执行侧 num_output = num_output_tokens + num_output_placeholders
发布上界              = num_computed_tokens - num_output_placeholders
```

### 3.2 三元状态：预留 / 已确认 / 作废

| 状态 | 谁产生 | 下游怎么用 |
|---|---|---|
| 预留（placeholder） | `_update_after_schedule` | 只用来算预算/行数；**不进**用户输出、**不进** prefix 发布、不参与停止判定 |
| 已确认（committed） | `_update_request_with_output` | 一切照旧（历史、停止判定、发布、用户输出） |
| 作废（stale） | 抢占时把**在飞**输出整体标记 | 结果照常交付（丢掉会扰动接受判决），但**不再改计数**（否则负占位） |

### 3.3 时序图（一页看全）

```
CPU（EngineCore.step_with_batch_queue）                GPU（Runner / 侧流）
────────────────────────────────────────────────      ────────────────────────────
schedule()  ── 用占位宽度排下一轮
execute_model(non_block)  ───────────────────────────▶ 前向（kernel 提交即返回）
sample_tokens(non_block)  ───────────────────────────▶ 采样内核
                                                       └─ 句柄构造：D2H 发到**侧流** + 记事件
appendleft((future, scheduler_output))                 （CPU 不等它）
if 队列未满:  return 空输出   ← CPU 已经推进了
（下一次调用）
schedule()  ── 再排一轮（此时上一轮结果仍未取）
execute_model(non_block)  ───────────────────────────▶ 入口先 wait_for_pending_async_output()
                                                       └─ 事件完成 → 解析 → 记账 + 提议
队列满 → pop 最早那轮
future.result() → handle.get_output()  ← **交付边界**（幂等；此处才真的要值）
scheduler.update_from_output(...)  ── 提交 token、判停、发布（上界已减占位）
post_step()  ── 异步下**不**取草稿（草稿留在执行侧）
```

---

## 4. 验收对照（需求 §4）

| 需求条目 | 落点 |
|---|---|
| 可控 fake executor + 事件门闩，证明 CPU 能推进且不提前读 buffer，不靠 sleep | `test_async_buffer_lifetime.py`（门闩挡住"昂贵后半段"，放行前断言"没做"）、`check_step70_async.py` C6/C7 |
| 全接受 / 全拒绝时占位数量闭合 | `test_async_spec_state.py`（加占位 → 结账 → 非负）、探针 B1/B2 |
| 结束于候选中间时只交付有效前缀 | `test_stop_truncates_before_closing_placeholders`、探针 B3 |
| 结果未回时 abort / 抢占后旧结果回传 / 复用 slot：无重复提交、无负占位、无错误发布 | 抢占：`test_preemption_marks_inflight_output_stale`、探针 B4；发布上界：`test_publish_bound_...`、B5；abort 与 ID 复用：**端到端未启用**（随 §6.2 一起待验） |
| seed 生命周期、异常状态与清理满足 57 的不变量 | 失败态：Runner 在 `get_output()` 里也记 `failure`（异步下记账发生在这里）；`EngineCore._step_with_batch_queue` 统一记 `failure`；同步路径回归 533 项全绿 |
| 实际 GPU 下同步/异步 greedy 一致 | **未实测**：端到端未接线（§6.2）。骨架层的一致性证据是"异步只改时机、不改实现"——结账与记账调用的是**同一段**代码（`_finish_async_output` 复用 `_bookkeeping_sync` + `_propose_draft_tokens`） |
| trace 显示重叠，区别"能运行"与"确实减少等待" | **未实测**（同上）。骨架层证据：`check_step70_async.py` C1（排队后不读结果）说明"调度不再被结果阻塞" |

---

## 5. 实测

```
pytest tests/step70                        → 14 passed（9 + 5，纯 CPU 确定性）
pytest tests/step58..70                    → 547 passed（68 关基线 505 + 69 关 28 + 本关 14）
benchmarks/check_step70_async.py           → 全部通过（17 项）
其余 check_step57_* / check_step58..69_*   → 全绿
```

---

## 6. 差异账本（含"为什么拒绝"的实测证据）

### 6.1 默认关（上游默认开）

`SchedulerConfig.async_scheduling=None` 在本仓库解析成 **False**。上游默认开，是因为它的
runner 把"上一轮采样的 token 与草稿"留在 GPU 侧、直接 scatter 进下一轮输入
（`input_ids.gpu.scatter_(..., src=input_batch.prev_sampled_token_ids)`），整条链都不需要 D2H；
本仓库的 `_prepare_inputs()` 是 CPU 向量化实现（读 `token_ids_cpu`），提议器也全部 CPU 驱动，
所以"不等结果就组装下一轮输入"这条前提不存在。

### 6.2 `async_scheduling=True` 端到端**明确拒绝**（本关最大的缺口）

`resolve_async_scheduling()` 在显式 True 时抛 `NotImplementedError`。这不是"没写"，而是
"照抄会静默算错"——实现过程中实测到的两条证据：

1. **与前缀缓存同开**：进度校正（`num_tokens_no_spec - 1`）与"命中长度"是两个来源，
   口径没对齐 → 输入里出现非法 token → `device-side assert triggered`
   （`Indexing.cu: indexSelectSmallIndex`，越界读词表）。
2. **长跑下偶发不一致**：同一份配置、多请求、跑若干轮后异步输出与同步**偶发**不同
   （实测一次 `{'A': [7, 0, 4, 4, 4, 4]}` vs 同步 `{'A': [7, 1, 1, 1, 1, 1]}`）——
   典型的"草稿值/行起点与调度器的乐观计划错位"。

要放开必须先做上游那一步：**执行侧 GPU 驻留**（`prev_sampled_token_ids` scatter +
`update_num_computed_tokens_for_batch_change` 式的进度校正）。那是 74/75 关（V2 speculator /
分块验证）那条路；本关保留明确边界，不用"偶尔错"换"看起来支持"。

### 6.3 异步 + 前缀缓存拒绝

同上第 1 条（前缀命中让"行起点"多了一个来源）。显式 True 时一并拒绝，报错信息指向本条。

### 6.4 没有 `_process_aborts_queue` / DP / PP / `drop_stale_output`

上游的这几块分别服务于多进程 abort 排队、数据并行、流水线并行与"同步恢复同一步"；
本仓库没有这些结构（单进程、无 PP/DP、无 same-step resume）。缺它们的后果：abort 会**立即**
结束请求，而 `update_from_output` 对已结束请求直接跳过（不会重复提交）——这条已在同步路径
长期成立；异步下的组合随 §6.2 一起待验。

### 6.5 `AsyncGPUModelRunnerOutput` 的"幂等交付"

上游的 `get_output()` 只允许取一次（取完即释放 device 张量）。本仓库有两个等待点
（下一步 `execute_model()` 入口、引擎的交付边界），两边都可能先到，所以 `get_output()`
把**记账**做成一次性、**结果**做成可重复读（返回缓存）。这不是放宽安全性：拷贝目标是我们
自己新分配的 pinned 缓冲，交付后不再被任何一方复用。

---

## 7. 回归与下一步

- 回归：`tests/step58..70` = **547 passed**；`check_step70_async.py` 17 项；其余脚本全绿。
- 下一步（70 关的收尾，落在 74/75 关那条路上）：执行侧 GPU 驻留（`prev_sampled_token_ids`
  scatter、`valid_sampled_tokens_count` 校正进度、草稿留在 worker）→ 去掉 §6.2/§6.3 两条拒绝，
  补"实际 GPU 下同步/异步 greedy 一致 + trace 显示重叠"两项验收。
