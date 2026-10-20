# 第七十九关：基于置信度与成本曲线的自适应验证

> 需求：`learning_notes/14_vllm_from_scratch/投机解码完整需求/079_基于置信度与成本曲线的自适应验证.md`
> 基线：本机 `vllm==0.28.0` 文件快照（2026-10-03）。执行路径：**V2 的 DSpark**（前置 74/75/77）。

## 0. 这次做了什么（两三句话）

DSpark 每轮仍然提 K 枚草稿，但 target **不再固定验证全部 K 枚**：CPU 用上一轮拷回的
**stale 置信度**估计"多验一枚能多换多少期望接受 token"，除以实测成本选出总预算 `b`；
GPU 用**当前**置信度做全局 top-k，把 `b` 个名额分给各请求的**连续前缀**，并写回真实的
`cu_num_logits` / `query_start_loc`。CPU 侧只保留一个"预算均分"的上界，真实布局只有 GPU 知道。

这一关**不是** 71 关的 batch-size→K：草稿宽度 K 不变，变的是 target 实际验证多少个草稿。

## 1. 调用链与数据流（逐轮）

```text
Scheduler
  └─ 仍按 1+K 排 token（本轮验证上一轮提的草稿；K 不变）
        ↓
GPUModelRunner.execute_model()
  ├─ gather_batch_req_state()
  │    └─ AdaptiveVerificationManager.get_num_tokens()
  │         CPU stale 置信度 → survival 前缀连乘 → 枚举 b
  │         期望接受数(b) / (draft_cost_ms[B] + verify_cost_ms[非草稿行+b])
  │         → _batch_budget = (每请求原草稿数, 每请求非草稿数, b)
  │         → BatchReqState.num_tokens 改成"裁剪后的 target 行数"
  ├─ prepare_inputs()
  │    ├─ compact_batch()         CPU 压紧：b=0 精确 offsets；中间预算只给均分上界
  │    ├─ reallocate_drafts()     GPU 当前置信度 → 全局 top-k → 每请求获准前缀
  │    │                          写回真实 cu_num_logits / query_start_loc
  │    └─ InputBatch(num_tokens=裁剪后, num_draft_tokens=b,
  │                   cu_num_logits_np_exact=是否精确)
  ├─ target forward（eager；varlen query 走 _forward_generic）
  └─ sample_tokens()
       ├─ 拒绝采样（或 block/synthetic）按真实 cu_num_logits 验证 b 枚
       └─ speculator.propose() → 用本步新草稿的置信度
            └─ record_confidences()：当前值写 GPU；D2H 到两个 stale 槽之一
```

调用链与上游一致：`record_confidences()` → `get_num_tokens()` → `compact_batch()` →
`reallocate_drafts()`；CPU 预算数组和 GPU 真实长度**不混为一份**。

## 2. 策略细节

### 2.1 survival 概率与全局 top-k

每个位置的 confidence 是"条件接受概率"（77 关的 confidence head）。前 i 枚都能被接受的
概率是前缀连乘：

```text
survival_i = c_0 × c_1 × ... × c_i
```

survival 沿位置单调不增，所以"全局 top-k survival"天然给出每个请求的连续前缀。
例（K=3）：

```text
A 置信度 [0.9, 0.8, 0.7] → survival [0.900, 0.720, 0.504]
B 置信度 [0.5, 0.4, 0.3] → survival [0.500, 0.200, 0.060]

全局 top-3 = A0(0.9), A1(0.72), A2(0.504) → A 验 3 枚、B 验 0 枚
全局 top-5 = A0,A1,A2,B0,B1            → A 验 3 枚、B 验 2 枚
```

**不能**按单点 confidence 排序后随便挑：那会选出"A 第 3 枚高、第 2 枚低"的非法前缀。
实现里的 `_assign_draft_token_budget()` 先把 `capacities` 之外的槽位打成 `-inf`，再
`topk`，最后按行求和写回 `capacities`（进来是"原有效候选数"，出去是"获准验证数"）。

### 2.2 CPU 预算：期望接受数 / 实测成本

`get_num_tokens()` 只看**上一轮已经落地**的 stale 置信度：

```text
num_sampling_requests = 真正会产出 bonus 的请求数（中间 prefill 块不算）
estimated(b) = num_sampling_requests + Σ_{i=1..b} survival_(i)   # 排序后的前 b 个
cost(b)      = draft_cost_ms[num_reqs] + verify_cost_ms[非草稿行总数 + b]
budget       = argmax_b estimated(b) / cost(b)
```

`+1` 的边界：`verify_cost_ms[非草稿行 + b]`，当 `b=0` 时就是"只有 bonus 行"的成本。
`max_draft_budget` 还受 `max_total_logits - num_reqs × bonus` 约束（拒绝采样必须能装进
一个 logits chunk）。tie 时 `np.argmax` 取**第一个**（更小预算）——这是参考实现的规则，
没有自创随机/最大预算。

手算例（测试 `test_budget_intermediate_is_hand_computable`）：

```text
A [0.9,0.8,0.7] → [0.9,0.72,0.504]；B [0.5,0.4,0.3] → [0.5,0.2,0.06]
非草稿行 = 2（每条 decode 一个 bonus）；draft=10ms；verify 在 5 行前都是 1ms
b=0: 2/(10+1) = 0.1818
b=1: 2.9/(10+1) = 0.2636
b=2: 3.62/(10+1) = 0.3291
b=3: 4.124/(10+1) = 0.3749  ← argmax
b=4: 4.624/(10+1000) ≈ 0.0046（超过第 5 行后成本陡增）
```

### 2.3 成本表：graph 内阶梯、graph 外插值

`build_cost_tables_from_curves()` 与上游逐行同义：

- **捕获上限内**：执行会补齐到下一个捕获档位，所以成本是"补齐后尺寸"的阶梯函数；
  在 pad bucket 内做平滑插值会发明不存在的边际成本。
- **捕获上限外**：不再补齐，成本连续；但只用**上限以上**的实测点插值，跨过上限本身仍是
  真实的不连续点。
- 超出最后一个实测点：按最后一段斜率线性外推，不一直卡在终值。
- 成本单调不降（profiling 噪声不能制造"多一行反而便宜"）；负成本裁到 0，verify 至少 1e-6 ms。

### 2.4 布局：前缀长度写回 GPU

`compact_batch()` 返回的每请求 token 数：

```text
budget == 全部草稿  原样返回（精确）
budget == 0         每请求只剩 bonus 行；cu_num_logits_np 精确 = [0,1,2,...]
其它                 非草稿行 + 把预算**均分**给有草稿的请求（只是上界）
```

`reallocate_drafts()` 才是真实分配：

```text
capacities = 原有效草稿数（GPU）
if 0 < budget < sum(capacities):
    top-k survival 胜者 → capacities（连续前缀）
cu_num_logits   = cumsum(capacities + bonus)
query_start_loc = cumsum(capacities + 非草稿行)
```

例：两条请求 A/B 各 3 枚、预算 3，当前置信度 A 高 B 低：

```text
capacities = [3, 0]
cu_num_logits   = [0, 4, 5]    # A: 3 草稿 + 1 bonus；B: 0 草稿 + 1 bonus
query_start_loc = [0, 4, 5]    # A: 3 草稿 + 1 非草稿；B: 1 非草稿
CPU 均分上界可能是 [0, 4, 8] 或 [0, 3, 5] —— **不能**拿来切 logits
```

`InputBatch.cu_num_logits_np_exact` 明确标记 CPU offsets 是否精确：中间预算是 `False`。
唯一需要真实行边界的消费者是 logprobs；`RejectionSampler._verify_in_chunks()` 在
"中间预算 + 用户要 logprobs"时做一次显式 D2H，取真实 `cu_num_logits` 再映射，避免把后一条
请求的 logits 算到前一条头上。普通验证/无 logprobs 不付这笔同步。

### 2.5 stale 双缓冲与一次性预算

```text
_confidence_probs        [max_num_reqs, K]  当前 GPU 值（record 写、reallocate 读）
_stale_confidences[0/1]  CPU 镜像           record 在侧流 D2H；预算读**上一轮已落地**那份
_copy_events[0/1]        blocking event     覆盖 ready 槽前先同步
_pending_resets          list[slot]         新请求/slot 复用把两代置信度重置成 1.0
_batch_budget            一次性 tuple        get_num_tokens 写、reallocate 取走并置 None
```

`add_request()` 同时重置 `_stale_confidences[当前读的那一槽]` 与 `_confidence_probs`，并
把 slot 放进 `_pending_resets`；下一次 `record_confidences()` 会把另一槽也重置。这样
取消/异常/slot 复用不会继续吃旧请求的置信度。

第一轮 decode 之前没有任何 D2H 数据，stale 槽是初始化的 1.0（上游同款）——所以第一轮
通常偏向满预算；这不是 bug，测试里明确写了。

## 3. 文件映射

| 本项目 | 上游 | 作用 |
|---|---|---|
| `minivllm/worker/gpu/spec_decode/adaptive_verification.py` | `v1/worker/gpu/spec_decode/adaptive_verification.py` | `AdaptiveVerificationManager` / `_assign_draft_token_budget` / `build_cost_tables_from_curves` / `maybe_create_...` |
| `minivllm/worker/gpu/attn_utils.py` | `v1/worker/gpu/attn_utils.py` 的两个入口 | `AttentionCGSupportInfo` / query-lens mismatch 检查 |
| `minivllm/worker/gpu/async_utils.py` | `v1/worker/gpu/async_utils.py` | `StepTimingSample` / `StepTimingEvents` / `StepTimingCollector` |
| `minivllm/worker/gpu/buffer_utils.py` | `v1/utils.py::CpuGpuBuffer` | CPU/GPU 双镜像 + 非阻塞 D2H |
| `minivllm/worker/gpu/model_runner.py` | `v1/worker/gpu/model_runner.py` | 预算接入、layout、record、真实 profiling |
| `minivllm/worker/gpu/input_batch.py` | `v1/worker/gpu/input_batch.py` | `cu_num_logits_np_exact` 标记 |
| `minivllm/worker/gpu/spec_decode/rejection_sampler.py` | 同名 | 中间预算 + logprobs 的真实 offsets 修正 |
| `minivllm/attention/backends/torch_sdpa.py` | 各后端 capability | `supports_device_cpu_query_lens_mismatch()` |
| `minivllm/config.py` | `config/speculative.py` | 只允许 DSpark；不再配置期拒绝 |
| `tests/step79/test_adaptive_budget.py` | — | 31 项：成本表/预算/top-k/compact/reallocate/双缓冲/能力 |
| `tests/step79/test_adaptive_layout.py` | — | 13 项：端到端满/0/中间、概率/block/synthetic、图、取消复用、logprobs |
| `benchmarks/check_step79_adaptive.py` | — | 23 项验收探针（含真实 profiling） |
| `docs/step79_cost_curves.json` | — | 本机实测成本曲线 + off/on 对比 |

## 4. 实测（2026-10-20）

### 4.1 测试与探针

```text
python -m pytest tests/step79 -q                 → 44 passed
    test_adaptive_budget.py                      → 31
    test_adaptive_layout.py                      → 13
python benchmarks/check_step79_adaptive.py       → 23/23
python -m pytest tests/step58 … tests/step79 -q  → 945 passed
benchmarks/check_step76_dflash.py                → 27/27
benchmarks/check_step77_dspark.py                → 28/28
benchmarks/check_step78_dflash2.py               → 31/31
```

### 4.2 手算预算（与冻结 manager 逐值）

| 场景 | 成本表 | 选出的预算 |
|---|---|---|
| 满预算 | draft 1ms、verify 恒 1ms | 6（=2 请求 × K=3） |
| 零预算 | verify = 1000×行数 | 0 |
| 中间预算 | verify 前 5 行 1ms、之后 1000ms | 3 |
| tie | 置信度全 0、成本恒定 | 0（argmax 取第一个） |
| chunk 上限 | `max_total_logits=5`、2 请求 | 3（=5−2×1） |

### 4.3 端到端布局

| 场景 | 观测 |
|---|---|
| 满预算 vs adaptive 关闭 | greedy 输出逐 token 相同：`{'a':[4,4,6,0], 'b':[1,1,1,1]}` |
| 预算 0 | `num_draft_tokens=0`，`cu_num_logits=[0,1,2]`，`query_start_loc=[0,1,2]` |
| 中间预算 3 | 真实 capacities 之和 = 3；每请求 `query_len == cap+1`；CPU 上界 `[0,4,8]` 未被当真实值 |
| 强制 A=1/B=0 置信度 | GPU 当前值给出 capacities `[3,0]`（不是 CPU 均分的 `[2,1]`） |
| 概率草稿 + 中间预算 | 仍走拒绝采样，输出正常 |
| block / synthetic + 中间预算 | 两条验证路径都正常消费真实 offsets |
| 草稿侧 FULL 图 | target eager 变长批 + 草稿图共存，布局仍正确 |
| 取消 a、新请求 c 复用 slot | c 正常产出；每轮 capacities 都是合法前缀 |
| 中间预算 + logprobs | 每个输出 token 的 logprob 条数正确；`cu_num_logits_np_exact=False` 时走真实 D2H |

### 4.4 真实 profiling（tiny 模型）

`capture_model()` 里用真实 GPU event 测 target forward 与整段 draft propose（含 K 步串行
采样 / 草稿图重放）。本次（`VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN=8`）：

```text
draft curve  (num_reqs, ms): [(1, 4.28), (2, 4.63)]
verify curve (target rows, ms): [(4, 2.18), (8, 4.84), (12, 4.60),
                                 (16, 4.15), (32, 4.68), (64, 3.73)]
```

关闭 / 开启自适应各跑 7 个 decode 步（同一组 prompt、greedy、8 token）：

```text
            提议数   验证数   每 decode 步均值
adaptive off  42       42        9.78 ms
adaptive on   42       42       14.89 ms
```

⚠️ 这组 tiny 曲线下 manager 选了**满预算**，所以提议/验证数相同；时间差是 tiny 模型噪声
（on 反而更慢），**不能**据此给出加速比。预算 0 / 中间的裁剪路径由 §4.2/§4.3 的功能测试
覆盖；真实 DSpark 权重下的收益属"集成待验"。

## 5. 差异账本

| 编号 | 差异 | 为什么 | 影响 |
|---|---|---|---|
| D-79-1 | V2 的 **target CUDA graph** 尚未实现，runner 调 `maybe_create_adaptive_verification_manager(require_varlen_cudagraph=False)`；默认参数仍要求 `AttentionCGSupport.ALWAYS` | 上游要求 ALWAYS 是因为它捕获 varlen decode 图；本仓库 target 走 eager 的 `_forward_generic()`，实际需要的是"不依赖 CPU query 长度"，Torch 后端已满足。**没有篡改 capability**：默认调用/单测仍按上游拒绝 `UNIFORM_BATCH` | target varlen 图接上时必须把 `_target_varlen_cudagraphs_enabled()` 改成真实判定；否则会在图里按捕获时的 query 长度切行（不报错、结果错） |
| D-79-2 | profiling 用 eager dummy step；`_dummy_run_adaptive()` 按 `1+K` 反推请求数；`_cudagraph_limit` 固定为 0 | 本仓库 target 无图，没有"补齐到捕获档位"的阶梯成本；若沿用 profile 网格最后值会凭空造 pad bucket | 成本表对 target 是"按实测点阶梯"；target 图接上后要改回真实 `captured_token_counts()` |
| D-79-3 | `set_initial_cost_curves()` 在没有 full-cudagraph 样本时回退用全部样本 | 本仓库 target 图不存在，严格按上游过滤会得到空 draft 曲线 | 有 full 样本时仍严格按上游过滤；回退只在本机 eager 路径发生 |
| D-79-4 | `set_cost_curves()` 的 TP 广播退化为"未初始化 distributed 时直通" | 本仓库单进程；上游 `get_tp_group().broadcast_object(...)` 在 83 关 TP 才需要 | 初始化了 `torch.distributed` 时仍按 rank 0 广播；单进程无行为差异 |
| D-79-5 | `CpuGpuBuffer` 放在 `worker/gpu/buffer_utils.py`，没有 `gpu_sync_allowed()` 调试钩子 | 本仓库没有上游的 GPU sync 调试守卫；`CpuGpuBuffer` 上游在 `v1/utils.py` | 语义相同：非阻塞 D2H + blocking event 同步 |
| D-79-6 | `InputBatch.cu_num_logits_np_exact` 是新增字段；logprobs 在中间预算时做一次真实 offsets D2H | 上游代码没有这个标记，若直接复用 CPU 均分上界会切错 logprobs 行 | 只影响"中间预算 + logprobs"；普通验证/无 logprobs 不增加同步 |
| D-79-7 | `AttentionCGSupportInfo` / mismatch 检查放在 `worker/gpu/attn_utils.py` | 本仓库只有一个注意力后端，不需要上游整套 selector | 检查语义与上游同名 helper 一致 |
| D-79-8 | 诊断字段 `last_scheduled_drafts` / `last_draft_budget` / `last_num_sampling_requests` 是本仓库加的 | 测试/探针要区分"提议/获准验证/实际接受"；上游没有公开计数 | 只是诊断，不参与决策；**获准验证数不是实际接受数** |

## 6. 边界与未做

- **本机没有真实 DSpark 权重**：端到端只用 tiny target + tiny DSpark；草稿质量、接受长度、
  真实加速比"集成待验"，不下载、不伪造。
- **target varlen CUDA graph 未接**：本关只验证"草稿侧 FULL 图 + target eager 变长批"共存；
  上游的 varlen decode 图属 target 图那条线。
- **TP/PP/DP**：单卡单进程；多 rank 广播、TP 局部 argmax 属 83 关，PP 在 DSpark 配置期拒绝。
- **指标**：`SpecDecodingStats`/Prometheus 属 84 关；本关只保证"获准验证数"不被误记为
  "实际接受数"（`last_draft_budget` vs 调度器 `num_accepted`）。
- **logprobs 的真实 D2H**：只在中间预算且用户请求 logprobs 时发生；这是正确性修正，不是
  性能路径。
- **synthetic**：不证明分布无损（75 关边界）；本关只验证裁剪后的布局能被它消费。

## 7. 与需求 §4 验收项对照

| 需求验收项 | 本关证据 |
|---|---|
| 人工成本表/置信度手算预算，逐值比较冻结 manager | `test_adaptive_budget.py` 的 budget 组 + 探针 B1–B6 |
| 全预算、0 预算、中间预算 | B1/B2/B3、E1/E3/E4 |
| 相同 confidence/tie、capacity 不同、零 confidence | `test_assign_budget_*`、B4 |
| prefill 混合 | B5（`last_num_sampling_requests`）、`compact_batch` 的验证请求前置断言 |
| CPU stale 与 GPU 新数据刻意不同 | `test_reallocate_uses_gpu_current_confidence_not_cpu_stale`、E 段强制置信度 |
| Graph 可变长度、logits chunk 上限 | 草稿 FULL 图测试；`max_total_logits` 上限 B6；target 图未接属边界 |
| 请求重排/取消 | `test_reallocate_is_order_independent_via_req_id_dicts`、取消/复用测试 |
| 支持检查失败 | `test_manager_rejects_*`、`test_backend_without_device_cpu_mismatch_support_is_rejected` |
| 至少一次真实 profiling + off/on 提议/验证/时间 | F1–F3；`docs/step79_cost_curves.json` |
| 不开放给所有 proposer | 配置只允许 DSpark；非 DSpark ValueError |
| 不把获准验证数误计成实际接受数 | `last_draft_budget` 与调度器 accepted 分开；文档/测试明确 |
