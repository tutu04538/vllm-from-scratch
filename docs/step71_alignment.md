# 第七十一关：动态投机长度与调度 Graph 兼容

**状态（2026-10-08）：已实现并自检通过。** 本文件按 `docs/README.md` 的固定结构写：
0 需求大概 → 1 改动内容 → 2 设计要点 → 3 钉死状态的端到端例子 → 4 验证 → 5 实测数字 →
6 差异账本（逐条） → 7 接口变化与遗留。

---

## 0. 需求大概

低并发时"一次猜 4 枚"划算，高并发时"猜 1 枚、甚至不猜"更快——投机唯一的收益是
**用一次权重读取喂多条 token**，并发越高、这一步本来就被摊薄。所以本关给一张
**批大小 → K** 的闭区间表（例：`[(1,2,4),(5,8,1)]` 读作"批 1~2 猜 4 枚、批 5~8 猜 1 枚"），
由 Scheduler 在**每一轮**按"这一轮实际被调度的请求数"查表，值随 `SchedulerOutput` 发给
执行侧。**不自行发明"按最近接受率调 K"的策略**（需求 071 §1 明确禁止）。

顺带解决两个兼容问题：

* **full CUDA graph** 冻结了"每请求 `1+K` 行"的形状，而动态 K 逐轮改这个宽度 → 含 full graph
  的模式一律**降级成 PIECEWISE**（上游 `_maybe_override_dynamic_sd_cudagraph_mode()`）；
* **DP>1** 时各 rank 可能选出不同的 K → 分歧/死锁，所以直接**关掉这张表**、退回固定 K
  （上游 `_maybe_disable_dynamic_sd_for_data_parallel()`）。

## 1. 改动内容

| 本项目文件 | 参考路径（vllm==0.28.0 快照） | 状态 |
|---|---|---|
| `minivllm/spec_decode/dynamic/{__init__,utils}.py`（新） | `v1/spec_decode/dynamic/utils.py` L7-148 | 两个函数**逐行对齐**：`validate_and_normalize_dynamic_sd_schedule` / `build_dynamic_sd_schedule_lookup`（含 `int()` 转换与报错文案） |
| `minivllm/config.py::SpeculativeConfig` | `config/speculative.py` L181、L1489-1490 | 新增字段 `num_speculative_tokens_per_batch_size`；新增 `uses_dynamic_speculative_decoding()`；新增 `_resolve_dynamic_sd()`（校验提前到配置期） |
| `minivllm/config.py::VllmConfig` | `config/vllm.py` L910-927、L929-946、L1414-1415 | 新增 `_maybe_override_dynamic_sd_cudagraph_mode()` 与 `_maybe_disable_dynamic_sd_for_data_parallel()`，在 `_resolve_cudagraph_config()` 里**档位表之前**调用（与上游调用点同序） |
| `minivllm/core/sched/output.py` | `v1/core/sched/output.py` L267-269 | `SchedulerOutput.num_spec_tokens_to_schedule: int = 0` |
| `minivllm/core/sched/scheduler.py` | `v1/core/sched/scheduler.py` L257-263、L1255-1258、L1289 | `dynamic_sd_lookup`（`__init__` 建）；本轮 K = `lookup[len(num_scheduled_tokens)]`；写进快照；trace 记一份。属性改名 `num_speculative_tokens` → **`num_spec_tokens`**（上游名，语义 = 最大容量） |
| `minivllm/core/sched/async_scheduler.py` | `v1/core/sched/async_scheduler.py` L19-22、L52 | 占位草稿列表按**本轮 K** 重建（不是最大 K） |
| `minivllm/worker/gpu_model_runner.py` | `v1/worker/gpu_model_runner.py` L5136、L5380 | `_propose_draft_tokens()` 读本轮 K 并断言 `0 ≤ K ≤ 最大 K`，把它传给 LLM 系提议者与 CPU ngram |
| `minivllm/spec_decode/draft_model.py` | `v1/spec_decode/llm_base_proposer.py` L510、L530、L615-632 | `propose(..., num_speculative_tokens=None)`：逐轮改写实例上的 K（上游同款），**K=0 仍跑完第一遍**（同步 draft KV）再返回空草稿 |
| `minivllm/spec_decode/ngram_proposer.py` | `v1/spec_decode/ngram_proposer.py` L135-150 | `propose_drafts(..., num_speculative_tokens=None)` 透传；`assert K <= self.k` 换成带说明的明确报错 |
| `minivllm/spec_decode/{ngram_proposer_gpu,suffix_decoding,medusa,extract_hidden_states}.py` | 同上（各自的 `propose`） | **只加注释**：固定 K 的断言保留，配置期已拒绝这个组合（需求 071 §3.6） |
| `tests/step71/{spec71_helpers,test_dynamic_sd}.py`（新） | — | 21 项单测 |
| `benchmarks/check_step71_dynamic_sd.py`（新） | — | 24 项探针 |

## 2. 设计要点

### 2.1 一张表、两个名字、一个语义

```
num_speculative_tokens                     最大容量（配置）：工作区 / KV lookahead / 图 / 掩码缓冲按它开
num_speculative_tokens_per_batch_size      区间表（配置）：[(start, end, K), ...]
dynamic_sd_lookup                          稠密表（Scheduler.__init__ 建）：dense[batch_size] = K，索引 0 不用
SchedulerOutput.num_spec_tokens_to_schedule 本轮值（每轮算）：= lookup[本轮实际被调度的请求数]
```

* 表里的 K 一律被**最大容量裁剪**（`min(max_K, K)`）：表只能把 K 调小，**不能放大容量**；
* **空隙与尾部**沿用前一段 / 最后一段的 K（`[(1,16,3),(32,128,2)]` 的 17~31 是 3、B>128 是 2）；
* `len(num_scheduled_tokens)` 是"**实际被调度的请求数**"，不是 waiting+running 总数
  （需求 071 §3.2；没排上的请求这一轮既不产生草稿也不该影响 K）；
* 空轮（`num_scheduled_tokens` 为空）**不查表**，字段保持 `num_spec_tokens`（上游同一句
  `if self.dynamic_sd_lookup is not None and len(num_scheduled_tokens) > 0`）。

### 2.2 两轮时序：本轮的 K 管"下一轮的候选"，不重解释"本轮验证的旧候选"

```
轮 t   : 调度 → 查表得 K_t → 执行/验证（验证的是轮 t-1 提的候选，宽度在 scheduled_spec_decode_tokens）
         采样后提议：按 K_t 提草稿
轮 t+1 : 调度采纳轮 t 提的前缀（≤ K_t）→ 验证它，同时再查 K_{t+1} 提新草稿
```

**推论**（需求 071 §3.3 点名）：改 K **不准**回头改旧候选的长度或概率行——旧候选的宽度由
`request.spec_token_ids` 决定，q 由 `pending_draft_probs` 按 `req_id` 的行偏移对齐
（`GPUModelRunner._get_spec_decode_draft_probs()`），两者都与本轮 K 无关。
K=0 的轮完全可能**正在验证上一轮 K=4 提的 4 枚候选**（实测见 §5.2 第 3 轮）。

### 2.3 两条配置期改写（都发生在档位表之前）

| 触发条件 | 结果 | 为什么 |
|---|---|---|
| 动态表 + `cudagraph_mode.has_full_cudagraphs()` | 降级为 `PIECEWISE` + warning | full 图冻结请求数/每请求行数，而动态 K 逐轮改 `1+K`；PIECEWISE 的键只锁 token 数（69b） |
| 动态表 + DP>1 | 清空表、退回固定 K + warning | 各 rank 数据不同 → K 不同 → 形状不一致 → 集合通信分歧/死锁 |

顺序不能反（DP 先关表，第 2 条自然不再触发），且都在 `_set_cudagraph_sizes()` **之前**：
所以降级后档位表按 PIECEWISE 算、**不会**再做"取整到 1+K 的倍数"（实测 §5.4）。

### 2.4 方法边界（三态矩阵：支持 / 明确拒绝 / 不属于本关）

| method | 动态表 | 依据 |
|---|---|---|
| `draft_model` / `eagle` / `eagle3` / `mtp` | **支持** | `llm_base_proposer.propose(num_speculative_tokens, ...)`：可变宽，K=0 时**先跑第一遍**再返回 `[B,0]` |
| `ngram` | **支持** | `ngram_proposer.propose` 收 K（`assert K <= self.k`），K=0 直接空草稿（它不写 KV，没有第一遍） |
| `ngram_gpu` | 配置期拒绝 | GPU 提议者 `assert K == self.k`、返回固定宽度 `[B,k]` |
| `suffix` | 配置期拒绝 | `assert K == self.num_speculative_tokens`（长度上界由树深 + K 共同决定） |
| `medusa` | 配置期拒绝 | K 就是 head 数，逐轮改 K 等于逐轮改 head 数 |
| `extract_hidden_states` | 配置期拒绝 | 恒 K=1（它不猜 token） |
| `custom_class` | 配置期拒绝 | 上游调用形态里**不传 K** → 表对它静默无效（本仓库不静默） |
| PARD / P-EAGLE / DFlash / DSpark | 不属于本关 | 72/76/77/78 关 |

"拒绝"落在 `SpeculativeConfig._resolve_dynamic_sd()`（配置期，报错里点名上游的断言位置）；
上游是懒校验 + 运行时断言，差异见 §6.2。

### 2.5 容量口径（不随 K 重建）

工作区、KV lookahead、图档位、结构化输出掩码缓冲全部按 **`num_speculative_tokens`（最大 K）**
开一次：

* Scheduler：`num_lookahead_tokens = num_spec_tokens`（逐轮 K 只会更小，预留不用改）；
* draft 提议者：`max_num_tokens` / 块表 / `input_ids|positions|slot_mapping|query_start_loc|seq_lens`
  都在 `__init__` 开好，每轮只覆盖前 `[:num_tokens]` 切片 → 地址恒定（实测 §5.5）；
* ngram（CPU 与 GPU）：`valid_ngram_draft` 是定宽 `[max_num_seqs, k]`，只有
  `valid_ngram_num_drafts[i]` 说的那几列算数——K 变小之后**旧的宽列不许被交出去**（实测 §5.6）。

### 2.6 与异步调度（70 关）的接缝

`AsyncScheduler._update_after_schedule()` 的占位草稿宽度必须用**本轮 K**：

```
self._spec_token_placeholders = [-1] * scheduler_output.num_spec_tokens_to_schedule
request.spec_token_ids = self._spec_token_placeholders
```

用最大 K 会让"下一轮要排几行"凭空多出来 → 预算 / 占位 / 验证长度三处同时错位。
⚠️ 端到端 `async_scheduling=True` 仍是 70 关的**明确拒绝**，本关不动它（实测 §5.3 的 C6 项）。

## 3. 钉死状态的端到端例子（K 跨 4 → 0 → 2）

**配置**（tiny qwen3 GQA，`vocab=11`，2 层，`max_model_len=64`）：
`max_num_seqs=4`、`max_num_batched_tokens=32`、`num_speculative_tokens=4`（最大 K）、
表 `[(1,1,4),(2,2,0),(3,8,2)]`、`method="draft_model"`、贪心采样。三条请求 A/B/C 依次加入。

**符号**：

| 字段 | 一句话含义 | 谁改它 |
|---|---|---|
| `dynamic_sd_lookup` | `dense[批大小] = K` 的稠密表 | Scheduler 建一次后只读 |
| `num_spec_tokens_to_schedule` | **本轮**选出的 K（= 本轮采完后要提几枚） | Scheduler 每轮算，随快照发出 |
| `scheduled_spec_decode_tokens[req]` | **本轮要验证**的旧候选（上一轮提的） | Scheduler 每轮按 `request.spec_token_ids` 装包 |
| `request.spec_token_ids` | 上一轮提的草稿（本轮被采用的前缀） | Scheduler（`update_draft_token_ids`） |
| `proposer.num_speculative_tokens` | 提议者这一轮**实际**用的 K | Runner 传参 → 提议者实例 |

**不变式**：`I1` 本轮 K == `lookup[本轮被调度请求数]`；`I2` 本轮验证的候选 == 上一轮提的那份
（逐值，含长度）；`I3` 本轮提的草稿数 ≤ 本轮 K；`I4` 发布上界 ≤ draft 第一遍已同步的进度
（199 §9）；`I5` 工作区地址与 K 无关。

**逐轮**（实测 trace，`tests/step71` 的 `_staggered_dynamic` 场景）：

| 轮 | 被调度（行数） | 请求数 | `I1`：查表得 K | `I2`：本轮验证 | `I3`：本轮提议 | draft 第一遍次数 |
|---|---|---|---|---|---|---|
| 0 | A:6（整段 prefill） | 1 | `dense[1]=4` | — | A:4 | 4（第一遍 + 3 次自回归） |
| 1 | A:5 | 1 | 4 | A:4（轮 0 提的） | A:4 | 4 |
| 2 | A:5, B:6 | 2 | `dense[2]=0` | **A:4（轮 1 提的，K 已经变 0 也不重解释）** | A:0, B:0 | **1（只有第一遍）** |
| 3 | A:1, B:1, C:6 | 3 | `dense[3]=2` | — | A:2, B:2, C:2 | 2（第一遍 + 1 次自回归） |
| 4 | A:3, B:3, C:3 | 3 | 2 | A:2, B:2, C:2 | A:2, B:2, C:2 | 2 |
| 5 | A:3, B:3, C:3 | 3 | 2 | A:2, B:2, C:2 | A:2, B:2, C:2 | 2 |
| 6 | A:3, C:3（B 结束） | 2 | 0 | A:2, C:2 | A:0, C:0 | 1 |
| 7 | 空轮 | 0 | 不查表 → 4 | — | — | 0 |

算术自查：轮 2 的 A 行数 `= tokens + 采用的草稿 − computed = 1 + 4 − 0 = 5` ✓（与表里一致）；
轮 2 的 draft 第一遍次数 = 1 = **"K=0 也照样同步 KV"** 的现场证据（跳过第一遍这里会是 0）。

**每一步由谁提供**：

* `minivllm/config.py::SpeculativeConfig._resolve_dynamic_sd()`：校验表 + 方法边界（配置期）；
* `minivllm/config.py::VllmConfig._maybe_override_dynamic_sd_cudagraph_mode()`：full → PIECEWISE；
* `minivllm/core/sched/scheduler.py::Scheduler.schedule()`：查表得本轮 K、装进 `SchedulerOutput`；
* `minivllm/core/sched/output.py::SchedulerOutput`：把 K 带过执行边界；
* `minivllm/worker/gpu_model_runner.py::_propose_draft_tokens()`：读 K、断言上界、传给提议者；
* `minivllm/spec_decode/draft_model.py::SpecDecodeBaseProposer.propose()`：按 K 提草稿，K=0 仍跑第一遍；
* `minivllm/spec_decode/dynamic/utils.py`：稠密表（唯一权威）。

**本例子在本仓库能跑到哪一步**：全流程可跑，被
`tests/step71/test_dynamic_sd.py`（`test_scheduler_picks_k_by_scheduled_request_count`、
`test_proposal_width_is_this_round_k_and_verification_is_previous_round`）与
`benchmarks/check_step71_dynamic_sd.py` 的 C1~C4 项覆盖。**就地标注的边界**：
端到端异步（§2.6）与 `ngram_gpu/suffix/medusa/extract/custom_class`（§2.4）在本关**明确拒绝**。

## 4. 验证

```bash
# 本关单测（21 项）
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/step71 -q
# 本关探针（24 项，退出码非 0 = 失败）
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmarks/check_step71_dynamic_sd.py
# 全量回归（58~71 一次跑）
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/step58 ... tests/step71 -q
```

覆盖矩阵（需求 071 §4 逐条）：

| 验收项 | 落点 |
|---|---|
| 边界配置 / 空隙 / 尾部 / 超最大 K 与参考工具函数逐值比较 | `test_lookup_matches_upstream_reference`（9 张表 × 7 个上界 × 5 个最大 K，逐值 + 报错种类）、探针 A1~A6 |
| B 跨区间（含 K=4→0→2）看 SchedulerOutput / proposal / 验证长度 / q 轮次 | `test_scheduler_picks_k_by_scheduled_request_count`、`test_proposal_width_is_this_round_k_and_verification_is_previous_round`、`test_q_rows_follow_this_round_proposal_width`、探针 C1~C4 / D6 |
| K=0 数轮后恢复 K>0：draft KV 不缺历史、greedy 输出不变 | `test_k0_rounds_still_sync_draft_kv_and_output_stays_identical`、探针 D1/D2 |
| Graph 转换 / DP fallback / 显式 async 组合 | `test_dynamic_sd_downgrades_full_graph_to_piecewise`、`test_data_parallel_fallback_disables_the_table`、`test_dynamic_sd_with_explicit_async_scheduling`、探针 B1~B6 / C5~C6 |
| 工作区不随 K 重分配、旧无效列不被消费 | `test_workspace_addresses_do_not_change_with_k`、`test_ngram_dynamic_k_does_not_consume_stale_columns`、探针 D3/D4 |

## 5. 实测数字

### 5.1 与上游工具函数逐值差分

9 张表（含需求 §3 的原例、带空隙、K=0 首段、超上限、重叠、负 K、空表）×
`max_batch_size ∈ {1,2,4,8,16,64,200}` × `max_K ∈ {1,2,3,4,8}` = 315 组：
**返回的稠密表与抛出的报错种类全部一致**（探针 A1）。

需求 §3 的例子：`[(1,2,4),(5,8,1)]`、`max_num_seqs=8`、最大 K=4 →

```
dense = [0, 4, 4, 4, 4, 1, 1, 1, 1]
         ↑  B=1..4 → 4（B=3/4 是空隙，沿用前段）  B=5..8 → 1
```

### 5.2 端到端两轮时序（tiny，CPU/GPU 皆可复现）

§3 那张表就是实测 trace：K 序列 `[4, 4, 0, 2, 2, 2, 0, 4]`（8 轮，最后一轮是空轮），
第 2 轮 K=0 时**仍在验证第 1 轮提的 4 枚**，且 draft 第一遍照样跑了 1 次。

### 5.3 输出正确性

同一组 prompt，三个引擎的贪心输出**逐 token 相同**：

```
不开投机    [7, 1, 1, 1, 1, 1]
固定 K=3    [7, 1, 1, 1, 1, 1]
动态表 [(1,1,0),(2,8,3)]（K 在 0 与 3 之间来回） [7, 1, 1, 1, 1, 1]
```

### 5.4 图模式改写

`cudagraph_mode` 请求值 → 最终值：`None / full / full_decode_only / full_and_piecewise → PIECEWISE`，
`piecewise → PIECEWISE`、`none → NONE`（探针 B1）。档位表随之变化：
动态 `[1,2,4,8,16]`（不再是 `1+K=5` 的倍数），静态 `full_decode_only` 是 `[5,10]`（全是 5 的倍数）——证明
"取整到 1+K" 只在 decode FULL 时发生，降级后不再发生（探针 B2）。

真实 Qwen3-1.7B（bf16）+ ngram 提议者 + 动态表 `[(1,1,4),(2,2,0),(3,8,2)]`，14 轮 trace：

```
B=1 → K=4（提 4，验证上一轮 4）      B=2 → K=0（不提，仍验证上一轮 4）
B=3 → K=2（提 2，验证上一轮 2）      B=1（其余请求结束后）→ 回到 K=4
图：14/14 轮命中 PIECEWISE（捕获 8 张图），FULL 一次都没出现
step 数 11、总耗时 0.70 s（含首轮 warmup；模型加载 1.7B bf16）
```

### 5.5 工作区地址

tiny 引擎在 K 从 0 变到 3 的整个过程里，`proposer.input_ids / positions / slot_mapping` 的
`data_ptr()` **只有一组取值**（探针 D3）——逐轮 K 不触发任何重新分配。

### 5.6 ngram 的定宽缓冲

历史 `[1,2,3,4,5,6,1,2,3,4,5,6]`、`prompt_lookup_min=max=2`：
K=4 → `[1,2,3,4]`；同一份缓冲再以 K=1 调用 → `[1]`（**旧的宽列没被交出去**）；K=0 → `[]`；
K=5（> 最大 K）→ 明确报错（探针 D4）。

## 6. 差异账本（逐条，不藏）

1. **校验时机**：上游对这张表是**懒校验**（`Scheduler.__init__` 建 lookup 时才炸），
   本仓库在 `SpeculativeConfig.__post_init__` 就校验并归一化（并把归一化结果写回字段）。
   规则集合、报错文案与转换行为完全一致，差别只在**报错时机更早**、以及"表只有一份权威表示"。
   好处：坏配置在启动期就停，而不是排到第一轮调度；表被排序/转 int 之后，测试与下游读到的是同一份。
2. **不支持的方法在配置期拒绝**（上游是运行时 `assert` 或静默无效，见 §2.4）。
   本仓库按"不静默降级"的约定提前拒绝，**不删上游断言**——四个固定 K 的提议者里都补了注释指回来。
3. **`use_v2_model_runner` 那半个条件不存在**：上游的图模式改写里还有
   `or self.use_v2_model_runner`（V2 runner 能在动态 K 下 capture 多组 query 长度），
   本仓库没有 V2 runner 这条轴（属 73/74 关），所以"含 full graph 就降级"恒成立。
   上游 V2 侧的多 `decode_query_lens` 捕获（`v1/worker/gpu/cudagraph_utils.py:197-207`）本关**不做**。
4. **DP 分支在本仓库的常规路径上不可达**：本项目是单进程单卡，没有 `ParallelConfig`。
   规则照抄（`getattr(..., "data_parallel_size", 1)`），测试用带 `data_parallel_size` 的替身
   钉住分支（探针 B4）——不是"写了不跑"的死代码，但也不假装本仓库支持 DP。
5. **上游的"每请求补齐到 `1+K`"不在本仓库**（69 关就记过的差异，见
   `docs/step69_alignment.md` §3.2 的注）：上游 `scheduler.py:938` 为了让 full graph 命中，
   会把新 decode 请求补成 `1+num_spec_tokens` 行，而且**只在 `dynamic_sd_lookup is None` 时补**；
   本仓库的紧凑输入本身就要求"每请求 `1+K` 行"，没有这一步。动态表 + PIECEWISE 之下这条
   补齐本来也没有意义（分段图不关心每请求行数），所以本关不引入它。
6. **publisher / 统计口径不改**：`SpecDecodingStats.new(num_spec_tokens)` 仍按**最大 K** 开桶
   （上游同款），逐轮 K 只影响观测到的位置数；`publish_bound`（异步的发布上界）也不因 K 变化。
7. **`docs/step69b_piecewise_cudagraph.md` 是本关的前置**：PIECEWISE 的键、能力协商、段间静态
   缓冲都在那一份里；本关只用到"PIECEWISE 不锁请求数/每请求行数"这一条事实。

## 7. 接口变化与遗留

* `Scheduler.num_speculative_tokens` **改名为** `Scheduler.num_spec_tokens`（对齐上游命名；
  语义 = 最大容量）。仓库内三个读点已同步；`SpeculativeConfig.num_speculative_tokens` 不变。
* `SpecDecodeBaseProposer.propose()` 与 `NgramProposer.propose_drafts()` 新增可选参数
  `num_speculative_tokens`（`None` = 用配置的 K）——旧的直接调用点语义不变。
* `SchedulerOutput` 新增字段 `num_spec_tokens_to_schedule`（默认 0）。**引擎路径上 Scheduler 总会填**；
  手工构造 `SchedulerOutput` 的非引擎调用方（少数基准脚本）如果不填，执行侧会按 K=0 提议——
  这与上游同一处语义相同（字段是唯一权威），本关不为它加"猜一个默认值"的后门。
* 遗留（按需求顺序，属后续关卡）：V2 runner 的多 query 长度全图捕获（73/74）、
  PARD/P-EAGLE 的并行提议与动态 K 的组合（72）、以及 63 关就记着的接受长度差距
  （上游 1.5637 vs 本项目 1.2190 tokens/step）——动态 K **不能**掩盖它，本关也不改接受率路径。
