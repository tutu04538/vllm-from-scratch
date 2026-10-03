# step58 对齐记录：draft 第一遍输入与双预算

- 对应代码：`minivllm/spec_decode/{utils.py,draft_model.py,ngram_proposer.py}`、
  `minivllm/core/sched/scheduler.py`、`minivllm/worker/gpu_model_runner.py`、
  `minivllm/config.py`、`minivllm/attention/backends/torch_sdpa.py`
- 参考基线：本机 `vllm==0.28.0`（2026-10-03 文件快照），关键文件
  `v1/spec_decode/llm_base_proposer.py`、`v1/spec_decode/utils.py`、
  `v1/spec_decode/draft_model.py`、`v1/core/sched/scheduler.py`、`config/speculative.py`
- 需求：`learning_notes/14_vllm_from_scratch/207_第五十八关_对齐Draft输入与双预算调度.md`
- 目录映射：本项目顶层 `minivllm/` ↔ 上游 `vllm/v1/`（`models/` ↔ `model_executor/models/`），
  这是 2026-10-03 重构后的约定（`stepNN/` 目录已取消，历史见 git 与 `docs/`）
- 交付：`tests/step58/`（41 项 pytest）、`benchmarks/check_step58_{input_budget,draft_inputs,workspace}.py`
  （10 + 11 + 15 项）、`docs/step58_results.json`、`docs/step58_models.json`

## 1. 三个数字（本关最容易混的地方）

| 数字 | 上游 | 本项目 | 限制什么 |
|---|---|---|---|
| token budget | `Scheduler.max_num_scheduled_tokens` | `Scheduler.max_num_scheduled_tokens`（= `max_num_batched_tokens`） | 本轮 target 最多执行多少 query token |
| input budget | `scheduler_config.max_num_batched_tokens` | `Scheduler.max_num_batched_tokens` | draft 第一遍工作区最多容纳多少行 |
| draft slots | `SpeculativeConfig.max_num_new_slots_for_drafting` | 同左（普通 draft=1，ngram=0） | **每条被调度请求**额外占几行输入 |
| KV lookahead | `VllmConfig.num_lookahead_tokens` | `Scheduler.num_lookahead_tokens`（draft_model=K） | 额外保留几个 **KV 位置**给草稿写 |

前三个是**输入行**的口径，第四个是 **KV 位置**的口径，不能互相顶替。本关没有 encoder
预算，所以两份预算的起点数值相同、含义不同。

## 2. 逐项对照

### 2.1 `SpeculativeConfig.max_num_new_slots_for_drafting`

| | 上游 `config/speculative.py:1421-1459` | 本项目 `config.py::SpeculativeConfig` |
|---|---|---|
| 结构 | `use_dflash()` → K；`parallel_drafting` → K 或 K-1；`uses_draft_model()` → 1；否则 0 | 只保留 `uses_draft_model()` → 1、否则 0 |
| 差异（记录） | 有 DFlash / PARD / P-EAGLE 分支 | 本关只有普通自回归 draft 与 ngram；`method` 只允许这两个值，其它**初始化即拒绝**（不静默降级） |
| 校验 | `config/vllm.py:1871-1876` 校验容量 | `Scheduler.__init__`：`max_num_batched_tokens < 1 + draft_slots` → `ValueError` |

### 2.2 Scheduler 双预算

| | 上游 `v1/core/sched/scheduler.py` | 本项目 |
|---|---|---|
| 初始化 | `draft_slots = spec.max_num_new_slots_for_drafting`；`input_budget = max_num_batched_tokens`（496-499） | 同左（`Scheduler.__init__`） |
| running 守卫 | `if input_budget <= draft_slots: break`（527） | 同左（`schedule()` 第 1 段开头） |
| waiting 守卫 | 同（752） | 同左（第 2 段开头） |
| 取小公式 | `min(needed, token_budget, input_budget - draft_slots, max_model_len - computed - num_sampled)`（566、927） | `_num_new_tokens(request, token_budget, input_budget, start)` |
| 扣减 | `token_budget -= n`；`input_budget -= n + draft_slots`（701、1134） | 同左（running 与 waiting 两处） |
| 抢占已排入的 victim | `token_budget += restored`；`input_budget += restored + draft_slots`；`req_to_new_blocks.pop`；**`scheduled_spec_decode_tokens.pop`**（658-676） | 同左（含草稿计划一起删） |
| 出包前断言 | `Σtarget ≤ max_num_scheduled_tokens`；`token_budget ≥ 0`；`input_budget ≥ 0`（1173-1174） | 同左，并额外断言 `Σ(n_i + draft_slots) ≤ max_num_batched_tokens`；余额另存 `last_token_budget/last_input_budget` 供测试对账 |
| 差异（记录） | 上游还有 encoder 预算、pause、skipped_waiting、async placeholder 等 | 本关没有这些；断言范围相应更小 |

### 2.3 第一遍输入（`set_inputs_first_pass`）

| | 上游 `llm_base_proposer.py:869-968` + `utils.py:308-455` | 本项目 `draft_model.py::set_inputs_first_pass` + `spec_decode/utils.py` |
|---|---|---|
| 缓冲 | `needs_extra_input_slots` 时预分配 `input_ids/positions/…/is_rejected_token_mask`，`max_num_tokens` 行 | 同左（`__init__` 里按 `max_num_batched_tokens` / `max_num_seqs` 开好，CPU staging + device 侧各一份） |
| 物理布局 | 每条请求 `[有效行] + [1 扩容行] + [被拒行]`，`shift_input_ids=False` 时 `num_valid = query_end - query_start + 1` | 同左（`expand_draft_inputs`，逐值对照见 §4） |
| 扩容行 token | `next_token_ids[req]`（partial prefill 用 `backup_next_token_ids`） | `TargetRows.next_token_id`：ready=最后一个新采样 token，未 ready=backup（本段最后一个 token） |
| 位置 | `positions = start_pos + j`；被拒行 0 | 同左 |
| mask | `is_rejected_token_mask`（只标被拒尾部）；`is_masked_token_mask` 只用于并行提议 | 同左；`is_masked_token_mask` 缓冲保留但恒 False |
| 采样行 | `token_indices_to_sample[i]` = 第 i 条请求扩容行的全局行号 | 同左（但**只填 ready 请求**：中间 prefill 块不提草稿，见差异） |
| slot mapping | `compute_new_slot_mapping`：正常行走块表；`pos ≥ max_model_len` 与 `is_rejected` → `PADDING_SLOT_ID(-1)` | 同左（`spec_decode/utils.py::compute_new_slot_mapping`，签名把 CAD 换成 `block_table + query_lens`） |
| query/seq | `extend_all_queries_by_N(N=1)`：`query_start_loc += N*arange`、`seq_lens += N`、`num_actual_tokens += B*N`、`max_query_len/max_seq_len += N` | 同左（本地 attention metadata 只有 `query_start_loc`/`seq_lens` 两个字段需要改，`num_tokens` 由调用方按 Σ(n+1) 传） |
| 首遍 forward 的取行 | `sample_hidden_states = last_hidden_states[token_indices_to_sample]` | 同左（`_sample_draft_tokens`） |
| 自回归 K-1 步 | 复用同一缓冲，`_update_positions_dependent_metadata` 每步 positions+1、slot 重算、seq_lens+1 | 同左（`_set_autoregressive_inputs`：同一工作区前 B 行，positions = `history_end + k - 2`，seq_lens = position+1，slot 用 target 同一份 `compute_slot_mapping`） |
| 差异（记录） | ① 上游用 Triton kernel 在 GPU 上拷贝+扩容；本项目是 Torch/Python 逐行实现（**语义逐值一致**，见 §4 的差分测试），原因是要支持 CPU；② 上游给**所有**请求都填 `token_indices_to_sample` 再让 Scheduler 丢未 ready 的草稿，本项目只填 ready 的（57E 起就记在差异账本：少跑 K 次试探性前向）；③ 上游的 `_get_slot_mapping`/cudagraph padding 分支本关不做（69 关） |
| 缓冲归属（结构差异） | 上游 proposer **不持有** `query_start_loc` / `seq_lens` / `block_table`：它们来自 runner 传进来的 `CommonAttentionMetadata`，首趟原地改写（shift 路径）或 `replace()` 生成新实例；本项目 proposer 自己持有这三样（本地 attention metadata 只有 4 个字段、runner 不往 draft 传 CAD）→ 影响：draft 的 metadata 由 proposer 自己拼。69 关接 CUDA Graph 时要与 runner 的 CAD/padding 口径对齐 |
| `token_indices_to_sample` 的分配 | 上游每次首趟 `torch.empty(batch_size * extra_slots, int32)` 新分配；本项目用固定工作区里算出来的行号 list → 语义相同，且连这个也不每轮新建（差一点点更省） |
| 异步占位符分支 | 上游有 `num_output_placeholders` 与 `pad_spec_decode`（新 decode 请求按 `1 + num_spec_tokens` 排、登记 `[-1] * num_spec_tokens` 占位），本项目没有（70 关：异步调度占位符与 GPU 结果回传） |
| 哨兵写入 | attention kernel 里 `slot >= 0` 才写 | `TorchAttentionImpl.write_kv` 先过滤 `slots >= 0`（否则 `index_copy_` 会把 -1 当最后一个槽位） |

### 2.4 "起点"的口径（本关修掉的那个 bug）

| | 57（旧） | 58（现在） |
|---|---|---|
| 第一遍起点 | `min(draft 自己记的 _draft_computed, 本轮边界)` → 新请求 = 0，**prefix 命中的前缀被整段重算** | `TargetRows.start` = 调度快照里的 `num_computed_tokens`（含命中起点） → 命中段不重算 |
| 边界来源 | 一个含糊的 `num_computed_tokens` 字典 | 显式的 `TargetRows(start, target_rows, num_rejected, history_end, next_token_id, ready)` |
| `_draft_computed` | 第二套权威 | 只作观测（保留字段名以便对账），**不参与任何决策** |

实测（`tests/step58/test_draft_inputs.py::test_prefix_hit_does_not_recompute_hit_range`）：
第二个相同 prompt 的请求命中 12 个 token，draft 第一遍只处理位置 `[12, 13]`；57 的实现会从
位置 0 处理到 13。

## 3. 状态所有权与故障策略

- 工作区（输入缓冲、块表工作区、mask）**只在 proposer 里**；`SchedulerOutput` 仍只传数据快照，
  不带 tensor / 活 Request；`Request` 上不挂工作区、第二份 KV 进度或 generator（需求 §5）。
- 输入预算不足时：Scheduler 少排（`input_budget - draft_slots` 取小），**不是**让 proposer 少写；
  proposer 仍保留"工作区放不下就报错"的内部检查（口径不一致 = 控制面 bug）。
- 容量不足的配置（M < 1 + draft_slots）在**初始化**拒绝，不进空转。
- 上下文末尾：自回归每写一枚前过 `max_model_len` 与 `BlockTable.covers`，过不了就少提几枚；
  第一遍的扩容行位置由调度侧的 `max_model_len - 1 - computed` 上限保证仍在范围内。

## 4. 验证

| 命令 | 结果 |
|---|---|
| `python -m pytest tests/step58 -q` | 41 passed（含 2 项 CUDA-only） |
| `python benchmarks/check_step58_input_budget.py` | 10 项（A 组） |
| `python benchmarks/check_step58_draft_inputs.py` | 11 项（B 组，含与上游 kernel 的 CUDA 差分） |
| `python benchmarks/check_step58_workspace.py` | 15 项（C 组，含 CUDA 端到端） |
| 15 个 step57 回归脚本 | 354 项（其中 `check_step57_draft_model.py` 32 项：双预算让轮数变了，期望值已同步） |
| 57 时代的 4 个验收探针 | rejection 2/2、inference 1/1 仍通过；`review_draft_boundaries` 与 `review_remaining_boundaries` **无法直接复跑**：前者的 spy 按旧 `propose(req_ids, all_token_ids, num_tokens_no_spec)` 签名写死，后者依赖"budget=4 时 B 也能被调度"（双预算下不成立）。两者覆盖的场景分别由 `check_step58_draft_inputs.py`（第一遍输入/prefix 复用）与 `check_step58_workspace.py`（生命周期）接管 |

**与上游 kernel 的逐值差分**（`shift_input_ids=False`、`num_padding_slots_per_request=1`，
两条请求其中一条带被拒行）：

```text
kernel ids   : [5, 6, 99, 0, 8, 9, 10, 111]
ours   ids   : [5, 6, 99, 0, 8, 9, 10, 111]
kernel pos   : [6, 7, 8, 0, 8, 9, 10, 11]        ours 相同
kernel rej   : [F, F, F, T, F, F, F, F]          ours 相同
kernel sample: [2, 7]                            ours 相同
kernel mask  : 全 False                          ours 相同
```

**prefix 复用实测**：命中 12 个 token 的第二次请求，第一遍只写位置 `[12, 13]`（2 行），
57 的实现是 `[0..13]`（14 行）。

**双预算实测**（纯 Scheduler）：M=8、两条请求各要 4 行 → 计划 `{A: 4, B: 2}`（合计 6 行 +
2 行额外 = 8，正好装满）；没有输入预算时会是 `{A: 4, B: 4}` 而 draft 需要 10 行。

## 5. 与 57 的行为差异（会改旧测试期望的地方）

| 变化 | 影响 |
|---|---|
| 每请求多占 1 行输入预算 | 同样 `max_num_batched_tokens` 下每轮能排的 target token 变少（`check_step57_draft_model.py` 的"中间 prefill 块"与 A/B 场景的期望值已按新口径改） |
| 抢占更难触发 | 原来的 `blocks=4` 配置不再抢占，回归用例改成 `blocks=3` |
| 第一遍多算"本轮起点已有效的少量位置" | 允许（需求 §6）；prefix 命中不再重算整段 |
| `propose()` 参数 | `req_ids + num_computed_tokens` → `rows: list[TargetRows]`（起点/终点写清楚）；`ready_req_ids` 由 `TargetRows.ready` 取代 |
| `_forward(rows, tokens, batch)` | → `_forward(num_tokens, num_reqs)`（读固定工作区） |
| `_sample()` | → `_sample_draft_tokens()`（上游同名） |

## 6. 留待后续（按总纲顺序）

- 编译 / CUDA Graph / padded 输入与 `cudagraph_runtime_mode`：69 关。
- EAGLE / MTP / 并行提议（PARD、DFlash）：63、65、72 关；那时 `max_num_new_slots_for_drafting`
  与 `is_masked_token_mask` 才会用到另一条分支。
- 异步调度占位符与 GPU 结果回传：70 关（本关仍是同步 eager）。
- 未支持组合（如 `method="eagle3"`）目前直接拒绝，按各关需求逐条接入。
