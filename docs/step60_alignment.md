# step60 对齐记录：CPU 与 GPU 的 Ngram 提议及历史增量维护

- 对应代码：`minivllm/spec_decode/ngram_proposer.py`（CPU，校准）、
  `minivllm/spec_decode/ngram_proposer_gpu.py`（新增：GPU kernel + proposer + 增量维护）、
  `minivllm/spec_decode/utils.py::update_scheduler_for_invalid_drafts`、
  `minivllm/worker/gpu_model_runner.py`（显存历史缓冲与接线）、
  `minivllm/core/sched/scheduler.py`（草稿交接时裁"占位→有效"）、
  `minivllm/config.py`（`prompt_lookup_min/max`、`method="ngram_gpu"`）、`minivllm/outputs.py`
- 参考基线：本机 `vllm==0.28.0`（2026-10-03 文件快照），关键位置
  `v1/spec_decode/ngram_proposer.py`（L12–L293）、`v1/spec_decode/ngram_proposer_gpu.py`（L28–L671）、
  `config/speculative.py`（L804–829、L1498）
- 需求：`投机解码完整需求/060_CPU与GPU的Ngram提议及历史增量维护.md`
- 交付：`tests/step60/`（91 项 pytest：`test_ngram.py` + `test_ngram_gpu_state.py`）、
  `benchmarks/check_step60_ngram.py`（17 项）、`docs/step60_results.json`、`docs/step60_models.json`

## 1. 这一关解决什么

历史里出现过 `… 北京的天气`，现在又走到同样的片段，就把上次后面跟着的 `预报`（+后续最多 K 枚）
抄来当草稿——不用第二个模型、不用额外 forward。两个痛点：**CPU 版要校准到与上游逐值一致**
（60 关之前本机的匹配规则和窗口都不一样），**GPU 版要新接**（历史常驻显存 + 增量写入 +
固定宽度候选 + 有效个数）。

## 2. 逐项对照

### 2.1 CPU `NgramProposer`

| | 上游 `v1/spec_decode/ngram_proposer.py` | 本项目 |
|---|---|---|
| 构造 | `NgramProposer(vllm_config)`；`min_n/max_n/k/max_model_len` 全部从配置取 | 同左（本机原来自己收 `num_speculative_tokens`，60 关改成同签名） |
| 匹配核心 | `_find_longest_matched_ngram_and_propose_tokens`：反转 + KMP 的 `lps` 数组，`prev_lps >= longest_ngram` 覆盖 `position` → **同长度取原序列最早那处** | **逐行照抄**（同名函数，本机可 import 对照） |
| `k` 的两个上限 | `k = min(k, max_model_len - total_token)`；再 `k = min(k, total_token - start_position)` | 同左 |
| 匹配长度窗口 | `[prompt_lookup_min, prompt_lookup_max]`（都没给 → 5/5；只给一个 → 另一个跟随；校验 `min ≤ max`） | 同左（本机原来固定 `max_ngram=3`、最小 1，现在走配置） |
| 跳过规则 | 本轮没采样出 token → 跳过；`num_tokens_no_spec >= max_model_len` → 跳过 | 同左（`ready` ↔ "采样出了 token"） |
| 批量入口 | `batch_propose(...)` + `batch_propose_numba`（`@njit(parallel=True)`） | `batch_propose(...)` 同签名，**逐请求循环**调同一个匹配函数（不引入 numba 依赖，见 §3） |
| `propose()` | 收 `(num_speculative_tokens, sampled_token_ids, num_tokens_no_spec, token_ids_cpu, slot_mappings=None)` → `list[list[int]]` | 同左；另加 `propose_drafts()` 适配本仓库统一的提议者协议（`rows` + `all_token_ids`），两者共用同一套匹配代码 |
| `load_model()` | 空操作 | 同左 |

### 2.2 GPU `NgramGPUKernel` / `NgramProposerGPU`

| | 上游 `v1/spec_decode/ngram_proposer_gpu.py` | 本项目 |
|---|---|---|
| 匹配实现 | `_find_first_and_extract_all_n_parallel`：`unfold` 滑窗 == 后缀、`argmax` 取**最早**匹配、在有匹配的 n 里取最长；`k` 的上限是"匹配点后面还有多少 token" | 逐行对应（同样的 torch 张量运算） |
| 编译 | `@support_torch_compile()` + 一整套 inductor 配置 | **不编译**（本机没有 torch.compile 支持；张量运算一致，60 关不涉及图捕获） |
| 输出 | `draft_tokens [B, k]`（尾部 `-1`）+ `num_valid_draft_tokens [B]`（前导连续有效个数） | 同左 |
| `propose()` | 把本轮新采样 token `scatter_` 进历史 → `num_tokens_tmp = 长度 + 有效采样数` → kernel；`assert num_speculative_tokens == self.k` | 同左（含固定 K 断言，动态 K 属 71 关） |
| `update_token_ids_ngram()` | 丢弃行整行 `-1`、越界 id 不计入有效数、无有效采样时回退历史末尾 token | 同左 |
| 设备 | 需要 CUDA（编译 + 图） | CPU 也能跑（只用 torch 张量运算），测试对两个设备都实跑 |
| 预热 | `_dummy_run()` 3 次触发编译 | `_dummy_run()` 3 次（先验形状/显存；没有编译要做） |

### 2.3 GPU 历史与增量维护

| | 上游 | 本项目 |
|---|---|---|
| 缓冲 | `token_ids_gpu_tensor [max_num_reqs, max_model_len]` int32 + `num_tokens_no_spec_gpu [max_num_reqs]` int32（Runner 分配） | 同左（`Runner._build_proposer()` 里按 `max_num_reqs/max_model_len` 分配） |
| 增量更新 | `update_ngram_gpu_tensors_incremental`：①行重排（历史整行搬家）②新请求/恢复整段拷一次 ③每步从 CPU 真值同步长度；pinned 缓冲 + `non_blocking` | 同左（去掉 pinned 缓冲，直接建 `num_reqs` 长度的索引张量；见 §3） |
| 长度同步 | `_sync_num_tokens`（`index_copy_` 到显存） | 同左（显式转 int32：本机 CPU 真值是 int64） |
| 上一轮行号映射 | 存在 `InputBatch.prev_req_id_to_index` | Runner 在 `_update_states()` 进入时快照一份局部变量（同一语义，不动 InputBatch） |
| 每步搬运量 | 新增 token + 长度（与历史长度无关） | 同左：**每步 2 次 D2H（草稿 `[B,K]` + 有效数 `[B]`），与历史长度无关**（实测历史 32 / 256 / 2048 token 都是 2 次）；"整段搬回"的方案是 `B×L×4` 字节（B=32/L=4000 → 500 KiB、约 0.066 ms/步，59 关实测） |
| **两个缓冲的更新时机** | 长度表：每步在 `_update_states` 里由 CPU 真值同步（`_sync_num_tokens`）；token 内容：每步在 `propose()` 里 `scatter_` 追加 | 同左（上游同款）。含义：提议那一刻"长度表 = **本轮起点**，token 内容 = 起点 + 本轮新 token"，两者要**配上本轮的 `counts`** 才等于权威的 CPU 镜像；不变量 `gpu_len + counts == cpu_len` 由 `test_gpu_length_is_round_start_snapshot` 盯着 |
| 批没填满（`num_reqs < max_num_reqs`） | 上游把采样结果也 padding 到 `max_num_reqs`，两边行数一致 | 本机采样结果是 `num_reqs` 行的参差 list，所以 `propose_drafts` 把两个 GPU 缓冲**切片到 `num_reqs`**（视图，就地 scatter 仍写进原缓冲）。漏了这一步的后果见 §5 的修复记录 |

### 2.4 调度侧：占位 ↔ 有效（`update_scheduler_for_invalid_drafts`）

| | 上游 | 本项目 |
|---|---|---|
| 时机 | **Runner 侧**，`_update_states` 里改 Runner 手里的 `scheduler_output` **副本**（上游为此专门 `replace` 复制一份） | **Scheduler 侧**，草稿交接时（`update_draft_token_ids`）裁一次 |
| 为什么位置不同 | 异步调度让 Scheduler 的计划是**乐观**的（按固定宽度 K 占位），必须等 GPU 的有效个数异步回来再裁；`prev_num_draft_len` 保留乐观值用于拒绝纠正 | 本机没有异步调度（70 关才有占位符）：草稿在交接那一刻就到齐了，直接裁掉更简单，而且预算/统计从一开始就只包含真实候选 |
| 效应 | `num_scheduled_tokens[req] -= (占位-有效)`、`total_num_scheduled_tokens -= ...`、`scheduled_spec_decode_tokens[req] = spec[:valid]`、有效数 0 就 pop | 同一效应（`spec_token_ids[:valid]`，有效数 0 → 空列表）；额外**滤掉 `-1`**：需求 §3.5 的不变量是"哨兵不许变成真实 token"，就算有效个数算错也不放出去 |
| 计数 | 上游把"占位未验证"的槽位仍计入 `num_draft_tokens`（统计会稀释，除非另有 `num_invalid_spec_tokens` 扣减） | 本机在交接处已裁，Scheduler 的统计（59 关 `SpecDecodingStats`）**只看到真实候选** ✓ |

## 3. 临时差异（逐条写清）

1. **不用 numba**：上游 `batch_propose` 在总 token 数超阈值时用 `@njit(parallel=True)` 多线程扫；
   本机不引入 numba JIT 依赖，`batch_propose` 是逐请求循环调同一个匹配函数。语义逐值一致
   （测试直接拿上游 `_find_longest_matched_ngram_and_propose_tokens` 与 `batch_propose_numba` 差分）；
   差别只在 CPU 大 batch 的吞吐。
2. **不做 torch.compile / CUDA Graph**：上游 `NgramGPUKernel` 带 `@support_torch_compile()`；
   本机不编译（69 关做固定形状与图捕获时再谈）。
3. **不做 pinned + 异步拷贝**：上游用 `copy_num_valid_draft_tokens` / `_copy_draft_token_ids_to_cpu`
   把草稿与有效数异步挪回 CPU（为异步调度服务）；本机是同步引擎、草稿最终要落到 CPU 交给
   Scheduler，所以直接同步取（数据量 `B×K + B` 个整数，不是整段历史）。
4. **行号映射不放 InputBatch**：上游把 `prev_req_id_to_index` 存在批镜像上（还有异步调度的
   `sampled_token_ids_cpu` 等用途）；本机只在 `_update_states()` 内用一次，所以用局部快照。
5. **CPU/GPU 的一个语义差别（上游同款，不是本机发明）**：CPU 版有 `k = min(k, max_model_len -
   total_token)`（不提议超出上下文上限的草稿）；GPU 版**没有**这个上限，它只抄"历史里匹配点后面
   已有的 token"。所以历史长度正好等于 `max_model_len` 时，CPU 给空、GPU 可能给草稿。
   测试 `test_gpu_does_not_cap_by_remaining_model_length` 把这个差别钉住。
6. **`ngram_gpu` 在 CPU 上也能跑**：上游那版要 CUDA（编译 + 图），本机实现只用 torch 张量运算，
   所以测试对 `cpu`/`cuda` 两个设备都实跑；不追求 CPU 上的性能。

## 4. 验证

| 命令 | 结果 |
|---|---|
| `python -m pytest tests/step60 -q` | **95 passed** |
| `python -m pytest tests/step59 -q` | **52 passed** |
| `python -m pytest tests/step58 -q` | **41 passed** |
| 三个目录一次跑（`pytest tests/step60 tests/step59 tests/step58 -q`） | **187 passed**（见 §6 的模块改名） |
| `python benchmarks/check_step60_ngram.py` | **17 PASS / 0 FAIL** |
| `check_step58_*` / `check_step59_rejection` | 10 + 11 + 15 + 21 = **57 PASS** |
| 15 个 `check_step57_*.py` | **350 PASS / 0 FAIL** |

关键用例：

- **逐值差分**：单条序列 75 组（含 tie / min_n / max_model_len 边界）+ `batch_propose` 对上游
  numba 版一致；GPU kernel 对同一批历史与上游冻结的 CPU 参考一致。
- **tie 规则**：`[1,2,3,1,2,9,1,2]`、n=2、k=3 → `[3,1,2]`（最早那处），不是 `[9,1,2]`。
- **宽度 vs 有效**：`[1,2,9,1,2]`、k=4 → `[9,1,2,-1]` + `num_valid=3`。
- **只读长度**：`propose()` 把新 token 写进历史后，`num_tokens_no_spec` 一位不变（否则同一输出
  会被累计两次）。
- **行重排**：`[A,B,C] → [C,A] → [D,C,A]` 三步之后，显存历史与长度都归属正确请求
  （`[4,4,4,4] / [3,3,3] / [1,1,1]`、长度 `[4,3,3]`）。
- **不整段 D2H**：`propose()` 自身 0 次 D2H；整条 `propose_drafts()` 每步 2 次（草稿+有效数），
  历史 32 / 256 / 2048 token 时都是 2 次。
- **哨兵不出门**：端到端跑 `ngram_gpu`，Scheduler 的 `spec_token_ids` 里 18 个草稿全部 `≥ 0`。
- **两个缓冲的时机不变量**：多请求 + 中途插入 + prefill 块的混合跑，`gpu_len + counts == cpu_len`
  每次都成立（实测 15/15；用例化后由 `test_gpu_length_is_round_start_snapshot` 守着）。
- **批没填满**：`max_num_seqs=4` + 2/1 条请求、`max_num_seqs=3` + 2 条请求都跑通（修复前会报错）。
- **端到端**：tiny 模型上 `ngram` 与 `ngram_gpu` 的 greedy 投机都 == 非投机，且两条路径输出一致。

## 5. 开发中发现并修掉的问题（60 关内）

| 问题 | 症状 | 修法 |
|---|---|---|
| `propose_drafts` 把**整块** GPU 缓冲（`max_num_reqs` 行）传给 `propose()`，而采样矩阵只有批里那 `num_reqs` 行 | 批没填满时 `write_positions` 是 `max_num_reqs` 行、采样 mask 是 `num_reqs` 行：`2 ≤ num_reqs < max_num_reqs` 时广播直接报错（`The size of tensor a (2) must match ... (3)`）；`num_reqs == 1` 更隐蔽——第 0 行的采样数据被广播进空闲行（当时没人读，但那一行被新请求占用后就是脏数据） | `propose_drafts` 开头把 `token_ids_gpu` / `num_tokens_no_spec_gpu` 切片到 `num_reqs`（视图，不影响就地写）；补两条回归：`test_partially_filled_batch_is_sliced`、`test_underfilled_batch_runs_end_to_end` |
| 端到端用例恰好都是"请求数 == `max_num_seqs`"，把上面那个组合躲开了 | — | 新增的端到端用例固定用 `max_num_seqs=4` + 2 条请求 |

## 6. 与之前关卡的行为差异

| 改动 | 原因 | 影响 |
|---|---|---|
| ngram 的匹配窗口从"固定 1..3"改成配置 `prompt_lookup_min/max`（默认 5/5） | 上游同款（默认 5/5 是上游的 arbitrary choice） | 短 prompt 上默认不再提草稿；`check_step57_spec_lifecycle.py` 显式传 `1/3`（它考的是时序，不是窗口） |
| 同长度多处匹配从"最近一次"改成"**最早**那处" | 上游源码的 tie 规则 | 同一历史可能给出不同的草稿（`check_step57_spec_lifecycle.py` 的期望随之更新为上游语义） |
| `NgramProposer.__init__` 从 `(num_speculative_tokens, max_ngram=3)` 改成 `(vllm_config)` | 上游同签名 | 只有 Runner 构造它（已改）；测试自己造配置替身 |
| 提议者协议入口：ngram 走 `propose_drafts()`；`propose()` 变成上游同签名的批量入口 | 两个入口都要保留（需求 §2） | `check_step57_spec_lifecycle.py` 的 spy 改成包 `propose_drafts` |
| `tests/step59/helpers.py` → `spec_helpers.py`、`tests/step60/helpers.py` → `ngram_helpers.py` | 三个测试目录同名 `helpers` 会在一次 pytest 里互相覆盖（59 关引入的坑） | 现在 `pytest tests/step60 tests/step59 tests/step58` 能一次跑完（184 passed） |
| `DraftTokenIds` 增加 `num_valid_draft_tokens`（CPU 提议者为 None） | GPU 提议者是固定宽度输出 | Scheduler 收草稿时裁"占位 → 有效" |

## 7. 留待后续

- **61 关**：suffix decoding 的请求内与跨请求历史（本关只做 ngram 的 CPU/GPU 两条路）。
- **62 关**：自定义 proposer 接入与配置分派边界（本关的 `method` 仍是白名单三分支）。
- **69/70 关**：固定形状输入与 CUDA Graph、异步调度占位符——届时 `update_scheduler_for_invalid_drafts`
  要挪回 Runner 侧（上游位置），并补 `prev_num_draft_len` 那套乐观计数。
- **71 关**：动态投机长度（现在 GPU proposer 的 `assert num_speculative_tokens == self.k` 是上游同款硬限制）。
