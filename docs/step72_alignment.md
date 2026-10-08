# 第七十二关：PARD 与 P-EAGLE 并行提议

**状态（2026-10-08）：已实现并自检。** 本文件按 `docs/README.md` 的固定结构写：
0 需求大概 → 1 改动内容 → 2 设计要点 → 3 **模型格式清单** → 4 验证 → 5 实测与待验 →
6 差异账本 → 7 接口变化与遗留。

---

## 0. 需求大概

串行提议要 forward **K 次**才能凑出 K 枚草稿（第一遍 + K−1 次自回归），而按并行草稿训练的
模型（**PARD** = draft model 的并行版、**P-EAGLE** = EAGLE3 的并行版）可以**一次 forward
同时给出 K 个位置**的 hidden。本关把这条输入协议接进来：

```
[有效行] [锚点] [K−1 个 mask token] [被拒尾部]      ← 一次 forward
     ↑ 锚点与 mask 行各采一枚 → [B, K]               ← 每轮每请求交回 K 枚
```

需求 072 §3 点名的四条差异必须保持：普通串行 draft 额外槽位 = **1**、PARD = **K**、
P-EAGLE = **K−1**（不能都写成 K）；PARD **不左移**、P-EAGLE 仍要与 target hidden 对齐；
`copy_and_expand_eagle_inputs_kernel` 填的两种 mask / hidden 映射 / 采样行必须同步；
槽位与 metadata 要**基于新 positions 重算**，不是只把 `seq_lens` 加个数字。

## 1. 改动内容

| 本项目文件 | 参考路径（vllm==0.28.0 快照） | 状态 |
|---|---|---|
| `minivllm/config.py` | `config/speculative.py` L168、L1421-1459 | 新增 `parallel_drafting` 字段；`max_num_new_slots_for_drafting` 补齐上游的**槽位表**（含 DFlash/DSpark 两行占位）；新增 `_resolve_parallel_drafting()`（方法边界 + K>0 + 把开关注入 draft 的 hf 配置） |
| `minivllm/spec_decode/utils.py` | `v1/spec_decode/utils.py` L308-454 | 新增 `expand_parallel_draft_inputs()`（+ `ParallelFirstPass`）：`copy_and_expand_eagle_inputs_kernel` 的**逐行 torch/Python 等价**，两种 shift 都支持 |
| `minivllm/spec_decode/draft_model.py` | `v1/spec_decode/llm_base_proposer.py` L112-119、L129-130、L350-379、L829-968、L1416-1424 | `__init__` 里两个槽位量 + `_init_parallel_drafting_params()`（mask token / mask hidden）；`_parallel_first_pass()`（并行第一遍）；`_write_parallel_hidden()`；`_maybe_fill_parallel_drafting_hidden_state()`；`propose()` 里"一次 forward 出 `[B,K]`"的早退分支 |
| `minivllm/spec_decode/eagle.py` | 同上（`EagleProposer` 只是构造参数不同） | 并行时第一遍改走共享的 `_parallel_first_pass()`（P-EAGLE） |
| `minivllm/models/qwen3_eagle3.py` | `model_executor/models/qwen3_eagle3.py` L316-323、L395-426 | `parallel_drafting=True` 时注册 `mask_hidden` buffer；加载权重里的 `mask_hidden`；**开了并行但权重里没有 → 加载期报错**（上游同款） |
| `minivllm/worker/gpu_model_runner.py` | `v1/worker/gpu_model_runner.py` | 两处判据加上并行：保留本轮 target 的 token/positions 缓冲（PARD 也要）、把它们传给提议者 |
| `minivllm/testing/tiny_models.py` | —（测试夹具） | `tiny_eagle3_dir(parallel_drafting=..., mask_token_id=...)`、新增 `tiny_pard_dir()` |
| `tests/step72/`、`benchmarks/check_step72_parallel_draft.py`（新） | — | 30 项单测 + 18 项探针 |

## 2. 设计要点

### 2.1 两个槽位量（**不要混**）

上游 `llm_base_proposer.py:112-119` 逐行照抄：

```
extra_slots_per_request        = 1 if not parallel_drafting else K      ← 这一块里"要采样"的行数
net_num_new_slots_per_request  = extra − (1 if 吃 target hidden 且不是 dflash else 0)
needs_extra_input_slots        = net > 0
```

- "吃 target hidden" ⇔ 左移 ⇔ 锚点**复用** target 块最后一行（位置与槽位不变）→ 净增 `K−1`；
- PARD 不左移：target 那一行原样当内容行，锚点 + (K−1) 个 mask 全部接在后面 → 净增 `K`。

采样行数两种都是 `extra = K`（1 锚点 + K−1 mask），所以"要采 K 行"与"多占几行"是两个量。

### 2.2 块内布局（同一算法两种 shift）

`expand_parallel_draft_inputs()` 逐行对应上游 kernel（j 从 0 数起）：

| 区间 | token | position | `is_rejected` | `is_masked` | 采样 |
|---|---|---|---|---|---|
| `[0, num_valid)` | target 的 token（shift：跳过第 0 个） | `start_pos + j` | 0 | 0 | 否 |
| `j == num_valid` | **锚点** = target 刚采出的 token | `start_pos + j` | 0 | 0 | **是** |
| `(num_valid, num_valid+extra)` | **mask token**（来自 checkpoint） | `start_pos + j` | 0 | **1** | **是** |
| 尾部 `num_rejected` 行 | padding | 0（don't care） | **1** | 0 | 否 |

`num_valid` = `target_rows − rejected − (1 if shift else 0)`；总行数 = `Σ(target_rows) + B×net`；
采样行走 `token_indices_to_sample`（按 `请求序 × 行内序` 排列 → 一次 forward 后天然是 `[B,K]`）。
**hidden 不跟着 token 移**：源行 i 的特征写到目标行 i（`hidden_state_mapping`），
**然后** mask 行才被替换成模型自带的 `mask_hidden` —— 顺序反了会在"被拒行与 mask 区重叠"时
把 mask 向量又盖回真实特征（探针 D3 盯着这条）。

### 2.3 一次 forward

`propose()` 里与上游同位置：

```
K == 0                     → 跑完第一遍，返回空草稿（71 关）
K == 1 or parallel_drafting → 在采样行上**一次**采样，view(-1, K) 返回
否则                        → 锚点一枚 + K−1 次自回归（串行路径，58 关）
```

实测（tiny、K=3）：并行 **1 次/轮**，串行 **3 次/轮**。

### 2.4 槽位与 metadata 的重算

并行第一遍的槽位**不是**"把 seq_lens 加个数字"：用 `compute_new_slot_mapping(num_new_tokens=net)`
按**新的 positions** 查块表（被拒行/越界行 → `PADDING_SLOT_ID`），
query/seq 长度用 `extend_all_queries_by_N(N=net)`。这两处都是 58 关就对齐过的同一套函数。

## 3. 模型格式清单（需求 §5 交付物）

| 方法 | config 必须有 | 权重必须有 | 本机有没有 |
|---|---|---|---|
| **PARD**（`method="draft_model"` + `parallel_drafting=True`） | `pard_token`（或 `mask_token_id` / `ptd_token_id` / `dspark_noise_token_id` / `dflash_config.mask_token_id`） | 无额外项（普通 LM 权重） | **没有**：本机只有 Qwen3-1.7B（target）与该模型的串行 EAGLE3 draft；没有按并行草稿训练的 PARD 权重 |
| **P-EAGLE**（`method="eagle3"` + `parallel_drafting=True`） | `mask_token_id`（同上五个来源任选其一） | **`mask_hidden`**，形状 `(1, hidden × 辅助层数)`；**缺了就加载失败** | **没有**：`models/Qwen3-1.7B-eagle3` 的 15 个权重键里没有 `mask_hidden`、config 里没有 `mask_token_id`（实测），它是**串行**训练的 |
| EAGLE3（串行，63 关） | `eagle_aux_hidden_state_layer_ids` | `fc/midlayer/norm/lm_head/d2t` | 有（`models/Qwen3-1.7B-eagle3`） |

**本仓库照抄上游的这条硬边界**：`parallel_drafting=True` 而权重里没有 `mask_hidden` →
**加载期报错**（上游 `qwen3_eagle3.py:421-426` 的原文：*"mask_hidden not found in weights but
model is configured for parallel drafting."*）。所以"拿串行权重开并行做性能验收"这件事在上游
本身就是被拒绝的——探针 E1 用一个"串行权重 + 并行开关"的 tiny 组合把这条钉住。

## 4. 验证

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/step72 -q            # 30 项
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmarks/check_step72_parallel_draft.py   # 18 项
```

| 需求 072 §4 的验收 | 落点 |
|---|---|
| B=2、K=1/4：输入/mask/positions/hidden 映射/采样行逐值对照（两请求长度故意不同） | `test_expand_matches_upstream_kernel`（**调上游真 Triton kernel**，6 组）、`test_pard_layout_shapes_and_masks`、`test_peagle_layout_reuses_the_last_row_and_maps_hidden`、探针 B1~B3 |
| K=1 时 P-EAGLE 额外槽=0、PARD=1；预算/KV 容量口径正确 | `test_slot_accounting`（6 组）、`test_scheduler_draft_slots_follow_the_same_table`、`test_parallel_k1_end_to_end_runs`、探针 A1~A4 |
| partial prefill / prefix hit / 首拒 / 中拒 / 全接受；图重放与 eager 一致 | 布局侧由 kernel 差分覆盖（含被拒 0/1/3/4 的用例）；端到端由 `test_parallel_greedy_matches_non_speculative`、`test_parallel_greedy_matches_with_two_requests_and_prefix_hit` 覆盖 partial prefill 与混批；**图重放**见 §5 待验 |
| trace 确认**一次**并行 forward，而不是 K 次伪并行循环 | `test_parallel_drafting_uses_one_forward_per_round`、`test_serial_drafting_still_uses_k_forwards`、探针 C1~C4 |
| 用匹配 PARD/P-EAGLE 训练格式的权重；普通 draft 权重不能做并行性能验收 | §3 的清单 + `test_serial_checkpoint_rejected_at_load_time`、`test_missing_mask_token_rejected_at_config_time`、探针 E1/E2（**质量/加速比本身待验**，见 §5） |

## 5. 实测与待验

### 5.1 实测

* **与上游 kernel 逐值一致**：`shift ∈ {True, False} × K ∈ {1,2,4}` × B=2（长度 4/2、被拒 3/1）
  以及 B=3（长度 3/1/5、被拒 2/0/4、base=17）——`input_ids` / `positions` / `is_rejected` /
  `is_masked` / `token_indices_to_sample` / `hidden_state_mapping` 全字段相等（探针 B1/B2）。
* **一次 forward**：tiny 引擎 K=3、6 步生成里，PARD/P-EAGLE **每轮 1 次** draft 前向；
  串行 draft 对照组每轮 3 次（探针 C1/C3/C4）。
* **正确性**：并行提议的 greedy 输出与非投机**逐 token 相同**（eagle3/draft_model × K=1/4）；
  B=2 混批（两请求长度不同）下并行与串行输出也相同（探针 D1/D2）。
* **mask 行真的换了 hidden**：8 个 mask 行的 hidden 全等于模型自带的 mask 向量（探针 D3）。

### 5.2 待验（**不当成通过**）

1. **草稿质量与加速比**：本机没有按并行草稿训练的 PARD/P-EAGLE 权重（§3），tiny 权重随机构造、
   并行草稿本身没有质量意义。所以"并行提议比串行快多少 / 接受长度多少"**没有测**——
   要测需要下载一份真正的 PARD 或 P-EAGLE checkpoint（需求 072 §4 最后一条明确禁止拿普通
   draft 权重充数）。
2. **CUDA Graph 重放**：drafter 侧的图属 74 关（上游只在 PIECEWISE 下给 drafter 做图），
   本关的并行第一遍是 eager；"图重放与 eager 一致"这条要等 74 关的 drafter 图。
3. **DFlash / DSpark**：需求 §5 明确"不能在这里以相同 mask 方式代替"——它们的输入协议不同
   （DFlash 是"1 个 bonus query + K 个 mask query"、DSpark 的 mask 与位置规则另有一套），
   属 76/77 关。槽位表里那两行已经按上游写好（`max_num_new_slots_for_drafting`），
   但配置期**不允许** `method="dflash"`/`"dspark"`。

## 6. 差异账本（逐条）

1. **内核实现方式**：上游用 Triton kernel（`copy_and_expand_eagle_inputs_kernel`）在 GPU 上展开；
   本仓库是**逐行 Python**（生产路径要在 CPU 上也能跑，58 关起就是这么做的）。
   语义以**差分测试**保证：探针 B1/B2 直接调上游真 kernel 逐字段比对。
2. **`_init_parallel_drafting_params()` 的开关来源**：上游模型从 live
   `vllm_config.speculative_config.parallel_drafting` 读；本仓库的模型只吃一个 config dict，
   所以由 `SpeculativeConfig._resolve_parallel_drafting()` 把开关**注入 draft 的 hf 配置**
   （与 64 关 `extract_hidden_states` 的做法一致）。这是管道差异，不是语义差异。
3. **配置期校验比上游严**：上游只在 docstring 里写"只与 EAGLE 和 draft model 兼容"，
   代码里没有检查（不兼容的方法会把并行开关静默忽略）；本仓库在配置期明确拒绝。
4. **`mask_hidden` 的加载规则**：上游"权重里有但没开并行 → warning + 跳过"；
   本仓库同样跳过（不打 warning，仓库没有 logger），但**开了并行却没有 → 报错**（与上游一致）。
5. **未接**：MTP + 并行的端到端只用 tiny 断言（`derive_mtp_draft_config()` 会把开关写进派生的
   hf 配置），本机没有带 MTP 层又能并行的 checkpoint；M-RoPE 未接（配置期报错，63 关就记着）。

## 7. 接口变化与遗留

* `SpeculativeConfig.parallel_drafting`（新，默认 False）；`max_num_new_slots_for_drafting`
  现在包含并行分支（PARD=K / P-EAGLE=K−1），调度侧的 `draft_slots` 自动跟着变。
* `SpecDecodeBaseProposer` 新增四个属性（`parallel_drafting` / `extra_slots_per_request` /
  `net_num_new_slots_per_request` / `needs_extra_input_slots`）与三个方法
  （`_init_parallel_drafting_params` / `_parallel_first_pass` / `_write_parallel_hidden`）。
* `minivllm/spec_decode/utils.py` 新增 `ParallelFirstPass` 与 `expand_parallel_draft_inputs()`。
* 遗留（属后续关卡）：drafter 侧 CUDA Graph（74）、DFlash/DSpark 协议（76/77）、
  PARD/P-EAGLE 真实权重下的质量与加速（需要 checkpoint，本机没有）、
  MTP+并行的真实权重端到端（80）。
