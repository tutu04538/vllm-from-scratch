# 第七十七关：DSpark 条件采样与实际 Proposal 分布

> 需求：[`077_DSpark条件采样与实际Proposal分布.md`](../../vllm-omni/learning_notes/14_vllm_from_scratch/投机解码完整需求/077_DSpark条件采样与实际Proposal分布.md)
> 基线：本机 `vllm==0.28.0` 的源码快照（`v1/worker/gpu/spec_decode/dspark/{speculator,utils}.py`、
> `model_executor/models/qwen3_dspark.py`、`config/speculative.py:241/298/1013-1142`）
> 状态：**只有 V2 一条执行路径**（上游 `use_v2_model_runner` 也只为 dspark 强制 V2），已实现并自检；
> ⚠️ **本机没有真实 DSpark 权重**，草稿质量与加速比未验；**自适应验证的阈值/预算属 79 关**。

## 0. 一句话

DSpark = **DFlash 的并行主干** + **串行 Markov 纠正**：一次非因果 query 前向给出每请求 K 个位置的
base logits，然后从左到右串行采 K 步，每步把"上一个实际选中的 token"经低秩头
（`markov_w1[token] → markov_w2 → V_draft` 偏置）加到当前 logits 上；**并且把每一步真正用的那份
logits 缓存成 `q`** 交给拒绝采样器。

本关就是把这条"并行出特征、串行出分布、q 与实际提议一致"的路径在 V2 里实现，
用**与上游内核逐值差分**钉住两种 anchor 布局，用**冻结参考**钉住"条件分布"与"缓存的 q"。

---

## 1. 为什么需要这一关（痛点）

1. **并行算出每个位置的基础特征 ≠ 各候选相互独立**：DFlash 的 K 枚草稿是同时从"互不知情"的
   分布里抽的；真实文本里第 3 个字取决于第 2 个字。DSpark 用 K 次极便宜的串行偏置把块内依赖
   补回来（一次 `[B,r]×[r,V]` 的量级，比主干便宜几个数量级）。
2. **验证器要的是"实际用的 q"**：拒绝采样无损的前提是 `q` = 实际提议分布。只在采样时加偏置、
   验证时用 base logits，接受概率就会算错，交付分布不再等于 target 的分布——**而且不会报错**。
   所以 `_sample_logits()` 把"采样"和"写缓存"做成同一份（`gumbel_sample(..., logits_cache=...)`）。
3. **全词表串行太贵**：每步要算一次 `[B,V]` 的低秩偏置（V=15 万时是笔真开销，而且 K 步是
   **串行**的、每步都要等上一步采样）。`dspark_draft_topk` 先把候选截到 k 个，串行步只对候选
   算偏置——**省的是偏置那一半**：截断后的分布仍交给普通稠密采样器，采样/softmax 照旧扫全词表
   （上游字段注释同样只承诺"只对候选求值"）。**代价是 q 也变成截断分布**，所以缓存必须缓存
   截断后的那一份（否则又是"q 与实际不一致"）。
4. **缩减词表**：draft 词表比 target 小，而验证器按 target 词表索引 q。所以散射要在**采样之前**
   把 draft logits 映回 target 宽度，且不存在的列必须恒 `-inf`（不能继承上一轮的值）。

---

## 2. 数据流

```
target 本轮（num_tokens 行，含被拒草稿）
   │  aux hidden ──combine_hidden_states(fc)──► draft hidden（num_tokens 行）
   │                                            │
   │                        precompute_and_store_context_kv（图外 eager，写草稿 KV）
   ▼
prepare_dflash_inputs（GPU 内核，SAMPLE_FROM_ANCHOR 决定布局）
   │  query 行（K 或 1+K）+ 采样映射（sample_indices / sample_pos / sample_idx_mapping）
   ▼
query 前向（FULL 图；非因果注意力）
   │  [num_query_tokens, hidden]
   ▼
_sample_sequential(num_reqs, head_hidden)                       ← 图里也录这一段
   │  base_logits = compute_draft_logits(hidden)   （**draft 宽度**、不含 d2t 散射）
   │  prev = input_ids[anchor 行]（= bonus token）
   │  for i in range(K):
   │      bias = logits_processor(markov_w2, markov_w1[prev])   （含 logit_scale）
   │      logits_i = base_logits[:, i] + bias   （top-k 时只给候选列加、其余 -inf）
   │      draft_i = _sample_logits(logits_i, …, step=i)
   │          └─ 缩减词表：先 scatter 到 target 宽度的 -inf 缓冲
   │             gumbel_sample(..., logits_cache=draft_logits[req, i])   ← q 在这里被写下
   │      prev = draft_i
   ▼
draft_tokens [num_reqs, K]（target 空间的 id）+ draft_logits [max_num_reqs, K, V_target]
```

`sample_from_anchor` 的两种布局（**同一个内核**、同一个 constexpr，见 §3）：

```
anchor-as-first（默认 True）          K 个 query：
  query 位置      last_valid_pos+1 … +K     第 0 行 = bonus token（真实），其余 = noise token
  采样行          **全部 K 行**，位置 = query 位置 + 1（每行预测下一个 token）
  seq_lens        last_valid_pos + 1 + K

fill-in（False，speculators 格式）    1+K 个 query：
  query 位置      last_valid_pos+1 … +K+1   第 0 行 = bonus token，其余 = mask token
  采样行          只有 mask 行（off ≥ 1），位置 = **它自己**的 query 位置（DFlash 口径）
  seq_lens        last_valid_pos + 1 + (1+K)
```

⚠️ 两种布局**草稿位置相同**（`sample_pos` 都是 10,11,12 这类"被预测 token 的位置"），差别只在
"谁坐在哪个 query 位置上"——探针 B3 与单测 `test_both_layouts_draft_the_same_positions` 钉的就是它。

---

## 3. 输入布局（与 DFlash 共用内核）

`_prepare_dflash_inputs_kernel` 在 77 关补上了 `SAMPLE_FROM_ANCHOR: tl.constexpr`（上游本来就有，
76 关按"配置期已拒绝"裁剪掉了）。新增/变化的只有采样段：

```python
sample_off = 0 if SAMPLE_FROM_ANCHOR else 1
is_sample  = is_query & (query_off >= sample_off)
sample_idx = req_idx * num_speculative_steps + (query_off - sample_off)
sample_pos = query_pos + 1 if SAMPLE_FROM_ANCHOR else query_pos
```

其余（context 行、query 行、槽位、`seq_lens`、补齐行 −1）一字未动。DFlash 传 `False`（默认值），
DSpark 传 `self.sample_from_anchor`。

| 量 | anchor-as-first（K=3，请求 0） | fill-in（K=3，请求 0） |
|---|---|---|
| `query_start_loc` | `[0, 3, 6, …]` | `[0, 4, 8, …]` |
| query 位置 | `9, 10, 11` | `9, 10, 11, 12` |
| `input_ids` | `[bonus, mask, mask]` | `[bonus, mask, mask, mask]` |
| 采样行（`sample_indices`） | `[0, 1, 2]` | `[1, 2, 3]` |
| `sample_pos` | `10, 11, 12` | `10, 11, 12` |
| `seq_lens` | `12`（= 8+1+3） | `13`（= 8+1+4） |
| 槽位预算 `max_num_new_slots_for_drafting` | `K−1`（上游表：DSpark = K−1） | 同（配置口径按方法定） |
| KV lookahead `num_lookahead_tokens` | `K`（上游 `VllmConfig.num_lookahead_tokens` 的 dspark 分支） | 同 |

---

## 4. 实现对照

| 本项目 | 上游 | 状态 |
|---|---|---|
| `minivllm/worker/gpu/spec_decode/dspark/speculator.py::DSparkSpeculator` | `v1/worker/gpu/spec_decode/dspark/speculator.py` | `__init__` / `load_draft_model` / `_sample_logits` / `_sample_sequential` / `_sample_sequential_topk` / `_generate_draft` 逐条对齐；`propose()` 继承 DFlash 的 |
| 同上 `utils.py::load_dspark_model` | `v1/worker/gpu/spec_decode/dspark/utils.py` | 共享 `embed_tokens`/`lm_head` + PP 拒绝；裁剪见 §5 |
| `minivllm/models/qwen3_dspark.py` | `model_executor/models/qwen3_dspark.py` | `DSparkMarkovHead` / `DSparkConfidenceHead` / `Qwen3DSparkModel` / `Qwen3DSparkForCausalLM` |
| `minivllm/models/qwen3_dflash.py`（共享部分） | 同上基类 | 新增 `logits_processor`（`logit_scale`）、`dflash_mask_token_id()`、`tie_embeddings` |
| `minivllm/worker/gpu/spec_decode/dflash/speculator.py::_prepare_dflash_inputs_kernel` | 同名内核 | 补 `SAMPLE_FROM_ANCHOR`；两种布局各 12 个输出逐值相等 |
| `minivllm/testing/torch_dspark.py` | —（本仓库的冻结参考） | 串行 Markov + top-k 截断 + d2t 散射，逐个函数可复算 |
| `minivllm/testing/tiny_models.py::tiny_dspark_dir` | —（测试权重生成器） | 两种布局、缩减词表（含非零 d2t 偏移）、top-k、confidence 头、省略 embed/lm_head |
| `SpeculativeConfig._resolve_dspark()` / `_resolve_adaptive_verification_field()` | `config/speculative.py:1013-1142` | 架构/K/n_predict/top-k/占位符/79 关边界 |

模型注册名沿用上游的"注册名 ≠ 类名"：checkpoint 的 `architectures` 是 `Qwen3DSparkModel`，
类名是 `Qwen3DSparkForCausalLM`（`models/registry.py` 里一处对应）。

**图**：`_generate_draft()`（主干前向 + K 步串行采样）整个录进 FULL 图；两种布局的
`decode_query_len` 分别是 K / 1+K。K 步之间是数据依赖，**不可能**退化成"各位置独立 argmax"
（那正好是 DFlash 的语义，也是本关要修的病）。

---

## 5. 差异账本（逐条）

| 编号 | 差异 | 为什么不改 | 影响 |
|---|---|---|---|
| D-77-1 | 只实现 **Qwen3** 一族的 DSpark；`Gemma4DSparkModel` / `K3DSparkModel` / `DSparkDraftModel`（DSV4）在配置期拒绝 | 上游这四族的头不同（K3 是 MLA、DSV4 复用 target 的完整配置与 `dspark_target_layer_ids`、Gemma4 要归一 `n_predict`），本机没有权重也无法验证 | 加载/配置期明确报错，不猜基座 |
| D-77-2 | `enable_adaptive_verification=True` 在配置期**拒绝**（79 关） | 需求 077 §5：阈值/预算决策放到 79；接受配置却不裁剪验证枚数就是静默降级 | 置信度**照算并落盘**（`draft_token_confidence_probs`），只是没有消费者。**79 关已放开配置**（DSpark 接受，其余方法拒绝），执行期由 Runner 检查后端能力；本行保留 77 关当时的状态 |
| D-77-3 | 没有 `attention_config.use_non_causal` / `kv_cache_dtype` / quant 覆盖 | 本仓库只有一个注意力后端（因果性由模型逐层决定）、无量化、无 KV dtype 轴 | 无行为差异（上游那两个 `replace(...)` 在本仓库没有对应字段） |
| D-77-4 | PP 检查放在**建模之前**（上游在建模之后） | 本仓库没有分布式 PP 轴；早报错少一次白建模型 | 语义相同（同一句 `NotImplementedError`） |
| D-77-5 | `confidence_head.proj` 用 `nn.Linear`，`load_weights()` 开始时把它升回 **fp32** | 本仓库没有 `ReplicatedLinear`（TP=1 下它就是 `nn.Linear`），而加载器会把整个模型 `.to(dtype)`；上游用 `params_dtype=torch.float32` | 数值等价（fp32 权重 + fp32 输入 + 上游同一条 `sigmoid`） |
| D-77-6 | `sample_from_anchor` 只读**顶层**配置（与上游相同）；`dflash_config.sample_from_anchor` 仍归 DFlash 的检查管 | 上游就是两个键；DFlash 的检查在 `method != "dspark"` 时才生效 | 若有人把 DSpark 的开关写进 `dflash_config`，DFlash 的检查会先报错（明确，不静默） |
| D-77-7 | **顺带修**：V2 草稿 KV 缓存的 dtype 取"第一个**浮点**参数" | 缩小词表的草稿把 `d2t` 注册成 int64 的 `nn.Parameter`，它在 `named_parameters()` 里排在子模块之前 → 草稿 KV 变整型，写 K/V 时 `index_copy_` 直接崩 | 76 关的缩小词表 DFlash 也一并受益（同一行代码） |
| D-77-8 | **顺带对齐**：`DFlashForCausalLM` 增加 `logits_processor`（`logit_scale`），`compute_logits()` 走它 | DSpark 的 Markov 偏置必须与 base logits 共用同一个 scale；复制一份不如共用 | `logit_scale` 缺省 1.0 → 76 关数值不变（回归全绿） |
| D-77-9 | DFlash 的 V2 加载**仍不**共享 target 的 `embed_tokens` / `lm_head`（76 关边界） | 要改就得同时改 V1（V1 也不共享），否则 V1/V2 分叉；属后续关 | checkpoint 省略 `embed_tokens` 时加载器**明确报错**（不静默留随机嵌入），不是静默错 |

---

## 6. 验证

```bash
PY=/home/user/anaconda3/envs/vllm-omni-dev/bin/python
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY -m pytest tests/step77 -q
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY benchmarks/check_step77_dspark.py    # 28 项
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    $PY -m pytest tests/step58 … tests/step77 -q                             # 全量回归
```

### 6.1 布局逐值（与上游内核）

| 对象 | 结果 |
|---|---|
| anchor-as-first（K=3）vs 上游 `_prepare_dflash_inputs_kernel(SAMPLE_FROM_ANCHOR=True)` | **12/12 输出逐值相等**（含 `sample_pos`、`sample_idx_mapping`、`seq_lens`、补齐行） |
| fill-in（1+K）vs 上游同内核 `SAMPLE_FROM_ANCHOR=False` | **12/12 输出逐值相等** |
| K=1 边界（两种布局） | 12/12 逐值相等；`seq_lens` = 10 / 11，采样行 = 1 枚 |
| 两布局的草稿位置 | 完全相同（`10, 11, 12`）；只有 query 行数与 `seq_lens` 差 1 |

### 6.2 条件采样与 q（冻结参考）

| 对象 | 结果 |
|---|---|
| 缓存的 q vs 沿**实际采样路径**复算的条件分布（probabilistic，温度 0 与 0.7，两种布局） | `max abs(Δ) = 0.0`（探针 C1 实测 `0.00e+00`） |
| q 是否真的含 Markov 偏置 | 是：`q != base_logits`（单测断言） |
| 同一份 base logits、换前一个 token | 第 1 步 q 从 `[2,1,0,…]` 变 `[2,4,0,…]`，选择从 0 变 1（探针 C2） |
| 缩减词表（`draft=7`、`d2t` 偏移 2） | 非 d2t 列恒 `-inf`；`softmax(q)` 在 target 宽度上和为 1；采样 id ⊆ `{2..8}`；与参考逐值相等 |
| 重复 batch（同一 slot 换输入） | q 整份被覆盖，与参考逐值相等（无残留） |
| top-k=3 | 每步恰好 3 个有限列；采样 id 必在候选内；与参考逐值相等 |
| top-k = 全词表 vs 不开 top-k | 草稿 token 逐值相同；q 只差浮点末位（`F.linear` vs `baddbmm`） |
| 采样是否从 q 抽（2000 个种子） | 经验分布 vs `softmax(q)` 的 **TV = 0.0249**（阈值 0.1） |
| confidence | `draft_token_confidence_probs == sigmoid(proj([hidden, markov_embed]))`（atol 1e-6），每位置一个 fp32 |
| greedy | `draft_logits is None`（点质量 q）；token == 参考 argmax |

### 6.3 端到端（tiny）

| 项 | 结果 |
|---|---|
| 两种布局贪心输出 | 相同（`{'a': [4,4,6,0,6,0], 'b': [10,4,4,4,4,0]}`） |
| 图 vs eager（probabilistic，温度 0.7，同 seed） | **逐轮草稿 token 相同**（4 轮）；捕获 `captured=1`、分派记录 3 条 |
| greedy vs probabilistic（温度 0） | 输出相同 |
| 缩减词表 + top-k + 概率草稿 | 能跑通 |
| 采样映射缓冲 | `max_num_reqs × K`；每轮只点亮 `num_reqs × K` 行（补齐行 −1 惰性） |

### 6.4 回归

| 命令 | 结果 |
|---|---|
| `pytest tests/step77 -q` | **40 passed**（布局 10 + 条件采样/q 15 + 边界与加载 15） |
| `benchmarks/check_step77_dspark.py` | **28/28** |
| `pytest tests/step58 … tests/step77 -q` | **836 passed**（= 76 关的 796 + 本关 40；无回归） |
| step76 的 27 项探针 + step58~75 的探针 | 全部通过（布局内核改成带 `SAMPLE_FROM_ANCHOR` constexpr 的同一份，76 关的 13 项布局测试不受影响） |

⚠️ **顺带修了一个既有的偶发失败**：`tests/step63/test_eagle_model.py::tiny_model()` 直接构造
`Eagle3ForCausalLM`，参数是 `torch.empty`（生产路径一定会被 `load_weights()` 覆盖）——全量跑时
这块内存可能恰好是 NaN，`test_compute_logits_is_identity_without_d2t` 的 `torch.equal`（比两次相同
计算）就会判成"不相等"（单独跑该文件永远通过）。现在用固定 seed 把 tiny 模型的参数填成确定且有限的
值，断言强度不变。与 77 关的行为无关，但会在全量回归里随机冒出来，所以在这里修掉并记录。

---

## 7. 已知边界

1. **没有真实 DSpark checkpoint**：只验证协议/数学/分布，草稿质量与加速比"集成待验"。
2. **只有 Qwen3 基座**（D-77-1）；Gemma4/K3/DSV4 在配置期拒绝。
3. **自适应验证未接**（D-77-2）：置信度算出来、落在 `draft_token_confidence_probs`，但没有消费者；
   阈值与预算属 **79 关**。
4. **DFlash 的 V2 加载不共享词嵌入**（D-77-9）：省略 `embed_tokens` 的 DFlash checkpoint 会明确
   报"参数没有被加载"，而不是静默用随机嵌入。
5. **top-k 的代价**：真值掉出候选就永远抽不到；本关只保证"截断后的 q 与采样一致"，不评估掉出率。
6. **`num_lookahead_tokens = K`**（上游 dspark 分支）：DFlash 那边上游是 `K+1`，本仓库沿用 76 关的
   `K`（已够覆盖 DFlash 的 `1+K` 行 query，见 §3 的说明）。
