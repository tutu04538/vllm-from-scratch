# step57E：投机架构回接

- 对应代码：`step57/spec_decode/{metadata,rejection_sampler,ngram_proposer,draft_model}.py`（新增；
  `SpecDecodeBaseProposer` 与 `DraftModelProposer` 同放在 `draft_model.py` 里，vLLM 是分两个文件），以及 `core/sched/scheduler.py`（草稿的采用与回退）、`worker/gpu_model_runner.py`
  （验证路径、下一轮提议）、`sample/metadata.py`（`spec_token_ids`）、`outputs.py`（`DraftTokenIds`）
- 包摘要 SHA256：`486b74007c8c2054…`（61 个 .py / 7114 行；口径 = 包内 `*.py` 按相对路径排序，
  每个文件取自身 sha256，拼成 `名字\0哈希\n` 再取 sha256）
- 验收脚本：`benchmarks/check_step57_{spec_metadata,rejection_sampler,spec_lifecycle,draft_model}.py`
  （对应需求里点名的 `test_spec_metadata.py` / `test_rejection_sampler.py` /
  `test_spec_lifecycle.py` / `test_draft_model.py`）
- 参考实现：本机 `vllm 0.28.0`（`v1/spec_decode/*`、`v1/sample/rejection_sampler.py`、
  `v1/core/sched/scheduler.py` 的草稿采用/回退、`v1/worker/gpu_model_runner.py` 的
  `_calc_spec_decode_metadata`）

## 0. 需求大概

199 + 200 §57E：把投机**按新架构的时序**接回来，并且"不能以关闭投机时通过宣布这段结束"。
判定点是那 10 条：greedy/random 混批、固定 p/q 与预设随机输入（首/中/全接受、K=0、ragged）、
独立 CPU 公式与统计对照、EOS 截断、q 随实际计划重排/截断、抢占清旧提议、
ngram 与真实 draft 模型都接通、target/draft 同 group 不互相读写 tensor、以及
**"轮 t 提议 → 轮 t+1 调度"的调用轨迹**。

## 1. 改动内容

| 文件 | 对应 vLLM | 它回答的问题 |
|---|---|---|
| `spec_decode/metadata.py` | `v1/spec_decode/metadata.py` + runner 的 `_calc_spec_decode_metadata` | 一轮里哪些行要 logits、验证行/bonus 行是哪几行（**两个坐标系**） |
| `spec_decode/rejection_sampler.py` | `v1/sample/rejection_sampler.py` 的 Torch 路径 | 草稿怎么被接受/拒绝、拒绝后用 `max(p−q,0)` 恢复 |
| `spec_decode/draft_model.py::SpecDecodeBaseProposer` | `v1/spec_decode/llm_base_proposer.py` | 提议的**步骤**：自己的输入缓冲、自己的 KV、自回归提 K 枚 |
| `spec_decode/draft_model.py` | `v1/spec_decode/draft_model.py` | 用哪个模型：加载、词表/KV 规格校验 |
| `spec_decode/ngram_proposer.py` | `v1/spec_decode/ngram_proposer.py` | 从历史里找重复片段当草稿（确定性、无 q） |
| `core/sched/scheduler.py` | 同名 | 下一轮采用几枚（`num_tokens_with_spec` 已经在预算里）、**被拒的草稿把进度退回来** |
| `worker/gpu_model_runner.py` | 同名 | 验证路径、草稿写进输入缓冲、`sample_tokens` 里顺手提下一轮草稿 |
| `outputs.py` | `v1/outputs.py::DraftTokenIds` | 执行侧交回"下一轮草稿"的最小数据（ID + token，**不带 `[P,V]` 概率**） |

## 2. 设计要点

### 2.1 一轮的形状：K+1 行

```text
请求的 query 行：  [ b ][ d1 ][ d2 ] ... [ dK ]        K+1 行
                    │     │      │         │
                    │     └──────┴─────────┴─→ 草稿（**它们是输入**）
                    └─ b = 上一个已提交但还没算 KV 的 token（普通 decode 也是这一行）
```

验证 d1 需要"b 位置的 logits"、验证 d2 需要"d1 位置的 logits"……最后一行 dK 的 logits 用来采
**bonus token**（全接受时白送一个）。所以 K=0 的请求退化成"一行的普通解码"——
**投机打开时不需要另写一条解码路径**（199 §6 的算例与本关的 `SpecDecodeMetadata` 逐值一致）。

### 2.2 两个坐标系（最容易搞错的地方）

```text
logits_indices          指向**扁平化的 forward 行**（取 logits 用）
target/bonus indices    指向**取完之后**的 [P+B, V] 张量（验证与 bonus 用）
```

所以 `target_logits_indices` 的前缀和用的是 `cu_num_sampled_tokens`（每请求 K+1 行）而不是行号。
`check_step57_spec_metadata.py` 的第一段就是拿 vLLM 注释里的算例逐值比。

### 2.3 时序：轮 t 提议 → 轮 t+1 采用

```text
轮 t  schedule（用已有的草稿）→ target forward → sample_tokens：验证 + 顺手提下一轮草稿
      → update_from_output（修正进度/提交/停止）→ post_step：take_draft_token_ids
轮 t+1 schedule 决定**采用几枚**（预算不够就截短，剩下的丢掉）
```

**提议不改本轮计划**：草案只存在请求的 `spec_token_ids` 上，下一轮 `schedule()` 才决定采用。
用例里有一条专门查这个（"提草稿的那一轮，包里没有它的草稿"）。

### 2.4 被拒的草稿要把进度退回来

`_update_after_schedule()` 是按"排了几行"推进的（K+1 行），但有效的只有"接受的 a 枚 + 最后那个
token"：被拒的草稿虽然算过 KV（下一轮会被同位置的写入覆盖），但它们不对应任何已提交 token。
`update_from_output()` 里按 `num_rejected = K - (len(generated) - 1)` 做减法（vLLM 在同一位置
做同样的事）。**不做这条的话下一轮会从错误的位置续算**——用例断言"每轮结束时
`num_computed_tokens == num_tokens - 1`"。

### 2.5 验证算法（199 §7）

```text
greedy 行：逐位置比对草稿 == target argmax；首个不同处用 argmax 顶替，后面全丢
random 行：接受概率 min(1, p[d]/q[d])，用一次均匀随机数判定；
           拒绝 → 用 max(p − q, 0) 采 recovered token（**不需要归一化**：指数竞赛只比大小）
全接受   → 追加 bonus
```

- `q` 必须是**实际提议时的分布**；ngram 这种点质量提议用 `draft_probs=None` 表示（`q[d] = 1`），
  **不能**把"没有 q"当"按 p 随便采"；
- `q[d] == 0` 防御性拒绝（vLLM 内核同样处理）；
- **历史条件**：第 j 个验证位置的 p 要按"已提交历史 + 草稿前缀 `[:j]`"应用惩罚与约束
  （`combine_outputs_with_spec_tokens` 造临时假设历史，不动权威历史）；被拒之后后面预计算的
  结果直接丢掉。

### 2.6 随机数（199 §8）

删掉了旧代码那套 counter RNG / 事件记账。现在只有：按 draft 段预生成均匀随机数（**K=0 的请求
不消耗随机数**）、recovered 用每请求一行噪声的指数竞赛。bonus 与 recovered 可能算了没用上，
**不回滚随机流**。测试可以注入固定的 uniform / recovered，把"实现差异"与"算法错误"分开。

### 2.7 draft 模型：不是第二套引擎

`SpecDecodeBaseProposer` 只有"历史进、草稿出"：

```text
第一遍 forward：把"上一轮新提交的那段"喂给 draft，把它的 KV 补到与 target 一致，
                最后一行的 logits 得到第一枚草稿
自回归 K-1 步：每步一行（上一枚草稿当输入）
```

- **有效边界**：上一轮为被拒草稿写过的 KV 还在物理缓冲里，但不是有效历史——每轮按"已提交到哪"
  重建 positions 与 slot_mapping（199 §9）；
- **KV**：与 target 共用逻辑块表与 slot 编号，但**每层绑自己的物理 tensor**
  （用例断言"同名层在两边是不同的 tensor、形状相同"）；
- **规格校验**：词表、dtype、KV head 数、head_dim、block_size 不一致 → 明确报错，
  **不给独立 pool 兜底**（199 §9 的原话）；
- 恢复（抢占）后重置 draft 侧的进度：旧物理编号上的 KV 已经不属于它了。

## 3. 与 vLLM 的差异账本

| 差异 | 说明 |
|---|---|
| `num_lookahead_tokens` 按 vLLM 规则实现（draft_model → K，ngram → 0）；仍没有 `input_budget` | 提议者写 target query 之外的 K 个位置，必须预留；输入缓冲不预分配，只检查位置边界（见对齐账本） |
| 没有独立的输入预算（`input_budget` / `max_num_new_slots_for_drafting`），也**没有预分配的定长输入缓冲** | 每轮按"实际要补多少 token"现搭张量，只检查位置落在 `[0, max_model_len)`。一轮的行数不小：**恢复之后 draft 要重算整段历史** |
| `RejectionSampler` 返回 padded `[B, max_spec_len+1]`（无效位 -1），Runner 裁掉 | 与 vLLM 同形；本关照做，但**不**为它准备 padded 的中间张量 |
| `draft_probs` 由执行侧按请求存、下一轮按"实际采用的前缀"重排（`_align_draft_probs`） | 199 §5 要求的 q 对齐。vLLM 也存概率并按请求重排（`take_last_draft_probs`），只是它按**批行号**索引，本关按**请求 ID**（并因此给 `SpecDecodeMetadata` 加了 `req_ids`，vLLM 没有这个字段） |
| 没有 synthetic mode / fp64 Gumbel / logprobs / 结构化输出过滤 | 实验与对照用途，或不属于 57E |
| **提议侧的 q 只做温度 + top-k/top-p，不做惩罚** | vLLM 的注释把这条写明了："we ignore most of the sampling parameters in generating the draft tokens. We only use the temperature. While this could degrade the acceptance rate, it does not affect the distribution of the generated tokens after rejection sampling."。拒绝采样对**任何** q 都成立，省掉约束只影响接受率。本关额外做了 top-k/top-p（与 target 的 p 更接近，接受率略高，代价是每次提议多一次排序）；**惩罚两边都不做** |
| 验证侧施加 min_tokens 的停止 token 屏蔽（vLLM 用 `MinTokensLogitsProcessor.apply_with_spec_decode`） | 本关复用普通采样器的同一条屏蔽逻辑，历史用"已提交 + 草稿前缀"的假设历史 |
| 只有一条 Torch 路径（vLLM 走 Triton 内核） | 199 §7 明确允许："第一版先 Torch 可读实现，函数边界对应源码" |
| EAGLE/MTP 的"左移一位输入"、draft 的 lookahead 输入槽不做 | 本关只有普通自回归 draft（199 §9 允许收窄并记录） |

## 4. 验证

| 脚本 | 项数 | 覆盖 |
|---|---:|---|
| `check_step57_spec_metadata.py` | 13 | vLLM 算例逐值对照（cu_num_draft / cu_num_sampled / logits / target / bonus）、K=0 退化、ragged、预填块、草稿数与行数不符要报错 |
| `check_step57_rejection_sampler.py` | 18 | greedy 首/中/全接受、random 接受与拒绝（注入 uniform/recovered）、边界 `u == p/q`、`q[d]=0` 防御性拒绝、与独立 CPU 公式逐值一致、recovered 分布 ∝ max(p−q,0) 的统计检查、K=0、ragged、点质量提议、greedy/random 混批、min_tokens 的停止 token 在投机路径上同样被屏蔽 |
| `check_step57_spec_lifecycle.py` | 12 | 提议不改本轮计划、下一轮才采用（逐枚对照）、K 裁剪（预算只够 1 行时一枚都不发）、进度回退不变量、抢占清空草稿、**greedy 下开/关投机输出逐 token 一致**、草稿位置被块表覆盖 |
| `check_step57_draft_model.py` | 19 | 词表/KV 规格不兼容明确报错（不给独立 pool 兜底）、draft 与 target 的 KV 是两份 tensor/一张块表、端到端出 token、草稿确实在被提、同 seed 可复现 |

十五个脚本全部通过（共 322 项；含验收修复后补的回归用例）。

真实模型演示（本机 Qwen3-1.7B 作 target + tiny fixture 作 draft 需要同词表，所以这里用
**同一个小模型的 1 层切片**当 draft，见用例）：

```text
draft 提议 15 枚，target 每轮验证 K+1 行；greedy 下开/关投机的输出逐 token 一致
```

**没有跑**性能测试：本关不设吞吐目标（199 的收尾原话就是"先把职责、概率条件、状态时序学对，
再讨论并行词表抽样"）。

## 5. 接口变化与遗留

- 新增可导入名字：`step57.spec_decode.{SpecDecodeMetadata, RejectionSampler, NgramProposer,
  SpecDecodeBaseProposer, DraftModelProposer, PLACEHOLDER_TOKEN_ID, expand_batch_to_tokens}`、
  `step57.outputs.DraftTokenIds`。
- `SamplingMetadata` 新增 `spec_token_ids`（验证路径要按"历史 + 草稿前缀"算惩罚）。
- `InputBatch` 新增 `spec_token_ids` 区、`num_tokens_with_spec()`、`update_req_spec_token_ids()`。
- `Scheduler.__init__` 多收一个 `speculative_config`；新增 `update_draft_token_ids()`
  （57A 就留好了 `post_step` 的调用点）；`schedule()` 现在真的会填
  `scheduled_spec_decode_tokens`。
- `SpeculativeConfig.method` 支持 `"ngram"` / `"draft_model"`；未知方法明确报错。
- 一处 57D 的潜在 bug 顺手修了：`SamplingMetadata.from_input_batch` 在"整批都不 ready"时
  会漏传 `top_k`/`top_p`（`TypeError`），现在返回一份空元数据，采样器在空张量上跑也不产出。
- **遗留**：EAGLE/MTP（左移一位、hidden states 输入、lookahead 输入槽）、
  结构化输出对草稿的过滤、KV 连接器/多 group 的投机、投机指标（接受率统计）、
  并行词表抽样的 kernel 化（57F 的源码回读之后再说）。
