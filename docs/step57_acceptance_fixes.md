# step57 验收修复：独立探针抓到的 6 个问题

- 对应代码：`step57/core/{kv_cache_manager.py,sched/scheduler.py}`、
  `step57/worker/{gpu_model_runner.py,block_table.py}`、
  `step57/spec_decode/{rejection_sampler.py,draft_model.py}`
- 包摘要 SHA256：`486b74007c8c2054…`（61 个 .py / 7114 行；口径 = 包内 `*.py` 按相对路径排序，
  每个文件取自身 sha256，拼成 `名字\0哈希\n` 再取 sha256）
- 验收输入：`vllm-omni/learning_notes/14_vllm_from_scratch/验收记录/step57_review_20261002/`
  （三个独立探针 + 我的 15 个用例的日志）
- 修复后：三个探针**全部通过**（`review_draft_boundaries` 6/6、`review_rejection_boundaries`
  2/2、`review_inference_boundary` 1/1），我自己的用例补到 16 个脚本 **322 项**全过

## 0. 探针报了什么

| # | 探针 | 现象 | 性质 |
|---|---|---|---|
| 1 | `review_inference_boundary` | `engine.step()` 之后 KV 缓存上有 `grad_fn=CopySlices`，**保留的图节点每步 +30**（34→64→94→124） | **显存随步数单调上涨** |
| 2 | `review_draft_boundaries` | 4 个用例报 `IndexError: 第 0 行的第 N 个块不在块表里`（prompt 恰好占满整块 / chunked prefill） | 提议者写到没分配的槽位 |
| 3 | `review_draft_boundaries` | CUDA 上 `Expected all tensors to be on the same device` | 块表在 CPU、索引在 GPU |
| 4 | `review_draft_boundaries` | 提议看到的历史里没有本轮刚采样的 token | 提议与记账的顺序反了 |
| 5 | `review_rejection_boundaries` | `repeats.size(0) = 2 and input.size(0) = 3`（K=[2,1] 混批） | 温度张量被展开两次 |
| 6 | `review_rejection_boundaries` | 有 seed 的请求，输出随**全局** RNG 种子变；generator 映射为空 | 展开元数据时把 generators 清了 |

## 1. 改动内容（逐条，都优先按 vLLM 的做法）

### 1.1 显存级：KV 写入不能留 autograd 图

```python
    @torch.inference_mode()          # execute_model / sample_tokens
```

vLLM 在同名位置也有这个装饰器（`gpu_model_runner.py` 上 `execute_model`/`sample_tokens`/
`load_model` 等 8 处 `@torch.inference_mode()`）。没有它时，模型的参数默认
`requires_grad=True`，而 KV 写入是 `index_copy_` → 每次写入记一个 `CopySlices` 反向图挂在
KV 缓存上，**并随步数累积**（推理引擎里就是显存泄漏）。

**为什么只在"每步入口"划线、不在 `load_model` 上也加**：在 `load_model` 的推理模式下创建的
权重会变成"推理张量"，而本关的测试会直接用这些权重做前向（`check_step57_model_logits.py`
的驱动路径）——vLLM 在那里加是因为它自己包办 warmup/dummy run，本关不需要，所以留着权重
是普通张量。这条差异写进了对齐账本。

### 1.2 提议者要写的槽位：预留 `num_lookahead_tokens`

vLLM 的 `VllmConfig.num_lookahead_tokens` 注释写得很直白（原文抄在代码注释里）：

> The drafter writes KV for positions **beyond the target model's query range**, so every
> component that reserves blocks must add this margin.

我原来在 57E 的差异账本里写了"草稿在本轮调度范围内，所以不需要预留" —— **这是错的**：
target 的 query 覆盖的是"上一轮采用的那 K 枚草稿"，而提议者这一轮要写的是**更后面** K 个
位置（本轮刚采样的 token 之后）。所以：

- `KVCacheManager.allocate_slots(..., num_lookahead_tokens=0)`，`num_tokens_need_slot =
  min(computed + num_new + lookahead, max_model_len)`（vLLM 同款公式）；
- `Scheduler.num_lookahead_tokens = num_speculative_tokens` —— **只有 draft_model 才预留 K 个，
  ngram 是 0**（vLLM 的规则：`use_eagle() or uses_draft_model()` 才返回 `num_speculative_tokens`，
  ngram 两者都不是）✓ 这也解释了为什么我原来的 ngram 用例一直没暴露这个问题；
- 提议者仍然**不假定**槽位一定够：`BlockTable.covers(row, position)` 问一句，没有槽位就少提几枚
  （上下文快满时 lookahead 会被 `max_model_len` 截掉——vLLM 的原话是 "不能 schedule 刚好塞满后
  让 proposer 越界"）。

### 1.3 中间 prefill 块不提草稿

`_propose_draft_tokens(ready_rows=...)` 只对**本轮已 ready**（历史算完）的请求提。
中间 prefill 块既没有"next token"，它的 KV 槽位也没分配完。vLLM 在
`Scheduler.update_draft_token_ids` 里有对应的一句：*"Ignore draft tokens for prefill chunks"*——
本关更早一步就不提。

### 1.4 提议与记账的顺序

```python
        output = self._bookkeeping_sync(state, sampled)     # 先记账
        self.pending_draft_token_ids = self._propose_draft_tokens(...)   # 再提草稿
        return output
```

反过来的话，提议看到的 `all_token_ids` 里没有本轮刚采样的 token（历史少一个），而且 draft 侧的
进度会比 target 落后一格——于是"已发布的完整块"可能还没被 draft 算过（prefix 只在该 group
**各层**都有有效 KV 时才能发布，199 §9）。这两点都是验收方的探针分别抓到的。

### 1.5 拒绝采样：混批时用逐请求的温度展开

```python
        is_greedy_rows = expand_batch_to_tokens(
            sampling_metadata.temperature < SAMPLING_EPS, metadata.num_draft_tokens)
```

`target_metadata.temperature` 已经是展开过的 `[P]`，再展开一次：**K 不相等时当场报错**，
K 相等时静默算错（我的用例原来只测了 K=[1,1]，所以没抓到）。

### 1.6 展开元数据时保留 generator

抽样按 `enumerate(num_draft_tokens)` 的**请求下标**取 generator（与 vLLM 的
`generate_uniform_probs(..., generators, ...)` 同一套键）。我原来在展开时写了
`updates["generators"] = {}` —— 于是有 seed 的请求退化成用全局 RNG，输出随全局种子变。

### 1.7 CUDA：槽位映射在 CPU 上算完再搬

块表镜像是 CPU 结构（`.cpu`），索引也必须是 CPU 张量；算完 `.to(device)`。原来直接把 CUDA
的 `positions` 传进 `compute_slot_mapping` → `Expected all tensors to be on the same device`。

### 1.8 补：`_with_histories` 的"行序契约"

这个方法把**逐请求**的参数摊成**逐验证行**，靠的是"行序与传入的 `sampling_metadata` 一致"这条
隐含前提，而代码里什么都没查、还用 `zip()` 摊平——`zip` 按短的那边**静默截断**：

```text
元数据 2 行、min_tokens 只给 1 项 → 以前：不报错，第 2 行的停止 token 屏蔽悄悄消失
                                   现在：ValueError「元数据自相矛盾：min_tokens 有 1 项…」
只对一部分行建元数据（比如"只对 ready 行建"）→ 以前：IndexError 或静默错位
                                   现在：ValueError「采样元数据有 1 行，但投机元数据里有 2 条请求…」
```

检查放在两处：`forward()` 校验"采样元数据的行数 == 投机元数据的请求数"（**行序契约**，
`sampling_metadata` 与 `spec_metadata.req_ids` 必须逐请求对应），`_with_histories()` 校验
"各参数的项数 == 行数"与"ΣK == 验证行数"。vLLM 那边是**按索引 gather 张量**
（`prompt_token_ids[repeat_indices]`）而不是摊平 Python 列表——数组索引天然会检查长度，
本关保持列表、但把长度检查显式写出来。

## 2. 验证

| 探针 / 脚本 | 修复前 | 修复后 |
|---|---|---|
| `review_inference_boundary.py` | 0/1（每步 +30 图节点） | **1/1**（`requires_grad=False`、`grad_fn=None`、0 节点） |
| `review_draft_boundaries.py` | 2/6 | **6/6** |
| `review_rejection_boundaries.py` | 0/2 | **2/2** |
| 我自己的 16 个脚本 | 314 项 | **325 项**（把上面这些类都补成了回归用例） |

新增的回归用例（免得下次再漏）：

- `check_step57_runner_inputs.py` §7：跑完端到端之后，每个 KV 缓存 `requires_grad == False`
  且 `grad_fn is None`；
- `check_step57_rejection_sampler.py` §4b/4c：混批 + **ragged K**（K=[2,1]）、以及
  "有 seed 的请求输出与全局 RNG 状态无关"；
- `check_step57_draft_model.py` §6：prompt 恰好占满一个/两个块、中间 prefill 块、CUDA 设备、
  "提议看到本轮刚采样的 token"、"发布的完整块不超过 draft 侧进度"。

## 3. 教训（为什么会漏）

| 漏掉的原因 | 对策（这次做的） |
|---|---|
| 用例的批是**同构**的（K 全相等、全贪心或全随机），掩盖了"展开两次"这类错 | 用例里加 ragged K + 混批 |
| 只测了 CPU 路径 | draft 侧补 CUDA 用例（`torch.cuda.is_available()` 时跑） |
| 只测了 ngram（不需要 lookahead）就下结论"不需要预留" | 用 draft_model 跑"prompt 恰好占满整块"这类边界 |
| 没查过 autograd 状态 | 端到端用例加一条 KV 缓存的 `grad_fn` 断言 |
| 把"时序"写在文档里但没写成断言 | 用 spy 抓提议看到的历史内容 |

## 4. 接口变化

- `KVCacheManager.allocate_slots(..., num_lookahead_tokens=0)` 新增参数（默认 0，
  纯分配用例不受影响）。
- `Scheduler.num_lookahead_tokens`（按 vLLM 的规则：draft_model → K，ngram → 0）。
- `BlockTable.covers(row_index, position)` 新增。
- `SpecDecodeBaseProposer.propose(..., input_batch, ...)` 的 `block_table` 参数改为
  `input_batch`（它同时要块表与采样参数）。
- `RejectionSampler._verify(..., sampling_metadata=...)` 新增参数（要逐请求的温度）。
- `GPUModelRunner.execute_model` / `sample_tokens` 加 `@torch.inference_mode()`。
- 对齐账本同步：删掉"不需要 lookahead 预留"那条**错误**声明，改成实际做法。
