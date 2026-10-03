# step57：让 drafter 与 target 每轮同步（撤销发布边界的夹取）

- 对应代码：`step57/worker/gpu_model_runner.py`（`_propose_draft_tokens`）、
  `step57/spec_decode/{draft_model.py,ngram_proposer.py}`（`propose`）、
  `step57/core/sched/scheduler.py`（`update_from_output` / `_publish_blocks`）、
  `step57/outputs.py`
- 触发：204 §6.2 的"修复要求"给了两条路，上一版选了第二条（暂不发布），这一版改走第一条
  （**同步计算 draft 的确定前缀**），理由是真实 vLLM 就是这条路——本档把它对上。

## 0. 需求大概

204 §6.2 的原文给了两个选项：

> chunked prefill 尚未生成 next token 时，也要**明确如何同步计算 draft 的确定前缀**；
> 或者**暂不发布**该段 group 缓存，不能把只有 target 算过的块作为双模型命中。

上一版（`step57_acceptance_fixes.md` §1.9）选的是第二条：执行端对中间 prefill 块**干脆不跑
提议者**，控制端把发布边界夹到 `min(target 进度, draft 进度)`。

这次改成第一条，与 vLLM 相同。核对本机源码（`vllm 0.28.0`）：

- `v1/core/kv_cache_manager.py::allocate_slots` 发布时只有
  `num_tokens_to_cache = min(total_computed_tokens + num_new_tokens, request.num_tokens)`
  ——按 target 进度截断，**不夹 draft**；
- `v1/spec_decode/llm_base_proposer.py` 里 drafter 每步跑的就是 target 本轮的 query 范围，注释
  写明即使不产出草稿也要跑一遍："The prefill forward pass above already ran to keep the
  drafter KV cache in sync"；
- 中间 prefill 块的草稿由 `v1/core/sched/scheduler.py::update_draft_token_ids` 丢掉
  （"Ignore draft tokens for prefill chunks"）。

也就是说：**vLLM 用"每步跑同一段位置"保证 group 各层同步，不需要谁去夹发布边界。** 本关原来
把"prefill 块忽略草稿"实现得更早一步（不跑提议者），于是两边进度真的会分开，才需要那层对账。

## 1. 改动内容

1. **执行端：每个被调度的请求都过一遍提议者**（`gpu_model_runner.py::_propose_draft_tokens`）。
   中间 prefill 块也同步 KV，边界取 `num_computed_tokens_cpu[row] + num_scheduled_tokens[req]`
   ——**不是**"已知的全部 token"（那正是 204 §6.1 的越界写）。ready 行仍用记账后的已提交历史。
2. **提议者：同步与提议分开**（`draft_model.py::propose`）。第一遍前向覆盖**所有** `req_ids`
   并把 `_draft_computed` 推进到边界；`_sample`（也就是第一枚草稿）只对 `ready_req_ids` 里的
   请求做，于是自回归循环自然跳过非 ready 行。`.propose(..., num_tokens_no_spec, ...)` 的第三个
   参数因此改名为 `num_computed_tokens`——它现在的含义是"target 本轮之后算到哪"。
3. **ngram 同签名**（`ngram_proposer.py`）：只对 ready 的请求提，其余原样返回空列表（它不写
   KV，没有"同步"这件事）。
4. **撤销控制端的夹取**：删掉 `ModelRunnerOutput.draft_computed_tokens`、
   `SpecDecodeBaseProposer.draft_computed()`、`Scheduler._publish_blocks()` 的 `num_tokens`
   参数。发布回到 `cache_blocks(request, request.num_computed_tokens)`。
5. **用例**（`benchmarks/check_step57_draft_model.py` §6）：原来那条"draft=0 的轮次恒不发布"
   已不成立，换成三条新断言——中间 prefill 轮次 draft 进度 == target 进度、发布边界 ≤ draft
   进度、**发布位置上的 draft KV 绝对值和非零**（验收探针抓的就是"KV 全零但 hash 已登记"）。

## 2. 设计要点

**为什么"同步"就够，不再需要夹。** 不变量是"一个块只有在 group 里**每一层**都写完时才能声明
完整可复用"（199 §9）。发布按 `floor(num_computed_tokens / block_size)` 取完整块，所以只要
`draft 进度 ≥ target 进度`，发布出去的每个位置在 draft 那一层都已经写过。同步这一遍保证了
这个不等式；夹取是在它不成立时兜底，现在它成立了，兜底就是多余的机制。

**为什么中间 prefill 块要同步 KV、却不提草稿。** 同步是给"发布"用的：块发布不管本轮是不是
ready，只要 target 算满了整块就会登记。提草稿是给"下一轮验证"用的：中间 prefill 块还没有可
验证的 next token，`Scheduler.update_draft_token_ids` 本来就会把它丢掉（vLLM 同款规则，本关这
条规则一直都在）。所以这里比 vLLM 少做一步：草稿根本不提，省掉 K 次试探性前向。

**边界值为什么必须按"算到哪"而不是"知道哪些 token"。** 中间 prefill 块的 `num_tokens(row)`
是整段 prompt（token 早就都知道），但 KV 槽位只分配到本轮算完的位置。用前者当边界就会往没分配
的槽位写（204 §6.1 的 `IndexError`）。

**恢复仍然重置。** `reset_req_ids`（抢占恢复：块表整表换过）照旧清 `_draft_computed`，否则会
拿旧物理编号上的 KV 当历史；清掉之后下一轮从 0 重算整段——这就是"draft 自己的进度与 target 的
进度分开维护"的意义。

## 3. 与 vLLM 的差异账本

| 差异 | 说明 |
|---|---|
| **发布边界不夹 draft 进度** | 与 vLLM 相同（上一版不同，已撤销）。见 `step57_alignment.md` §5 的这一条 |
| 中间 prefill 块**不提草稿**，只同步 KV | vLLM 跑完 drafter 再让 Scheduler 丢草稿；本关在提议阶段就不提，少跑 K 次前向。副作用是**没有** vLLM 那两处兜底机制：vLLM 的 prefill lookahead token 会污染该块的 draft KV，它靠 `num_reprefillable_tokens` 排除 + EAGLE/MTP 命中时丢最后一块来修；本关不提就不写，所以不需要 |
| 仍然没有 `input_budget` / `max_num_new_slots_for_drafting` | 每轮按实际要补多少 token 现搭张量，只检查位置落在 `[0, max_model_len)` |
| `_draft_computed` 只在执行端（proposer）维护 | 与 vLLM 一致：控制端的 `Request` 上没有第二个进度对象（199 §9 明确要求"临时执行进度存在 Runner/proposer"） |

## 4. 验证

| 脚本 / 探针 | 结果 |
|---|---|
| `benchmarks/check_step57_draft_model.py` | 25 项全过（§6 换成新不变量） |
| `benchmarks/check_step57_{spec_lifecycle,spec_metadata,runner_inputs,rejection_sampler,prefix_cache,scheduler_basic,preemption,engine_protocol,request_progress}.py` | 9 个脚本全过（exit=0，共 202 项） |
| 验收探针 `review_draft_boundaries.py` | **6/6**（含"同一 KV group 发布的完整块必须覆盖 draft 层"） |
| 验收探针 `review_rejection_boundaries.py` | **2/2** |
| 验收探针 `review_inference_boundary.py` | **1/1** |

关键观察值（prompt 8、budget=2、block_size=4、K=1、prefix 开）：

```text
(target 进度, draft 进度, 缓存块数, 发布位置上的 draft KV 绝对值和)
(2,  2, 0, None)      # 还没有完整块
(4,  4, 1, 29.07)     # 中间 prefill 块：draft 已同步 → 完整块正常发布
(6,  6, 1, 29.07)
(8,  9, 2, 28.45)
(10, 11, 2, 28.45)
(11, 12, 2, 28.45)
```

对照上一版：`draft=0` 的那几轮一个块都不发布（`cached_blocks=0`）。现在两边每轮都对得上，
发布也不再推迟。验收探针那条独立检查（直接读 `_draft_computed` 与 `request.num_computed_tokens`）
从"靠夹取满足"变成"靠同步满足"：每一步都是 `published ≤ draft_computed`（4≤7、8≤9、8≤11、
12≤13）。

没跑的：`benchmarks/check_step57_vllm_boundaries.py`、`compare_step57_vllm.py` 这类需要真实
vLLM 引擎/真实权重的对照（按约定等通知再跑）。

## 5. 接口变化与遗留

- `SpecDecodeBaseProposer.propose(req_ids, all_token_ids, num_computed_tokens, input_batch,
  ready_req_ids=None, reset_req_ids=None)`：第三参数改名（`num_tokens_no_spec` →
  `num_computed_tokens`，含义从"该行已知多少 token"变成"target 本轮算到哪"），新增
  `ready_req_ids`。
- `NgramProposer.propose(...)`：同签名（只对 ready 的请求提）。
- **删除**：`ModelRunnerOutput.draft_computed_tokens`、`SpecDecodeBaseProposer.draft_computed()`、
  `NgramProposer.draft_computed()`、`Scheduler._publish_blocks(request, num_tokens=None)` 的参数。
- 遗留：草案仍然"提前 K 个位置写 KV"（lookahead 预留 + `BlockTable.covers` 检查），
  只是非 ready 行不再产生草案；EAGLE/MTP 的输入槽与多 group 仍未做（199 §9 允许收窄）。
