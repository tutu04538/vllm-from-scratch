# step57 架构与源码对照（57F）

- 对应代码：`step57/`（不改代码，本文是**回到真实源码**的对照）
- 对照对象：本机 `vllm 0.28.0`（`/home/user/anaconda3/envs/vllm-omni-dev/lib/python3.12/site-packages/vllm`）
- 轨迹脚本：`python benchmarks/trace_step57.py`
- 数学对照：`benchmarks/compare_step57_vllm.py`（A/B/C/D 四段）
- 边界练习：`benchmarks/check_step57_vllm_boundaries.py`
- 差异账本：见 [`step57_alignment.md`](step57_alignment.md)

## 0. 需求大概

200 §57F：**回到真实源码，不只自测自洽**。交付本文件与差异账本，并完成三种对照：

  A. **源码结构与状态轨迹**：同一小场景的逐轮状态（计划前/计划后/结果后的进度、调度数、
     块表、Runner 输入、最终提交），并对本机 vLLM 的对应分支逐段读；
  B. **数学与真实 vLLM 的定向对照**：相同权重/dtype/位置比 logits；固定 p/q 比拒绝采样；
     本地真实模型跑 greedy 短输出；
  C. **一次 issue/PR 练习**：挑一个边界，写最小复现与不变量，在真实 vLLM 上确认，
     若上游没问题就解释它如何避免。

性能矩阵不是本阶段前置条件（199/200 都写明了）。

## 1. 分层与状态归属

| 我们 | 本机 vLLM | 拥有什么状态 | 谁可以写 |
|---|---|---|---|
| `request.py::Request` | `v1/request.py::Request` | prompt/已提交输出、优先级、`num_computed_tokens`、`spec_token_ids`、块 hash 链 | 只有 `append_output_token_ids()` 写 token；进度由 Scheduler 写 |
| `core/sched/scheduler.py` | `v1/core/sched/scheduler.py` | `requests`/`waiting`/`running`/`finished_req_ids` | 只有它自己 |
| `core/kv_cache_manager.py` | `v1/core/kv_cache_manager.py` | 请求 → 逻辑块、发布进度 | 只有它自己（块池在下一层） |
| `core/block_pool.py` | `v1/core/block_pool.py` | 物理块、`ref_cnt`、空闲队列、hash 索引 | 只有它自己 |
| `core/sched/output.py` | `v1/core/sched/output.py` | **快照**（无活对象） | 生产者写完就不再改 |
| `worker/gpu_model_runner.py` | 同名 | `requests`（镜像）、`InputBatch`、`kv_caches`、`execute_model_state` | 只有它自己；镜像的进度**由协议校正** |
| `worker/gpu_input_batch.py` | 同名 | 定长缓冲：token/进度/块表/采样参数/generator/草稿区 | 只有 Runner |
| `attention/*` | `vllm/attention/*` | 每层绑定的 `kv_cache` 物理张量 | Runner 绑定、后端写 |
| `sample/*`、`spec_decode/*` | `v1/sample/*`、`v1/spec_decode/*` | 无请求状态；只有按行的元数据与自己的随机流 | 无（纯计算） |

三条跨层约定（本关反复验证过的）：

1. **跨执行边界只传快照**：`SchedulerOutput` 里不许有 `Request`/Scheduler/块对象，也不许有张量
   （`check_step57_runner_inputs.py` 的 §7 逐轮扫描）。
2. **控制端权威、执行端镜像**：镜像的进度**每轮由协议覆盖**，执行侧不自增（自增会出现两个真相）。
3. **模型层不认识请求**：`forward(input_ids, positions)` 的签名里没有 Request/KV 池/采样参数，
   metadata 走 `set_forward_context()` 这个"环境参数"通道。

## 2. 状态轨迹（对照 A）

`benchmarks/trace_step57.py` 的场景：两条 8-token prompt、池子 4 块（块大小 4）、预算 8、
ngram 投机 K=3、priority 策略。**一轮不漏地覆盖五种事件**：

```text
step 1  prefill        A 一次算完 8 个 token（预算 8）；B 还在 waiting
        计划 num_scheduled={'A': 8} spec={}
        Runner input_ids=[1,2,3,4,1,2,3,4] positions=[0..7] slot_mapping=[0..7] seq_lens=[8]
        KV 空闲=2  A:{blocks:[0,1], ref:[1,1]}        提交 A:[10]

step 2  prefill + 投机  A 续跑并采用 3 枚草稿（K+1=4 行）；B 首次进批
        计划 num_scheduled={'A': 4, 'B': 4} spec={'A': [1,2,3]}
        Runner input_ids=[10,1,2,3, 5,6,7,8] positions=[8,9,10,11, 0,1,2,3]
               slot_mapping=[8..11, 12..15] seq_lens=[12, 4]
        KV 空闲=0                                    提交 A:[10,10]（3 枚只接受 1 枚）

step 3  抢占          A 长大要第 3 块 → 分配失败 → 抢 victim（priority 最大 = B）
        计划 num_scheduled={'A': 1} spec={}          B: PREEMPTED, preempt=1, 块表清空
        KV 空闲=1  A:{blocks:[0,1,2]}                 提交 A:[10,10,5]

step 4  投机验证      A 再提 1 枚草稿（K=1 → 2 行）
        计划 num_scheduled={'A': 2} spec={'A': [10]}
        Runner input_ids=[5,10] positions=[10,11]   提交 A:[10,10,5,3]

step 8  恢复          B 被重新接纳（它的块早被 A 拿走，所以从位置 0 重算）
        计划 num_scheduled={'B': 8}（首次接纳走 NewRequestData，带**整张**块表）

step 14 自抢占        B 是唯一 running，自己长大到装不下 → 抢自己 → 放回 waiting
        计划 num_scheduled={} preempted=['B']

step 15+ 恢复并跑完   B 从 position 0 重算，最后 提交 B:[6,10,4,6,5,3,7,5]
```

读这张轨迹时的三处关键（都在上面的行里看得见）：

- `num_computed_tokens` 与 `num_tokens` 的关系：**每轮结束恰好差 1**（最后一个已提交 token
  还没算）。投机把这条变得更微妙：被拒的草稿要把进度退回来（见 §5 的边界 1）。
- **槽位与块表同步增长**：`slot_mapping` 连续（8,9,10,11 …），因为块表是按"本轮要算到哪"补齐的。
- **抢占与恢复是两次独立的协议事件**：抢占 = 释放 + 放回 waiting（本轮不再接纳等待者）；
  恢复 = 首次接纳路径（`NewRequestData` + 整张块表）——执行侧因此能整表替换。

## 3. 七个问题 × 五种事件（194 §8 的最终验收）

| 问题 | prefill | decode | 抢占 | 恢复 | 投机拒绝 |
|---|---|---|---|---|---|
| **算几个 token** | 我们 `Scheduler._num_new_tokens()`（三重取小）／vLLM `schedule():L476` 的 `num_new_tokens = min(...)` | 同左（差 1） | 抢占本身**不排**（victim 放回 waiting）／vLLM 同一分支 | 恢复走 waiting 路径的 `_num_new_tokens()` | 投机请求差值是 `K+1`；预算不够时 `spec_token_ids[:num_spec]` 截短 |
| **分配物理块** | 我们 `KVCacheManager.allocate_slots()` → `FullAttentionManager.allocate_new_blocks()` → `BlockPool.get_new_blocks()`／vLLM `kv_cache_manager.py:L347` → `single_type_kv_cache_manager.py` → `block_pool.py:L647` | 同上（只补新增块） | `Scheduler._preempt_request()` → `kv_cache_manager.free()`（逆序归还，尾部先淘汰）／vLLM `_preempt_request():L1336` | 整张新表（`get_blocks()`） | 草稿的槽位就在本轮调度范围内（`num_tokens_with_spec` 已含） |
| **算 slot_mapping** | 我们 `BlockTable.compute_slot_mapping()`／vLLM `v1/worker/block_table.py::compute_slot_mapping`（Triton 内核 `L182`） | 同上 | —（不写 KV） | 恢复后按新块表重算 | 草稿行也要（它们是 query） |
| **真正写 KV** | 我们 `TorchAttentionImpl.write_kv()`（`index_copy_`）／vLLM 各后端实现（AttentionImpl 里 `reshape_and_cache`） | 同上 | 不写 | 重算时重写 | 写（草稿的 KV 先写上，被拒的下一轮被覆盖） |
| **得到 logits / token** | 我们 `GPUModelRunner.compute_logits` + `Sampler.forward()`／vLLM `Sampler.forward():L74` | 同上 | 不采 | 同上 | `RejectionSampler.forward()`（验证 + bonus/recovered）／vLLM `rejection_sampler.py:L92` |
| **提交输出 + 修正进度** | 我们 `Scheduler.update_from_output()` + `_update_request_with_output()` ＋ `_update_after_schedule()` 先前推进／vLLM `update_from_output():L1733`、`_update_request_with_output():L2182`、`_update_after_schedule():L1379` | 同上 | 进度归零、草稿清空 | 进度按协议给的重算起点 | 再补一步：`num_computed_tokens -= num_rejected` |
| **同一个位置在 vLLM 的哪段代码** | 上表"／"后面那列就是；行号取自 200 §7 的源码快照，函数内容以本机源码为准 | 同 | 同 | 同 | 同 |

一个具体的"同一个位置"的例子（投机）：**"第 j 个验证行的分布"**在我们这边是
`spec_decode/rejection_sampler.py::_apply_penalties`（历史 = 已提交 + 草稿前缀 `[:j]`），
在本机 vLLM 是 `sample/rejection_sampler.py::RejectionSampler.apply_logits_processors`
里那句 `self._combine_outputs_with_spec_tokens(output_token_ids, spec_token_ids)` ——
**同一个语义，两处代码形状不同但可以逐行对上**。

## 4. 与真实 vLLM 的数学对照（对照 B/C）

`benchmarks/compare_step57_vllm.py`。**运行环境说明**：WSL2 下 vLLM 默认关掉 pinned memory，
`is_uva_available()` 因此为假、引擎起不来，必须带

```bash
VLLM_WSL2_ENABLE_PIN_MEMORY=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 python ...
```

（前者打开 WSL2 的 pinned memory——内核 6.6 ≥ 4.19.121 ✓；后者让引擎留在本进程，
否则子进程无法 re-exec `<stdin>`。）

| 对照 | 结果 |
|---|---|
| **A. 模型 logits**（tiny 权重，vLLM 的 `prompt_logprobs` vs 我们的 `compute_logits`） | 每个位置的 argmax 一致；vLLM 给的每个 (token, logprob) 与我们**逐值差 ≤ 2.4e-7** |
| A. 增量位置：同一条 prompt 分 3 块 / 逐 token decode | 与全量一次算完差 ≤ 9e-8（绝对位置、KV 复用都对齐） |
| **B. 增量位置对 vLLM**（它 greedy 生成 6 步的 `logprobs` vs 我们逐 token decode） | 每一步的 top-5 **逐值差 ≤ 2.4e-7**，argmax 全一致 |
| **C. 拒绝采样**（固定 p/q、ragged K=[2,1]，各 600 次） | 平均接受数：解析 0.550 ｜ vLLM 0.549 ｜ 我们 0.557；首位置接受次数 144 vs 153；recovered 分布两侧都 ∝ `max(p−q,0)`（与解析值偏差 < 0.05） |
| **D. 端到端 greedy**（真实 Qwen3-1.7B） | **fp32：16/16 token 与 vLLM 完全一致**；bf16：公共前缀 5/16，分歧点两候选在我们这边差 0.25、共同前缀上逐值最大差 0.50 —— bf16 的 logits 网格量级（见下） |

关于 D 的 bf16 分歧（200 §B 要求"先检查误差与最大值间隔，不能直接归因浮点"）：

- 在同一段输入上，我们与 vLLM 的**每步 log-prob 差最大 0.50**——这是 **bf16 的 logits 量化网格**
  （|logit| ≈ 64 时 bf16 的步长是 0.25，两侧各一格就是 0.5）；
- 分歧点上，两个候选在我们这边的 logprob 差 0.25、在 vLLM 那边 0.00（它的两个候选量化成了同一个
  bf16 值）——即**两个候选落在同一格网格里**，argmax 翻转是量化噪声，不是逻辑差异；
- 对照组 fp32 **逐 token 一致**，把这条结论钉死了。

**没有做的对照**（记录原因）：与 vLLM 的**内核级逐位**对照（它的拒绝采样走 Triton、随机流
按自己的方式消费），所以 C 用**统计 + 解析值**对照（199 §8 明确允许）；CUDA Graph / 多进程 /
FlashAttention 后端都不在本关范围。

## 5. 一次边界练习（对照 C）

`benchmarks/check_step57_vllm_boundaries.py` 在真实 vLLM 上跑两条我自己踩过的边界。

### 边界 1：被拒草稿的进度回退

- **最小复现**：真实 vLLM + ngram 投机（K=3），一条 8-token prompt 生成 12 个 token；
- **预期不变量**：每轮结束时 `request.num_computed_tokens == request.num_tokens - 1`
  （排的是 K+1 行，有效的只有"接受的 a 枚 + 最后一个 token"）；
- **结果**：11 轮，**零违例**；
- **vLLM 怎么避免的**：`Scheduler.update_from_output()`：

  ```python
  num_accepted = max(len(generated_token_ids) - num_sampled, 0)
  num_rejected = num_draft_tokens - num_accepted
  request.num_computed_tokens -= num_rejected
  ```

  它先按"排了几行"推进（`_update_after_schedule`），再在结果处理时把被拒的减回来。
- **我们的第一版**漏了这一步，请求会卡在 `computed > num_tokens` 上（被空转保护报出来）——
  这正是"自测自洽"抓不到、对源码才看得清的那类问题。

### 边界 2：投机验证路径上的 `min_tokens` 屏蔽

- **最小复现**：直接调用 vLLM 的 `MinTokensLogitsProcessor.apply_with_spec_decode`
  （可隔离入口），两条请求 `min_tokens=[3, 0]`、K=[2,1]；
- **预期不变量**：`min_tokens` 未到的请求，它的**每个草稿行**都不许出现停止 token；
- **结果**：按草稿行的屏蔽情况 `[True, True, False]` ✓（r0 的两行被屏蔽、r1 的那行没有）；
- **vLLM 怎么做的**：它按 `num_draft_tokens` 算出每个请求占的草稿行，逐行 `index_put_` 成 `-inf`；
- **我们的第一版**同样漏了（只做了惩罚与温度/top-k/top-p），草稿可以在 `min_tokens` 之前把 EOS
  送进提交序列——修法就是复用普通采样器的那条屏蔽逻辑（`docs/step57e_speculative.md` 记着）。

两条都属于 200 §C 说的"**确认上游没问题 → 解释它的代码如何避免**"，同样是合格结果；
没有发现需要提 issue 的上游问题，所以不做 PR。

## 6. 未验收项与环境限制

- **WSL2 的 pinned memory**：不设 `VLLM_WSL2_ENABLE_PIN_MEMORY=1` 时 vLLM 引擎起不来
  （`UVA is not available`）——这是环境限制，不是实现差异。
- **显存**：本机 24 GB，四个引擎（vLLM/我们 × bf16/fp32）不能同进程共存，所以 D 用**子进程**
  分 dtype 跑；脚本里打印了每段的空闲显存。
- **未做**：内核级逐位对照、CUDA Graph、多进程/RPC（都在差异账本里，标了"以后何时消除"）。
- **性能矩阵**：按 199/200，本阶段不要求；本关只报功能与数值正确性。
