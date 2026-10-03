# step57：投机收尾——逻辑上限、请求生命周期、失败态

- 对应代码：`step57/spec_decode/{draft_model.py,ngram_proposer.py}`、
  `step57/worker/gpu_model_runner.py`、`step57/core/sched/scheduler.py`、
  `step57/engine/core.py`、`step57/model_loader/base_loader.py`
- 触发：验收笔记
  [`205_第五十七关复验`](../../vllm-omni/learning_notes/14_vllm_from_scratch/205_第五十七关复验_原补测全过但生命周期尚未闭合.md)
  的三类阻塞（逻辑边界 / 生命周期 / 失败态）与 §6 的整理项。
- 上一轮的同步改动见 [`step57_draft_lockstep.md`](step57_draft_lockstep.md)（205 §2 判定它成立、保留）。

## 0. 需求大概

205 的复验结论是"方向对、生命周期没闭合"：

1. **物理块覆盖 ≠ 逻辑位置合法**：块表按块向上取整（10 个位置能给出 12 个槽位），`covers()`
   只回答物理容量。`max_model_len=10`、prompt 8、K=3 时草稿要写 position=10，被 `_forward()`
   的断言抓住 → 正常生成在上下文末尾失败。
2. **请求生命周期与 batch 成员变化混在一起**：`_drop_stale()` 把"不在本轮 req_ids"当成
   "请求结束"，于是（a）预算不够没排上的请求丢了随机流；（b）它的草稿还在、q 却没了，
   再入批直接报"q 对不上"；（c）真的结束的请求反而没清，复用同一 ID 会继承旧进度、跳过
   draft 前向、把一个 draft KV 全零的块登记成 prefix 命中。
3. **sample/propose 异常没有失败态**：只有 `execute_model()` 的一段代码记 `failure`；
   提议抛异常后镜像已更新、权威输出未提交，下一轮却照常调度并返回空输出。

这几条都属于 57E 的收尾，不新增功能。

## 1. 改动内容

### 1.1 草稿写入的两个边界（`draft_model.py::propose`）

自回归循环每写一枚草稿前，位置必须同时满足**模型逻辑上界**与**物理槽位覆盖**：

```python
if not 0 <= position < self.max_model_len:          # 逻辑：模型自己的位置范围
    continue
if not input_batch.block_table.covers(row, position):  # 物理：块表覆盖（lookahead 可能被截掉）
    continue
```

过不了就**少提几枚**（草稿只是候选）：轮末正常返回 target 已经采出的结果，不抛异常。
`_forward()` 的越界断言保留，作为**内部错误检查**——正常路径不该用它来停止。

对照本机 vLLM：`gpu_model_runner.py::_input_fits_in_drafter()` 判断"序列长度 + 提议所需位置"
是否超过 drafter 的逻辑上限，据此决定还跑不跑 drafter；它同样不把块表向上取整后的容量当模型长度。

### 1.2 请求生命周期的三种事件（`_drop_stale` → `_reset_requests` + `remove_requests`）

| 事件 | draft 进度 / KV 假设 | 请求 generator | 待验证 proposal / q |
|---|---|---|---|
| 本轮没被调度（块仍有效） | **保留** | **保留**（同一对象、状态不动） | 按"草稿只活一轮"的契约成对作废（§1.3） |
| 抢占恢复（块表整表换过） | 重置并重算 | 保留（请求还活着，不重新 seed） | 旧草稿与 q 作废（`_preempt_request` 已清） |
| finished / abort | 删除 | 删除 | 删除 |

实现：

- `SpecDecodeBaseProposer.remove_requests(req_ids)`：显式删除进度、随机流、草稿概率；
  `NgramProposer.remove_requests()` 是空操作（无状态），只为**接口一致**——Runner 不需要知道
  用的是哪种提议者；
- `GPUModelRunner._update_states()` 处理 `finished_req_ids` 时调用它。**0-token 的结束清理轮
  也会走到这里**（`execute_model` 先 `_update_states`），所以最后一条请求结束后状态是真的空的；
- `_reset_requests(reset_req_ids)` 只做恢复重置。**"不在 batch 里"不再等于"结束"**。

### 1.3 草稿与 q 同寿命（`scheduler.py::_invalidate_stale_drafts`）

契约（205 §4.2 建议的简单路径）：**草稿只活一轮**。轮 t 提的草稿只允许在轮 t+1 被采用。
执行端只保留"最近一轮提的那批"概率 q；如果请求在 t+1 没排上，t 的 q 已经丢了，而
`spec_token_ids` 还留在请求上——采用它就会在拒绝采样里拿错误的 q 算接受率。

所以 `schedule()` 在**计算采用数、打包 SchedulerOutput 之前**先做一次清理：

```python
for request in self.running:
    if request.spec_token_ids and request.request_id not in self.prev_step_scheduled_req_ids:
        request.spec_token_ids = []      # 断代：本轮按 K=0 正常算一次，之后会重新提一批
```

`prev_step_scheduled_req_ids` 本来就是 Scheduler 的字段（`_make_cached_request_data` 用它判断
要不要带完整历史）。只清 running：抢占走 `_preempt_request()`（那里已清），新请求本来没有草稿。
代价是少一轮投机机会，换来"token IDs 与 q 要么成对存在、要么成对消失"。

另一种做法（保留跨轮草稿）要求 token IDs、q、请求归属、对应历史四样一起跨轮保存——205 明确
要求"不能只留 token IDs"，本关按简单契约实现并在此记录。

### 1.4 失败态（`sample_tokens` + `EngineCore.step`）

- `GPUModelRunner.sample_tokens()` 把正常路径抽到 `_sample_and_propose()`，外面包一层：
  任何异常 → 清掉未交付的 `pending_draft_token_ids` / `pending_draft_probs`、
  记 `self.failure = "类型: 原因"`、原样抛出。保留最初异常原因，不做 GPU KV 回滚。
- `EngineCore.failure` + `_check_alive()`：`step()` **第一件事**就是检查失败标记，
  在 `schedule()` **之前**拒绝；`schedule → execute → sample → update` 整段也在 try 里，
  任何一环抛出都记 failure 再抛。下一轮不会返回空输出假装正常。

对照本机 vLLM：`_update_states()` 明确分开"finished 删除请求状态"与"unscheduled 移出 batch、
保留请求状态"两件事（约 1252–1288 行）——本关按同一口径拆开了 §1.2 的三种事件。

### 1.5 整理项

- `draft_model.py` 顶部不再写"没有 lookahead 预留"（与新实现冲突），改为记实际规则；
  `input_budget` 明确写成**对齐差异**：动态张量避免了定长缓冲写越界，但**不等于**有输入预算，
  普通 draft 同样需要那份核算；并写明 prefix 命中后 draft 仍要整段重算的成本。
- `BaseModelLoader.load_model()` 返回前 `model.eval()`（204 §6 的遗留项）：
  eval 管模块行为（dropout/BN 等训练态分支），`torch.inference_mode()` 管梯度记录，两者分工不同。
- 回归用例：`check_step57_draft_model.py` §7 新增七条（见 §3）。

## 2. 设计要点

- **"没排上"与"结束"是两种事件**，混在一个判断里必然错一边：前者要留（块还有效、历史没变），
  后者要删（状态与物理块都归还了）。控制端的结束通知是唯一权威。
- **进度数字足够大，还必须确认它属于当前这次请求及当前历史**（205 §2 原话）：这就是 4.3 的
  复用 ID 场景——旧进度 9 套到新请求上，`start == boundary` 直接跳过 draft 前向。
  清理挂在 `finished_req_ids` 上才有"代际"意义。
- **失败必须是状态，不是一次次异常**：半轮状态（镜像已更新、权威输出未提交）不可能靠重试自愈，
  所以在调度之前拒绝，而不是让 Scheduler 继续往前走。
- **发布不变量的前提是"每轮同步"**（上一轮改动）：本轮的三种生命周期事件都不改变它——
  未调度请求的块不会被发布（它的 `num_computed_tokens` 不前进），恢复请求整表换过会重算。

## 3. 验证

| 脚本 / 探针 | 结果 |
|---|---|
| `验收记录/step57_recheck_20261003/review_remaining_boundaries.py`（205 的 9 个收尾场景） | **9/9**（原 4/9）：逻辑上限 K=3 与非投机同为 `[6,10]`；B 的 generator 同对象保留、再入批跑完；复用 ID 后 `_draft_computed={}`、generators 为空、新请求 draft 前向 9 行、新 prefix 第 3 位 draft KV=32.36；注入异常后 `runner.failure` 有值、下一轮在调度前被拒；抢占 1 次且输出与非投机一致；prefix 命中 4 token 且开/关输出一致；CUDA 8 token 与非投机一致 |
| 上一轮三个独立探针（`review_draft_boundaries` / `review_rejection_boundaries` / `review_inference_boundary`） | 6/6、2/2、1/1 |
| 15 个功能回归脚本 | 全过（exit=0，共 **354 项**；`check_step57_draft_model.py` 从 25 项增到 **32 项**） |
| `check_step57_draft_model.py` §7（新） | 逻辑上限 9/10/11/12 与非投机逐 token 一致；最后一轮刚好到上限；未调度请求保留 generator（第二轮 batch=`['A']`、B 的 generator 是同一对象且状态不变）；结束清理后 `({}, [])` 且复用 ID 重算；失败态；真实抢占 `num_preemptions=1` 且输出一致；prefix 命中 4 token 且输出一致 |

复跑验收脚本时**先拷到 `/tmp`**：它把结果写在 `Path(__file__).with_name(...)`，在原地跑会覆盖
验收记录（这次发生了一次，已在 `remaining_boundary_results.json` 的 `_note` 里说明，
修复后的结果另存 `remaining_boundary_results_after_fix.json`）。

没跑的：需要真实 vLLM 引擎/真实权重的对照（`check_step57_vllm_boundaries.py`、
`compare_step57_vllm.py`），按约定等通知再跑。

## 4. 接口变化与遗留

- 新增 `SpecDecodeBaseProposer.remove_requests(req_ids)`、`NgramProposer.remove_requests(req_ids)`。
- `SpecDecodeBaseProposer._drop_stale()` → `_reset_requests(reset_req_ids)`（语义收窄为"恢复重置"）。
- 新增 `Scheduler._invalidate_stale_drafts()`（`schedule()` 内部调用）。
- 新增 `EngineCore.failure` / `EngineCore._check_alive()`；`GPUModelRunner.sample_tokens()` 拆出
  `_sample_and_propose()`（失败态只包一层 try）。
- `BaseModelLoader.load_model()` 现在会 `model.eval()`。
- 遗留：`input_budget` / 定长输入缓冲仍未做（对齐差异，接 EAGLE/MTP 前必须补）；
  EAGLE/MTP 不做；跨轮草稿保留未做（当前契约是"只活一轮"，见 §1.3）；不做故障恢复（失败即停摆，
  要重建引擎）。
