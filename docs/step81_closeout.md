# 第 81 关收口记录：投机解码单卡主线

> 2026-10-28 按新需求 81 重写范围。旧 `docs/step81_alignment.md` 的 7 项多组子集证据继续有效；
> 本文件记录本轮新增的生命周期闭环、正式统计链与支持边界。
>
> **本关只覆盖：单 GPU、同步调度、纯文本、已有模型/后端。**
> 不覆盖 Mamba/混合状态、多模态、PD/KV transfer、量化、TP/DP/PP/CP/EP、异步端到端。

## 1. 已有证据（不重做）

| 条目 | 现有实现/测试 |
|---|---|
| FullAttention / SlidingWindow spec、group 配置 | `core/kv_cache_interface.py`；`test_spec_kv_groups.py` |
| 多组原子分配、共同 prefix 命中 | `core/kv_cache_coordinator.py`；`test_spec_kv_groups.py` |
| packed shared slab / 逐层 view | `worker/kv_cache_alloc.py`；`test_packed_kv_layout.py` |
| 多 group / 异构 head_dim / SWA / Q-only | V1/V2 Runner、`attention/`；`test_gemma4_real_integration.py` |
| Gemma4 文本 target + assistant | `models/gemma4.py`、`gemma4_mtp.py`、专用 proposer |
| 多模块 MTP 输入/拒绝修复/Graph | `worker/gpu/spec_decode/multi_module_mtp/`；80A/B/C |

旧 81 的 7 项测试保留，不覆盖历史数字。

## 2. A：生命周期闭环

新增 `tests/step81/test_spec_lifecycle_closeout.py`，12 passed：

1. **拒绝 + max_tokens 截断**：K=3 验证接受 3，`max_tokens=2` 只交付 2；
   统计仍是截断前口径；
2. **EOS / stop_token_ids 截断**：第二个 token 命中停止，验证统计仍记 3；
3. **部分 prefill + decode**：长 prompt 在 `budget=8` 下第一步不输出；
   短请求正常 decode，最终两条都完成；
4. **两组 prefix 全有或全无**：删 sliding 组第一块 hash 后共同命中 0；
5. **V2 tiny MTP 同步抢占**：`blocks=4` 真触发 2 次抢占；恢复输出与
   `blocks=64` 冷算逐 token 相同；
6. **取消 + 同 ID 复用**：取消后清理轮结束，新请求不继承旧草稿/输出；
7. **失败即停摆**：execute 注入异常后，下一次 step 在 schedule 前拒绝；
8. 停止模式参数化（EOS / stop_token_ids）各一条；
9. 直接驱动 Scheduler 的 accepted=0/1/2/3 参数化：首拒/中拒/全接受
   的统计分母、位置数组与交付长度逐值一致。

代表选择：

```text
V2 tiny MTP      -> 自有 draft KV 的拒绝修复与抢占
V1 tiny 投机     -> 截断/统计/取消/失败路径
控制面 manager   -> 两组 prefix 边界
```

测试断言 `num_preemptions > 0`，不是“给了小池子但根本没抢占”。

## 3. B：正式统计输出

新增：

```text
minivllm/metrics/stats.py              SchedulerStats
minivllm/spec_decode/metrics.py         SpecDecodingLogging.observe/log/reset
minivllm/outputs.py                     EngineCoreOutputs.scheduler_stats
minivllm/core/sched/scheduler.py        make_stats + 每轮附着
minivllm/engine/llm_engine.py           observe + last_scheduler_stats
```

数据流：

```text
Scheduler.update_from_output
  -> Scheduler.make_stats
  -> EngineCoreOutputs.scheduler_stats
  -> LLMEngine.step
  -> SpecDecodingLogging.observe
  -> log/reset
```

统计字段：

```text
num_drafts
num_draft_tokens
num_accepted_tokens
num_accepted_tokens_per_pos
num_draft_tokens_per_pos
```

固定例子（测试逐值）：

```text
K_i=[3,1], a_i=[2,0]
num_drafts=2, num_draft_tokens=4, num_accepted_tokens=2
accepted_per_pos=[1,1,0], draft_per_pos=[2,1,1]
接受率=50%, 平均接受长度=2
```

`committed_output_tokens` 不塞进 SchedulerStats：它由基准层按
`RequestOutput.token_ids` 增量统计，不能拿它反推 accepted。

`test_spec_metrics_pipeline.py`（5 passed）覆盖：

- 计数/位置数组固定例子；
- `SpecDecodingLogging` 复制快照（旧 stats 被原地改也不污染）；
- 零分母 `None`，不伪造 0%/100%；
- 真实 V1 引擎的 `SchedulerStats -> LLMEngine` 链与 `log()` 聚合；
- 中间 prefill 无用户输出但 stats 仍存在；
- 当前包 hash 与 `step81.closeout` 一致。

## 4. C：支持边界

新增 `tests/step81/test_spec_scope_boundaries.py`（3 passed）与
`docs/step81_experiment_manifest.json`：

1. `async_scheduling=True` 明确 `NotImplementedError`，不静默降级；
2. V1 Runner 对不同 group block_size 明确拒绝；
3. manifest 模板固定 V2/同步/Torch/BF16/纯文本/单卡，
   列出本轮不支持的组合，固定 target/assistant revision。

保留的边界：

```text
async 不支持
V1 多组图不支持
Gemma4 Q-only 非 Torch backend 不支持
执行侧不同 group block_size 不支持
滑窗掩码已实现但历史块不回收
packed 不是全动态 page 分配
BF16 near-tie 是风险，不是忽略错误的许可证
```

`test_step80_capability_matrix.py` 已改为：80H 的历史 package hash 只校验摘要形状，
当前包 hash 由 `step81.closeout` 校验；不再用 81 的工作区现码去改 80H 历史结论。

## 5. 命令与结果

```bash
PATH=/home/user/anaconda3/envs/vllm-omni-dev/bin:$PATH \
PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m pytest tests/step81 -q
# 27 passed, 14 warnings

PATH=... python -m pytest tests/step58 … tests/step81 -q
# 1132 passed, 149 warnings in 177.30s
```

## 6. 明确移出本关

```text
量化（权重/KV）、TP/DP/PP/CP/EP、LoRA
多模态、PD/KV transfer
Mamba/混合状态
torch.compile、异步端到端
滑窗块回收、不同 group block_size 执行
新模型族/稀疏 MLA、Prometheus 全服务化
```

当前结论只能叫：

> **已声明单卡文本投机主线的生命周期/统计/边界已收口。**
> 不是“全部 vLLM 投机特性与部署组合已完成”。
