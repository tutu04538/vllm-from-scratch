# 第七十四关补充：V2 target 的 FULL_DECODE_ONLY CUDA Graph（Phase 1）

> 需求背景：74 关只做了 V2 **提议者侧**的图，target 一直是 eager。本文件记录 V2 target 图的
> 第一个里程碑：统一 decode 批的 `FULL_DECODE_ONLY`。
> 参考实现：上游 `vllm/v1/worker/gpu/cudagraph_utils.py::ModelCudaGraphManager` 与
> `vllm/v1/worker/gpu/model_runner.py::capture_model / execute_model`。

## 0. 这次做了什么

V2 的 **target** 现在也会捕获并重放 CUDA Graph，但范围严格限定：

```text
FULL_DECODE_ONLY：每请求恰好 1+K 行的统一 decode 批 → FULL 图
混合 prefill/decode 批 → 回退 eager
自适应验证（varlen decode）→ Phase 1 不接，target 图关闭，仍走 eager
（Phase 2 已接上 varlen decode 图，见 `docs/step74_target_graph_varlen.md`）
```

这样做的好处是先把最容易验证的一档跑通：同一套 target 模型、同一套 KV/块表、同一套
speculator 图；混合批和 varlen 都属于后续 phase。

## 1. 与上游的对应关系

| 上游 V2 | 本仓库 Phase 1 |
|---|---|
| `ModelCudaGraphManager`（target） | `worker/gpu/cudagraph_utils.py::ModelCudaGraphManager` |
| `prepare_inputs_to_capture()` | 复用已有 `prepare_inputs_to_capture()`；回调 `_build_target_capture_attn_metadata()` |
| `InputBatch.make_dummy()` + dummy block table/slot mapping | 复用 |
| `BatchExecutionDescriptor` / `_is_compatible` / `capture` / `run_fullgraph` | 复用已有 `CudaGraphManager` |
| target 先捕获、speculator 后捕获 | `capture_model()` 同序 |
| 静态 `hidden_states` / `aux_hidden_states` | `ModelCudaGraphManager` 持有并在 `run_fullgraph()` 返回切片 |
| 运行时 padding、`num_tokens_after_padding` / `num_reqs_padded` | `prepare_inputs()` 新增 `batch_desc` 参数并补齐 |
| 图分派 | `execute_model()` 调 `target_cudagraph_manager.dispatch()`；FULL 只 replay |

## 2. 捕获流程

`initialize_cudagraph_capture()` 在模式协商之后建 target manager：

```text
mode.has_full_cudagraphs()
    → ModelCudaGraphManager(
          cudagraph_mode=FULL_DECODE_ONLY,   # Phase 1 只认 FULL 那一半
          decode_query_len=1+K,
          allow_shape_padding=True)
```

`capture_model()`：

```text
set_cudagraph_capturing_enabled(True)
    target_manager.capture(
        model, input_buffers, block_tables,
        _build_target_capture_attn_metadata,
        use_aux_hidden_state_outputs=...)
    speculator.capture()
set_cudagraph_capturing_enabled(False)
```

`ModelCudaGraphManager.capture()` 的 `create_forward_fn()`：

```text
num_reqs = desc.num_reqs or min(num_tokens, max_num_reqs)
attn_state = prepare_inputs_to_capture(...)
    InputBatch.make_dummy()            写常驻输入缓冲
    get_dummy_block_tables()           块表行清零
    get_dummy_slot_mappings()          全填 PAD_SLOT_ID(-1)
    _build_target_capture_attn_metadata()  造图内 metadata
forward_fn：
    set_forward_context(..., cudagraph_runtime_mode=FULL)
    model(input_ids, positions)
    把 hidden/aux 拷进静态输出缓冲
```

### 2.1 为什么必须有静态输出缓冲

图里记的是捕获时张量的**地址**。重放时如果模型返回新分配的张量，`sample_tokens` 读到的
还是捕获时那块内存——不报错，结果错。所以：

```text
self.hidden_states = torch.empty_like(hidden_states)     # 只分配一次
self.aux_hidden_states[i] = torch.empty_like(aux)        # 只分配一次
run_fullgraph() → hidden_states[:desc.num_tokens]        # 每次重放同一块
```

## 3. 运行时 padding 与重放

`execute_model()`：

```text
num_reqs = len(batch_req_state.req_ids)
max_query_len = max(num_scheduled_tokens)
uniform_tok_count = get_uniform_decode_token_count(...)
batch_desc = target_cudagraph_manager.dispatch(
    num_reqs, num_tokens, uniform_tok_count, max_query_len=...)

input_batch = prepare_inputs(scheduler_output, batch_req_state, batch_desc)
block_tables, slot_mappings = prepare_attn(input_batch)
if batch_desc.cg_mode == FULL:
    model_output = target_cudagraph_manager.run_fullgraph(batch_desc)
else:
    model(...)   # eager / 混合批
```

`prepare_inputs()` 的 padding 规则：

| 项 | 真实行 | padding 行 |
|---|---|---|
| `input_ids` | 正常写 | 清零（合法 token id 0） |
| `positions` | 正常写 | 清零（合法位置 0） |
| `query_start_loc` | 真实前缀和 | 指向真实总行数 → query_len=0 |
| `seq_lens` | 真实值 | 0（`prepare_pos_seq_lens` 的尾部 block 负责清零） |
| `slot_mapping` | 按块表算 | `PAD_SLOT_ID(-1)` |
| `block_table` | gather 真实块 | 行清零（0 号块/垃圾桶） |
| `is_padding` | False | True |

例：K=1，捕获键 `(num_tokens=4, num_reqs=2, uniform=2)`，真实批只有 1 请求 2 行：

```text
真实: qsl=[0,2], seq_lens=[...], 2 行
捕获: qsl=[0,2,2], seq_lens=[...,0], input_ids 后 2 行=0, slot 后 2 行=-1
```

图重放仍然按 4 行/2 请求跑，padding 请求 `query_len=0`，不写真实 KV、不参与采样。

## 4. 测试

新增 `tests/step74/test_target_graph.py`（5 项）：

- 统一 decode 的 target 图 vs eager：greedy 输出逐 token 相同；
- 真实 1 请求 × 2 行命中捕获的 2 请求 × 4 行图：输出不变；
- 混合批 `uniform_token_count=None`：dispatch 回退 NONE；
- target 先捕获、speculator 后捕获；`hidden_states` 地址在重放前后不变；
- 非投机 decode（`decode_query_len=1`）也走 target 图。

实测：

```text
pytest tests/step74/test_target_graph.py -q   → 5 passed
pytest tests/step58 … tests/step79 -q         → 950 passed（945 + 5）
check_step74_fused_draft.py                   → 23/23
check_step76_dflash.py                        → 27/27
check_step77_dspark.py                        → 28/28
check_step78_dflash2.py                       → 31/31
check_step79_adaptive.py                      → 23/23
```

## 5. 差异账本（Phase 1）

| 编号 | 差异 | 为什么 | 影响 |
|---|---|---|---|
| D-74T-1 | target manager 固定用 `FULL_DECODE_ONLY`，即使配置是 `FULL_AND_PIECEWISE` | Phase 1 不做 PIECEWISE；FULL 那一半独立可用 | 混合批回退 eager；Phase 3 再换回完整模式 |
| D-74T-2 | 没有 `ModelState` 抽象；用一个回调 `_build_target_capture_attn_metadata()` | 本仓库只有 Qwen3 一个基座、一个 KV group | 行为等价，少一层模型适配 |
| D-74T-3 | Phase 1 的 target 图与自适应验证**互斥**：adaptive=True 时 target manager 为 None | 自适应需要 varlen decode 图，属 Phase 2 | **Phase 2 已接上**（varlen decode 图 + 图安全 varlen 注意力）；本行保留 Phase 1 当时状态 |
| D-74T-4 | 混合批没有 target PIECEWISE，dispatch 返回 NONE | 本仓库 V2 尚未接分段图 | 混合批性能不变（eager），正确性不变 |
| D-74T-5 | 没有 LoRA/DP/PP/offloader/多模态的图分支 | 本仓库没有这些能力 | 不生成对应图键；以后加能力时按上游扩展 |
| D-74T-6 | 静态输出只有 hidden/aux，没有 PP 的 intermediate tensors | 本仓库单卡、无 PP | 单卡路径完整 |

## 6. Phase 1 明确没做

- varlen decode 图（自适应验证）→ **Phase 2 已实现**，见 `docs/step74_target_graph_varlen.md`；
- PIECEWISE / 混合 prefill+decode 图 → Phase 3；
- `AttentionCGSupport.ALWAYS`：Torch 后端仍是 `UNIFORM_BATCH`；
- 目标图的成本 profiling：Phase 1 的 target 图不参与 adaptive，仍用 eager 成本表；
- 真实模型加速比：本机没有真实 DSpark 权重，只验证协议/数值/图。
