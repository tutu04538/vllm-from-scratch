# 第七十四关补充 Phase 2：varlen decode target 图与 adaptive 合并

> 接 Phase 1（`docs/step74_target_graph.md`）：Phase 1 只做统一 decode 的
> `FULL_DECODE_ONLY`。Phase 2 把 **自适应验证** 接进 target 图。
> 参考实现：上游 `CudaGraphManager(..., varlen_decode=True)`、
> `BatchExecutionDescriptor(max_query_len=1+K)`、`BatchExecutionDescriptor._is_compatible`，
> 以及上游 FlashAttention/Triton 的 varlen decode kernel。

## 1. 这次做了什么

自适应验证裁剪后，每条请求的 query 行数是 `1..1+K`，不再统一。Phase 2 做了三件事：

1. **target manager 支持 varlen decode 图键**：
   `BatchExecutionDescriptor(cg_mode=FULL, num_tokens=..., num_reqs=...,
   max_query_len=1+K)`，不再要求 `num_tokens` 是 `1+K` 的倍数；
2. **图安全 varlen 注意力**：`TorchAttentionImpl._forward_varlen_padded()`，
   所有每请求长度从 GPU 的 `query_start_loc`/`seq_lens` 读，无 `.item()`、无 Python 请求循环；
3. **adaptive profiling 改走 target 图**：用 `captured_token_counts()` 生成成本曲线；
   `_cudagraph_limit` 是真实最大捕获 token 数（不再是 0），`full_cudagraph` 样本为 True。

## 2. 图键与运行时

### 2.1 捕获

`initialize_cudagraph_capture()` 在 adaptive 开启时：

```python
target_manager = ModelCudaGraphManager(
    ..., cudagraph_mode=FULL_DECODE_ONLY,
    decode_query_len=1+K,
    varlen_decode=True,
)
```

`_init_candidates()` 生成：

```text
capture_sizes = [4, 8, ...]
max_decode_tokens = max_num_reqs × (1+K)

desc = (FULL, num_tokens, num_reqs=min(num_tokens, max_num_reqs),
        max_query_len=1+K)
```

不再把 `num_tokens` 取整到 `(1+K)` 的倍数；捕获的 dummy 批由
`InputBatch.make_dummy(num_reqs, num_tokens, max_query_len=1+K)` 生成，
`prepare_inputs_to_capture(varlen=True)` 把 `uniform_query_len` 清成 `None`，
metadata 带上 `max_query_len`。

### 2.2 运行时

```text
adaptive get_num_tokens() → 真实 target 行数可能 5（A=3 行、B=2 行）
dispatch(num_reqs=2, num_tokens=5, uniform_token_count=None, max_query_len=4)
    → 命中捕获的 desc(num_tokens=8, num_reqs=2, max_query_len=4)
prepare_inputs(batch_desc) → 真实 5 行 + 3 行 padding
prepare_attn() → 真实 slot + padding slot=-1
run_fullgraph(desc) → 只 replay，不跑 model()
```

`prepare_inputs()` 对 padding 行的规则与 Phase 1 相同；额外把
`InputBatch.uniform_query_len` 保持 `None`、`max_query_len=desc.max_query_len`，
让 `_build_attn_metadata()` 生成 varlen metadata。

**重要 gate**：target 图只服务 **decode** 批。混合 prefill/decode / 纯 prefill 的
`query_len` 可能恰好 `<= 1+K`（短 prompt），不能误命中 varlen decode 图；
`execute_model()` 在 `batch_req_state.has_prefill` 时直接回退 eager。

## 3. 图安全 varlen 注意力

`TorchAttentionImpl.forward()`：

```python
if self._graph_dispatched():
    if attn_metadata.uniform_query_len is not None:
        return self._forward_padded(...)        # 统一 decode
    if attn_metadata.max_query_len is not None:
        return self._forward_varlen_padded(...) # 自适应/varlen decode
return self._forward_generic(...)               # eager
```

`_forward_varlen_padded()` 做的事：

```text
q_lens = query_start_loc[1:] - query_start_loc[:-1]      # GPU 张量
q_idx  = arange(max_query_len)                          # Python int 定形状
row    = query_start_loc[:-1,None] + q_idx[None,:]      # [B, max_q]
q      = query[row].view(B,max_q,H,D)                   # 定形 gather
K/V    = 按块表 gather [B, P, Hk, D]                    # 与 uniform 路径同款
scores = q @ K^T
mask   = (key_pos >= seq_lens) | causal(key_pos > q_pos)
         | (q_idx >= q_lens)                            # 无效 q 行整行 mask
out    = softmax(scores) @ V                             # [B,max_q,H*D]
out_flat.index_copy_(idx, out)                           # 无效行写 scratch
```

关键点：

- `max_q` 是 Python int（捕获时定死），所以 kernel/张量形状固定；
- `q_lens`、`seq_lens`、`query_start_loc` 都是 GPU 张量，重放时可变；
- 无效 `(b,i)` 的 scatter 索引统一指向 `out_flat[N]` 这个 scratch 行，
  不覆盖真实行；返回 `out_flat[:N]`；
- 全程无 `.item()`、无 Python `for req`、无动态 shape 分配。

## 4. adaptive profiling

`profile_adaptive_verification()` 现在：

```text
captured = target_manager.captured_token_counts()   # 例 [4, 8]
sizes = captured or _adaptive_profile_sizes()
for batch in manager.batches_to_profile(sizes):     # 4/8 + 尾部 eager 尺寸
    _dummy_run_adaptive(**batch)
manager._cudagraph_limit = max(captured)            # 例 8
manager.set_initial_cost_curves(timings)
```

`_dummy_run_adaptive()` 对命中 target 图的尺寸：

```text
dispatch(...) → desc FULL varlen
InputBatch.make_dummy(desc.num_reqs, desc.num_tokens, max_query_len=desc.max_query_len)
uniform_query_len = None
dummy block table / slot_mapping
target_manager.run_fullgraph(desc)   # 图重放
speculator.propose(dummy_run=True)   # 草稿照常跑
StepTimingSample(full_cudagraph=True)
```

超过捕获上限的尾部尺寸仍 eager（`full_cudagraph=False`）。因此
`set_initial_cost_curves()` 的 draft 曲线只取 full 样本；verify 曲线包含图内外两类。

## 5. 测试与实测

```text
pytest tests/step74/test_varlen_attention.py -q   → 4 passed
pytest tests/step74/test_target_graph.py -q       → 5 passed
pytest tests/step79 -q                            → 45 passed
pytest tests/step58 … tests/step79 -q             → 955 passed
check_step74_fused_draft.py                       → 23/23
check_step79_adaptive.py                          → 23/23
```

新增/关键用例：

- `_forward_varlen_padded` vs `_forward_generic`：因果/非因果，`max diff = 0.0`；
- padding 请求（q_len=0）不影响真实行；
- 图内路径禁止 `.item()`（monkeypatch 后仍能跑）；
- adaptive 中间预算：真实 5 行命中捕获 8 行 varlen 图，输出与 eager 一致；
- target 图只服务 decode，prefill 批走 eager；
- profiling 的 `_cudagraph_limit` 是目标图真实最大捕获尺寸（例 8）。

## 6. 差异账本（Phase 2）

| 编号 | 差异 | 为什么 | 影响 |
|---|---|---|---|
| D-74V-1 | varlen 注意力用 Torch 稠密 gather+mask，不是上游 FlashAttention/Triton varlen kernel | 本仓库只有 Torch SDPA 后端；不假装有 Triton 后端 | 语义/图安全性对齐；显存/算力按 `B × max_query_len × P` 物化，长上下文更贵 |
| D-74V-2 | 后端 capability 仍是 `UNIFORM_BATCH`，不是上游 `ALWAYS`；`maybe_create_adaptive_verification_manager` 仍传 `require_varlen_cudagraph=False` | 本仓库没有 mixed prefill/decode 的图内 full 路径（Phase 3 breakable 才处理混合批） | target varlen decode 图可用，但 `FULL` 混合批仍会被降级/回退；不能宣称 ALWAYS |
| D-74V-3 | varlen target 图只服务 **decode** 批 | 短 prefill 的 query_len 可能 <= 1+K，误命中会按 decode 语义算 | 混合/纯 prefill 走 eager；正确性优先 |
| D-74V-4 | profiling 的静态输出缓冲按**最大捕获 token** 分配 | 捕获顺序从大图开始，但显式按 max 分配更稳 | 所有图共用一块 hidden/aux；重放地址稳定 |
| D-74V-5 | target 图的 `max_query_len=1+K` 固定；budget 裁剪后的真实 q_lens 只作为张量值 | 捕获时形状固定，重放时长度可变 | 与上游 varlen decode 的语义一致 |
| D-74V-6 | `batches_to_profile` 仍会补捕获上限外的 eager 尺寸 | 与上游一致：图外成本用实测而不是纯外推 | 成本表在 limit 以上按实测点插值/外推 |

## 7. 后续 phase 状态（Phase 2 当时明确没做）

- 混合 prefill/decode 的 target 图（PIECEWISE / breakable）→ **Phase 3 已实现**，见 `docs/step74_target_graph_breakable.md`；
- `AttentionCGSupport.ALWAYS`；
- `FULL_AND_PIECEWISE` 的 target 混合批 → **Phase 3 已实现**；
- 真实 DSpark 权重下的收益/加速比（本机无权重，仅 tiny 协议与数值）。
