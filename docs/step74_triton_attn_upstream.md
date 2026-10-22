# 第七十四关补充：上游 TRITON_ATTN 核心子集

> 目标：把上一版“本地 Triton paged kernel”升级成**上游 `TRITON_ATTN` 的
> `unified_attention` 路径**，而不是只改配置名。
>
> 上游基线：`vllm==0.28.0`
> `vllm/v1/attention/backends/triton_attn.py` +
> `vllm/v1/attention/ops/triton_unified_attention.py`。

## 1. 范围：核心子集

已接：

```text
q / k / v 分离的 paged KV
GQA
causal / 非 causal
varlen / 混合 prefill+decode
block_size >= 16
head_dim >= 16
fp16 / bf16 / fp32
CUDA Graph（2D unified kernel）
```

明确拒绝 / 未接：

```text
量化 KV（FP8 / INT8 / INT4 / FP4 / TurboQuant）
sliding window
cascade attention
sinks
MM prefix
R-SWA
tensor descriptor（use_td）
3D softmax 分段路径
```

这些不是“悄悄降级”：要么由 impl 参数固定为 `None/False`，要么由上游 kernel 的
入口断言拒绝；本子集不声称完整 TRITON_ATTN。

## 2. Vendored 文件

```text
minivllm/attention/ops/triton_unified_attention.py     # 上游 1189 行，适配 import
minivllm/attention/ops/triton_attention_helpers.py     # 上游 440 行，适配 import
minivllm/attention/ops/kv_quant.py                     # KVQuantMode 枚举子集
minivllm/attention/backends/triton_attn.py             # 新 backend
```

适配点（不改变 kernel 数学）：

| 上游 | 本仓库 |
|---|---|
| `vllm.triton_utils` | `minivllm.triton_utils` |
| `vllm.envs.VLLM_BATCH_INVARIANT` | 恒 `False` |
| `vllm.platforms.current_platform` | `_LocalPlatform`：FP8 dtype / capability 查询 |
| `vllm.v1.kv_cache_interface.KVQuantMode` | `attention/ops/kv_quant.py` |
| `vllm.logger` | 无（kernel 不使用 logger） |

## 3. KV 布局

本仓库的 KV cache 仍是：

```text
[2, num_blocks, block_size, num_kv_heads, head_size]
 0 = K, 1 = V
```

上游 kernel 需要：

```text
k: [num_blocks, block_size, num_kv_heads, head_size]
v: [num_blocks, block_size, num_kv_heads, head_size]
```

这两个视图与本地 `kv_cache[0]` / `kv_cache[1]` **完全同形同 stride**，所以不需要
packed KV、不需要拷贝：

```python
unified_attention(q=q, k=kv_cache[0], v=kv_cache[1], out=out, ...)
```

与上游 `TritonAttentionImpl` 的 packed `[blocks, Hk, bs, 2*D]` 布局不同；这是本子集
与上游的明确差异（见 §7）。

## 4. 运行路径

```text
Attention.forward
  → TritonAttentionImpl.forward
      write_kv（仍用本地分页 slot_mapping）
      if graph_dispatched:
          block_size >= 16 and head_dim >= 16
              → _forward_unified()        # 上游 unified_attention 2D
          else
              → _forward_paged()          # 本地逐行 fallback
      else:
          → _forward_generic()            # eager / PIECEWISE 的 Torch 参考实现
```

`_forward_unified()` 的调用参数：

```python
unified_attention(
    q=q, k=kv_cache[0], v=kv_cache[1], out=out,
    cu_seqlens_q=query_start_loc,
    max_seqlen_q=num_tokens,                    # Python int 上界
    seqused_k=seq_lens,
    max_seqlen_k=block_table.shape[1] * block_size,
    softmax_scale=self.scale,
    causal=attn_metadata.causal,
    window_size=(-1, -1),                       # 不支持 sliding
    block_table=block_table,
    softcap=0.0,
    q_descale/k_descale/v_descale=None,
    seq_threshold_3D=None,                      # 强制 2D
    ...,
    kv_quant_mode=KVQuantMode.NONE,
    use_td=False,
)
```

`max_seqlen_q/k` 都是 Python int（捕获时固定），不使用 `.item()`。

## 5. 配置名

```python
attention_backend="triton_attn"   # 上游子集（推荐）
attention_backend="triton"        # 同上，保留旧别名
attention_backend="triton_paged"  # 上一版本地逐行 kernel
attention_backend="torch"         # 默认 Torch 参考后端
```

`triton_attn` / `triton` 在 block_size<16 或 head_dim<16 时会自动回退到本地逐行
kernel；这是为了兼容 tiny 测试（block4），不是上游行为。**要验证上游 kernel，必须
用 block_size>=16。**

## 6. 测试与实测

```text
pytest tests/step74/test_triton_paged.py -q       → 6 passed（本地 fallback kernel）
pytest tests/step74/test_triton_backend_e2e.py -q → 3 passed
  - torch vs triton（block4 fallback）
  - EAGLE3 target 图
  - block16 + head16 → 上游 unified_attention
pytest tests/step79/test_adaptive_layout.py -q    → 47 passed
pytest tests/step58 … tests/step79 -q             → 967 passed
```

关键用例：

- `test_triton_attn_upstream_kernel_block16_matches_torch`：
  非投机混合批、block_size=16、head_dim=16，target `mode="full"`，
  输出与 Torch eager 逐 token 相同；`runner.attention_backend.get_name() == "TRITON_ATTN"`。
- `test_adaptive_triton_attn_unified_block16_matches_eager`：
  DSpark adaptive + block16，target varlen FULL 图走上游 unified kernel；
  `_target_varlen_cudagraphs_enabled() == True`，输出与 eager 相同。

## 7. 差异账本

| 编号 | 差异 | 影响 |
|---|---|---|
| D-UTA-1 | 只接 unified attention 的 2D 路径；`seq_threshold_3D=None`，没有 3D softmax 分段 workspace | 语义等价；decode 少一条 3D 优化路径，性能不等价 |
| D-UTA-2 | 没有 `TritonAttentionMetadata` 的全部字段（cascade/scheduler metadata/RSWA/MM prefix/sinks 等） | 这些特性不支持；不会静默启用 |
| D-UTA-3 | KV 布局是本地 `[2, blocks, bs, Hk, D]`，不是上游 packed `[blocks, Hk, bs, 2D]` | kernel 可接受同形 K/V；少了 packed 布局的写缓存优化 |
| D-UTA-4 | block_size<16 / head_dim<16 回退本地逐行 kernel | tiny 测试可跑；不是上游 TRITON_ATTN 行为 |
| D-UTA-5 | 不支持量化 KV / sliding / cascade / sinks / MM prefix / RSWA / TD | 这些配置不能声称“已对齐上游” |
| D-UTA-6 | eager/PIECEWISE 仍走 Torch 参考路径 | 只有 FULL 图内 attention 用上游 kernel；PIECEWISE 的 attention 仍是 eager |
| D-UTA-7 | 只接 V2 target；草稿层仍 Torch | 草稿图/DFlash 不走 TRITON_ATTN |

## 8. 一句话

```text
现在 attention_backend="triton_attn" 的 FULL 图内注意力
走的是 vendored 上游 unified_attention（2D、paged、GQA、varlen），
不是上一版的本地逐行 kernel；
但它是 TRITON_ATTN 的核心子集，不包含量化/sliding/cascade/sinks/3D 等高级路径。
```
