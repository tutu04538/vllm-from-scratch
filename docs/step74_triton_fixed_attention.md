# 第七十四关补充：固定场景 Triton paged attention

> 目标：不再维护 1199 行 vendored 上游 unified_attention，只保留一个固定场景的
> 单 kernel：**Qwen3 target、block16、head_dim 16/32/64/128、causal、GQA、
> paged KV、fp16/bf16/fp32**。
>
> 这个后端不是完整上游 `TRITON_ATTN`；配置名 `triton_attn`，`triton` 是别名。

## 1. 固定场景

```text
模型：Qwen3 target（单卡）
block_size = 16
head_dim = 16 / 32 / 64 / 128
dtype = fp16 / bf16 / fp32
mask = causal
GQA
paged KV
q 行按 query_start_loc 打包
prefill / decode / 混合批都支持
```

明确不支持：

```text
量化 KV（FP8/INT8/INT4/NVFP4/TurboQuant）
sliding window
cascade attention
sinks
ALiBi / softcap
MM prefix / RSWA
非 causal
tensor descriptor
```

这些由后端拒绝或直接不提供入口；不是“静默降级”。

## 2. 代码结构

```text
minivllm/attention/ops/fixed_attention.py     # 一个 kernel + launcher，约 230 行
minivllm/attention/backends/triton_attn.py    # builder + impl + fallback，约 100 行
minivllm/attention/backends/triton_paged.py   # block4/head4 的旧逐行 fallback，约 190 行
```

已经没有：

```text
triton_unified_attention.py（1199 行）
triton_attention_helpers.py（440 行）
kv_quant.py
```

## 3. query 分块

一个 program 负责：

```text
(query block, kv head)
```

GQA 的 `GROUP_SIZE` 个 query head 被打包进同一个 `BLOCK_M=16` 行：

```text
BLOCK_Q = BLOCK_M // GROUP_SIZE     # 每个 program 覆盖的 query 位置数
BLOCK_M = 16                        # Q tile 行数 = BLOCK_Q × GROUP_SIZE
```

grid：

```text
decode（有 max_query_len）：
    (num_reqs * ceil(max_query_len / BLOCK_Q), num_kv_heads)

prefill/混合（没有单请求上界）：
    (floor(N / BLOCK_Q) + num_reqs, num_kv_heads)
```

`N = q.shape[0]`（本轮 token 行数，图路径里是补齐后的宽度）。

prefill 的 grid 用**上游同款的上界**：

```text
sum_i ceil(q_len_i / BLOCK_Q)
  <= floor(N / BLOCK_Q) + num_reqs
```

第 0 维是全局 block id，kernel 内用 GPU 上的 `query_start_loc` 做二分，
把它映射回 `(req, q_blk)`：

```text
mid_val = query_start_loc[mid] // BLOCK_Q + mid
q_block_local = pid - (query_start_loc[req] // BLOCK_Q + req)
```

不需要 host 侧知道每个请求的真实 block 数，也不会给所有请求都分配
`ceil(N/BLOCK_M)` 个 block。多出来的少量 block 由
`q_blk * BLOCK_Q < q_len` guard 跳过。

decode 有 `uniform_query_len` / `max_query_len`（Python int），直接用它得到
更紧的 grid；这是本仓库相对上游统一二分的一个小优化。

- decode：`q_len` 通常 1..K+1，只有一个 query block；
- prefill：`q_len` 可达多块，每个 query block 独立 program，**并行执行**；
- 多出来的 block 在 kernel 里用 `q_blk * BLOCK_M < q_len` 直接跳过；
- 每个 block 内：

```text
Q tile: [BLOCK_M, D]
K tile: [BLOCK_N, D]
S = Q @ K^T                     # tl.dot
causal mask: log_n <= q_pos
online softmax
O = P @ V
```

`q_pos = seq_len - q_len + q_blk * BLOCK_M + i`，无效 q 行和未来 key 都 mask。

## 4. 与旧 `triton_paged` 的区别

| 维度 | 旧 `triton_paged` | 固定 `fixed_attention` |
|---|---|---|
| program | 一行 query 一个 head | **一个 query block × GQA 组** |
| Q 分块 | 无，Q 只有 1 行 | `BLOCK_M=16`（`BLOCK_Q×GROUP_SIZE`） |
| 矩阵乘法 | 逐元素 `sum(K*q)` | `tl.dot(Q, K^T)` |
| 每请求 q 行 | 每行独立 program | 一个 block 覆盖多行 |
| decode K/V | 每行重读 | 每请求/head 读一次 |
| prefill | 每行独立 | query block 并行 |
| 代码量 | 190 行 | ~230 行（固定场景） |

decode 的 K+1 行现在在同一个 program 里共享一次 K/V 扫描；prefill 的多个 query
block 并行。这个固定 kernel 的目标就是这个场景，而不是通用 TRITON_ATTN。

### 还会浪费吗？

不会再有“每请求固定 `ceil(N/BLOCK_M)` 个 block”的浪费：

- decode：`max_query_len` 已知，grid 按 `ceil(max_query_len/BLOCK_Q)`；
- prefill/混合：grid 上界 `floor(N/BLOCK_Q)+num_reqs`，kernel 内二分映射；
- 上界之外只可能有至多 `num_reqs` 个空 block（数学上界），由 guard 跳过。

例如：

```text
B=4, N=64, q_lens=[1,1,1,61], BLOCK_Q=8（GQA group=2）
sum ceil = 1+1+1+8 = 11
grid 上界 = floor(64/8)+4 = 12
→ 最多 1 个空 block（旧的“每请求 8 个”会是 32 个）
```

上游也是同样的思路；它没有把每请求 block 数写到 Python 侧，而是把
`query_start_loc` 传进 kernel 做二分。

## 5. 配置

```python
attention_backend="triton_attn"   # 固定 kernel
attention_backend="triton"        # 同上
attention_backend="triton_paged"  # 旧逐行 kernel（fallback/对照）
attention_backend="torch"         # 默认 Torch 参考
```

block4/head4 自动回退 `triton_paged`（tiny 测试兼容）；block16/head>=16 使用
`fixed_attention`。

## 6. 测试与实测

```text
pytest tests/step74/test_triton_paged.py -q       → 7 passed
  - 旧 kernel vs Torch（6 项）
  - query-tiled prefill（20+7 行，>BLOCK_M）1 项
pytest tests/step74/test_triton_backend_e2e.py -q → 3 passed
  - block4 fallback、EAGLE3 target、block16 fixed kernel
pytest tests/step79 -q                            → 47 passed
  - DSpark adaptive + block16 fixed kernel
pytest tests/step58 … tests/step79 -q             → 968 passed
```

## 7. 差异账本

| 编号 | 差异 | 影响 |
|---|---|---|
| D-FIX-1 | 只支持固定场景；量化/sliding/cascade/sinks/MM/RSWA/非 causal 不支持 | 其他模型/配置回退 Torch 或旧 kernel |
| D-FIX-2 | 没有上游 2D/3D、softmax 分段、tensor descriptor | 性能不等价于完整 TRITON_ATTN |
| D-FIX-3 | prefill 的 K/V 会被多个 query block 重读 | decode 优先；prefill 不如上游 tile 复用 |
| D-FIX-4 | block4/head4 回退旧逐行 kernel | tiny 测试兼容，不是固定 kernel 行为 |
| D-FIX-5 | 只接 V2 target；草稿层仍 Torch | 草稿图不走此 kernel |
| D-FIX-6 | `get_name()` 是 `TRITON_ATTN_FIXED` | 不冒充完整上游 |

## 8. 一句话

```text
固定场景 kernel = 一个 program 一个 query block/head，
Q tile 走 tl.dot，query block 并行，decode 的 1..K+1 行共享 K/V，
代码 ~230 行；不支持量化/sliding/cascade/sinks/3D。
```
