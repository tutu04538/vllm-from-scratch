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
(request, query_block, query_head)
```

grid：

```text
(num_reqs * blocks_per_req, num_heads)
```

其中：

```text
BLOCK_M = 16
N = q.shape[0]（本轮 token 行数，图路径里是补齐后的宽度）

blocks_per_req =
    ceil(max_query_len / BLOCK_M)   若 metadata 有 uniform_query_len / max_query_len
    ceil(N / BLOCK_M)               否则（混合 prefill 没有单请求上界）
```

`max_query_len` 是 Python int（uniform decode = 1+K，varlen decode = 1+K 上界），
所以用它可以把每请求的 block 数收紧；混合 prefill 的 `max_query_len=None`，只能用整批
`N` 作为上界。grid 里多出来的 block 由 kernel 的 `q_blk * BLOCK_M < q_len` guard 跳过。

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
| program | 一行 query 一个 head | **一个 query block 一个 head** |
| Q 分块 | 无，Q 只有 1 行 | `BLOCK_M=16` |
| 矩阵乘法 | 逐元素 `sum(K*q)` | `tl.dot(Q, K^T)` |
| 每请求 q 行 | 每行独立 program | 一个 block 覆盖多行 |
| decode K/V | 每行重读 | 每请求/head 读一次 |
| prefill | 每行独立 | query block 并行 |
| 代码量 | 190 行 | ~230 行（固定场景） |

decode 的 K+1 行现在在同一个 program 里共享一次 K/V 扫描；prefill 的多个 query
block 并行。这个固定 kernel 的目标就是这个场景，而不是通用 TRITON_ATTN。

### 每请求固定 block 数会浪费吗？

会，尤其在每请求 q_len 不均时：

```text
B=8, decode q_len=4, N=32
  ceil(N/16)=2  → grid 16 个 program
  实际每请求只需要 1 个 block → 8 个 program 空转（50%）

有 max_query_len=4 后：
  ceil(4/16)=1 → grid 8 个 program，没有空转
```

混合 prefill 仍用 `ceil(N/16)` 作为每请求上界。例如：

```text
B=4, N=64, q_lens=[1,1,1,61]
  每请求 4 个 block → grid 16
  实际需要 1+1+1+4=7 个 → 9 个空转
```

空转 program 只读 qsl/seq_lens 后走 guard 返回，不写显存，正确性不受影响；浪费的是
grid 调度/occupancy。要彻底消除，需要“block → request”的映射（总 block 上界
`ceil(N/16)+B`，每个 program 查自己的 request/q_start），属于下一步优化，当前没做。

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
