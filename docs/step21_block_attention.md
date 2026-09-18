# step21：不拼完整历史，按块计算 attention

- 对应代码：`step21.py`（新增，未提交）
- 源码 SHA256：`c427cca1672c3d7ff9b72b7ec6e6ede068c5a4c9a2c5f1f78273b346ed59eb9b`
- 基线：`step20.py`（SHA256 `9c6d204b…`）原样保留、未修改

## 0. 需求大概

step20 的 KV 搬运已经是批量操作，但 attention 仍然把整条历史拼成连续矩阵：

```text
块表 [5,2,7] → gather 出完整 K/V → Q @ K.T → softmax → weights @ V
```

本关要求**不复制出完整历史 K/V，也不保存覆盖全部历史的 score 矩阵**，结果仍然正确。做法是按逻辑块读 KV，用 online softmax 把各块的中间结果合并起来。

只换 attention 的计算方式：调度器、分配器、packed 输入、slot mapping、原位 KV、chunked prefill、prefix cache/LRU、停止与回调全部不变。这不解决"Python 逐请求调用 attention"的瓶颈，也不是高性能 GPU kernel——两件事的区别是另一篇的内容。

## 1. 改动内容

| 位置 | 改动 |
|---|---|
| `KVCachePool.block_view` | 新增：返回池里某个物理块的有效 K/V 切片，是池存储的**视图**，不复制 |
| `KVCachePool.gather` | 保留，但注明只作参考/调试；attention 路径不再调用 |
| `TinyCausalLM.block_attention` | 新增：按逻辑块遍历，online softmax 合并，返回 `out[Q,D]` |
| `TinyCausalLM._forward_append` | 逐请求的 "gather + 完整 score + softmax + @V" 换成一次 `block_attention(q_slice, cache, pool, query_positions)` |

## 2. 设计要点

### 2.1 会算错的做法：每块各自 softmax 再合并

分数都是 0、value 分别是 `[2,4]` 和 `[9]` 时，全局 softmax 三个权重都是 `1/3`：

```text
正确：(2+4+9)/3 = 5
错误：第一块 softmax 得 3，第二块得 9，再平均得 6
```

每块各自归一化会把权重和变成 1，丢掉"这块在全部 key 里占多少"的信息。**难点不是遍历块，而是合并方式。**

### 2.2 三个累计量，和一次合并公式

每个 query 保留：

| 量 | 含义 | 形状 |
|---|---|---|
| `m` | 已见最大分数（防 `exp` 溢出） | `[Q,1]` |
| `z` | 以 `m` 为基准的指数权重和（未归一化） | `[Q,1]` |
| `u` | 同一基准下的加权 value 和 | `[Q,D]` |

读入一个块（分数 `s[Q,B]`、value `V[B,D]`）时：

```python
block_max = s.max(dim=-1, keepdim=True).values      # [Q,1]
m_new = torch.maximum(m, block_max)
scale = torch.exp(m - m_new)                        # 旧结果换到新基准
p = torch.exp(s - m_new)                            # [Q,B]，未归一化

z = scale * z + p.sum(dim=-1, keepdim=True)
u = scale * u + torch.matmul(p, v_block)
m = m_new
```

最后 `out = u / z`。`p` 是**每个 query 对块内每个 key** 的权重，不是整块一个标量——`p @ V` 的 `[Q,B] @ [B,D] → [Q,D]` 正是加权求和。

### 2.3 初始状态用 `-inf / 0 / 0`，第一块自动完成初始化

```python
m = torch.full((Q, 1), float('-inf'))
z = torch.zeros((Q, 1))
u = torch.zeros((Q, D))
```

第一块进来时 `m_new` 是有限的 `block_max`，于是：

- `scale = exp(-inf - 有限值) = 0` → 旧的 `z`、`u`（都是 0）被乘 0，新块直接填进去；
- `p = exp(s - m_new)` 正常。

**不需要为"第一块"写特判**——`-inf` 初始值天然让第一块成为唯一的贡献者。

### 2.4 全被 mask 的行不能出 NaN

尾块可能对某些 query 全部不可见（比如 query 位置 `[2,3,4]`、尾块只有 key 位置 `4`，前两行全被 mask）。这时 `block_max = -inf`：

```text
m_new = max(m, -inf) = m            （m 已经有限）
scale = exp(m - m)   = 1            旧状态原样保留
p     = exp(-inf - m) = 0           该块贡献零
```

所以这一行既不产生 NaN，也不改变已有统计量。

**前提是 `m` 在第一次合并后一定有限**：第一个逻辑块包含逻辑位置 0，而所有 query 位置都 ≥ 0，所以每个 query 至少能看到位置 0 的 key。危险的是 `exp(-inf - (-inf))` 这种 `-inf - -inf`，只有"第一块就整行全 mask"才会出现，而它不会发生。

### 2.5 块 K/V 是视图，因果用逻辑位置

```python
def block_view(self, cache, logical_block, count):
    block_idx = cache.block_table[logical_block]
    return self.k_cache[block_idx][:count], self.v_cache[block_idx][:count]   # 视图
```

`k_cache[block_idx]` 是 dim 0 的 select、`[:count]` 是基本切片，两者都返回共享存储的视图，`data_ptr` 与池一致。所以这一关的"当前块 K/V"连 `[S,D]` 的临时副本都不需要。

因果 mask 用的是**逻辑 key 位置**，不是物理块号：

```python
key_positions = torch.arange(block_start, block_start + count)
score = score.masked_fill(key_positions.unsqueeze(0) > query_positions.unsqueeze(-1), float('-inf'))
```

`block_start = logical_block * block_size`，与物理块编号无关——块表乱序（`[7,1]`）不影响正确性。尾块只读 `min(block_size, cache.length - block_start)` 行；预留但没写入的块根本不进循环（循环上界是 `ceil(length / block_size)`）。

### 2.6 中间数据规模

| 中间量 | 本关实际 |
|---|---|
| 当前块 K/V | `[count, D]` **视图**，0 额外分配 |
| 当前块 score / `p` | `[Q, count]`，`count ≤ block_size` |
| `u` | `[Q, D]` |
| `m`、`z` | `[Q, 1]` |

不把各块 score 或 K/V 存进列表再 `cat`——那只是换个地方 gather。

## 3. 验证

conda `vllm-omni-dev`；模型与池落在 cuda:0，float32。

### 3.1 正确性

- **需求 §2 的手算例子**：分块合并得 `5.0`；错误做法（每块各自 softmax 再平均）得 `6.0`。
- **与完整 attention 对照**：`block_table=[5,2]`、length=7、2 个 query，最大误差 `0.000e+00`（完全一致）。
- **全 mask 的行**：query 位置 `[2,3,4]`、尾块只有 key 位置 `4`，前两行对该块全被 mask——输出全部有限，与完整 attention 最大误差 `0.000e+00`。
- **大幅正负分数**：K/V 与 q 都放大 50 倍，输出仍全部有限。
- **确实不走 gather**：把 `KVCachePool.gather` 换成抛异常的桩，跑完整个 3 请求场景（含 prefix cache 命中、chunked prefill）没有触发。
- **不改写池**：`block_attention` 调用前后逐值比对，池的 K/V 完全未变（共享前缀只读）。
- **block_view 是视图**：与池共享同一块 storage；尾块返回 `(2, 8)`。
- **step20 vs step21**：300 组随机场景（并发 1–3、预算 1/2/4/8、块大小 1/2/4、池 2–8 块、缓存在 1/3 场景关闭），**逐轮返回与回调顺序不一致 0 例**，结束后 `block_usage` 全零。
- **独立参考**：逐步 KV 核对 353 次不符 0；193 个生成 token 全部等于参考模型在同一前缀上的 argmax。
- **prefix cache 回归**：条目内容不变量（120 组，不符 0）、索引双射（300 组 / 60027 次 step）、开关输出对照（300 组，不一致 0）在 step21 上全部通过。

### 3.2 中间数据规模（CUDA 峰值，预热后测量）

`d_model=64`、`block_size=128`、`Q=16`，只测 attention 调用本身：

| 历史长度 | step20 中间峰值 | step21 中间峰值 | 完整 score 理论值 |
|---:|---:|---:|---:|
| 512 | 328.0 KiB | 39.5 KiB | 32.0 KiB |
| 2048 | 1312.0 KiB | 39.5 KiB | 128.0 KiB |
| 8192 | 5248.0 KiB | 39.5 KiB | 512.0 KiB |

step20 的峰值随历史**线性增长**（约 0.64 KiB/位置：K/V 各一份 `[L,D]` 加上 `[Q,L]` 的 score 与权重），step21 **恒定 39.5 KiB**（`[Q,block_size]` 的 score 与 `p`、`[Q,D]` 的 `u`，加 mask 与 softmax 临时量）。8192 位置时相差 133 倍。

第一次测量没预热时出现过 33 MB 的读数，那是 CUDA 缓存分配器首次扩容的假象，不作为结论。

### 3.3 时间（粗测，attention 层）

`d_model=64`、`block_size=128`、`Q=16`、CPU：

| 历史长度 | step20 完整 attention | step21 分块 online | 倍数 |
|---:|---:|---:|---:|
| 256 | 0.056 ms | 0.140 ms | 2.51x |
| 1024 | 0.145 ms | 0.602 ms | 4.15x |

**分块更慢**：一次大矩阵乘法被拆成多个小块乘法，外加块循环、指数运算和累计量更新。这与"用更少的临时空间、花更多时间"的预期一致。**这里只证明"分块合并结果正确"，不是加速结论**；完整 Engine 的性能由验收方按相同负载测。

## 4. 接口变化与遗留

- 新增 `KVCachePool.block_view(cache, logical_block, count)`。
- 新增 `TinyCausalLM.block_attention(q, cache, kv_cache_pool, query_positions) -> out[Q,D]`。
- `KVCachePool.gather` 保留、签名不变，仅供参考与调试；Engine 的 attention 路径不再调用它。
- `TinyCausalLM.forward`（完整历史参考实现）未改动，仍可用于对照。
- `_forward_append` 的签名与返回值未变。
- 遗留：仍是逐请求 Python 循环 + 逐块循环，没有把多个请求并行、也没有 kernel 融合；块循环里每块一次 `matmul`，小模型下算子启动开销占比高。
- 未提交。
