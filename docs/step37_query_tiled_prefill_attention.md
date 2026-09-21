# step37：query 分块的分页 prefill attention

- 对应代码：`step37/`（新增，从 `step35/` 复制）
- 包摘要 SHA256：`ca06af2a7d074f1c97dad7241bb610b91e17e7f4f3604c6b509d6a5e3263fc26`（13 个 .py / 2350 行）
  - 完成时的指纹是 `e7601c9e…`；此后**只**改了一行：`tiled_paged_attention` 里未被使用的
    `num_rows` 改成 `_`（编辑器告警）。行数、行为、本文所有结论都不受影响。
- 基线：`step35/`，指纹 `a00a46d9…`（13 个 .py / 2132 行），原样保留未改
- 改动文件：`attention.py`、`model.py`、`__init__.py`、入口改名 `step37.py`

## 0. 需求大概

第三十六关把差距定位到一个 kernel：`prefill_c8` 里 attention 占 GPU 时间 86%，
同一批 56 次调用我们 421.81 ms、vLLM 7.18 ms。这一关只改这一个方向。

原来的 `_paged_attention_kernel`：

```text
一个 program → 一行 query、一个 head
读 K/V → 逐元素乘 + 归约求分数 → online softmax → 逐元素乘 + 归约求输出
```

prefill 时同一请求有很多 query，相邻 query 读的 K/V 几乎相同却各自重读一遍；
而且 `tl.sum(k * q)` 是手写点积，全程 CUDA core，没用矩阵乘法单元。

改成：

```text
一个 program → 同一请求的一小组 query、一个 head
读一块 K/V → 一次矩阵乘法算出这组 query 对这块 K/V 的分数 → 每行各自更新 online softmax
```

**不是重新发明 softmax，是把已有的 online softmax 从「一行」扩展到「多行」。**

需求划定的范围：只做 BF16 + 每请求本轮追加多个 token 的路径；旧 kernel 完整保留，
作为参考、纯 decode 路径和回退。第一版固定 `BLOCK_M=32`、`BLOCK_N=64`，不做自动调优。
不做调度、KV 分配、QKV/MLP 融合、采样重写、split-KV。

## 1. 改动内容

| 位置 | 改动 |
|---|---|
| `attention.py` | 新增 `_tiled_paged_attention_kernel` 与 `tiled_paged_attention`；`AttentionMetadata` 增加 tile 缓冲；`fill()` 增加 tile 构建 |
| `model.py` | 新增 `self.use_tiled` 路径选择；图缓存键加入该标志；`_attention_out` 按路径分派 |
| `engine.py` | **未改** |
| `cache.py` / `scheduler.py` / `norm.py` / `formats/` | **未改** |

## 2. 设计要点

### 2.1 tile 划分：一个 program 处理谁

tile 是「**同一请求内**连续的至多 `BLOCK_M` 行 query」。请求之间绝不合并：

```text
请求 A 的 20 行 | 请求 B 的 50 行
A → 1 个 tile（20 行）
B → 2 个 tile（32 行 + 18 行）
```

所以不能拿打包前 32 行当 A 的 tile——后 12 行属于 B。范围来自 CPU 侧构建的 tile 表。

### 2.2 元数据：为什么要在 CPU 侧建表

`gpu_forward` 在图内运行，只能读**固定地址的 GPU 缓冲**，拿不到 Python 侧的
`num_scheduled_tokens`。所以 tile 归属必须在图外算好、整段上传。

`AttentionMetadata` 仍是**同一段连续 int32 存储**，一次 `copy_` 上传。新增四个区域：

```text
seq_lens | block_tables | token_to_req | query_pos
        | num_tiles | tile_req | tile_q_start | tile_q_count
```

`tile_q_start` 是该 tile 首行的**打包行号**；首行的真实位置直接从已有的
`query_pos[tile_q_start]` 读，不再单独存一份位置。

tile 容量上界 = `ceil(max_num_query_tokens / BLOCK_M) + max_num_seqs`
（每个请求至少占一个 tile，最坏情况是每个请求都浪费不到一整块）。这个上界与
「本轮怎么切分」无关，**所以 grid 大小固定**。

### 2.3 因果掩码用真实位置，不用 tile 内行号

已有 5 个 KV、本轮追加 3 个 token 时：

```text
tile 内行 0 → 真实位置 5 → 只能看 key 0～5
tile 内行 1 → 真实位置 6 → 只能看 key 0～6
```

kernel 里算的是 `q_pos = q_pos0 + offs_m`，掩码条件是 `log_n <= q_pos[i]`。
拿 `0/1/2` 当位置会算错。

尾块的 padding 行（`offs_m >= q_count`）没有真实位置。给它们一个**确定的**可见位置
（0），避免整行被屏蔽时出现 `-inf - (-inf) = NaN`；这些行的输出不会被写回
（store 带 `valid_m` 掩码）。有效行永远至少能看到 key 0，所以不会整行被屏蔽。

### 2.4 计算块 ≠ 物理 KV 块

`BLOCK_N=64` 而物理 `block_size=16`，一个计算块横跨 4 个物理块，且它们的物理块号
**不连续**（可能是 `[7, 2, 19, 4]`）。所以每个 key 都各自换算：

```text
逻辑位置 → 块号 blk_idx / 块内偏移 blk_off → block_table[req, blk_idx] → 物理地址
```

不能拿到首块地址后连续读 64 个 token——那样读到的是别的请求的数据。

### 2.5 状态从一行扩到多行

| 状态 | 逐行版本 | 分块版本 |
|---|---|---|
| 当前最大值 `m` | 标量 | `[BLOCK_M]` |
| softmax 分母 `z` | 标量 | `[BLOCK_M]` |
| 未归一化输出 `acc` | `[D]` | `[BLOCK_M, D]` |

每行独立保持自己的 `m/z`。**不能对整个 `[BLOCK_M, BLOCK_N]` 分数矩阵只求一个最大值**——
那是把 32 行的 softmax 混成一行。

PV 的 P 输入要先转成 KV 的运行精度才能进 `tl.dot`，这一步会引入额外舍入。
需求明确不要求与旧版逐位相同，但要单独做数值对照（见 §4.1）。

### 2.6 路径选择进图缓存键

这是本关最容易出错的地方。**相同 token 总数不代表请求划分相同**：

```text
8 条请求各 512 行      → 全是 prefill，走分块
8 条请求各 1 行        → 全是 decode，走逐行
```

两者 N 都是 4096，但 kernel、grid 都不同。所以图缓存键从

```python
(num_tokens, num_samples, sample_rows_given)
```

变成

```python
(num_tokens, num_samples, sample_rows_given, use_tiled)
```

另一种做法是把路径判断也放进 GPU 元数据、两条 kernel 都发射然后各自空转，但那样
每层都要多发射一个空 kernel。需求允许「纳入图缓存键」这条路，选了它。

grid 固定 + tile 表在 GPU 上，这两件事合起来才使得**同一张图能重放不同的请求划分**：
同一个 N=64 的图，既能跑「32+32」，也能跑「40+24」。

### 2.7 什么时候走新路、什么时候回退

```python
self.use_tiled = (dtype == bfloat16
                  and 本轮有请求
                  and 每条请求的 num_scheduled_tokens > 1)
```

- **纯 decode**（每请求 1 行）：分块没有可复用的 query，回退逐行。
- **混合 prefill/decode**：整批回退。需求允许，这一版不强求它提速。
- **FP32**：回退。分块 kernel 用 `tl.dot`，FP32 输入会走 TF32，精度与旧版不一致，
  不该悄悄降精度。
- **attention_backend != triton**：走原有的 torch 路径，与本次改动无关。

## 3. 验证

### 3.1 算子数值（`benchmarks/check_step37_tiled_attention.py`）

物理 `block_size=4`、`head_dim=16`、6 q heads / 2 kv heads（GQA，group=3），
物理块号打乱、故意不连续。请求覆盖：(17,20) 单 tile、(5,100) 多 tile、(0,64) 整除、
(0,1) 单 token、(33,31) 尾块、(50,3) 历史远长于追加。

误差判据是**先验推导**的，不是看到结果再放宽：

- 两个 kernel 的输入 Q/K/V 都是 BF16，输出也存回 BF16；
- 参考用 FP32 算同一批 BF16 输入，所以「旧 kernel vs 参考」的差主要就是
  **BF16 输出存储的量化台阶**（最大元素量级 × 2⁻⁹）；
- 分块 kernel 多一步 P 量化，其贡献与输出量化同量级，因此上限取旧 kernel 的 **2 倍**。

| | 最大绝对差 | 相对参考 RMS |
|---|---:|---:|
| 旧 kernel | 0.003887 | 1.32e-02 |
| 新 kernel | 0.004940 | 1.68e-02 |
| 最新/旧误差比 | **1.27**（上限 2.0） | |

两者都落在 BF16 台阶（0.004578）之内，说明误差就是量化本身。

### 3.2 引擎语义（`benchmarks/check_step37_engine_paths.py`，15/15 通过）

| 检查 | 结果 |
|---|---|
| prefill（每条 32 行、BF16）走分块 | PASS |
| prefill 走分块、下一步 decode 回退逐行 | PASS |
| FP32 回退逐行 | PASS |
| 混合 prefill/decode 整批回退 | PASS |
| **A(32+32) 与 C(40+24) 的 eager/Graph 各一致** | PASS ×4 |
| **A 之后跑 C：A 不受污染、C 结果正确（复用 A 的图）** | PASS ×2 |
| chunked prefill 与整段 prefill 结果一致 | PASS |
| M=0 不报错 | PASS |
| prefill 两条路径的 logits 差 ≤ 4× BF16 台阶 | PASS（差 0.000977，台阶 0.000481，2.0 倍） |
| prefill 两条路径 argmax 一致 | PASS |
| 单行 query 时分块 kernel 仍正确 | PASS |

**关于判据的一处修正**：最初写的判据是「两条路径生成的文本逐 token 相同」，结果
失败（`[10,8,6,10]` vs `[10,0,0,0]`）。查下来 logits 只差 0.000977（约 2 个 BF16 台阶），
argmax 完全相同——**分叉来自小模型上 BF16 量级的差翻转了后续 argmax，不是 kernel 错误**。
把判据改成比 logits（需求 §6.1 的口径）后通过。用「文本逐 token 相同」当标准，
会把正常的舍入差误报成 bug。

### 3.3 性能（`benchmarks/bench_step37_prefill.py`）

同一份输入、同一配置（BF16 / triton attention / triton norm / Graph 开 / KV 512 块），
只换 attention 实现。

| 测点 | step35（逐行） | step37（分块） | 变化 |
|---|---:|---:|---:|
| **prefill_c8** | 0.5023 s | **0.0984 s** | **−80.4%（5.1×）** |
| **prefill_c1** | 0.0598 s | **0.0107 s** | **−82.1%（5.6×）** |
| short_c1 | 0.1278 s | 0.1185 s | −7.3% |
| short_c8 | 0.2152 s | 0.2094 s | −2.7% |
| decode_c1 | 0.5260 s | 0.5135 s | −2.4% |
| decode_c8 | 0.7807 s | 0.7860 s | +0.7% |

后四个点都走逐行路径，差异在测量波动内，**没有回退**。

与 vLLM 对照（vLLM 数字取自第三十六关 v2 矩阵）：

| 测点 | 我们 | vLLM | 改前 | 改后 |
|---|---:|---:|---:|---:|
| prefill_c8 | 0.0984 s | 0.0764 s | 6.51× | **1.29×** |
| prefill_c1 | 0.0107 s | 0.0157 s | 4.14× | **0.68×（反超）** |

### 3.4 热点复核（需求 §6.4）

`prefill_c8` 的 profiler 分类：

| 类别 | step35 | step37 | vLLM |
|---|---:|---:|---:|
| **attention** | **390.09 ms (86.0%)**，56 次 | **9.61 ms (11.1%)**，56 次 | 7.18 ms (10.0%) |
| GEMM/GEMV | 43.58 ms，394 次 | 56.23 ms，394 次 | 58.37 ms，226 次 |
| elementwise/其他 | 19.71 ms | 21.05 ms | 6.24 ms |
| GPU 忙碌 | 453.4 ms | **86.9 ms** | 71.8 ms |

**attention 从 390.09 ms 降到 9.61 ms（40.6×）**，而且 profile 形状变得与 vLLM 一致：
改前 attention 占 86%、GEMM 占 9.6%；改后 attention 占 11.1%、GEMM 占 64.7%
（vLLM 是 10.0% / 81.3%）。这正是第三十六关预测的结果。

一处**没有解释的观察**：GEMM 从 43.58 ms 变成 56.23 ms（+29%，调用次数都是 394）。
可能是 profiler 运行间的波动，也可能与分块 kernel 抢占同一个执行单元有关。
不影响主结论（墙钟净快 5.1×），但记在这里不下结论。

## 4. 接口变化与遗留

### 4.1 接口变化

**新增**：

| 接口 | 说明 |
|---|---|
| `attention.tiled_paged_attention(...)` | 分块路径入口，签名与旧版平行 |
| `attention._tiled_paged_attention_kernel` | 分块 Triton kernel |
| `attention.BLOCK_M` / `BLOCK_N` / `MIN_DOT_DIM` | 分块尺寸常量 |
| `AttentionMetadata.tile_capacity` | tile 容量上界（固定） |
| `AttentionMetadata.count_tiles(num_scheduled_tokens)` | 静态方法，只数不建 |
| `AttentionMetadata.fill(..., build_tiles=False)` | 新增关键字参数 |
| `AttentionMetadata.validate(..., build_tiles=False)` | 新增关键字参数 |
| `TinyCausalLM.use_tiled` | 本轮走哪条路径 |

**未改**：`paged_attention`、`_paged_attention_kernel`、`Engine` 全部公开接口、
`AttentionMetadata` 原有的五个区域与 `upload()`。

`MIN_DOT_DIM = 16` 是因为 `tl.dot` 的 K 维下限；`head_dim < 16` 时把 K 维补零到 16 再算
（补的部分乘出来是 0，不影响结果）。Qwen3 的 `head_dim=128` 不受影响。

### 4.2 遗留

1. **`BLOCK_M` / `BLOCK_N` 未调优**，固定 32 / 64。需求明确第一版不要求自动调优。
2. **混合 prefill/decode 批次整批回退**，那部分没有提速。需求允许，但这是一块还没吃的收益。
3. **FP32 走不了分块路径**（会降到 TF32 精度），没有为它单独做数值验证。
4. **换请求划分会多捕获一张图**：图键带了 `use_tiled`，同一个 N 在 prefill/decode
   两种划分下各有一张图。图数量上限仍是 O(不同 N) 的常数倍，但没有实测图的内存开销。
5. **单行 query 也走分块 kernel 的路径没有被正常选择逻辑触发**，只在测试里强制验证过。
6. **GEMM 时间上升 29% 未解释**（§3.4）。
7. **未做预热/steady-state 区分**：以上都是含首次图捕获之外的稳态数据，但只测了单进程。
