# triton_unified_attention 学习问答

> 记录用户在学习 `minivllm/attention/ops/triton_unified_attention.py` 时提出的问题与答案。
> 按提问顺序追加，不把“结论”写成源码复述；每个答案尽量给代码位置、证据和边界。

## 0. 代码快照

```text
文件：minivllm/attention/ops/triton_unified_attention.py
当前 HEAD：f084053（Astra stride 修复 + 行内 type ignore 之后）
Triton：3.7.1
vLLM 参考：0.28.0
GPU：RTX 5090 Laptop / SM120
目标 dtype：BF16
```

当前生产策略（先记结论，细节在 Q&A 中展开）：

```text
K/V 最后一维 stride：tl.constexpr（恢复 AxisInfo 连续性）
K/V load：K 在 dot 前 tl.trans
BF16/FP16 全注意力、head<=256：TILE32 + BLOCK_M16
head=512：BLOCK_M8
FP32：保留旧 TILE/BLOCK_M + IEEE dot
SWA：保留旧 TILE64/BLOCK_M8
3D：TILE16，原行分组
```

## 1. 阅读地图

```text
unified_attention(...)              Python wrapper：选择 2D/3D、TILE、BLOCK_M、warp
  └─ kernel_unified_attention       主 kernel：每条请求/每个 q block 扫 KV tile
       ├─ _resolve_seq              从 query_start_loc/seq_lens 解析请求
       ├─ _kv_seq_mask              causal / sliding window / 非 causal 的 key mask
       ├─ _softmax_step             online softmax 的 M/L/P/alpha
       └─ reduce_segments           3D 分段 softmax 的最终合并

kernel_reshape_and_cache_flash     写 KV cache
```

2D 路径：

```text
grid = (query_blocks, num_kv_heads)
每个 program 负责一个 query block + 一个 KV head
串行扫 seq_len 范围内的 KV tile
```

3D 路径：

```text
grid = (query_blocks, num_kv_heads, num_par_softmax_segments)
把 KV 分成多段并行，再 reduce_segments 合并
```

## 2. 问题索引

| 编号 | 问题 | 关键词 |
|---|---|---|
| Q1 | VSCode 为什么无法解析 `tl.int64`，提示 Variable not allowed in type expression？ | Pylance、Triton 注解、`# type: ignore` |
| Q2 | `workspace_pool` 的 key 有哪些？3D 的 5 个参数各自做什么？ | 3D workspace、分段 softmax、reduce_segments |
| Q3 | `window_size` 为什么是 tuple？第二个元素是干什么的？ | sliding window、left/right、causal/non-causal |
| Q4 | `tile_size` 这个 tile 是指什么 tile？ | KV tile、BLOCK_SIZE、BLOCK_M、循环 |
| Q5 | `seqused_k` 是什么？ | KV 长度、query 长度、context_len、投机 |
| Q6 | `tl.full([], 0, tl.int32)` 是什么？ | 0-D tensor、scalar、二分查找 |
| Q7 | 为什么有 `offs_m < BLOCK_Q * num_queries_per_kv`？BLOCK_M 不是等于 BLOCK_Q * group 吗？ | GQA、padding 行、BLOCK_M/BLOCK_Q |
| Q8 | `max_prefix` 是什么？ | causal、KV 前缀上界、tile_mask、循环上界 |
| Q9 | 为什么 `max_prefix` 不写成 `context_len + (q_block_local+1)*BLOCK_Q`？ | 等价性、padding、保守上界 |
| Q10 | f3bd003 里 `k_ptrs` load 出 `[H, TILE_SIZE]`，是转置后的，有意为之吗？ | K layout、dot 操作数、cache 布局 |
| Q11 | `input_precision="ieee"` 是什么？ | tl.dot、TF32、FP32、数值精度 |
| Q12 | `v_keep = tl.max(tl.where(keep, 1, 0), axis=0) > 0` 在做什么？ | V 清零、NaN、keep 列归约 |
| Q13 | V load 已经 `other=0.0`，为什么还会出现 NaN？ | tl.load mask、other 语义、未初始化槽位 |
| Q14 | K 和 V 的 mask 一样，为什么只对 V 做 v_keep？ | 数据流位置、post-dot mask、NaN |
| Q15 | “所有 query 都看不到的 V 列”具体长什么样？ | sliding window、keep 表、V 行/列 |
| Q16 | 从 basic 版到当前版，后续每一步优化了什么？ | spill、layout、TILE、stride constexpr、async、SWA |
| Q17 | 阶段 1 的 2D 为什么能去掉 v_keep？和之前说“必须清零”矛盾吗？ | 2D cache 契约、3D 保留、NaN |
| Q18 | 上游 vLLM 是不是也删掉了 v_keep？ | vLLM 0.28.0、sw_mask_v、SWA |
| Q19 | block-table 尾块 clamp 是不是多余？不是有 tile_mask 吗？ | unmasked load、clamp、tile_mask |
| Q20 | 为什么 `BLOCK_M=8 → BLOCK_Q=1` 是 3 个 program？ | grid 上界、active program、num_seqs |
| Q21 | 什么是 `load/shared layout`？ | layout、sizePerThread、shared、convert_layout |

## 3. Q&A

### Q1. VSCode 为什么无法解析 `tl.int64`，提示 Variable not allowed in type expression？

**问题**

Triton kernel 参数写成：

```python
def kernel(..., stride_k_cache_3: tl.int64, ...):
```

VSCode/Pylance 报：

```text
Variable not allowed in type expression
```

**结论**

这是 Pylance/Pyright 对 Triton DSL 的误报，不是 kernel 错误。

- Python 类型注解要求类型表达式里是类/类型别名；
- `tl.int64` 是 Triton 的 dtype 值对象，不是 Python 类；
- `tl.constexpr` 也是 Triton JIT 的 DSL 标记，不是普通 typing 类型；
- Triton JIT 会读取这些注解做参数类型/特化，Pylance 按 PEP 484 检查，所以报 warning。

**处理**

保留 Triton 注解，只加行内忽略：

```python
block_table_stride: tl.int64,  # type: ignore
num_seqs: tl.int32,            # type: ignore
```

覆盖文件：

```text
minivllm/attention/ops/triton_unified_attention.py
minivllm/attention/ops/triton_reshape_and_cache_flash.py
```

提交：

```text
f084053 step88：忽略 Triton dtype 注解的 Pylance 误报
```

**不要做的事**

- 不要把 `tl.int64` 改成 `int`：Triton JIT 可能因此改变参数类型/生成代码；
- 不要加 `from __future__ import annotations`：注解变成字符串，Triton JIT 读取注解时可能出问题。

**如果仍然报警**

右键 warning 看 Diagnostic Code；若是 `reportInvalidTypeForm`，可改用：

```python
stride_k_cache_3: tl.int64,  # pyright: ignore[reportInvalidTypeForm]
```

或在 `.vscode/settings.json` 中关闭该诊断：

```json
{
  "python.analysis.diagnosticSeverityOverrides": {
    "reportInvalidTypeForm": "none"
  }
}
```

**边界**

这是编辑器诊断问题，不改变 kernel 行为；相关测试：

```text
tests/step88/test_attention_async_stride.py：14 passed
step81 package hash 测试：1 passed
```

### Q2. `workspace_pool` 的 key 有哪些？3D 的 5 个参数各自做什么？

**问题**

`TritonAttentionWorkspacePool` 按什么分桶？
`seq_threshold_3D`、`num_par_softmax_segments`、`softmax_segm_output`、
`softmax_segm_max`、`softmax_segm_expsum` 分别是什么？

**结论**

池的 key 是：

```python
key = (
    _device_key(device),          # (device.type, device.index)
    int(num_heads),               # query heads
    int(head_size),               # 原始 head_size（内部再 pad 到 16/2^n）
    int(seq_threshold_3D),        # 最多支持多少个 sequence
    int(num_par_softmax_segments),# 每条 sequence 切成几段 KV
)
```

`_device_key` 返回：

```python
(device.type, -1 if device.index is None else device.index)
```

例如：

```text
(('cuda', 0), 8, 256, 128, 16)
```

**为什么是这 5 个**

3D workspace 的实际形状是：

```text
softmax_segm_output:
  (seq_threshold_3D, num_heads, num_par_softmax_segments, head_padded)

softmax_segm_max / expsum:
  (seq_threshold_3D, num_heads, num_par_softmax_segments)
```

所以每个 key 都对应一个 shape 维度：

| key | 对应什么 |
|---|---|
| `device` | tensor 分配在哪个 GPU |
| `num_heads` | 第二维，每个 query head 独立保存分段结果 |
| `head_size` | 最后一维；`max(16, next_power_of_2(head_size))` |
| `seq_threshold_3D` | 第一维，最多容纳多少个 query token/sequence |
| `num_par_softmax_segments` | 第三维，每条 sequence 的 KV 分段数 |

不放进 key 的东西：

- `num_kv_heads`：影响 grid，不改变 workspace 形状；
- `dtype`：workspace 固定 FP32；
- `sliding_window`：只影响 tile 范围，不改变 workspace 形状；
- 实际 `num_seqs` / `seq_len`：workspace 按上界分配，跨 batch 复用；
- `block_size` / `TILE_SIZE`：只影响 tile 循环，不改变 workspace 的三个维度。

**一个容易误解的点**

`seq_threshold_3D` 不是“sequence length 阈值”，而是 **batch/sequence 数阈值**。

3D 只在：

```text
max_seqlen_q == 1
num_seqs > 0
num_seqs <= seq_threshold_3D
```

时启用。此时一条 sequence 只有一个 query token，所以 workspace 第一维用
query token 下标，也就等于 sequence 下标；这也是它名字里带 `seq` 的原因。

`TritonAttentionImpl` 当前设置：

```python
_MIN_LAUNCH_GRID_SIZE_2D = 128
_NUM_PAR_SOFTMAX_SEGMENTS = 16

seq_threshold_3D = max(1, 128 // num_kv_heads)
num_par_softmax_segments = 16
```

例如 `num_kv_heads=1` → threshold=128；`num_kv_heads=2` → 64。
batch 超过阈值时退回 2D，不创建/使用这个 workspace。

**3D 的 5 个参数**

1. `seq_threshold_3D`

```text
作用：3D 是否启用 + workspace 第一维大小
位置：Python 侧 can_use_3d / make_3d_workspace
不是：最大序列长度
```

它只读 Python int，不读 GPU 张量，所以 Graph capture 安全。

2. `num_par_softmax_segments`

```text
作用：把一条 sequence 的 KV tile 切成多少段并行
grid：第三维
workspace：第三维
```

kernel 里：

```python
tiles_per_segment = cdiv(seq_len, NUM_SEGMENTS_PER_SEQ * TILE_SIZE)
loop_lo = max(segm_idx * tiles_per_segment, tile_start)
loop_hi = min((segm_idx + 1) * tiles_per_segment, tile_end)
```

每个 `(query_block, kv_head, segment)` program 只扫自己那段 KV。

3. `softmax_segm_output`

```text
形状：(seq_threshold_3D, num_heads, segments, head_padded)
dtype：FP32
内容：每个 segment 的未归一化 output
      acc = sum_j exp(score_j - M_seg) * V_j
```

它**不是最终 output**，而是分段的部分和。

4. `softmax_segm_max`

```text
形状：(seq_threshold_3D, num_heads, segments)
内容：每个 segment 的局部最大值 M_seg
```

5. `softmax_segm_expsum`

```text
形状：(seq_threshold_3D, num_heads, segments)
内容：每个 segment 的局部 exp-sum L_seg
      L_seg = sum_j exp(score_j - M_seg)
```

**最后怎么合并**

`reduce_segments` 对每个 query token/head：

```text
overall_max = max_s M_s
L_s' = L_s * exp(M_s - overall_max)
acc' = sum_s acc_s * exp(M_s - overall_max)
out = acc' / sum_s L_s'
```

对应代码：

```python
segm_expsum = segm_expsum * tl.exp(segm_max - overall_max)
segm_out = segm_out * tl.exp(segm_max - overall_max)[:, None]
acc_sum = tl.sum(segm_out, axis=0)
acc = acc_sum / overall_expsum
```

**2D 时这些参数在哪**

2D 路径：

```text
is_3d = False
segm_* = None
NUM_SEGMENTS_PER_SEQ = 1
```

kernel 里 `IS_3D` 是 `tl.constexpr`，相关 store/reduce 分支会被裁掉；
所以 2D 不分配 workspace，也不执行 `reduce_segments`。

**代码位置**

```text
minivllm/attention/backends/triton_attn.py
  _device_key
  TritonAttentionWorkspacePool.get
  TritonAttentionImpl.__init__（threshold/segments）
  TritonAttentionImpl._run（pool.get + 3D 参数）

minivllm/attention/ops/triton_unified_attention.py
  make_3d_workspace
  can_use_3d / use_3d_path
  kernel_unified_attention 的 IS_3D / segm_*
  reduce_segments
```

**边界**

- 3D 只用于 `max_seqlen_q == 1`、`num_seqs <= seq_threshold_3D`；
- K=2 的 target 有 K+1 行 query，`max_seqlen_q > 1`，因此走 2D；
- workspace 是 FP32，与模型 dtype 无关；
- `seq_threshold_3D` 过小只会让更多 batch 回退 2D，不会算错；
- `num_par_softmax_segments` 过大会增加 workspace 和 reduce 开销，不保证更快。

### Q3. `window_size` 为什么是 tuple？第二个元素是干什么的？

**问题**

`unified_attention(..., window_size: tuple[int, int] = (-1, -1), ...)`
为什么不是单个整数？`window_size[1]` 在 kernel 里做什么？

**结论**

这是沿用上游/vLLM 的二元组约定：

```text
window_size = (left, right)
```

- `left`：向左能看到的 key 范围；
- `right`：向右能看到的 key 范围（双向/non-causal 才会用到）；
- 全注意力用 `(-1, -1)`；
- causal sliding window 用 `(w - 1, 0)`，表示可见 key 数 `w`。

**但当前 mini 只使用第一个元素。**

wrapper：

```python
if window_size[0] is not None and window_size[0] >= 0:
    sliding_window = int(window_size[0]) + 1
```

kernel 收到的 `SLIDING_WINDOW` 是这个 `sliding_window`，mask 条件：

```python
dist = query_abs_pos - seq_offset
seq_mask = seq_mask & (dist < SLIDING_WINDOW)
```

所以：

```text
window_size = (w - 1, 0)
→ SLIDING_WINDOW = w
→ visible keys = w
→ 实际左窗口 = w - 1
```

**第二个元素现在的作用**

```text
mini 当前不读取 window_size[1]。
```

backend 生产路径只传：

```python
window_size = (window - 1, 0)  # causal sliding
window_size = (-1, -1)         # full attention
```

对 causal 来说右窗口就是 0，所以第二元素不需要参与 kernel。

对 non-causal + sliding，kernel 里的 mask 是：

```python
seq_mask = seq_mask & (dist < SLIDING_WINDOW)
seq_mask = seq_mask & (seq_offset - query_abs_pos < SLIDING_WINDOW)
```

也就是**左右都用同一个 `SLIDING_WINDOW`**，而不是 `window_size[1]`。

因此：

```text
causal + (w-1,0)            ：正确，左窗口 w
non-causal + (l,r) 且 l==r  ：正确，左右对称
non-causal + 非对称 l!=r    ：当前实现不支持右窗口差异
```

**为什么还保留 tuple**

1. 对齐上游 `window_size` 的公共签名/调用约定；
2. causal 和 non-causal 共用同一个接口；
3. 以后要支持真正的双向/非对称窗口时，不需要改函数签名。

**代码位置**

```text
minivllm/attention/ops/triton_unified_attention.py
  unified_attention：只读 window_size[0]
  _kv_seq_mask：causal/非 causal + SLIDING_WINDOW 的 mask
  _tile_loop_bounds：sliding 的 tile 起止

minivllm/attention/backends/triton_attn.py
  _run：window_size = (window - 1, 0) 或 (-1, -1)
```

**边界**

- 当前生产路径 `window_size[1]` 恒为 `0` 或 `-1`，所以没有实际影响；
- 如果外部直接调用 `unified_attention` 并传入非零 `window_size[1]`，这个值不会生效；
- 要支持非对称右窗口，需要把 `window_size[1]` 传进 kernel，并同时改 mask 和 tile bounds，不能只改 Python wrapper。

### Q4. `tile_size` 这个 tile 是指什么 tile？

**问题**

`tile_size` 的 tile 是 query tile、cache block，还是别的？
为什么有的地方还有 `BLOCK_SIZE`、`BLOCK_M`、`BLOCK_Q`？

**结论**

`TILE_SIZE` 是 **KV 序列维度上的分块大小**：

```text
一次循环处理 TILE_SIZE 个连续的 key/value 位置
```

不是 query tile，也不是物理 cache block。

在 `f3bd003` 的 kernel 里：

```python
offs_t = tl.arange(0, TILE_SIZE)

for j in range(loop_lo, loop_hi):
    seq_offset = j * TILE_SIZE + offs_t
```

这里：

```text
j            第几个 KV tile
TILE_SIZE    每个 tile 多少个连续 KV 位置
seq_offset   这个 tile 覆盖的绝对 KV 位置
```

例如 `TILE_SIZE=32`：

```text
j=0: seq_offset = 0..31
j=1: seq_offset = 32..63
j=2: seq_offset = 64..95
```

然后：

```python
physical_block = block_tables[... + seq_offset // BLOCK_SIZE]
slot_in_block = seq_offset % BLOCK_SIZE
```

一个 `TILE_SIZE=32` 的 KV tile 会跨越多个物理 cache block（`BLOCK_SIZE=16` 时是 2 个）。

**三个容易混的维度**

| 名字 | 维度/对象 | 作用 |
|---|---|---|
| `TILE_SIZE` | KV 序列位置 | 一次循环扫多少 key/value |
| `BLOCK_SIZE` | 物理 KV cache 行 | 一个 cache block 有多少 slot；block table 的粒度 |
| `BLOCK_M` | query 行 | 一个 program 一次处理多少 query/GQA 行 |
| `BLOCK_Q` | query 位置 | 一个 program 覆盖多少个 query token |

所以：

```text
TILE_SIZE 切的是 KV 这一侧
BLOCK_M/BLOCK_Q 切的是 query 这一侧
BLOCK_SIZE 是 cache 的物理存储粒度
```

**tile 和 mask 的关系**

`tile_mask` 也是 KV 维：

```python
tile_mask = seq_offset < max_prefix
```

`S` 的形状是：

```text
[BLOCK_M, TILE_SIZE]
```

即每个 query 行对当前 KV tile 的 score。

**3D 里的 tile**

`tiles_per_segment` 的单位也是 KV tile，不是 cache block：

```python
tiles_per_segment = cdiv(seq_len, NUM_SEGMENTS_PER_SEQ * TILE_SIZE)
```

每个 segment 分到一段 `j` 范围。

**f3bd003 的取值**

```python
tile_size = 16 if is_3d else 32
```

- 2D：`TILE_SIZE=32`
- 3D：`TILE_SIZE=16`

这是当时的选择；后续 Astra 的 stride/async 改动又调整了 full/SWA/FP32 的 tile 策略，但 tile 的语义没变。

**代码位置**

```text
f3bd003:minivllm/attention/ops/triton_unified_attention.py
  offs_t = tl.arange(0, TILE_SIZE)
  for j in range(loop_lo, loop_hi)
  seq_offset = j * TILE_SIZE + offs_t

minivllm/attention/ops/triton_unified_attention.py
  _tile_loop_bounds
  kernel_unified_attention
```

**边界**

`TILE_SIZE` 不是越大越好：

- 大 tile：循环次数少，但 K/V tile 更大，shared/寄存器压力高；
- 小 tile：shared 小，但循环和 block-table/load 次数多；
- 3D 用更小 tile，是为了让分段并行 + workspace/shared 可控。

### Q5. `seqused_k` 是什么？

**问题**

`unified_attention` 的 `seqused_k` 是什么？和 `cu_seqlens_q`、`seq_lens` 有什么关系？

**结论**

`seqused_k` 是每个请求在**本轮之后**实际使用的 KV 长度：

```text
seqused_k[i] = 第 i 条请求本轮 attention 要看的 K/V token 数
             = 已计算的历史 token 数 + 本轮 query token 数
```

形状：

```text
[num_reqs]
```

dtype：

```text
int32
```

它在 kernel 里就是 `seq_lens_ptr` 指向的数组。

**在 kernel 里怎么用**

`_resolve_seq` 读：

```python
q_start = query_start_loc[seq_idx]
q_end = query_start_loc[seq_idx + 1]
q_len = q_end - q_start
seq_len = seqused_k[seq_idx]
```

然后：

```python
context_len = seq_len - q_len
```

含义：

```text
seq_len    本轮之后的总 KV 长度
q_len      本轮 query token 数
context    本轮之前的上下文长度
```

所以 KV 位置范围是：

```text
0 .. seq_len - 1
```

当前 query 对应的 key 也在里面；不是“只看历史 KV”。

**举例**

假设一条请求原来有 1024 个 token，本轮 query 有 3 个 token：

```text
q_len       = 3
seqused_k   = 1024 + 3 = 1027
context_len = 1027 - 3 = 1024
```

本轮要看的 K/V 是：

```text
position 0 .. 1026
```

其中 0..1023 是历史，1024..1026 是本轮 query 自己。

**和 `cu_seqlens_q` 的区别**

```text
cu_seqlens_q  [num_reqs + 1]  query 行的前缀和
seqused_k     [num_reqs]      每条请求本轮之后的 KV 长度
```

- `cu_seqlens_q` 决定“这一条请求的 query 行在扁平 q 张量里的起止”；
- `seqused_k` 决定“attention 要扫到多长的 K/V”。

它们不是同一个东西：

```text
q 长度：q_end - q_start
KV 长度：seqused_k[i]
```

**为什么叫 seqused_k**

名字来自上游接口，强调它是 **K/V 侧实际使用的序列长度**。
不要把它理解成“请求 ID”或“query 长度”。

**在 mini 里的来源**

V2 `prepare_pos_seq_lens` 的 kernel：

```python
num_computed_tokens = ...
query_len = q_end - q_start
seq_len = num_computed_tokens + query_len
seq_lens[req_id] = seq_len
```

也就是：

```text
seq_lens = num_computed_tokens + query_len
```

`num_computed_tokens` 是本轮之前已经写进 KV cache 的 token 数；
`query_len` 是本轮真正送进 attention 的 query token 数。

**投机解码时**

K=2 的 target：

```text
q_len = K + 1 = 3
seqused_k = 之前的 context + 3
context_len = seqused_k - 3
```

3D 单 token decode：

```text
q_len = 1
seqused_k = 之前的 context + 1
context_len = seqused_k - 1
```

**它影响什么**

```text
- tile 循环上界（_tile_loop_bounds）
- causal / sliding mask
- block table 的读取范围
- 3D 的 tiles_per_segment
```

block table 至少要覆盖：

```text
ceil(seqused_k[i] / BLOCK_SIZE)
```

个逻辑块；tail 由图中的 mask 挡住。

**常见错误**

如果调用方传的是“本轮之前的 context”，而不是“context + q_len”：

```text
seqused_k = context
→ context_len = context - q_len
→ 当前 query 的 key 被漏掉，甚至 context_len 为负
```

所以低层直接调用 `unified_attention` 时，`seqused_k` 必须是**本轮之后的总 KV 长度**，不是历史长度。

**代码位置**

```text
minivllm/attention/ops/triton_unified_attention.py
  unified_attention(seqused_k=...)
  _resolve_seq
  _tile_loop_bounds
  _kv_seq_mask

minivllm/worker/gpu/input_batch.py
  prepare_pos_seq_lens
  _prepare_pos_seq_lens_kernel

minivllm/attention/metadata.py
  seq_lens 字段注释
```

### Q6. `tl.full([], 0, tl.int32)` 是什么？

**问题**

`tl.full([], 0, tl.int32)` 里的空 list 是什么？它和 Python 的 `0` 有什么区别？

**结论**

```python
tl.full(shape, value, dtype)
```

创建一个指定 shape、用同一个值填满的 Triton tensor。

```python
tl.full([], 0, tl.int32)
```

是：

```text
shape = []      零维 / rank-0 tensor
value = 0
dtype = tl.int32
```

也就是一个 **0-D 的 int32 scalar tensor**，不是 Python `int`，也不是空数组。

对比：

```python
tl.full([], 0, tl.int32)      # 0-D，标量 tensor
tl.full([1], 0, tl.int32)     # 1-D，长度 1 的 tensor
tl.full([BLOCK_M], 0, tl.int32)  # 长度 BLOCK_M 的 vector
```

`[]` 不是“列表没有元素”，而是“没有维度”，即 shape `()`。

**在 `_find_seq_idx` 里的用途**

```python
left = tl.full([], 0, tl.int32)
right = tl.full([], num_seqs, tl.int32)

while left < right:
    mid = (left + right) // 2
    ...
    if val <= target_idx:
        left = mid + 1
    else:
        right = mid
return left - 1
```

这是对 `query_start_loc` 前缀和做二分查找，找 target 属于哪条请求。

这里为什么要用 `tl.full`：

1. `left` / `right` 必须是 **Triton tensor**，不能只是 Python int；
2. 循环里会执行：
   ```python
   left = mid + 1
   right = mid
   ```
   `mid` 是 tensor，`left/right` 也必须是同类型 tensor；
3. 显式 `tl.int32` 保证和 `num_seqs: tl.int32` 的类型一致；
4. `return left - 1` 返回的是 0-D int32 tensor，后续可以参与指针偏移/索引。

**为什么不直接写 `left = 0`**

Triton 对 while 循环的循环携带变量要求类型稳定。
Python 字面量 `0` 在 Triton 里可能被当成 constexpr/不同整数类型；显式 `tl.full([], 0, tl.int32)` 把“这是一个运行时 int32 标量 tensor”钉死。

**注意**

- 0-D tensor 不是 Python 标量，不能拿去做普通 Python 控制流；
- 它可以参与 Triton 算术、比较、`tl.load` 指针偏移；
- 如果写成 `tl.full([1], 0, tl.int32)`，就是长度为 1 的向量，后续 `left < right` 的形状和返回类型都会变。

**代码位置**

```text
minivllm/attention/ops/triton_unified_attention.py
  _find_seq_idx
```

### Q7. 为什么有 `offs_m < BLOCK_Q * num_queries_per_kv`？BLOCK_M 不是等于 BLOCK_Q * group 吗？

**问题**

```python
row_mask = (
    (offs_m < BLOCK_Q * num_queries_per_kv)
    & (query_pos < q_len)
    & (query_offset_1 < num_query_heads)
)
```

既然 `BLOCK_M = BLOCK_Q * num_queries_per_kv`，第一个条件不是恒真吗？

**结论**

**不是恒等。** `BLOCK_M` 是 Triton 的 2 的幂行容量，`BLOCK_Q` 是向下取整：

```python
BLOCK_Q = BLOCK_M // group
```

所以一般有：

```text
BLOCK_Q * group <= BLOCK_M
```

只有 `group` 整除 `BLOCK_M` 时才相等。多出来的行是 **padding 行**，必须用
`offs_m < BLOCK_Q * group` 挡掉。

**举例**

当前目标 `group=8`：

```text
BLOCK_M = 8
BLOCK_Q = 8 // 8 = 1
BLOCK_Q * group = 8
→ 相等，mask 恒真
```

但不是所有 GQA group 都这样：

```text
group=6,  BLOCK_M=8,  BLOCK_Q=1  → 6 < 8，多 2 行
group=12, BLOCK_M=16, BLOCK_Q=1  → 12 < 16，多 4 行
group=24, BLOCK_M=32, BLOCK_Q=1  → 24 < 32，多 8 行
```

f3bd003 里 `group<=16` 用 `BLOCK_M=16`：

```text
group=6, BLOCK_M=16, BLOCK_Q=2 → 12 < 16，多 4 行
```

**多出来的行会映射成什么**

行号 `offs_m` 到 `(query_pos, query_head)` 的映射是：

```python
query_pos     = q_block_local * BLOCK_Q + offs_m // group
query_offset_1 = kv_head_idx * group + offs_m % group
```

如果 `offs_m >= BLOCK_Q * group`，这些行已经超出本 program 负责的 query
位置范围，会映射到下一个 query 位置或重复的组内 head。

它们不是合法的 `(query_pos, head)` 对，所以必须 mask。

**为什么还会被 `query_pos < q_len` 挡住？**

有些 padding 行映射出的 `query_pos` 可能仍然 `< q_len`，
因为 `q_len` 是整条请求的 query 长度，不是当前 `BLOCK_Q` 块的长度。

例如当前 block 只负责 query 位置 0，padding 行可能映射到位置 1；
如果 `q_len >= 2`，`query_pos < q_len` 仍然是 true。
所以仅靠 `query_pos < q_len` 不够，必须保留：

```python
offs_m < BLOCK_Q * group
```

**这个 mask 用在哪**

`row_mask` 会用于：

- Q load 的 mask；
- S 的 keep mask；
- 最终 output store 的 mask。

所以它保证 padding 行不会：

- 读到别的 query；
- 污染 softmax；
- 写到别的 query 位置。

**为什么 BLOCK_M 不直接设成 group * BLOCK_Q**

Triton 的 block shape 通常要求 2 的幂；GQA group 不一定是 2 的幂。
因此只能：

```text
BLOCK_M = 2 的幂
BLOCK_Q = BLOCK_M // group
padding = BLOCK_M - BLOCK_Q * group
```

用 mask 处理 padding，而不是让 shape 变成非 2 的幂。

**目标场景**

Gemma4 target 是：

```text
num_query_heads = 8
num_kv_heads   = 1
group          = 8
```

当前 2D full attention 用 `BLOCK_M=8, BLOCK_Q=1`，所以：

```text
8 == 1 * 8
```

这个 case 的 `offs_m < BLOCK_Q * group` 确实恒真，但 kernel 保留它是为了
支持非 2 的幂 group；不能因为 target case 恒真就删掉。

**代码位置**

```text
minivllm/attention/ops/triton_unified_attention.py
  unified_attention：BLOCK_M/BLOCK_Q 选择
  kernel_unified_attention：offs_m、query_pos、row_mask
```

### Q8. `max_prefix` 是什么？

**问题**

`_tile_loop_bounds` 里的 `max_prefix` 是什么？为什么不是直接用 `seq_len`？

**结论**

`max_prefix` 是：

> 当前 query block 在 causal 情况下**可能看到的最大 KV 前缀长度（开区间上界）**。

它是 **KV 绝对位置的上界**，不是 query 长度，也不总是 `seq_len`。

代码：

```python
max_prefix = context_len + q_block_local * BLOCK_Q + (
    (BLOCK_M - 1) // GROUP) + 1

if USE_CAUSAL:
    max_prefix = tl.minimum(max_prefix, seq_len)
else:
    max_prefix = seq_len
```

然后：

```python
num_tiles = _cdiv(max_prefix, TILE_SIZE)
tile_mask = seq_offset < max_prefix
```

**causal 时怎么算**

```text
context_len
+ 当前 query block 的第一个 query 位置（q_block_local * BLOCK_Q）
+ 按 BLOCK_M 行容量推得的最大 query 偏移（(BLOCK_M - 1) // GROUP）
+ 1（开区间）
```

即：

```text
max_prefix = context + 行容量允许的最大 query 偏移 + 1
```

如果 `BLOCK_M` 正好等于 `BLOCK_Q * group`，这个偏移就是本 block 最后一个
query 位置；如果 group 不整除 `BLOCK_M`，padding 行会让它略微**高估**。
高估只会多扫一点，mask 仍会挡住，所以它是安全上界，不是精确值。

含义：

```text
KV 位置 0 .. max_prefix - 1 才可能被这个 block 用到
```

例如：

```text
context_len = 1024
q_len = 3
BLOCK_Q = 1
GROUP = 8
BLOCK_M = 8

query block 0：query 绝对位置 1024
max_prefix = 1024 + 0 + 0 + 1 = 1025
可见 KV：0..1024

query block 1：query 绝对位置 1025
max_prefix = 1024 + 1 + 0 + 1 = 1026
可见 KV：0..1025

query block 2：query 绝对位置 1026
max_prefix = 1024 + 2 + 0 + 1 = 1027
可见 KV：0..1026
```

最后 `min(..., seq_len)` 防止超过实际序列长度。

**为什么不用 seq_len**

如果每个 query block 都扫到 `seq_len`：

- causal decode 的早期 query 会读未来 key；
- 虽然 mask 能挡住，但会多扫 tile、多做 load；
- tile 循环上界变大，浪费。

所以 causal 用“这个 query block 最多能看到哪里”来收紧循环上界。

**non-causal 时**

```python
max_prefix = seq_len
```

因为每个 query 都可能看到完整 KV 范围。

**和 sliding window 的关系**

`max_prefix` 只是 causal/non-causal 的**外层上界**。
sliding window 还会进一步收窄 tile 范围：

```python
tile_start = first_key // TILE_SIZE
tile_end = min(last_key // TILE_SIZE + 1, num_tiles)
```

真正的 mask 还会在 `_kv_seq_mask` 里按 `SLIDING_WINDOW` 再挡一次。

所以：

```text
max_prefix：这个 query block 的 KV 扫描上界
tile_start/tile_end：sliding window 进一步收窄后的 tile 范围
seq_mask：每个 query 行对每个 key 的精确可见性
```

**tile_mask 的作用**

```python
tile_mask = seq_offset < max_prefix
```

它挡住：

- 最后一个 tile 中超过 `max_prefix` 的 KV 位置；
- causal 下当前 query block 不需要的未来位置；
- 避免读越界的 block table / tail。

`tile_mask` 也用于 K/V load 的 mask。

**它不是**

```text
不是 q_len
不是 seq_len（causal 时通常更小）
不是 block table 长度
不是 sliding window 大小
```

**代码位置**

```text
minivllm/attention/ops/triton_unified_attention.py
  _tile_loop_bounds：max_prefix 计算
  kernel_unified_attention：tile_mask = seq_offset < max_prefix
  _kv_seq_mask：逐 query/key 的精确 mask
```

### Q9. 为什么 `max_prefix` 不写成 `context_len + (q_block_local+1)*BLOCK_Q`，再和 `seq_len` 取 min？

**问题**

当前写法：

```python
max_prefix = context_len + q_block_local * BLOCK_Q + (
    (BLOCK_M - 1) // GROUP) + 1
```

如果 query block 覆盖 `BLOCK_Q` 个位置，为什么不直接写：

```python
max_prefix = context_len + (q_block_local + 1) * BLOCK_Q
max_prefix = min(max_prefix, seq_len)
```

**结论**

在目标场景下，两者**数学等价**。

因为当前 wrapper 保证：

```python
BLOCK_Q = BLOCK_M // GROUP
```

当 `BLOCK_M` 正好等于 `BLOCK_Q * GROUP` 时：

```text
(BLOCK_M - 1) // GROUP
= (BLOCK_Q * GROUP - 1) // GROUP
= BLOCK_Q - 1
```

代回：

```text
context_len + q_block_local * BLOCK_Q + (BLOCK_Q - 1) + 1
= context_len + (q_block_local + 1) * BLOCK_Q
```

所以：

```python
context_len + (q_block_local + 1) * BLOCK_Q
```

就是当前公式在“没有 padding 行”时的等价写法。

**为什么代码写成 `(BLOCK_M - 1) // GROUP`**

它是从行号 `offs_m` 直接推出来的：

```python
query_pos = q_block_local * BLOCK_Q + offs_m // GROUP
```

最大行 `offs_m = BLOCK_M - 1`，所以最大 query 偏移是：

```text
(BLOCK_M - 1) // GROUP
```

这个写法不依赖“BLOCK_M 必须等于 BLOCK_Q*GROUP”的假设。

**有 padding 行时**

如果 `BLOCK_M > BLOCK_Q * GROUP`：

```text
(BLOCK_M - 1) // GROUP 可能 > BLOCK_Q - 1
```

例如：

```text
GROUP = 6
BLOCK_M = 8
BLOCK_Q = 1

(BLOCK_M - 1) // GROUP = 7 // 6 = 1
```

而当前 block 实际只覆盖 `BLOCK_Q=1` 个 query 位置（偏移 0）。

所以当前 `max_prefix` 会**高估 1 个位置**：

```text
当前写法：context + q_block_local + 2
你的写法：context + q_block_local + 1
```

但这是**安全上界**：

- 多扫的 key 会被 `_kv_seq_mask` 挡掉；
- 不会越界（最后还有 `min(..., seq_len)`）；
- 不会改变输出，只是可能多一个 tile 的循环/load。

**你的写法更紧**

```python
max_prefix = context_len + (q_block_local + 1) * BLOCK_Q
max_prefix = tl.minimum(max_prefix, seq_len)
```

它在所有 GQA group 下都精确对应当前 block 的 query 位置范围，
比 `(BLOCK_M - 1) // GROUP` 更紧，尤其是 group 不整除 `BLOCK_M` 时。

**那为什么现在不改成你的写法**

没有正确性上的阻碍；在目标 case 下两者完全一样，
在非 2 的幂 group 下你的写法只是更精确，不会更不安全。

如果改，需要注意：

1. `BLOCK_Q` 必须是当前 program 实际覆盖的 query 位置数；
2. 最后仍要 `min(..., seq_len)`；
3. 要用测试覆盖 `group` 不整除 `BLOCK_M` 的情况，确认 mask 不变。

当前 kernel 选择保守写法，不是因为它“必须这样”，而是因为它直接从
`offs_m` 的行容量推上界，避免依赖 `BLOCK_M == BLOCK_Q*GROUP` 这个关系。

**代码位置**

```text
minivllm/attention/ops/triton_unified_attention.py
  _tile_loop_bounds
  kernel_unified_attention：tile_mask
```

### Q10. f3bd003 里 `k_ptrs` load 出 `[H, TILE_SIZE]`，是转置后的，有意为之吗？

**问题**

f3bd003 里：

```python
k_ptrs = (
    key_cache_ptr
    + physical_block[None, :] * stride_k_cache_0
    + slot_in_block[None, :] * stride_k_cache_1
    + kv_head_idx * stride_k_cache_2
    + offs_d[:, None] * stride_k_cache_3
)
```

`physical_block` / `slot_in_block` 是 tile 轴，`offs_d` 是 head 轴，
所以 `tl.load(k_ptrs)` 的形状是 `[H, TILE_SIZE]`。
这是有意为之吗？

**结论**

是故意的。

这不是“先加载 `[T,H]` 再额外 transpose”，而是**直接按 `tl.dot` 需要的操作数布局去 gather**：

```text
Q: [BLOCK_M, H]
K: [H, TILE_SIZE]
S = Q @ K
→ [BLOCK_M, TILE_SIZE]
```

`K` 的 `[H, T]` 是逻辑布局；`tl.load` 只是按地址把元素装进这个逻辑张量，
没有显式的 `tl.trans`。

**K cache 的实际内存布局**

```text
K cache: [num_blocks, block_size, num_kv_heads, head_size]
```

固定一个物理块、一个 KV head 后，一个 slot 的 head 向量是连续的：

```text
slot 0: h0 h1 h2 ... h255
slot 1: h0 h1 h2 ... h255
```

`k_ptrs` 的两个轴：

```text
offs_d[:, None]          → head 轴，stride = 1
slot_in_block[None, :]   → tile 轴，stride = 256
```

所以 load 出来的逻辑形状是：

```text
[H, T]
```

- 沿 H 方向地址连续；
- 沿 T 方向跨 slot 跳 256 个元素。

**为什么 K 要这样，V 却相反**

因为两个 dot 的操作数方向不同：

```text
S = Q @ K
Q [M,H] @ K [H,T] -> S [M,T]

O = P @ V
P [M,T] @ V [T,H] -> O [M,H]
```

所以 f3bd003 里：

```text
K: [H, T]   ← 直接作为左乘的右操作数
V: [T, H]   ← 直接作为右乘的右操作数
```

这是 attention 里很常见的非对称布局：

```text
K 沿 head 连续，按 [H,T] 读
V 按 [T,H] 读
```

**为什么不说它是“转置后的 K”**

物理上 K 的一个 block 确实是：

```text
[block_size, head_size] = [T,H]
```

但 `k_ptrs` 用 `offs_d[:,None]` 和 `slot[None,:]` 把它**直接索引成 `[H,T]`**。

这不是：

```python
K = tl.load(... [T,H])
K = tl.trans(K)
```

而是：

```python
K = tl.load(... [H,T])
```

所以它是“逻辑上的转置视图”，不是一次额外的 transpose 操作。

**如果写成 `[T,H]` 会怎样**

也可以：

```python
K = tl.load(k_ptrs_T_H)   # [T,H]
S = tl.dot(Q, tl.trans(K))
```

但 `tl.trans(K)` 会引入 layout conversion / shared 读写，未必免费。
f3bd003 选择 `[H,T]`，是为了让 K 直接成为 dot 的合法操作数，避免显式 transpose。

**后续代码为什么又变成 `[T,H] + tl.trans`**

这是另一个阶段的性能取舍：

- `[H,T]` K + `[T,H]` V 会让 K/V 的 load layout 分叉；
- 后续为了统一 K/V load layout、减少 shared/layout conversion，改成 K 也按 `[T,H]` load，再 `tl.trans`；
- 再后面 Astra 通过 `stride constexpr` 恢复 K/V coalesced/async。

所以：

```text
f3bd003 的 K [H,T]：直接服务 dot，避免显式 transpose
后续 K [T,H]+trans：服务统一的 load layout / shared 优化
```

两者都不是 bug，是不同的布局取舍。

**代码位置**

```text
f3bd003:minivllm/attention/ops/triton_unified_attention.py
  k_ptrs
  V = tl.load(v_ptrs)
  S = tl.dot(Q, K)
  acc += tl.dot(P.to(V.dtype), V)
```

### Q11. `input_precision="ieee"` 是什么？

**问题**

```python
S = tl.dot(Q, K, input_precision="ieee")
```

这个参数控制什么？ieee 是“用 FP64 算”吗？

**结论**

不是 FP64。

`input_precision` 控制 **f32 × f32 的 dot 在 Tensor Core 上如何乘**：

```text
"tf32"    默认；f32 输入截断成 TF32（19-bit）再算
"tf32x3"  用 3 次 TF32 近似 f32
"ieee"    不做 TF32 截断，按 IEEE FP32 乘
```

它只对 **f32 输入**有意义；对 BF16/FP16 输入会被忽略。

**TF32 是什么**

TF32 不是一种存储 dtype，而是 Tensor Core 的一种输入格式：

```text
FP32: 1 sign + 8 exponent + 23 mantissa
TF32: 1 sign + 8 exponent + 10 mantissa
```

所以 TF32 保留了 FP32 的指数范围，但尾数只有 10 位。
f32 输入进 Tensor Core 时会被截断到 TF32，尾数精度下降。

默认：

```python
tl.dot(Q_f32, K_f32)  # input_precision 默认 "tf32"
```

结果可能和真正的 FP32 乘法有差异。

**`"ieee"` 做什么**

```python
tl.dot(Q_f32, K_f32, input_precision="ieee")
```

不做 TF32 截断，按 IEEE FP32 乘法语义算。

注意：

```text
accumulator 仍然是 FP32
不是 FP64
不是无限精度
只是 f32 输入不被截成 TF32
```

**三个选项的取舍**

| 选项 | 输入精度 | 速度 | 数值 |
|---|---|---|---|
| `"tf32"` | 截断到 10-bit mantissa | 最快 | 可能偏离 FP32 |
| `"tf32x3"` | 3 次 TF32 近似 FP32 | 中等 | 比 tf32 准，接近 f32 |
| `"ieee"` | 完整 FP32 乘 | 通常最慢 | 最接近 IEEE FP32 |

`"ieee"` 往往不能用 TF32 Tensor Core 的那条快路径，所以 FP32 路径会慢一些。

**mini 里为什么用**

代码：

```python
if Q_IS_FP32:
    S = tl.dot(Q, K, input_precision="ieee") * scale
else:
    S = tl.dot(Q, K) * scale
```

原因：

```text
FP32 路径要和 Torch/参考实现的 IEEE FP32 对齐；
如果默认 tf32，测试会出现 1e-3 级别的差异。
```

第二个 dot 也一样：

```python
if Q_IS_FP32:
    acc += tl.dot(P.to(V.dtype), V, input_precision="ieee")
else:
    acc += tl.dot(P.to(V.dtype), V)
```

因为 FP32 时 `P` 和 `V` 都是 f32，`P @ V` 也必须是 IEEE。

**BF16/FP16 时**

```python
S = tl.dot(Q, K)
```

没有传 `input_precision`，因为：

```text
输入已经是 bf16/fp16，
Tensor Core 直接按 bf16/fp16 乘，
input_precision 对非 f32 输入不起作用。
```

**常见误解**

```text
ieee ≠ FP64
ieee ≠ 更高精度累加
ieee = f32 输入不被截成 TF32
```

**代码位置**

```text
minivllm/attention/ops/triton_unified_attention.py
  Q_IS_FP32 分支
  S = tl.dot(...)
  acc += tl.dot(...)
```

### Q12. `v_keep = tl.max(tl.where(keep, 1, 0), axis=0) > 0` 在做什么？

**问题**

```python
v_keep = tl.max(tl.where(keep, 1, 0), axis=0) > 0
V = tl.where(v_keep[:, None], V, 0.0)
```

这两行在干什么？为什么要按 `axis=0` 取 max？

**结论**

它在做：

> 对当前 KV tile 的每一列（每个 key），判断“是否存在至少一个 query 行能看到它”。
> 如果整列所有 query 都看不到，就把这一列的 V 清零。

目的是避免：

```text
P = 0 时，0 * NaN 仍然是 NaN
```

**形状**

```text
keep:        [BLOCK_M, TILE_SIZE]
axis=0:      query 行维
结果 v_keep: [TILE_SIZE]
```

对每个 key 列：

```text
v_keep[t] = OR over m: keep[m, t]
```

`tl.max(..., axis=0) > 0` 就是“这一列是否存在 1”。

**手算例子**

```text
keep =
  row0: [1, 1, 0, 0]
  row1: [0, 1, 0, 0]
  row2: [0, 0, 0, 0]

max(axis=0) =
  [1, 1, 0, 0]

v_keep =
  [True, True, False, False]
```

然后：

```python
V = tl.where(v_keep[:, None], V, 0.0)
```

把第 3、4 列的 V 清零；第 1、2 列保留。

**为什么需要清零 V**

`S` 被 mask 后：

```python
S = tl.where(keep, S, -inf)
P = exp(S - M)
```

被 mask 的位置：

```text
P = 0
```

但 V 可能存在未写入/未初始化的槽位，值为 NaN。
后面的 dot 是：

```text
P @ V
```

即使 `P=0`：

```text
0 * NaN = NaN
```

NaN 会污染整个 output。
所以必须在 dot 之前把“所有 query 都看不到的 V 列”清零。

**为什么不直接 `tl.max(keep, axis=0)`**

`keep` 是 bool tensor；`tl.max` 对 bool 的支持/返回类型不如整数直接。
先转成 0/1 再取 max 更稳：

```python
tl.where(keep, 1, 0)  # bool -> int
tl.max(..., axis=0)   # 每列是否存在 1
> 0                   # 转回 bool
```

**为什么 axis=0**

`keep` 的第一维是 `BLOCK_M`（query 行），第二维是 `TILE_SIZE`（key 列）。

“这个 key 是否对任何 query 可见”是**沿 query 行归约**，所以是 `axis=0`。

如果按 `axis=1`，就变成“这个 query 是否有任何可见 key”，语义完全不同。

**为什么不是 sum**

只需要“是否存在”，max 就够：

```text
max = 1 → 至少一个 query 看到
max = 0 → 全部 query 都看不到
```

sum 也能判断 `>0`，但 max 更直接，也不会关心有多少行看到。

**当前代码里还在不在**

f3bd003 里这段是**所有路径都执行**。

后续 2D 优化中，这段被移除了：

- 2D 的 K/V load 已用 `tile_mask` 把 tail 置 0；
- causal/window 挡掉的 key 都是已写入的有限值；
- 所以 2D 不再需要逐 tile 做 `v_keep` 归约。

当前代码只在 `IS_3D` 分支保留：

```python
if IS_3D:
    v_keep = tl.max(tl.where(keep, 1, 0), axis=0) > 0
    V = tl.where(v_keep[:, None], V, 0.0)
```

**代码位置**

```text
f3bd003:minivllm/attention/ops/triton_unified_attention.py
  keep / v_keep / V 清零

minivllm/attention/ops/triton_unified_attention.py
  IS_3D 分支的 v_keep
```

### Q13. V load 已经 `other=0.0`，为什么还会出现 NaN？

**问题**

f3bd003 里：

```python
V = tl.load(
    v_ptrs,
    mask=tile_mask[:, None] & dim_mask[None, :],
    other=0.0,
)
```

既然 `other=0.0`，V 为什么还可能有 NaN？

**结论**

`other=0.0` 只对 **mask=False 的 lane** 生效：

```text
mask=False → 返回 other（0.0）
mask=True  → 返回内存里的真实值
```

它**不会**检查真实值是不是 NaN，也不会把 NaN 改成 0。

所以：

```text
mask=True + 未初始化/被污染的 cache → NaN
other=0.0 完全管不到
```

**一个最小例子**

```text
mask   = [True, True, False]
memory = [NaN, 1.0, NaN]

load = [NaN, 1.0, 0.0]
```

- 第一个 lane：mask=True → 返回 memory[0]=NaN；
- 第二个 lane：mask=True → 返回 1.0；
- 第三个 lane：mask=False → 返回 other=0.0。

**f3bd003 的 V mask 不包含 seq_mask**

V load 的 mask 是：

```python
tile_mask[:, None] & dim_mask[None, :]
```

它没有包含：

```text
seq_mask   causal / sliding 的逐 query-key 可见性
row_mask   query 行是否有效
```

而 `keep` 是：

```python
keep = row_mask[:, None] & tile_mask[None, :] & seq_mask
```

所以存在这种情况：

```text
tile_mask[t] = True       → V[t] 会被 load
seq_mask[m,t] = False     → 所有 query 都看不到这个 key
```

如果这个 key 的 cache 槽位没有写过，它的 bit 可能是 NaN；
V load 仍然返回 NaN，`other=0` 不会生效。

**为什么 v_keep 能解决**

```python
v_keep = tl.max(tl.where(keep, 1, 0), axis=0) > 0
V = tl.where(v_keep[:, None], V, 0.0)
```

- `keep` 是逐 query-key 的语义可见性；
- `v_keep` 把它归约成“这一列 key 是否对任何 query 可见”；
- 对完全不可见的列，强制把 V 清零。

这样即使 load 回来的是 NaN，dot 前也会变成 0。

**为什么不直接把 seq_mask 加进 V load mask**

因为 `seq_mask` 的形状是：

```text
[BLOCK_M, TILE_SIZE]
```

而 V 的形状是：

```text
[TILE_SIZE, HEAD]
```

V 的一个 lane 对应一个 key，不对应某个 query 行。
要直接用 seq_mask，必须先沿 query 行做 OR/any 归约，得到 `[TILE_SIZE]`；
这正是 `v_keep` 做的事，只是它放在 load 之后。

**这是不是意味着 cache 里一定有 NaN**

不一定。

在正常、完整写入的 cache 上：

```text
seq_mask 挡掉的 key 通常也是已写入的有限值
```

`v_keep` 是一种防御：

- paged cache 的未写槽位；
- tail / 预留槽位；
- 3D 分段/Graph 路径；
- 未来如果 mask 与写入范围不完全一致。

它保证“即使读到 NaN，也不会污染 dot”。

**当前代码为什么 2D 不需要了**

后续 2D 优化里：

- K/V load 用 `tile_mask` 把 tail 置 0；
- causal/window 挡掉的 key 都是已写入的有限值；
- 因此 2D 不再做 `v_keep`，减少一次 `[BLOCK_M,TILE] → [TILE]` 的归约。

3D 仍保留，因为分段/边界路径更复杂。

**代码位置**

```text
f3bd003:minivllm/attention/ops/triton_unified_attention.py
  V = tl.load(..., mask=tile_mask ... , other=0.0)
  keep = row_mask & tile_mask & seq_mask
  v_keep / V 清零

minivllm/attention/ops/triton_unified_attention.py
  IS_3D 分支的 v_keep
```

### Q14. K 和 V 的 mask 一样，为什么只对 V 做 v_keep？

**问题**

f3bd003 里 K/V load 的 mask 本质一样：

```python
K: dim_mask[:, None] & tile_mask[None, :]
V: tile_mask[:, None] & dim_mask[None, :]
```

既然 K 没有额外清零，为什么 V 要做：

```python
v_keep = ...
V = tl.where(v_keep[:, None], V, 0.0)
```

**结论**

因为 K 和 V 在数据流里的位置不同：

```text
K → S = Q @ K → S = tl.where(keep, S, -inf) → softmax
V → acc += P @ V
```

K 的无效值会被**后置的 S mask** 清掉；
V 的无效值会直接参与 dot，之后没有“逐 key 的 where”可以清掉它。

**K 为什么安全**

```python
S = tl.dot(Q, K) * scale
S = tl.where(keep, S, float("-inf"))
```

如果某个 key 对某个 query 不可见：

```text
keep[m,t] = False
```

那么无论 `S[m,t]` 是正常值还是 NaN，都会被替换成：

```text
-inf
```

`tl.where` 是选择，不是算术：

```text
where(False, NaN, -inf) = -inf
```

NaN 不会被保留。

所以 K 不需要在 load 后逐列清零；它的无效项在 S 阶段被 mask 掉。

**V 为什么不行**

V 在 dot 里是右操作数：

```text
acc = P @ V
```

P 是 `[BLOCK_M, TILE]`，V 是 `[TILE, HEAD]`。
dot 会对 key 维求和：

```text
acc[m,h] = Σ_t P[m,t] * V[t,h]
```

对于所有 query 都不可见的 key `t`：

```text
P[m,t] = 0
```

但 dot 里仍然是：

```text
0 * NaN = NaN
```

这个 NaN 会进入求和，污染整个 `acc[m,h]`。

而且 dot 之后没有“按 key 的 where”可以把它拿掉，因为 key 维已经被求和掉了。

**所以必须清零 V**

```python
v_keep = tl.max(tl.where(keep, 1, 0), axis=0) > 0
V = tl.where(v_keep[:, None], V, 0.0)
```

`v_keep[t]=False` 表示：

```text
所有 query 对 key t 都不可见
```

这些列在数学上权重为 0，物理上可以直接置 0，避免 NaN 参与 dot。

**为什么不是把 P 清零就够了**

因为：

```text
0 * NaN = NaN
```

把 `P` 的对应项设成 0 并不能消掉 V 的 NaN；必须把 V 也变成有限值。

**为什么 K 不做同样的列清零**

可以做，但不是必需：

- K 的无效项会经过 `S` 的 where mask；
- K 不会在“已经乘完并求和”之后才暴露 NaN；
- 对 K 做列清零只会多一次归约/where，不解决新问题。

**如果 key 是可见的但 V 是 NaN**

`v_keep` 也救不了：

```text
v_keep[t]=True
→ V[t] 保留
→ 仍然是 NaN
```

这说明“可见 key 的 K/V 必须已经正确写入”是更底层的 cache 契约。
`v_keep` 只处理“整列都不可见”的防御场景。

**一句话**

```text
K 的 NaN 在 S 阶段被 where 后置清掉；
V 的 NaN 在 P@V 里已经被乘进求和，必须前置清掉。
```

**代码位置**

```text
f3bd003:minivllm/attention/ops/triton_unified_attention.py
  S = tl.where(keep, S, -inf)
  v_keep / V 清零
  acc += tl.dot(P, V)
```

### Q15. “所有 query 都看不到的 V 列”具体长什么样？

**问题**

“所有 query 都看不到的 V 列”很抽象，能不能画出来？

**先纠正一个说法**

严格说，`V` 是：

```text
V [TILE_SIZE, HEAD_SIZE]
```

`v_keep[t]` 清掉的是 **V 的第 t 行**（一个 key 的完整 value 向量）。

但 `keep` 是：

```text
keep [BLOCK_M, TILE_SIZE]
```

`t` 在 `keep` 里是**列**，在 `V` 里是**行**。

所以“V 列”这个说法不准确；准确说是：

> `keep` 的 key 列，对应 `V` 的 key 行。

**具体例子：sliding window 的 tile 对齐**

设：

```text
context_len = 574
q_len = 2
query 绝对位置 = 574, 575
window = 512
BLOCK_M = 8
BLOCK_Q = 2
GROUP = 8
TILE_SIZE = 32
```

每个 query 能看到的最早 key：

```text
query 574: first_key = 574 - 512 + 1 = 63
query 575: first_key = 575 - 512 + 1 = 64
```

`tile_start` 按 tile 对齐：

```text
63 // 32 = 1
```

所以第一个要扫的 tile 是：

```text
j = 1
keys = 32 .. 63
```

**这个 tile 的 keep 表**

`BLOCK_M=8` 行对应两个 query 位置、每个位置 8 个 GQA head。
为了方便，先只看两个 query 位置（每个位置只画一个 head）：

```text
key:        32  33  34 ... 62  63
query 574:  0   0   0  ... 0   1
query 575:  0   0   0  ... 0   0
```

为什么：

- query 574 的第一个可见 key 是 63，所以 32..62 全不可见，63 可见；
- query 575 的第一个可见 key 是 64，所以 32..63 全不可见。

**v_keep**

```text
v_keep[t] = OR over query rows
```

```text
key:        32  33  34 ... 62  63
v_keep:     0   0   0  ... 0   1
```

所以：

```text
key 32..62：所有 query 都看不到 → v_keep=False
key 63：query 574 能看到 → v_keep=True
```

**V 清零**

`V` 的每一行是一个 key 的 value：

```text
V:
  key32: [v0 v1 ... vH]  → 清零
  key33: [v0 v1 ... vH]  → 清零
  ...
  key62: [v0 v1 ... vH]  → 清零
  key63: [v0 v1 ... vH]  → 保留
```

代码：

```python
V = tl.where(v_keep[:, None], V, 0.0)
```

`v_keep[:, None]` 形状 `[T,1]`，广播到 `[T,H]`，按 key 行清零。

**为什么这些 key 会被 load**

V 的 load mask 是：

```python
tile_mask[:, None] & dim_mask[None, :]
```

对 tile `keys=32..63`：

```text
tile_mask = True   # 都在 max_prefix 之前
```

所以 V 会去读这些 key。
但 `seq_mask` 不参与 V load mask；key 32..62 虽然被 load，
对当前所有 query 行都不可见。

如果这些 cache 槽位是未初始化/NaN，load 回来就是 NaN；
`v_keep` 在 dot 前把它们清零。

**如果 key 是可见的**

比如 key 63：

```text
v_keep[63] = True
V[63] 保留
```

如果 V[63] 本身是 NaN，`v_keep` 救不了；
这说明“可见 key 必须正确写入”是更底层的契约。

**为什么 P 不能解决**

对 key32..62：

```text
P[:,32:62] = 0
```

但 dot：

```text
acc = Σ_t P[:,t] * V[t,:]
```

如果 `V[t,:]` 是 NaN：

```text
0 * NaN = NaN
```

所以必须在乘法前把 `V[t,:]` 变成 0。

**一句话图**

```text
keep 的 key 列：
  [key32 ... key62] [key63]
      全 0            有 1
        │              │
        ▼              ▼
V 的 key 行：
  [key32 ... key62] [key63]
      清零            保留
```

已追加到：

```text
docs/step88_triton_unified_attention_qa.md 的 Q15
```

没有 commit。

### Q16. 从 basic 版到当前版，后续每一步优化了什么？

**先建立心智模型**

basic kernel 的主循环：

```text
一个 program 负责一个 query block + 一个 KV head
for 每个 KV tile:
    1. 算 seq_offset / block table
    2. load K/V
    3. S = Q @ K
    4. mask + online softmax
    5. acc += P @ V
最后 out = acc / L
```

优化不是“把所有 knob 调一遍”，而是依次回答：

```text
1. 有没有 spill？
2. 有没有不必要的 layout conversion/shared？
3. tile 和 program 并行度是否合适？
4. global load 有没有被 compiler 做成 async？
5. 改 tile 后数值还一样吗？
```

**总时间线**

| 阶段 | 主要问题 | 改动 | q3/ctx1024 |
|---|---|---|---:|
| f3bd basic | 无优化 | TILE32，K `[H,T]`，V `[T,H]`，v_keep | 0.1903 ms |
| 1. 去 spill | 255 regs / 58 spills | 2D 去 v_keep、尾块 clamp、warps 调整 | 0.1522 ms |
| 2. layout | K/V 两套 shared layout | K 也 `[T,H]` load，dot 前 trans | 0.068 ms |
| 3. TILE64 | scalar load 次数多 | 64 load + 2×32 精确 subtile，BM8 | 0.0522 ms |
| 4. stride | K/V 不 async | `stride_*_cache_3: tl.constexpr` | async 出现 |
| 5. retile | async 后旧 tile 不再最优 | full BF16 TILE32/BM16；hd512 BM8 | 0.0411 ms |

注意：第 3 阶段后来被第 4/5 阶段部分取代；当前只在 SWA/FP32 保留 TILE64。

---

#### 阶段 0：basic 版

```text
TILE_SIZE = 32（2D）
BLOCK_M = 16（group<=16）
K load: [H, T]
V load: [T, H]
每个 tile 都做 v_keep
```

它在数学上正确，但：

- 255 regs / 58 spills；
- 每个 program 串行扫很多 KV tile；
- K/V 没有 async；
- 所以 q3/ctx1024 约 0.1903 ms。

#### 阶段 1：先定位 spill

同一份输入和 vLLM 对比：

```text
mini: 255 regs / 58 spills
vLLM: 158 regs / 0 spills
```

vLLM 在这个 case 也走 2D，所以差距不是“3D 并行”。

spill 是什么：

```text
每个线程寄存器有上限（这里约 255）
超过后 compiler 把变量塞到 local memory
local memory 实际上在显存里
→ 每次读写都变慢
```

第一版修复：

1. **2D 去掉逐 tile 的 `v_keep`**
   - 2D 的 K/V load 已经用 `tile_mask` 把 tail 置 0；
   - causal/window 挡掉的 key 是已写入的有限值；
   - `v_keep` 的 `[BLOCK_M,TILE] -> [TILE]` 归约增加 live range。
2. **block-table 尾块 clamp**
   - 避免尾块越界读。
3. **调整 warps**
   - 当时 8 warps，减少每线程行数/寄存器压力。

结果：

```text
198 regs / 0 spills
q3/ctx1024 0.1903 → 0.1522 ms
```

**具体代码改动**

这次修复对应 commit `f8e132a`，核心是三类改动。

1）2D 去掉 `v_keep`，3D 保留：

```diff
- # 被窗口/tail mask 掉的 key 可能落在未写过的槽位；P=0 乘 NaN 仍是
- # NaN，所以先把这些 V 清零，再进 dot。
- v_keep = tl.max(tl.where(keep, 1, 0), axis=0) > 0
- V = tl.where(v_keep[:, None], V, 0.0)
+ # 2D 的 K/V load 已用 tile_mask 把 tail 槽位置 0；causal/window
+ # mask 掉的 key 仍是已写入的有限值，P=0 不会产生 NaN。
+ # 因此 2D 不再逐 tile 做 v_keep 归约（它增加寄存器压力）。
+ # 3D 保持原路径。
+ if IS_3D:
+     v_keep = tl.max(tl.where(keep, 1, 0), axis=0) > 0
+     V = tl.where(v_keep[:, None], V, 0.0)
```

原理：

```text
v_keep 会生成 [BLOCK_M,TILE] 的 keep 中间量
再做 axis=0 归约，得到 [TILE]
这两个 tensor 都会拉长 live range
→ 更多寄存器
```

2D 去掉了它，因为 2D 的 tail 已经由 `tile_mask` 清零。

2）2D block-table 尾块 clamp：

```diff
- physical_block = tl.load(
-     block_tables_ptr + block_table_offset
-     + seq_offset // BLOCK_SIZE,
-     mask=tile_mask,
-     other=0,
- ).to(tl.int64)
+ if IS_3D:
+     physical_block = tl.load(
+         block_tables_ptr + block_table_offset
+         + seq_offset // BLOCK_SIZE,
+         mask=tile_mask,
+         other=0,
+     ).to(tl.int64)
+ else:
+     # 2D 尾块可能读到 seq_len 之后的槽位；clamp 到块表最后一列。
+     block_idx = tl.minimum(
+         seq_offset // BLOCK_SIZE, block_table_stride - 1)
+     physical_block = tl.load(
+         block_tables_ptr + block_table_offset + block_idx,
+     ).to(tl.int64)
```

原理：

```text
2D 的 tail tile 可能越过 block table 的最后一列
→ 用 clamp 把索引压到最后一列
→ 避免越界；K/V 的 tail 仍由 tile_mask 清零
```

3）2D 用 8 warps：

```diff
+ num_warps = 4 if is_3d else 8
...
+ num_warps=num_warps,
```

原理：

```text
BLOCK_M=16、HEAD=256 时，acc/Q/K/V 每线程的寄存器很多
4 warps 时每线程要扛更多行
→ 255 regs + spill

8 warps 把行/元素摊到更多线程
→ 每线程寄存器压力下降
→ spill 消失
```

注意：warps 不是越多越好；后面 K/V layout 改了以后，又回到 4 warps。

这一步优化的是：**寄存器压力**。

#### 阶段 2：统一 K/V load layout

basic 版：

```text
K [H,T]  → 直接给 Q @ K
V [T,H]  → 直接给 P @ V
```

两个方向都符合 dot，但 K/V 的 **load/shared layout 不同**，compiler 需要两套布局和转换。

改成：

```text
K 也用 [T,H] load
K = tl.trans(K) 再 dot
V 保持 [T,H]
```

代价是 K 多一次 transpose；收益是：

- K/V load layout 统一；
- 少一套 shared layout / `convert_layout`；
- regs 和 shared 下降。

结果：

```text
约 140 regs / 0 spills，shared 25,600 B
q3/ctx1024 约 0.068 ms
```

**具体代码改动**

对应 commit `b88d94c`。

K 的指针从 `[H,T]` 改成 `[T,H]`：

```diff
  k_ptrs = (
      key_cache_ptr
-     + physical_block[None, :] * stride_k_cache_0
-     + slot_in_block[None, :] * stride_k_cache_1
+     + physical_block[:, None] * stride_k_cache_0
+     + slot_in_block[:, None] * stride_k_cache_1
      + kv_head_idx * stride_k_cache_2
-     + offs_d[:, None] * stride_k_cache_3
+     + offs_d[None, :] * stride_k_cache_3
  )
```

K 的 load 加 transpose，mask 方向也转：

```diff
- K = tl.load(
-     k_ptrs,
-     mask=dim_mask[:, None] & tile_mask[None, :],
-     other=0.0,
- ).to(Q.dtype)
+ K = tl.trans(tl.load(
+     k_ptrs,
+     mask=tile_mask[:, None] & dim_mask[None, :],
+     other=0.0,
+ )).to(Q.dtype)
```

V 不变：

```python
V = tl.load(
    v_ptrs,
    mask=tile_mask[:, None] & dim_mask[None, :],
    other=0.0,
).to(Q.dtype)
```

原理：

```text
旧：
  K load [H,T] → 自己的 shared layout
  V load [T,H] → 另一套 shared layout
  → 两套 layout + convert

新：
  K load [T,H] ┐
  V load [T,H] ┘ 同一类 load layout
  K 在 dot 前 tl.trans
  → 只保留一套 load/shared 路径
```

transpose 不是免费的，但它比“两套 layout + 多次 convert”便宜。

同时 S 的 keep 去掉 `tile_mask`：

```diff
- keep = row_mask[:, None] & tile_mask[None, :] & seq_mask
+ keep = row_mask[:, None] & seq_mask
```

原理：

```text
tail 已经由 seq_mask 挡住
tile_mask 再塞进 [BLOCK_M,TILE] 的 keep
只会多一个 mask 的 live range
```

BLOCK_M 从 16 改 8（commit `4ccedf7`）：

```diff
- block_m = 16 if group <= 16 else _next_power_of_2(group)
+ block_m = 8 if group <= 8 else (
+     16 if group <= 16 else _next_power_of_2(group))
```

原理：

```text
q3、group=8：
  BLOCK_M=16 → BLOCK_Q=2 → grid 2，2 个 active program
  BLOCK_M=8  → BLOCK_Q=1 → grid 4，3 个 active program
                            （第 4 个 q_block 立即 return）

更多 program：
  每个 program 的 acc/BLOCK_M 更小，shared 更小
  长上下文/并行度更好
代价：
  短上下文会多 program 启动/重复扫描
```

num_warps 从 8 回到 4（commit `b88d94c`）：

```diff
- num_warps = 4 if is_3d else 8
+ num_warps = 4
```

原理：

```text
K/V layout 统一后，K/V 的 shared/layout 压力下降
8 warps 反而让每个 program 的并行粒度更碎
4 warps 在目标 shape 上更快
```

后来再选 4 warps、BLOCK_M8、去掉冗余 `tile_mask`：

```text
168 regs / 0 spills，shared 20,992 B
0.0647 ms
```

这一步优化的是：**layout conversion / shared 压力**，不是 FLOPs。

#### 阶段 3：TILE64 + 2×32 精确 subtile

当时 K/V 仍是 scalar load，没有 async：

- 每个 tile 的 load 次数多；
- 循环次数多；
- 长上下文更明显。

做法：

```text
一次 load 64 个 K/V
→ reshape/permute/tl.split 成两个 32 宽 subtile
→ 按 TILE32 的顺序做两次 softmax/acc
```

为什么不是直接 TILE64 做一次 softmax？

```text
online softmax 的 tile 合并顺序变了
→ BF16 舍入顺序变了
→ 输出不是逐元素 0 差
```

拆成两个 32 宽 subtile 后，顺序和 TILE32 一致，所以输出 0.0 差。

**具体代码改动**

对应 commit `644744e`。

wrapper 改 TILE：

```diff
- tile_size = 16 if is_3d else 32
+ tile_size = 16 if is_3d else (64 if head_size <= 256 else 32)
```

loop 里增加 `TILE_SIZE == 64` 分支：

```python
if TILE_SIZE == 64:
    # K: [TILE, HEAD] -> [HEAD, 2, 32] -> [HEAD, 32, 2]
    K3 = tl.permute(
        tl.reshape(K, [HEAD_SIZE_PADDED, 2, 32]), 0, 2, 1)
    K0, K1 = tl.split(K3)

    # V: [TILE, HEAD] -> [2, 32, HEAD] -> [32, HEAD, 2]
    V3 = tl.permute(
        tl.reshape(V, [2, 32, HEAD_SIZE_PADDED]), 1, 2, 0)
    V0, V1 = tl.split(V3)

    # 第一个 32 宽 subtile
    K_h, V_h = K0, V0
    seq_offset_h = j * TILE_SIZE + 0 + tl.arange(0, 32)
    ... softmax/acc ...

    # 第二个 32 宽 subtile
    K_h, V_h = K1, V1
    seq_offset_h = j * TILE_SIZE + 32 + tl.arange(0, 32)
    ... softmax/acc ...
else:
    ... 原来的 TILE32 路径 ...
```

原理：

```text
一次 tl.load 64 宽 K/V
→ 减少 block-table load / 循环次数 / load 指令

但 online softmax 必须按 32 宽顺序更新
→ reshape/permute/tl.split 把 64 拆成两个 32
→ 先处理 key 0..31，再处理 32..63
→ 和 TILE32 的合并顺序一致
```

为什么 head512 不跟着用：

```text
64×512 的 K/V
→ 128 KiB shared
→ 超出预算，304 spills
```

结果：

```text
255 regs / 0 spills，shared 20,480 B
q3/ctx1024 0.0522 ms
```

这一步优化的是：**在没有 async 的情况下，摊薄循环/load 次数**。

#### 阶段 4：找到真正的 async 根因

Astra 发现 basic/中期版本里 K/V 最后一维 stride 是：

```python
stride_k_cache_3: tl.int64
stride_v_cache_3: tl.int64
```

而 vLLM 是：

```python
stride_k_cache_3: tl.constexpr
stride_v_cache_3: tl.constexpr
```

因果链：

```text
runtime int64 stride
→ TTIR 保留 offset = arange * stride 的动态乘法
→ AxisInfo 证不出相邻元素连续
→ Coalesce 选 sizePerThread=[1,1]
→ BF16 每线程 1 个元素 = 16 bit
→ 小于 cp.async 的 32 bit 门槛
→ AssignLatencies 不标 async
→ K/V 不进 pipeline
```

改成 `tl.constexpr`（调用点仍传真实 stride，不是硬编码 1）：

```text
动态乘法消失
→ contiguity 恢复
→ load layout 变成 [1,8]
→ BF16 每线程可搬多元素
→ pipeline pass 选中
→ ttg.async_copy_global_to_local / cp.async
```

只改这两行：

```diff
- stride_k_cache_3: tl.int64,
+ stride_k_cache_3: tl.constexpr,
- stride_v_cache_3: tl.int64,
+ stride_v_cache_3: tl.constexpr,
```

调用点不变，仍传真实 stride：

```python
stride_k_cache_3=k.stride(3),
stride_v_cache_3=v.stride(3),
```

原理：

```text
tl.int64 是运行时标量：
  kernel 参数里有 stride
  IR 里保留 offs_d * stride 的动态乘法
  compiler 不知道 stride 一定是 1
  → 不能证明 offs_d 连续

tl.constexpr 是编译期特化：
  外层仍传 k.stride(3)（本次是 1）
  compiler 用 1 生成这一份 kernel
  offs_d * 1 被折叠成 offs_d
  → AxisInfo 能证明连续
  → Coalesce 选 [1,8]
  → async 资格成立
```

注意：

```text
这不是把 stride 硬编码成 1。
如果传入 stride=2，会编译另一份 stride=2 的 kernel，
地址仍按 2 计算；测试覆盖了非单位 stride。
```

```text
async 出现了，但 q3/ctx1024 仍约 0.050 ms
```

因为旧 TILE64 workaround 在 async 下不再最优。

这一步优化的是：**编译期连续性证明 / async 资格**。

#### 阶段 5：async 后重新选 tile

有了 async 后，full BF16 head<=256 改成：

```text
TILE32
BLOCK_M16（BLOCK_Q=2）
```

为什么：

- TILE64 的 64 宽 load 不再必要；
- TILE32 shared 更小；
- BLOCK_M16 让两个相邻 query 位置共享一次 KV 扫描；
- 但 head512 的 acc 更大，仍用 BLOCK_M8。

**具体代码改动**

对应 commit `2b5316d`。

```diff
  sliding_window = 0
  if window_size[0] is not None and window_size[0] >= 0:
      sliding_window = int(window_size[0]) + 1
- tile_size = 16 if is_3d else (64 if head_size <= 256 else 32)
+
+ # FP32/SWA 保留原 dot 布局和窗口起点
+ preserve_tiling = q.dtype == torch.float32 or sliding_window > 0
+
+ # full BF16/FP16、head<=256：TILE32 + BLOCK_M16
+ if not is_3d and not preserve_tiling and head_size <= 256:
+     block_m = 16 if group <= 16 else _next_power_of_2(group)
+     block_q = block_m // group
+
+ tile_size = (16 if is_3d else
+              (64 if preserve_tiling and head_size <= 256 else 32))
  num_warps = 4
```

原理：

```text
async 前：
  scalar K/V load
  TILE64 用来摊薄 load/循环次数
  BLOCK_M8 用来控制寄存器

async 后：
  K/V load 已经是 global→shared 的异步流水
  不再需要 TILE64 的大块摊薄
  TILE32 让 shared 更小
  BLOCK_M16 让两个 query 位置复用同一份 K/V
  → 0.04113 ms
```

为什么 SWA 不切：

```text
SWA 的 tile 起点和全 mask subtile 会影响 online softmax 的浮点历史
→ preserve_tiling=True
→ 继续 TILE64/BM8
```

为什么 FP32 不切：

```text
FP32 需要 IEEE dot
TILE32/BM16 在 FP32 上出现过 1e-7 级差异
→ 不满足逐位 0 差
→ preserve_tiling=True
```

为什么 head512 不切 BM16：

```text
head512 的 acc/BLOCK_M16 寄存器太大
→ 继续 BM8
```

结果（Astra 同轮）：

```text
D256 full:  0.04988 → 0.04113 ms；vLLM 0.04007
D512 full:  0.09501 → 0.05797 ms；vLLM 0.06153
D256 ctx2048: 0.09567 → 0.07902 ms
regs 155 / 0 spills，shared 42,112 B
```

但 SWA 不能直接跟：

```text
D256 SWA512: 0.02752 → 0.02968 ms（变慢）
```

原因不是纯性能，是**数值边界**：

```text
query 绝对位置 574，window 512
第一个可见 key = 63

旧 TILE64 从 0 开始：
  key 0..31 全 mask，M 先被钳到 0
直接 TILE32 从 32 开始：
  key 32..63 中 63 有效，直接按它的 score 更新 M
```

如果第一个有效 score 是负的，旧路径 M=0，新路径 M=-2.56；
`P` 先转 BF16，`exp(-2.56)` 和 `1` 的舍入不同，后面缩放无法保证逐位一致。

所以：

```text
SWA：保留 TILE64/BLOCK_M8
FP32：保留 TILE64/BLOCK_M8 + IEEE dot
3D：保持 TILE16
```

这一步优化的是：**在 async 可用后重新平衡 tile/并行度，同时保护数值边界**。

#### 最终数字

单 kernel（Astra 同轮）：

```text
q3/ctx1024: 0.04113 ms（vLLM 0.04007）
ctx2048:    0.07902 ms（vLLM 0.07705）
hd512:      0.05797 ms（vLLM 0.06153）
```

真实端到端（我们的 A/B，1024 prompt，128 output，Graph）：

```text
旧 mini: 158.28 tok/s，TTFT 302.8 ms，TPOT 3.991 ms
新 mini: 170.22 tok/s，TTFT 301.4 ms，TPOT 3.615 ms
```

同轮 mini vs vLLM：

```text
mini: 175.55 tok/s，TTFT 280.5 ms，TPOT 3.563 ms
vLLM: 165.44 tok/s，TTFT 327.0 ms，TPOT 3.491 ms
```

注意：

- TTFT 基本不变：这个优化主要打在 decode 的 TPOT；
- 跨引擎输出第 112 token 分歧，性能对照不等于质量等价；
- 轮次波动大，不能把单轮 +6% 当成永久结论。

#### 为什么不是一步到位

因为每一步解决的是不同瓶颈：

```text
spill        → 寄存器
layout       → shared/layout conversion
TILE64       → 循环/load 次数（无 async 时）
stride       → async 资格
retile       → async 下的 shared/并行度
```

如果顺序错了，比如先上 `num_stages`：

```text
K/V 仍是 scalar load
→ pipeline 只能选 block-table
→ 时间不会降
```

负结果也验证了：

- TILE64 hd512 → 304 spills；
- K 改回 `[H,T]` → 252 spills；
- cache `.cg`/evict → 2048 更慢；
- BLOCK_M4 → 长上下文无收益。

#### 给初学者的方法论

```text
1. 先保证输出 0.0 差
2. 看 regs/spills/shared
3. 看 TTGIR/PTX：有没有 convert_layout、async_copy
4. 一次只改一个假设
5. 改 tile 后重做数值对照
6. 最后才看真实模型 tok/s/TTFT
```

最重要的两个教训：

```text
layout 和编译期信息，往往比多写几个 tl.dot 更重要；
async 不是手写 cp.async，而是让 compiler 有资格选它。
```

### Q17. 阶段 1 的 2D 为什么能去掉 v_keep？和之前说“必须清零”矛盾吗？

**问题**

前面 Q12–Q15 说：

```text
没有被任何 query 看到的 V key 要清零
否则 P=0 时 0 * NaN = NaN
```

但阶段 1（commit `f8e132a`）只给 2D 去掉了 `v_keep`，3D 还保留。
这不是前后矛盾吗？

**结论**

不矛盾，但之前说“必须清零”不够准确。

准确说法是：

```text
basic 版/general 情况下：需要防御性清零
2D 在 mini 的 cache 契约下：清零是冗余的
3D：当前代码保守保留
```

**basic 版为什么需要 v_keep**

f3bd003 里 V 的 load mask 是：

```python
mask = tile_mask[:, None] & dim_mask[None, :]
```

它没有包含 `seq_mask`。
所以一个 key 可能：

```text
tile_mask = True   → V 被 load
seq_mask  = False  → 所有 query 都看不到
```

如果这个 V cache 槽位是未初始化/NaN，load 回来就是 NaN；
`P=0` 乘 NaN 仍是 NaN。
所以 basic 版用 `v_keep` 把“所有 query 都看不到”的 V 行清零。

这部分解释没有变。

**2D 为什么可以去掉**

关键是 2D 的 cache 契约：

```text
V load 的位置必须满足：
  seq_offset < max_prefix
  max_prefix <= seq_len

也就是说：
  seq_offset < seq_len
```

而 2D 的 `seq_len` 是：

```text
seq_len = num_computed_tokens + query_len
```

它覆盖的每一个位置 `0 .. seq_len-1`：

- 要么是之前已经写入 KV cache 的历史 token；
- 要么是本轮 query token，attention 前已经由 `reshape_and_cache` 写入。

所以对 2D：

```text
tile_mask=True 的 V 位置
→ 都在 seq_len 内
→ 都已写入
→ 都是有限值
```

tail 呢？

```text
tile_mask=False
→ tl.load 返回 other=0.0
→ 不可能是 NaN
```

因此 2D 里：

```text
“所有 query 都看不到”的 key
仍然是已写入的有限值
→ P=0 乘有限值 = 0
→ 不需要 v_keep
```

这不是“NaN 不存在”，而是：

```text
2D 的 load 范围被 max_prefix/seq_len 限制
未写槽位根本不会被 mask=True 地 load
```

**为什么 3D 还保留**

当前代码：

```python
if IS_3D:
    v_keep = tl.max(tl.where(keep, 1, 0), axis=0) > 0
    V = tl.where(v_keep[:, None], V, 0.0)
```

注释写的是：

```text
3D 保持原路径，避免改变单 token decode 的分段语义。
```

这不是说 3D 一定有 NaN，而是：

- 3D 走的是分段/workspace/reduce 路径；
- 阶段 1 的目标是修 2D spill；
- 没有对 3D 做同等证明和数值对照；
- 所以保守保留，避免顺手改出另一个边界。

**更准确的表述**

之前：

```text
“没有被任何 query 看到的 V 要清零”
```

应该补全为：

```text
“如果 V load mask 不能保证这些位置已写入，
  就需要清零；
  2D 的 mask 范围 + cache 写入契约可以保证，
  所以 2D 可以去掉；
  3D 当前选择保守保留。”
```

**代码改动**

f8e132a：

```diff
- v_keep = tl.max(tl.where(keep, 1, 0), axis=0) > 0
- V = tl.where(v_keep[:, None], V, 0.0)
+ if IS_3D:
+     v_keep = tl.max(tl.where(keep, 1, 0), axis=0) > 0
+     V = tl.where(v_keep[:, None], V, 0.0)
```

当前代码只在 3D 分支保留。

**这个结论依赖什么**

2D 去 `v_keep` 依赖：

```text
1. max_prefix <= seq_len
2. seq_len 内的 K/V 位置都已经写入
3. tail 位置由 tile_mask=False + other=0 处理
```

如果以后出现：

- cache 槽位未初始化但 seq_len 声称有效；
- block table 指向未写页；
- 新的 Q-only/共享 KV 路径破坏写入契约；

那 2D 也可能重新需要防御性清零。

所以这不是“v_keep 永远没用”，而是：

```text
在当前 2D cache 契约下，它是冗余的；
换成不满足契约的路径，它仍然有意义。
```

### Q18. 上游 vLLM 是不是也删掉了 v_keep？

**问题**

mini 阶段 1 在 2D 去掉了 `v_keep`。
本机 vLLM 0.28.0 的 unified attention 是不是也没有 `v_keep`？

**结论**

本机安装的 vLLM 0.28.0：

```text
没有通用的 v_keep
```

但它不是“完全不管 V 的不可见列”，而是有一个 **SWA 专用** 的 `sw_mask_v`：

```python
if SLIDING_WINDOW:
    qpos_lo = q_block_local_idx * BLOCK_Q
    dist = context_len + qpos_lo - seq_offset[:, None]
    if USE_CAUSAL:
        sw_mask_v = dist < SLIDING_WINDOW
    else:
        sw_mask_v = ((dist < SLIDING_WINDOW)
                     & (dist > -SLIDING_WINDOW))
    V = tl.where(sw_mask_v, V, 0.0)
```

它和 mini 的 `v_keep` 不一样：

```text
v_keep:
  对 [BLOCK_M, TILE] 的 keep 做 axis=0 归约
  = 每个 key 列是否被任意 query 看到

sw_mask_v:
  只看 qpos_lo（block 内最早 query）
  用距离判断这个 key 是否在 sliding window 外
```

**上游 full attention 为什么没有 v_keep**

vLLM 的 V load mask 也是：

```python
mask=dim_mask[None, :] & tile_mask[:, None]
```

同样不包含 `seq_mask`。
它的 full/2D 路径依赖和 mini 2D 相同的 cache 契约：

```text
tile_mask=True 的位置在 seq_len 内
→ 已经写入
→ 是有限值
tail → mask=False → other=0
```

所以 full attention 不需要通用 `v_keep`。

**上游 SWA 为什么有 sw_mask_v**

sliding window 的 tile 起点会按 `TILE_SIZE` 对齐，
所以一个 tile 里可能包含“窗口外”的 key。

上游用 `qpos_lo` 做距离判断，把这些 key 的 V 清零。
对 causal SWA，最早 query 的窗口左边界最靠左，
所以窗口外的列可以直接清零；这不是通用的 any-query 归约。

**mini 和上游的对应关系**

```text
mini 2D full：没有 v_keep        ≈ 上游 full 2D
mini 2D SWA ：没有 v_keep        ≠ 上游有 sw_mask_v（实现不同）
mini 3D     ：保留通用 v_keep    ← 上游没有这个通用逻辑
```

严格说：

```text
上游 0.28.0 没有 v_keep；
mini 的 2D full 去掉 v_keep 与上游行为一致；
mini 的 3D 保留 v_keep 是额外的保守防御。
```

**是否“删掉”**

我只能确认本机 vLLM 0.28.0 当前源码没有 `v_keep`。
它历史上是否曾经有、何时删除，我没有版本历史证据，不能断言。

**代码位置**

```text
vLLM 0.28.0:
  vllm/v1/attention/ops/triton_unified_attention.py
    SLIDING_WINDOW 分支的 sw_mask_v

mini:
  minivllm/attention/ops/triton_unified_attention.py
    2D 无 v_keep
    3D 保留 v_keep
```

### Q19. block-table 尾块 clamp 是不是多余？不是有 tile_mask 吗？

**问题**

basic 版已经有：

```python
tile_mask = seq_offset < max_prefix
```

而且 block-table load 也是 masked：

```python
physical_block = tl.load(
    block_tables_ptr + block_table_offset
    + seq_offset // BLOCK_SIZE,
    mask=tile_mask,
    other=0,
)
```

那阶段 1 为什么还要 clamp：

```python
block_idx = tl.minimum(
    seq_offset // BLOCK_SIZE, block_table_stride - 1)
physical_block = tl.load(
    block_tables_ptr + block_table_offset + block_idx,
)
```

不是多余吗？

**结论**

在 basic 版那种“masked block-table load”里，clamp 确实是冗余的。

但阶段 1 的 2D 路径把 block-table load 改成了 **unmasked load**：

```diff
- physical_block = tl.load(
-     block_tables_ptr + block_table_offset
-     + seq_offset // BLOCK_SIZE,
-     mask=tile_mask,
-     other=0,
- ).to(tl.int64)
+ if IS_3D:
+     physical_block = tl.load(
+         block_tables_ptr + block_table_offset
+         + seq_offset // BLOCK_SIZE,
+         mask=tile_mask,
+         other=0,
+     ).to(tl.int64)
+ else:
+     block_idx = tl.minimum(
+         seq_offset // BLOCK_SIZE, block_table_stride - 1)
+     physical_block = tl.load(
+         block_tables_ptr + block_table_offset + block_idx,
+     ).to(tl.int64)
```

2D 的 block-table load 没有 mask 了，所以必须靠 clamp 保证索引不越界。

**tile_mask 保护的是谁**

`tile_mask` 仍然在保护 K/V load 和 S：

```python
K/V load: mask=tile_mask[:, None] & dim_mask[None, :]
S:        keep = row_mask & seq_mask
```

tail 的 K/V：

```text
tile_mask=False → other=0.0
```

所以 clamp 把尾块 block 索引压到最后一列后，
即使读到了“最后一个 block”的 physical_block，
后面的 K/V load 仍会被 `tile_mask` 清零，不参与计算。

**为什么可以去掉 block-table 的 mask**

masked load 需要：

```text
每个 lane 的 predicate
predicate 参与 load
可能增加 mask/live range
```

去掉 mask、用 clamp 后：

```text
block_idx 始终在 [0, stride-1]
load 是普通 unmasked load
```

这可以减少一个 mask 相关 live range。
不过这个改动和“去 v_keep / 8 warps”是同一个 commit 一起做的，没有单独做 clamp-only 的消融，所以不能把收益全部归因于 clamp。

**什么时候 clamp 才真的多余**

如果保留 basic 版的 masked block-table load：

```python
physical_block = tl.load(..., mask=tile_mask, other=0)
```

那么：

```text
mask=False 的尾块不会读内存
→ 不需要 clamp
```

此时 clamp 才是冗余的。

**当前代码**

```text
2D：unmasked block-table load + clamp
3D：masked block-table load，无 clamp
```

所以不是“tile_mask 已经够了”，而是：

```text
tile_mask 保护 K/V 和 S；
2D block-table 本身不再用 mask，改用 clamp 保护索引。
```

**代码位置**

```text
minivllm/attention/ops/triton_unified_attention.py
  kernel_unified_attention
    IS_3D 分支的 physical_block
    2D 分支的 block_idx/clamp
    K/V load 的 tile_mask
```

### Q20. 为什么 `BLOCK_M=8 → BLOCK_Q=1` 是 3 个 program？

**问题**

q_len=3，`BLOCK_Q=1`，为什么会说 3 个 program？
`total_q_blocks` 算出来不是 4 吗？

**结论**

准确说：

```text
grid 是 4 个 program
实际有 3 个 active program
第 4 个 q_block 立即 return
```

不是 grid 只有 3 个。

**grid 公式**

wrapper：

```python
total_q_blocks = num_tokens // block_q + num_seqs
grid = (total_q_blocks, num_kv_heads)
```

target：

```text
num_tokens = 3
block_q = 1
num_seqs = 1
num_kv_heads = 1
```

所以：

```text
total_q_blocks = 3 // 1 + 1 = 4
grid = (4, 1)
```

确实会 launch **4 个** q_block × 1 个 KV head。

**为什么是上界**

`total_q_blocks` 是上界，不是精确的 active 数：

```text
sum ceil(q_len_i / BLOCK_Q)
<= floor(N / BLOCK_Q) + num_seqs
```

对单请求 q_len=3、BLOCK_Q=1：

```text
真实需要：ceil(3/1) = 3
上界：3//1 + 1 = 4
```

多出来的 1 个是每个请求的“可能 partial block”。

为什么不在 CPU 精确算：

```text
q_len 在 GPU 的 query_start_loc 里
要在 CPU 精确算就得同步 GPU
→ Graph/性能都受影响
```

所以用保守上界，多 launch 几个，让 kernel 自己退出。

**kernel 里怎么退出**

`_resolve_seq`：

```python
q_block_local = q_block_global_idx - q_block_start
q_len = q_end - q_start
```

然后：

```python
if q_block_local * BLOCK_Q >= q_len:
    return
```

target：

```text
q_block_local=0 → 0*1 >= 3? false → active，处理 query 0
q_block_local=1 → 1*1 >= 3? false → active，处理 query 1
q_block_local=2 → 2*1 >= 3? false → active，处理 query 2
q_block_local=3 → 3*1 >= 3? true  → return
```

所以：

```text
grid = 4
active = 3
```

**为什么 BLOCK_M=16 是 2 个**

`BLOCK_M=16`、group=8：

```text
BLOCK_Q = 16 // 8 = 2
```

grid：

```text
total_q_blocks = 3 // 2 + 1 = 2
```

两个都 active：

```text
q_block 0：query 0,1
q_block 1：query 2
```

所以：

```text
BLOCK_Q=2 → grid 2，active 2
BLOCK_Q=1 → grid 4，active 3
```

**为什么多 program 能更快**

`BLOCK_Q=1` 时：

```text
3 个 query token → 3 个 active program
每个 program 只处理 1 个 query 位置 × 8 个 GQA head
```

好处：

```text
每个 program 的 acc/BLOCK_M 更小
shared 更小
3 个 program 可以并行扫 KV
```

代价：

```text
短上下文多 launch/重复扫描
```

**注意 KV head**

target：

```text
num_kv_heads = 1
```

所以 q_block 数就是总 program 数。

如果：

```text
num_kv_heads = 2
```

grid 是：

```text
(total_q_blocks, 2)
```

总 program 数要再乘 2。

**代码位置**

```text
minivllm/attention/ops/triton_unified_attention.py
  unified_attention：total_q_blocks / grid
  kernel_unified_attention：q_block_local 越界 return
  _resolve_seq
```

### Q21. 什么是 `load/shared layout`？

**问题**

前面一直说：

```text
load layout
shared layout
convert_layout
```

这些 layout 到底指什么？和 shape 有什么区别？

**一句话**

```text
shape：逻辑上有几个元素、怎么索引
layout：这些元素物理上分给哪个线程、放在 shared 的哪个位置
```

同一个 shape 的 tensor，可以有不同的 layout。

**一个直观例子**

逻辑张量：

```text
X: [2, 4]
```

它固定是 8 个元素。
但“哪个线程拿哪几个元素”可以有两种分配：

```text
layout A（按行分）：
  thread0: X[0,0] X[0,1] X[0,2] X[0,3]
  thread1: X[1,0] X[1,1] X[1,2] X[1,3]

layout B（按列分）：
  thread0: X[0,0] X[1,0]
  thread1: X[0,1] X[1,1]
  ...
```

shape 都是 `[2,4]`，但：

- 线程拿到的元素不同；
- 全局访存是否连续不同；
- 是否要用 shared 重排不同；
- 性能不同。

**load layout 是什么**

`tl.load` 产生一个 tensor。
这个 tensor 在寄存器里怎么分布到线程，就是它的 load layout。

TTGIR 例子：

```text
#blocked = #ttg.blocked<{
    sizePerThread = [1, 1],
    threadsPerWarp = [1, 32],
    warpsPerCTA = [1, 4],
    order = [1, 0]
}>
```

字段含义：

| 字段 | 含义 |
|---|---|
| `sizePerThread` | 每个线程在每个维度上拥有几个元素 |
| `threadsPerWarp` | 一个 warp 的线程怎么排到各维 |
| `warpsPerCTA` | CTA 的 warp 怎么排 |
| `order` | 哪一维是连续/最快变化的 |

例如：

```text
sizePerThread = [1, 8]
```

表示每个线程在第二维拿 8 个连续元素，可以做成向量化 load。
而：

```text
sizePerThread = [1, 1]
```

表示每个线程只拿 1 个元素，通常是 scalar load。

**shared layout 是什么**

`tl.dot` / `tl.trans` 需要把操作数搬到 shared memory。
元素在 shared 里怎么排、怎么 swizzle、怎么避免 bank conflict，就是 shared layout。

TTGIR 例子：

```text
#shared = #ttg.swizzled_shared<{
    vec = 8,
    perPhase = 1,
    maxPhase = 8,
    order = [1, 0]
}>
```

它决定：

- shared 里的元素顺序；
- 每个 phase 的 swizzle；
- 向量宽度；
- 后续 `ldmatrix` / `stmatrix` 怎么读。

**为什么需要 layout conversion**

如果 load 出来的布局和 dot/shared 需要的布局不一致，compiler 会插入：

```text
ttg.convert_layout
```

它通常意味着：

```text
寄存器 -> shared -> barrier -> 重新读进寄存器
```

可能带来：

- 额外 shared 占用；
- `st.shared` / `ldmatrix`；
- barrier；
- 寄存器 live range 变长。

所以：

```text
layout conversion 不是免费的
```

**K/V 的例子**

basic 版：

```text
K [H,T] → 直接给 Q @ K
V [T,H] → 直接给 P @ V
```

两个 tensor 都符合 dot，但：

```text
K 的 load layout
V 的 load layout
```

不是同一类，compiler 更容易生成两套 shared layout / 转换。

改成：

```text
K [T,H] load
K = tl.trans(K)
V [T,H] load
```

让 K/V 共用同一种 load layout，再对 K 做一次小块 transpose。
transpose 本身也有成本，但比两套 layout 的转换便宜。

**为什么 `tl.trans` 不是免费的**

逻辑上：

```text
K [T,H] -> K [H,T]
```

但线程持有的元素没变；要让 dot 以 `[H,T]` 使用，通常需要：

```text
写 shared（按目标 layout）
-> barrier
-> ldmatrix 读
```

所以 `tl.trans` 可能触发 shared layout conversion。
它不是 C++ 里一个指针偏移。

**和 stride constexpr 的关系**

Astra 的发现是：

```text
runtime stride
→ 编译器证不出连续
→ load layout 选 [1,1]（scalar）
```

改成 constexpr：

```text
stride 已知
→ 证出连续
→ load layout 选 [1,8]（向量化）
→ async 资格成立
```

所以 layout 不只影响“快不快”，还影响：

```text
这个 load 有没有资格进 cp.async pipeline
```

**一句话总结**

```text
shape 是“数学上长什么样”；
layout 是“线程和 shared 实际怎么摆”；
convert_layout 是“摆法不对，搬一次”；
很多 attention kernel 优化，本质是在减少这种搬运。
```

已追加到：

```text
docs/step88_triton_unified_attention_qa.md 的 Q21
```

没有 commit。

## 4. 追加模板

后续每问一个，按下面格式追加：

```markdown
### Qn. 问题标题

**问题**

用户原问题（可压缩，但不改变意思）。

**结论**

一句话先回答。

**代码位置**

文件 + 函数/代码段；必要时给最小代码。

**机制/为什么**

按“输入 → 状态 → 代码动作 → 结果”讲，不用源码行号堆砌。

**证据/复现**

测试、benchmark、TTGIR/PTX、真实模型差分；没有证据就写“未验证”。

**边界**

哪些 dtype/shape/路径不适用；是否改变数值。
```

## 5. 相关文档

```text
docs/step88_attention_async_resolution.md   Astra 对 stride/async 的完整定位
docs/step88_attention_open_problem.md       最初的开放问题与验收标准
docs/step88_attention_pipeline.md           早期 TTGIR/PTX 诊断与负结果
docs/step88_vllm_kernel_diff.md             22-case mini vs vLLM kernel 对照
```
