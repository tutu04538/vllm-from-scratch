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
