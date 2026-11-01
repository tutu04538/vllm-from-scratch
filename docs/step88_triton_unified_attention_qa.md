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
