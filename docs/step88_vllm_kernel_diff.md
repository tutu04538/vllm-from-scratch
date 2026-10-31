# 88 后续：为什么同一 2D 路径 mini 比 vLLM 慢

> 目标：回答“在这个 B=1、K=2、1024 上下文、FULL Graph 场景，为什么 mini 比 vLLM 慢？”
> 脚本：`benchmarks/check_step88_vllm_kernel_diff.py`
> 原始结果：`benchmarks/results/step88/vllm_kernel_diff/summary.json`（gitignored）

## 1. 先纠正上一轮的方向

上一轮只证明了：

```text
mini 2D → mini 3D（q_len>1 prototype）可快 2.7–4.6×
```

它没有回答 mini 为什么比 vLLM 慢。实际上，**vLLM 的目标模型在这个场景也走 2D**：

```text
vLLM TritonAttentionImpl（从 EngineCore 钩子读取）
  max_query_len       = 3
  head_size           = 256 / 512
  TILE_SIZE           = 32
  BLOCK_M             = 16
  BLOCK_Q             = 2
  IS_3D               = false
  NUM_SEGMENTS_PER_SEQ= 1
  USE_CAUSAL          = true
```

所以“vLLM 用了 3D、mini 只会 2D”不是正确答案。两边是同一套 2D 数学、同一组 tile
参数；差异在 **kernel 实现/寄存器压力**。

## 2. 同输入同 2D 的直接对照

不加载 engine，同一组 q、paged K/V、block table、窗口和 `max_seqlen_q=3`：

| 配置 | mini ms | vLLM ms | vLLM 快 | 输出差 |
|---|---:|---:|---:|---:|
| q3 ctx128, hd256 full | 0.0391 | 0.0514 | 0.76×（mini 快） | 0.0 |
| q3 ctx256, hd256 full | 0.0553 | 0.0627 | 0.88×（mini 快） | 0.0 |
| q3 ctx512, hd256 full | 0.1003 | 0.0541 | 1.85× | 0.0 |
| q3 ctx1024, hd256 full | 0.1903 | 0.0530 | **3.59×** | 0.0 |
| q3 ctx2048, hd256 full | 0.3706 | 0.0839 | **4.42×** | 0.0 |
| q3 ctx1024, hd256 sliding512 | 0.0920 | 0.0442 | 2.08× | 0.0 |
| q3 ctx1024, hd512 full | 0.3508 | 0.0762 | **4.61×** | 0.0 |

**输出差全是 0.0**：不是算法/数值不同，是同一 2D 数学的执行效率不同。

## 3. 直接证据：mini 2D kernel 在 spill

读取两边 `kernel_unified_attention` 的 Triton `CompiledKernel`：

| 配置 | mini registers | mini spills | vLLM registers | vLLM spills |
|---|---:|---:|---:|---:|
| hd256 2D | 255 | **58** | 158 | 0 |
| hd512 2D | 255 | **280**（该编译变体） | 255 | **6** |

`255` 是每线程寄存器上限；mini 已经顶到上限并大量 spill 到 local memory，vLLM 的
hd256 kernel 是 158 regs / 0 spill，hd512 也只 6 spill。这个差异和 3.6–4.6× 的耗时差
方向一致。

同一份 trace 里：

```text
mini Graph:  kernel_unified_attention 50.52 ms / 8 steps
GEMM:        39.80 → 39.92 ms（上下文变化几乎不变）
RMSNorm:     2.58 → 2.59 ms
RoPE:        0.46 → 0.46 ms
```

上下文增加时只有 attention 显著变重；Graph 又把 CPU launch 从 2584 降到 88，因此
在 Graph 下暴露的是 **mini 2D attention kernel 的 spill/执行效率**。

## 4. 为什么 eager 时 mini 反而略快

1024 上下文：

```text
eager:  mini 62.30 vs vLLM 58.33（mini 1.068×）
Graph:  mini 125.63 vs vLLM 169.50（mini 0.741×）
```

eager 两边都有大量前端/调度开销，mini 的 host 路径更轻，掩盖了 kernel 的 spill；Graph
把 CPU 发起/等待压掉后，kernel 成为主项，vLLM 的 158/0-spill kernel 就领先。这也解释了
为什么“同一注意力数学”的差距只在 Graph 列明显。

## 5. `v_keep` 不是全部原因

mini kernel 每个 tile 额外把被 mask 的 V 清零：

```python
v_keep = tl.max(tl.where(keep, 1, 0), axis=0) > 0
V = tl.where(v_keep[:, None], V, 0.0)
```

A/B 删除它：

```text
orig:    0.185 ms, 255 regs, 58 spills
no_vkeep:0.178 ms, 255 regs, 48 spills
```

只快约 4%，还剩 48 spills。因此不能把问题简化成“删掉 v_keep 就好”；真正的问题是
**mini 2D kernel 整体的 live range/寄存器分配**比 vLLM 版本差。

## 6. 和 3D prototype 的关系

3D q_len>1 prototype 是**另一个可选优化**：

- 它把每 program 的 KV tile 数从约 33 降到约 5，长上下文单 kernel 可快 2.7–4.6×；
- 但 vLLM 在目标模型 decode 上并没有用 3D，它仍用 2D；
- 所以 3D 不能解释“为什么 mini 比 vLLM 慢”，也不能替代“把 mini 2D kernel 的 spills 降下来”。

正确优先级：

```text
1. 先修 mini 2D kernel 的 register pressure / spills（或对齐 vLLM 的优化实现）
2. 再看 3D/split-KV 是否还有额外收益
3. 最后才考虑 compile/异步
```

## 7. 下一步

1. 对比 mini 与 vLLM 2D kernel 的 PTX/SASS live range，定位 255/58 的具体来源；
2. 优先尝试与 vLLM 一致的代码结构（尤其去掉不必要的 V mask、减少同时 live 的 2D mask，
   或对齐 vLLM 的 `_cast_kv_tile`/softmax helper）；
3. 用 A/B 检查 `num_warps` / tile / V-mask 方案，要求输出逐元素仍为 0.0 差；
4. 在真实 Gemma4 B=1 K=2、1024 上下文上验证 Graph 吞吐是否从 125.6 向 169.5 靠近；
5. 只有 2D spill 问题解决后，3D q_len>1 仍可作为长上下文补充，而不是当前主因。

## 8. 复现

```bash
PY=/home/user/anaconda3/envs/vllm-omni-dev/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

$PY benchmarks/check_step88_vllm_kernel_diff.py --warmup 50 --repeats 200 --rounds 10
$PY -m pytest tests/step88/test_vllm_kernel_diff.py -q
```

结果目录（不跟踪）：

```text
benchmarks/results/step88/vllm_kernel_diff/summary.json
```
