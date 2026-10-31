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

## 6. 已实施的第一版 2D 修复

只改 2D，3D 路径不动：

```text
1. 2D launch 用 8 warps（3D 仍 4）
2. 2D 去掉每 tile 的 v_keep / V 清零（3D 保留）
3. 2D block table 尾块 clamp 到最后一列，去掉 masked load
   （K/V 仍由 tile_mask 清零，尾块不参与 softmax/dot）
```

效果：

| 场景 | 修复前 | 修复后 | 变化 |
|---|---:|---:|---:|
| q3 ctx1024 hd256 | 0.1903 ms | 0.1522 ms | **-20.0%** |
| q3 ctx2048 hd256 | 0.3706 ms | 0.2952 ms | -20.4% |
| q3 ctx1024 hd512 | 0.3508 ms | 0.2895 ms | -17.5% |
| mini 2D registers | 255 | 198 | — |
| mini 2D spills | 58 | **0** | — |
| 真实 Graph 1024 tok/s | 125.63 | 135.55 | +7.9% |

真实 Graph 四格（同协议、repeats=1）：

```text
mini 135.55 tok/s vs vLLM 160.16（0.846×）
修复前：125.63 vs 169.50（0.741×）
```

**spill 已消除，但这只是第一步。** 修复后 q3/ctx1024/hd256 仍是 mini
0.1522 ms vs vLLM 0.053–0.066 ms，约 2.3–2.9×；剩余差距不是 spill，而是
kernel codegen/流水线：vLLM 的 PTX 有 `cp.async.cg.shared.global`、
`ldmatrix`，mini 仍是同步 load + 更多 `st.shared`。所以下一步要优化
2D 的 load→shared→MMA 流水，而不是继续调 register 或上 3D。这一阶段的 0.152 ms
已在下一节的 vendored 方案中被 0.047 ms 取代。

## 7. 最终方案：vendored vLLM 2D kernel

第一版本地优化只到 0.152 ms，仍比 vLLM 的 2D kernel 慢。最终做法是把 vLLM 0.28.0
的 2D kernel **vendored 到本地**（`triton_unified_attention_vllm.py` +
`triton_attention_helpers.py`，保留 Apache-2.0 头；不是 import 上游包），
由 `unified_attention` 在 2D BF16 路径直接调用。

结果（同输入、同 2D、同 `max_seqlen_q=3`）：

| 场景 | mini | vLLM | mini/vLLM |
|---|---:|---:|---:|
| q3 ctx128 | 0.0748 ms | 0.0759 ms | 0.985 |
| q3 ctx512 | 0.0455 ms | 0.0449 ms | 1.013 |
| q3 ctx1024 | 0.0472 ms | 0.0536 ms | 0.880 |
| q3 ctx2048 | 0.0839 ms | 0.0833 ms | 1.008 |
| hd256 full | 0.0531 ms | 0.0516 ms | 1.029 |
| hd256 sliding | 0.0496 ms | 0.0442 ms | 1.122 |
| hd512 full | 0.0662 ms | 0.0664 ms | 0.997 |

编译结果：

```text
mini vendored 2D: 157 regs, 0 spills, cp.async 51
vLLM 2D:          158 regs, 0 spills
```

**2D attention 的差距已基本消除**：直接 kernel 在多数 shape 上为 0.88–1.12×，
剩下的差异在噪声范围；不是此前 2.3–3.6×。

真实 Gemma4 B1 K2 1024 context（同协议 repeats=1）：

```text
mini Graph: 173.74 tok/s
vLLM Graph: 138.23 tok/s（本轮；此前 160.16，vLLM 自身波动）
mini/vLLM:  1.257
修复前:     125.63 vs 169.50（0.741）
```

这轮 vLLM 比此前慢，不能把 1.257 当作稳定领先；能确定的是 attention kernel 已对齐，
剩余是 engine 噪声/FP32 fallback/3D 等边界，不是主路径。

保留的 fallback：

- **FP32 q**：走本地 `ieee` dot，满足 step86 的 2e-5 精度；
- **head_size=512 + sliding**：vLLM 2D 组合 shared memory 超限，走本地 kernel；
- **3D**：保持本地 kernel，不改变单 token decode 分段。

## 8. 和 3D prototype 的关系

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

## 9. 下一步

2D attention 主路径已基本对齐 vLLM，下一步不再是继续调 kernel：

1. 保留 vendored 文件的来源/许可说明，后续升级 vLLM 时同步核对；
2. 观察 vLLM 自身 138–160 tok/s 的波动，用多轮同协议重测，不拿单轮比值下结论；
3. FP32 的本地 fallback 只服务测试；真实 target 是 BF16；
4. hd512+sliding 是合成边界，真实 Gemma4 是 hd256 sliding + hd512 full；
5. 3D 仍留在本地路径，是否启用由后续需求决定，不与本次 2D 优化混在一起。

## 10. 复现

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
