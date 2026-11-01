# 88 后续：vLLM 2D attention 的 software pipeline 到底优化了什么

> 脚本：`benchmarks/check_step88_attention_pipeline.py`
> 原始结果：`benchmarks/results/step88/attention_pipeline/summary.json`（gitignored）
> 当前生产：本地 2D kernel，255 regs / 0 spills，20,480 B shared；**没有 vendored vLLM 代码**。
> 最终优化是 K/V `[TILE,HEAD]` layout + BLOCK_M8 + TILE64 load/2×32 精确 subtile（§10–§11.4），不是 cp.async。

## 1. 先看结论

同一组 q、paged K/V、block table，`q_len=3/context=1024/head_dim=256`。

**诊断阶段（K 还是 `[HEAD,TILE]` load layout）**：

```text
mini:  0.1511 ms，198 regs，0 spills
vLLM:  0.0825 ms，158 regs，0 spills
输出差：0.0
```

**最终（K/V 都按 `[TILE,HEAD]` load，K 再 transpose；§10）**：

```text
mini:  0.0522–0.0553 ms，255 regs，0 spills，20,480 B shared
vLLM:  0.0486–0.0969 ms（轮次波动）
输出差：0.0
```

短/中上下文大多数 shape 已不慢于 vLLM；长上下文仍约 1.1–1.5×。

vLLM 快的核心不是 3D、不是 grid/tile，而是 **Triton 把 block-table/K/V load 编成了
`cp.async` 的 software pipeline**：

```text
vLLM TTGIR:
  ttg.async_copy_global_to_local   10 次
  ttg.async_commit_group           10 次
  ttg.async_wait                    3 次
  ptx cp.async                     51 条
  ptx st.shared                    14 条

mini TTGIR:
  ttg.async_copy_global_to_local    0 次
  ptx cp.async                      0 条
  ptx st.shared                    72 条
```

## 2. 机制：为什么 cp.async 值 2 倍

mini 当前每个 KV tile 是：

```text
global load K/V
  -> 寄存器
  -> 写 shared（st.shared 72 条）
  -> ldmatrix
  -> MMA
```

下一步 tile 的 global load 必须等当前 tile 的 dot 结束附近才能开始；2 个 program 时
没有足够其他 warp 掩盖这个延迟。

vLLM 是：

```text
cp.async 把下一 tile 的 K/V/block-table 直接写 shared
  -> 当前 tile 继续 MMA
  -> async_wait 到点后消费 shared
```

也就是 **global load 与 MMA 重叠**。TTGIR 里的 `async_copy_global_to_local` /
`async_commit_group` / `async_wait` 就是这件事，不是某个手写公式。

## 3. 不是这些原因

已排除：

- **3D**：vLLM 目标 decode `max_query_len=3`，`IS_3D=false`；
- **BLOCK_M/BLOCK_Q/TILE_SIZE**：两边 16/2/32；
- **`v_keep`**：删除只从 58 spills 降到 48，约 4%；
- **num_warps**：4/8 都试过；
- **mask 表达式 / int1 / L=1 / S=zeros / K/V offset 写法**：都不触发完整 pipeline；
- **tile16 + 标量 block-table**：本地 0.1256 ms（约 -17%），但仍有 0.000976 的输出差，
  且 cp.async 仍是 0，不能作为生产替代；
- **`tl.range(num_stages=3)`**：能触发 8–14 条 cp.async，但本地耗时约 0.14–0.18 ms，
  比原 kernel 还慢；
- **`warp_specialize=True`**：当前 Triton 3.7.1 的 pipeline pass 直接失败。

## 4. 当前仓库状态

```text
生产 2D: 本地 kernel，255 regs / 0 spills，20,480 B shared
q3/ctx1024: 0.0522–0.0553 ms
vLLM 同输入: 0.0486–0.0969 ms（轮次波动）
```

K/V 统一 load layout 后，2D 直接 kernel 在短/中上下文、B>1 多数场景已不慢于 vLLM；
长上下文（ctx1024/2048）仍约 1.1–1.2×。真实 Graph 1024 同协议为 150.87 vs
vLLM 168.42 tok/s（单轮 0.896×；vLLM 153–168 波动）。之前 vendored vLLM kernel 的 0.047 ms 只作为
“目标上限”证据，已经 revert，不在生产代码里。

## 5. 当时的下一步（后来被 §10 的 layout 优化替代）

1. 让 block-table/K/V 三类 load 都成为 `ttg.async_copy_global_to_local`；
2. 先做 **2D-only、BF16-only** 的循环，排除 3D/FP32 分支对 pipeline pass 的干扰；
3. 把 `tl.range(num_stages=...)` 只作为实验开关，找到“14 条 cp.async”为什么没有
   覆盖 K/V 的原因（对比 vLLM 的 51 条）；
4. 检查循环里同时 live 的 `physical_block` 两种 layout：vLLM 用 shared 缓冲一份
   block-table，本地 TTGIR 出现两次 `tt.load`；这可能是 pipeline pass 的阻碍点；
5. 保持输出逐元素为 0.0 差，再谈真实 Graph。

## 6. 本轮追加的负结果

这些实验都保持输出逐元素差 0.0（除 tile16 明确标注），但没有触发完整 pipeline：

| 实验 | 结果 | 说明 |
|---|---|---|
| K/V 共用 base offset | 0.1509 ms，async 0 | 减少 block-table 两种 layout 的使用，但 compiler 仍复制 |
| 每 tile 两个标量 block load | 0.1528 ms，async 0 | 避免向量 block-table load，仍不 pipeline |
| 精确 query-mask / 去掉 keep tensor | 0.1504 ms，async 0 | 只略快，不触发 async |
| 去掉 3D + FP32 两个编译期分支 | 0.1505 ms，async 0 | 排除“分支阻止 pipeline” |
| vLLM 侧换入 mini 的 softmax / `L=0` / `S` 直接赋值 / S mask 加 tile_mask | async 10 不变 | 这些不是 vLLM pipeline 的触发点 |
| 2D `BLOCK_M=32`（q3 共享一次 KV 扫描） | 0.1752 ms，更慢 | K/V 读取减半但 acc/Q 翻倍，寄存器/占用率更差 |
| `TILE_SIZE=16` + 标量 block-table | 0.1256 ms，async 0 | 约 -17%，但输出差 0.000976，且没有 async |
| `TILE_SIZE=64` | 0.1411 ms，255 regs/4 spills | 略快，但没有 async，且 shared 增大到 73 KiB |

因此目前不能把差距归因于单一 `v_keep`、mask、`L`、`S`、query-mask 或 warps。
vLLM 的 pipeline 更像是 **整段 loop body 的代码形状** 触发的；下一步应保持 2D-only/BF16-only，
先把 loop body 缩到最小，再逐步加回 `K/V` load 和 block table，观察 `async_copy` 从 0 到 10
的那一步。

## 7. `tl.range` stage 扫描

同一个本地 kernel，只改 `tl.range(..., num_stages=N)`：

| stages | TTGIR async_copy | PTX cp.async | 时间 |
|---:|---:|---:|---:|
| 2 | 4 | 10 | 0.1575 ms |
| 3 | 6 | 14 | 0.1592 ms |
| 4 | 8 | 18 | 0.1589 ms |
| 5 | **10** | **22** | 0.1606 ms |
| vLLM | 10 | **51** | 0.0825 ms |

stage5 已经把 TTGIR 的 `async_copy` 数追到和 vLLM 一样的 10，但 PTX
`cp.async` 只有 22（vLLM 51），时间仍约 0.16 ms。说明：

- “有 async_copy”不等于“K/V 的三类 load 都被多 stage 覆盖”；
- 下一步不能只看 TTGIR 计数，要看每个 `async_copy` 的 operand 和 stage 深度；
- 目前仍是同步回退，生产不使用 `tl.range`。

## 8. operand 级发现：stage5 只流水化了 block-table

把 stage5 的每个 `async_copy` 按 operand 拆开：

```text
stage5:
  async_copy(physical_block) x 10
  async_copy(K)               x 0
  async_copy(V)               x 0

vLLM:
  async_copy(physical_block)
  async_copy(K_load)
  async_copy(V_load)
  ... 多 stage 重复
```

这解释了两件事：

1. 为什么 stage5 的 TTGIR `async_copy` 也是 10，但 PTX `cp.async` 只有 22：本地只拷贝了
   32 个 int32 的 block-table，没有拷贝 256×32 的 K 和 32×256 的 V；
2. 为什么 stage5 时间反而约 0.16 ms：pipeline 只包住了很小的 block-table load，
   K/V 仍然同步，stage 增加只增加了调度/shared 开销。

所以下一步目标非常具体：

```text
让 K_load / V_load 也变成 async_copy
```

而不是继续增加 block-table 的 stage 数。候选原因：

- K/V load 的 pointer/mask layout 没有让 Triton 推断出 `contiguity=8`（vLLM 的 async
  copy 带这个属性，本地 stage5 的 K/V 连 async 都没进）；
- K/V load 的 `tile_mask`/`dim_mask` 可能让 pipeline pass 只保留 block-table；
- 需要在 **2D-only/BF16-only** 的干净循环中逐项移除 mask/layout，找到 K/V 进入 async 的条件。

## 9. K/V async 尝试矩阵

所有变体都保持输出逐元素差 0.0（除 tile16 另行标注）：

| 变体 | async block-table | async K | async V | cp.async | 时间 |
|---|---:|---:|---:|---:|---:|
| stage5 基线 | 10 | 0 | 0 | 22 | 0.1606 ms |
| K/V 完全去 mask | 10 | 0 | 0 | 22 | 0.1705 ms |
| K/V 只留 head mask | 10 | 0 | 0 | 22 | 0.1530 ms |
| 去 mask + head contiguity hint | 10 | 0 | 0 | 22 | 0.1530 ms |
| tile16 + stage5 | 10 | 0 | 0 | 22 | 0.1411 ms |
| tile16 + BLOCK_M=8 + stage5 | 10 | 0 | 0 | 22 | 0.1557 ms |
| block-table 行 gather（loop 内无块表 load） | 0 | 0 | 0 | 0 | 0.1566 ms |

结论：

```text
去掉 K/V mask、缩小 tile、降低 BLOCK_M、给 head 加 contiguity hint、
把 block-table 改成寄存器 gather，都不能让 K/V 进入 async_copy。
```

所以 K/V 是否 async 不是 mask/shared/tile/block-table 单点决定，而是 K/V load 本身的
pointer/layout 表达没有满足 Triton pipeline pass 的条件。vLLM 的 K/V async copy 带
`contiguity=8`；本地所有变体的 K/V 都是普通 `tt.load`。

## 10. 突破：统一 K/V 的 load layout

之前 K 以 `[HEAD, TILE]` 指针加载，V 以 `[TILE, HEAD]` 加载；虽然数学一样，但
Triton 为两者生成不同的 shared layout，带来：

```text
198 regs / 40,960 B shared / 0.151 ms
```

改成 **K 也用 `[TILE, HEAD]` 加载，再 `tl.trans` 给 dot**：

```python
k_ptrs = key_cache + physical_block[:, None] * stride0 + ...
K = tl.trans(tl.load(k_ptrs, mask=...)).to(Q.dtype)
```

K/V 现在共用一种 load layout，编译器不再为 K 单独保留 `[HEAD, TILE]` 的
shared 布局。效果：

| 版本 | regs | shared | q3/ctx1024 | 输出差 |
|---|---:|---:|---:|---:|
| K `[HEAD,TILE]` + w8 | 198 | 40,960 | 0.151 ms | 0 |
| K `[TILE,HEAD]` + w8 | 140 | 25,600 | 0.068 ms | 0 |
| K `[TILE,HEAD]` + w4 | 180 | 25,600 | 0.066 ms | 0 |
| + 去掉 S mask 的冗余 tile_mask | 168 | 25,600 | 0.066 ms | 0 |

真实 Graph（1024 context，同协议）：

```text
w8 + K 原 layout：140.39 tok/s
w4 + K [TILE,HEAD]：157.39 tok/s（vLLM 163.23，0.964×）
```

**这里没有 cp.async**：K/V 仍是同步 load。收益来自 **消除 K/V 两套 layout 的
shared 转换和寄存器压力**。所以“vLLM 有 cp.async”是事实，但本地不一定非要
先复刻 cp.async 才能接近它；先把数据布局做对，已经能把 0.151 ms 降到 0.066 ms。

当前 2D 直接 kernel 在短/中上下文、B>1 多数场景已不慢于 vLLM；长上下文
（ctx1024/2048）仍约 1.1–1.5×，是下一阶段（K/V async 或长上下文专用策略）。

## 11. 长上下文选项：tile64 的收益与代价

在 K/V 统一 layout 后，试了更大的 `TILE_SIZE=64`：

| 版本 | ctx1024 | ctx2048 | 数值差 |
|---|---:|---:|---:|
| TILE32，当前生产 | 0.066 ms | 0.127 ms | 0 |
| TILE64 | 0.0595 ms | 0.107 ms | 0.0005–0.002 |
| TILE64 + BLOCK_M=8 | **0.053 ms** | **0.0996 ms** | 0.0005–0.002 |

tile64 确实把长上下文拉近了（2048: 0.127→0.107，BM8 到 0.0996），但：

- 输出不再是逐元素 0.0 差，而是 BF16 的 5e-4–2e-3；
- TILE64 的 shared 43,008 B，regs 255/2 spills；
- 它改变的是 online-softmax 的 tile 合并顺序，不是纯性能优化。

因此**没有生产化**：当前生产仍是 TILE32 + 精确 0.0 差。若未来接受 BF16 舍入差，
tile64+BM8 是长上下文的下一个候选，但必须先过真实模型/近 tie token 的差分门禁。

### 11.1 64 load + 两个 32 subtile（精确版）

为了同时拿到“64 宽 load”的吞吐和 TILE32 的数值顺序，用
`reshape + permute + tl.split` 把 K/V 的 64 宽 tile 拆成两个 32 宽 subtile，
按原顺序做两次 `_softmax_step`：

| ctx | TILE32 当前 | 64+2×32 精确 subtile |
|---|---:|---:|
| 128 | 0.0365 ms | **0.0260 ms** |
| 1024 | 0.0663 ms | 0.0680 ms |
| 2048 | 0.1277 ms | 0.1293 ms |

输出差：**0.0**。但代价是：

```text
regs: 255, spills: 18
shared: 24,576 B
```

短上下文有收益；**1024/2048 没有收益**，还引入 spill。说明长上下文的瓶颈不是
“tile 循环次数”，而是 K/V global load 延迟/async 覆盖；仅拆 subtile 不够。
这条实验只作为学习记录，未生产化。

### 11.2 为什么当前布局更快，但 K/V 仍不 async

当前生产的 K/V load 在 TTGIR 里是：

```text
K/V load: tensor<32x256xbf16, #blocked>
#blocked: sizePerThread=[1,1], order=[1,0]
```

它是**非合并的 scalar-per-thread 布局**，但省掉了 K/V 两套 shared layout 的转换，
所以整体仍然更快（198→168 regs，40.96→25.6 KiB shared）。

对照 vLLM：

```text
K_load: tensor<256x32, #blocked1> sizePerThread=[8,1]
V_load: tensor<32x256, #blocked2> sizePerThread=[1,8]
→ async_copy，带 contiguity=8
```

试过把 V 也改成 `[HEAD, TILE]` load 再 transpose：

```text
0.150 ms，255 regs/22 spills，async 0
```

更慢，因此没有采用。结论：

- **当前最优不是“coalesced load”**，而是“没有 layout conversion 的 load”；
- K/V async 需要的是 **既 coalesced、又能被 pipeline pass 选中** 的 load；
- 下一步可以用 `tl.make_block_ptr` 或显式 layout hint 做实验，而不是继续切 transpose。

### 11.3 BLOCK_M=8：用更多 program 换长上下文

在 K/V layout 后，把 GQA group8 的 `BLOCK_M` 从 16 改成 8（`BLOCK_Q=1`），
q3 从 2 个 program 变成 3 个 program：

| ctx | BLOCK_M16 | BLOCK_M8 |
|---|---:|---:|
| 128 | 0.0507 ms | 0.0567 ms（变慢） |
| 256 | 0.0269 ms | 0.0454 ms（变慢） |
| 1024 | 0.0662 ms | **0.0647 ms** |
| 2048 | 0.1271 ms | **0.1210 ms** |
| hd512 full | 0.1128 ms | **0.0985 ms** |
| hd512 sliding | 0.0935 ms | **0.0622 ms** |

输出差：**0.0**；shared 25,600 → 20,992 B；regs 168 不变。

真实 Graph 1024：

```text
BLOCK_M8：148.83 vs vLLM 153.42（0.89–0.96×，不同轮）
BLOCK_M16：145.35–157.39 vs vLLM 163.07（0.89–0.96×）
```

这是一个**明确 tradeoff**：短上下文多 program 有启动/重复扫描成本，长上下文和
hd512 受益。目标场景是 1024 context，因此生产选 BLOCK_M8；短上下文回退是已知边界。

### 11.4 最终生产：TILE64 load + 2×32 subtile + BM8

把 11.1 的精确 subtile 与 BM8 结合后，长上下文终于有收益，且输出仍是 **0.0 差**：

| ctx | TILE32 + BM8 | TILE64 subtile + BM8 |
|---|---:|---:|
| 128 | 0.0373 ms | **0.0317 ms** |
| 256 | 0.0454 ms | **0.0364 ms** |
| 1024 | 0.0647 ms | **0.0522 ms** |
| 2048 | 0.1210 ms | **0.0990 ms** |
| hd512 full | 0.0985 ms | 0.0975 ms（head>256 仍 TILE32） |

```text
regs: 255 / spills: 0 / shared: 20,480 B
输出差: 0.0
```

机制：

```text
一次 load 64 宽 K/V
→ reshape/permute/split 成两个 32 宽 subtile
→ 按 TILE32 的顺序做两次 softmax/acc
```

这样 K/V load 次数减半，但 online-softmax 的合并顺序没变，所以不是 tile64 的
BF16 舍入差。最终 22-case：

```text
ctx128 0.0403 vs 0.0457 ms（mini 快）
ctx256 0.0447 vs 0.0575 ms（mini 快）
ctx512 0.0542 vs 0.0783 ms（mini 快）
ctx1024 0.0553 vs 0.0492 ms（约 1.12×）
ctx2048 0.1036 vs 0.0862 ms（约 1.20×）
hd512  0.1015 vs 0.0741 ms（约 1.37×）
```

真实 Graph 1024：mini 150.87 vs vLLM 168.42 tok/s（0.896×）；vLLM 本轮偏高
（153–168 波动），绝对 mini 从 148.83 提到 150.87。短/中上下文已不慢于 vLLM，
剩余是长上下文/hd512 的 1.2–1.4×。

### 11.5 hd512 的 BM4 实验

对 head_size=512 试 `BLOCK_M=4`：

| 版本 | hd512 full | hd512 sliding B4 |
|---|---:|---:|
| BM8 / w4（当前） | 0.0979 ms | 0.0594 ms |
| BM4 / w4 | 0.0987 ms（255 regs / 28 spills） | 0.0594 ms |
| BM4 / w8 | 0.1055 ms（198 regs / 0 spills） | **0.0562 ms** |

结论：

- BM4 在 hd512 full 上没有收益；
- BM4+w8 只对 sliding B4 略快 5%，但 full 更慢；
- 因此没有生产化，hd512 仍保持 BM8/4 warps + TILE32。

这条负结果说明：hd512 的剩余差距不能靠继续减小 BLOCK_M 解决，还是 load/流水问题。

### 11.6 K/V cache policy 实验

给 K/V load 加 `.cg`、`evict_first`、`evict_last`（输出全部 0.0 差）：

| ctx | 默认 | `.cg` | `evict_first` | `evict_last` |
|---|---:|---:|---:|---:|
| 1024 | 0.0709 ms | 0.0662 | 0.0591 | **0.0582** |
| 2048 | 0.0986 ms | 0.0982 | 0.1106 | 0.1110 |

1024 上 evict 看似更快，但同一轮默认值本身波动到 0.0709（历史 0.0522）；2048 上
evict 明显更慢。结论：

- cache policy 不是稳定收益；
- 不能为了单轮 1024 数字生产化；
- K/V 长上下文瓶颈仍是 load 延迟/async，而不是 L1/L2 policy。

### 11.7 tail-mask 优化尝试

把 2D 的 K 改成完全无 mask load，V 先无 mask load 再
`tl.where(tile_mask, V, 0)`：

| ctx | 默认 | 无 K mask + V where |
|---|---:|---:|
| 128 | 0.0457 ms | **0.0397 ms** |
| 1024 | 0.0586 ms | **0.0546 ms** |
| 2048 | 0.0994 ms | 0.0993 ms |
| hd512 | 0.0976 ms | 0.1029 ms（变慢） |

输出差都是 0.0，regs 254/0 spills。短上下文和 1024 有 7–13% 收益，但 2048 持平、
hd512 变慢，且 tail 安全性依赖 `V=0` 的 where。由于不是全局收益，**未生产化**；
当前仍保留带 mask 的 K/V load，尾块语义更直接。

## 12. 复现

```bash
PY=/home/user/anaconda3/envs/vllm-omni-dev/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
$PY benchmarks/check_step88_attention_pipeline.py \
  --case qlen3_ctx1024 --warmup 50 --repeats 200 --rounds 10
```

输出会同时打印 mini/vLLM 的 regs、spills、shared、TTGIR async 计数、PTX cp.async 计数
和单 kernel 时间，便于继续做源码对照。
