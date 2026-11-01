# 88 后续：vLLM 2D attention 的 software pipeline 到底优化了什么

> 脚本：`benchmarks/check_step88_attention_pipeline.py`
> 原始结果：`benchmarks/results/step88/attention_pipeline/summary.json`（gitignored）
> 当前生产：本地 2D kernel，198 regs / 0 spills；**没有 vendored vLLM 代码**。

## 1. 先看结论

同一组 q、paged K/V、block table，`q_len=3/context=1024/head_dim=256`：

```text
mini:  0.1511 ms，198 regs，0 spills
vLLM:  0.0825 ms，158 regs，0 spills
输出差：0.0
```

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
生产 2D: 本地 kernel，198 regs / 0 spills
q3/ctx1024: 0.1511 ms
vLLM 同输入: 0.0825 ms（本轮；此前 0.053–0.083 波动）
```

因此当前是 **约 1.8×**，不是已追平。之前 vendored vLLM kernel 的 0.047 ms 只作为
“目标上限”证据，已经 revert，不在生产代码里。

## 5. 下一步（本地实现，不复制）

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

## 10. 复现

```bash
PY=/home/user/anaconda3/envs/vllm-omni-dev/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
$PY benchmarks/check_step88_attention_pipeline.py \
  --case qlen3_ctx1024 --warmup 50 --repeats 200 --rounds 10
```

输出会同时打印 mini/vLLM 的 regs、spills、shared、TTGIR async 计数、PTX cp.async 计数
和单 kernel 时间，便于继续做源码对照。
