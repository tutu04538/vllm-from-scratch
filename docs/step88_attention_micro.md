# 88 后续：2D 长上下文 attention 并行度 microbenchmark

> 目标：把 `docs/step88_execution_report.md` §8 的“2D 长上下文 attention”假设变成可测的结论。
> 脚本：`benchmarks/check_step88_attention_micro.py`
> 原始结果：`benchmarks/results/step88/attention_micro{,_flush}/summary.json`（gitignored）
> **注意：本文件回答“mini 2D → 3D 是否有收益”，不回答“mini 为什么比 vLLM 慢”。**
> 后者见 `docs/step88_vllm_kernel_diff.md`：vLLM 实际也走 2D，修复前 mini 2D kernel
> 255 regs/58 spills 对 vLLM 158 regs/0 spills；2D 第一版修复后为 198 regs/0 spills。

## 1. 测什么

只测 `minivllm/attention/ops/triton_unified_attention.py::unified_attention` 本身，不加载
模型、不跑 engine，避免把 scheduler/输入组装/Graph 混进来。

```text
q_len      1 / 2 / 3 / 4 / 8
context    128 / 256 / 512 / 1024 / 2048
batch      1 / 2 / 4 / 8
head_dim   256 / 512
window     full (-1) / sliding 512
segments   4 / 8 / 16 / 32
```

两边输入完全相同：paged K/V cache、连续 block table、causal、BF16、同一 3 个随机 seed。
比较：

- `2D`：生产当前路径，`max_seqlen_q=q_len`；
- `3D_forced`：**只为诊断**传 `max_seqlen_q=1`，强制走 3D segment+reduce；
  production `can_use_3d()` 仍拒绝 `q_len>1`，本报告不改变生产行为。

计时：50 次 warmup；10 轮 × 20 次 call 的 CUDA event burst，报每轮的中位数；另跑一组
每轮前用 32 MiB buffer 清 L2 的 `--flush-l2` 对照。

## 2. 先看 grid 数学

Gemma4 target：8 个 query head / 1 个 KV head → GQA group=8，`BLOCK_M=16`，
`BLOCK_Q=2`。

B=1、q_len=3：

```text
2D: q_blocks = 3 // 2 + 1 = 2
    programs = 2 × 1 KV head = 2
    每个 program 串行扫 ceil((1024+3)/32) ≈ 33 个 KV tile

3D: programs = 2 × 1 × 16 segments = 32
    每个 program 只扫 ceil(1027 / (16×16)) ≈ 5 个 tile
    再 reduce_segments 合并
```

B=4、q_len=3：

```text
2D: 12 // 2 + 4 = 10 programs
3D: 10 × 1 × 16 = 160 programs
```

**sequence 不是 2D grid 的维度**；上下文变长只增加每个 program 的串行 tile 数。这正是要
验证的“并行度”问题。

## 3. 正确性

`3D_forced` 在原 production 入口之外绕过 `max_seqlen_q==1` 检查，因此必须先证明它不只是
“快而错”。22 个 case 每个跑 3 个 seed，和 2D 输出比较：

```text
全部 case: correctness=True
max_abs  ≤ 0.002
mean_abs ≤ 0.0002
```

覆盖了 q_len=1/2/3/4/8、B=1/2/4/8、context 128..2048、head_dim 256/512、
full/sliding 512。**至少在这个 synthetic 分布上，3D kernel 支持 q_len>1。**

## 4. 结果

### 4.1 q_len 固定，context 扫描（q_len=3，B=1，hd=256，full）

| context | 2D ms | 3D ms | speedup（无 flush） | speedup（flush） |
|---:|---:|---:|---:|---:|
| 128 | 0.0421 | 0.0695 | **0.607** | **0.532** |
| 256 | 0.0540 | 0.0599 | 0.901 | 0.818 |
| 512 | 0.0989 | 0.0583 | 1.696 | 1.336 |
| 1024 | 0.1900 | 0.0700 | 2.716 | 3.239 |
| 2048 | 0.3688 | 0.0799 | **4.614** | **4.370** |

结论：**短上下文 3D 有 segment/reduce 固定开销，反而更慢；约 512 起转正，1024/2048 明显
占优。** 这解释了为什么不能无条件把 K=2 改成 3D。

### 4.2 q_len / batch（context=1024，hd=256，full）

| 配置 | 2D ms | 3D ms | speedup（无 flush） |
|---|---:|---:|---:|
| q_len=1, B=1 | 0.1849 | 0.0647 | 2.859 |
| q_len=2, B=1 | 0.1881 | 0.0677 | 2.779 |
| q_len=4, B=1 | 0.1897 | 0.0667 | 2.843 |
| q_len=8, B=1 | 0.1900 | 0.0771 | 2.464 |
| q_len=3, B=2 | 0.1908 | 0.0771 | 2.475 |
| q_len=3, B=4 | 0.1912 | 0.0714 | 2.676 |
| q_len=3, B=8 | 0.1911 | 0.0707 | 2.705 |

2D 时间在 0.185–0.191 ms 附近几乎不变：**增加 query 行/batch 没有把单 program 的串行 KV
循环变短**；3D 把 KV 分段后稳定降到约 0.06–0.08 ms。

### 4.3 head_dim / sliding window（context=1024，B=1，q_len=3）

| 配置 | 2D ms | 3D ms | speedup（无 flush） | speedup（flush） |
|---|---:|---:|---:|---:|
| hd256 full | 0.1897 | 0.0650 | 2.918 | 3.393 |
| hd256 sliding512 | 0.0928 | 0.0340 | 2.734 | 1.614 |
| hd512 full | 0.3536 | 0.0785 | **4.505** | **5.996** |
| hd512 sliding512 | 0.1842 | 0.0486 | 3.791 | 2.682 |

head_dim 越大，2D 的串行负载越重，3D 分段收益越大；sliding window 把 KV 扫描减半，但 3D
仍占优（尤其 hd512 full）。

### 4.4 segment 数（context=1024，q_len=3，B=1，hd=256）

| segments | 3D ms（无 flush） | speedup |
|---:|---:|---:|
| 4 | 0.0713 | 2.670 |
| 8 | 0.0756 | 2.508 |
| 16 | 0.0945 | 2.019 |
| 32 | 0.0783 | 2.434 |

这组抖动较大，**没有足够证据说 4/8/16/32 中哪个稳定最优**；当前生产默认 16 不属于明显
错误，但生产化时应把 segment 数作为独立调参项，而不是从这张表直接定死。

## 5. 这次定位到了什么

### 结论

**实际要改的是 2D kernel 的 KV 串行扫描，不是 GEMM、RMSNorm、RoPE，也不是再减少 CPU launch。**

证据：

1. 单 kernel microbench 里，2D q_len=3 / context 1024 约 0.19 ms，3D 约 0.06–0.07 ms；
   context 2048 达到 4.4–4.6×；
2. 2D 时间对 q_len（1→8）和 B（1→8）几乎不敏感，说明瓶颈不是“query 行不够”本身，
   而是每个 program 内部扫完整 KV 的串行长尾；
3. 3D 把 KV 切成 16 段后，program 数从 2 变成 32（B=1）或 10 变成 160（B=4），
   单 program tile 数从 33 降到 5；
4. 在真实 engine trace 里，同一现象表现为 `kernel_unified_attention` 4.17→50.52 ms，
   而 GEMM/RMSNorm/RoPE 基本不变（`docs/step88_execution_report.md` §6.3）。

### 不能直接说“改 3D 就完了”

- 本 microbench 是**单 kernel、synthetic、无 Graph**；真实 target 有 35 层串行、
  sliding/full 混合、Graph replay、metadata/workspace 所有权；
- `can_use_3d()` 当前硬性要求 `max_seqlen_q==1`；放开它需要同时改 metadata/graph
  capture 和 3D workspace 生命周期；
- 短上下文 3D 更慢，必须按 `seq_len`（或 2D tile 数）设阈值；
- segment 数没有稳定最优，不能从这组 10 轮数据直接选 4 或 32；
- 数值只在 3 个 seed、synthetic 分布上通过；生产前需要真实 Gemma4 target/assistant
  eager/Graph 对照。

## 6. 建议的下一步

**不是立即改生产 kernel**，而是先做一个受控的真实 engine 原型：

1. 保留 2D 作为默认；只在 `seq_len >= 512` 且 q_len>1 时尝试 3D；
2. 复用现有 `TritonAttentionWorkspacePool`，确认 metadata builder、Graph capture、
   reduce_segments 都在图内可重放；
3. 跑真实 Gemma4 B=1 K=2，短/1024/2048 三种上下文，eager 与 FULL Graph 都比 2D/3D；
4. 先看 `kernel_unified_attention` 是否真的从 50 ms 降到约 15–20 ms（按 3D 单 kernel
   收益估算），再决定是否合入；
5. 若真实 Graph 没有收益，问题可能转向 workspace 拷贝/Graph bucket/短 tile 开销，
   不要继续盲目调 segment 数。

## 7. 复现

```bash
PY=/home/user/anaconda3/envs/vllm-omni-dev/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

$PY benchmarks/check_step88_attention_micro.py --warmup 50 --repeats 200 --rounds 10
$PY benchmarks/check_step88_attention_micro.py --warmup 50 --repeats 200 --rounds 10 --flush-l2
$PY -m pytest tests/step88/test_attention_micro.py -q
```

结果目录（不跟踪）：

```text
benchmarks/results/step88/attention_micro/summary.json
benchmarks/results/step88/attention_micro_flush/summary.json
```

## 8. 一句话

**2D 长上下文的瓶颈不是“query 行数”或 launch，而是每个 program 串行扫完整 KV；在
q_len=3 上用 3D 分段原型可把单 kernel 在 1024/2048 上下文加速约 2.7–4.6×，但 128/256
会变慢，且生产化还要过 Graph/metadata/真实模型三道门。**

补充：这并不能解释当前 mini-vs-vLLM 的 Graph 差距；vLLM 目标 decode 也走 2D
（`max_query_len=3`、`TILE_SIZE=32`、`BLOCK_M=16`、`IS_3D=false`），真正的差距是
mini 2D kernel 修复前 255 regs/58 spills 对 vLLM 158 regs/0 spills；2D 第一版修复后为
198 regs/0 spills，见 `docs/step88_vllm_kernel_diff.md`。
