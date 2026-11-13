# step111：四卡公平性能矩阵与通信归因

状态：初次矩阵与回归已完成，存在 3 组失败，尚未完成验收。前置 110 已完成；本关的任何速度结果都必须先经过正确性门禁。

## 冻结源码与项目职责

冻结 vLLM 0.28 的 `v1/executor/multiproc_executor.py::MultiprocExecutor.execute_model` 将同一份 Scheduler 输出发往所有 Worker，并只从 output rank 收取回复；项目 `minivllm/executor/multiproc_executor.py` 对应同一进程边界。冻结 `v1/worker/gpu/model_runner.py::GPUModelRunner.execute_model` 先校正请求镜像、块表与本轮 batch 描述，再分派 eager 或 Graph 并执行本地模型；项目 V2 Runner 具有相同职责。输入按本轮实际调度 token 展平，logits 按需采样行组织，参数与 KV 在 TP rank 本地分片；Scheduler 独占请求和逻辑 KV，benchmark 只观察结果，不能重建或替换热路径输入。

| 所有者 | 状态 | 测量边界 |
|---|---|---|
| Scheduler | 请求、逻辑 KV、提交 token | 只统计最终提交的输出，不能把每个 rank 或被拒候选重复计数。 |
| Worker / Runner | 本地权重、KV、工作区、图 | 每 rank 单独采样内存；capture 和加载不计入稳态推理。 |
| Executor | rank 生命周期、RPC 与通信 | 每个 case 独立进程，串行、超时有界，退出后检查资源。 |

固定模型 8B TP1/2/4、14B TP2/4，BF16、同步、prefix off、无 compile、单副本；同一模型/TP/GPU 集合与 token trace 下比较 mini 和冻结上游。延迟、单副本吞吐、吞吐/GPU 和容量分别报告。warmup 至少一次、正式重复至少三次，并记录每次结果；trace 独立于正式计时，异步 inclusive 耗时不能直接相加。历史 105 的 Worker 关闭超时仍是未定位风险。

正在实现的测量驱动使用独立正确性进程：先重复冻结参考、登记阈值，再与 mini 的每 rank 采样状态及张量比较，只有通过的 case 才进入正式计时。K0 复用 103 探针，K2 复用 106/109 探针；计时进程不安装这些 D2H 诊断 hook。真实 8B TP1 B1 K0 eager 和 8B TP2 B1 K2 Graph 的短输入试运行已通过门禁及各三次计时；这两点不代替完整矩阵。

首次 8B TP1、1024 输入诊断使用了同 TP 重复所得到的 `1e-5` 门槛，mini 的第 0 层有效 KV 有 11 个 BF16 元素不同，后续 hidden/logits 差异被放大，32 个贪心输出 token 相同；该轮失败保留于 `benchmarks/results/step111/smoke_8b_tp1_1024*`。随后冻结参考在同一历史下的 TP1/2/4 对照也出现累积差异。这个诊断揭示初始登记漏掉了 002 要求的跨 TP 参考，不据此调整阈值或把失败改为通过。

正式长输入与混合负载改用预先固定的 `step111-v2` token ID 序列：对 1024 行相对首次诊断的固定循环序列起点移动一位，短输入保持原序列。先仅运行冻结参考的所有合法 TP、每个 TP 两次，按相同历史的 hidden、logits 与按 head 拼接的逻辑有效 KV 登记每对象最大绝对误差和相对 L2 门槛，并保存全部参考的哈希；校准完成后才允许 mini 运行该新 fixture。登记算法不读取 mini 结果；初次诊断和正式 fixture、产物必须分别保留。物理块在不同 allocator 下通过稳定的一一映射核对，原始块号和不同清理轮次都保留；这与 [103](step103_alignment.md) 的逻辑 KV 比较口径一致。

16 请求负载首次冻结参考发现 TP1/2/4 的部分贪心输出在第 9 个 token 或之后分歧，而各自同 TP 重复一致；初版校准因此失败，记录保留在 `benchmarks/results/step111/calibration_8b_b16.json`。当前只在每请求前四个输出历史均一致时登记对应阶段的数值误差，若更早分歧则必须先做固定历史重放；后续跨 TP 的输出分歧逐请求保留，不能写成跨 TP 逐 token 等价。同 TP 比较若完整输出不同，必须通过独立的首分歧检查，才能列为非严格等价观察。

8B TP1 16 请求首次同 TP 对照中，前四轮数值均在预登记阈值内，5 个请求在第 9 个 token 或之后分歧。独立重跑逐请求保存首分歧处每 rank 的 top-2、logits dtype 和输入/输出历史，确认这五处至少一侧是精确的 BF16 分数平局（例如请求 15 的 mini 两个候选均为 35.25）；原失败与独立诊断分别留在 `smoke_8b_tp1_mixed16_v2.json`、`firstdiff_8b_tp1_mixed16_*_v3.json`。这只能支持将相应性能单列为 `strict_equivalent=false` 的观察，不能写成逐 token 相同。

正式 `calibration_full_v3.json` 包含四组、20 次冻结运行。8B 的两组阈值与各自在首次 mini 比较之前登记的合同逐项一致，没有增大阈值；历史合同及首次失败仍保留。14B 两组此前未运行 mini。合同及比较结果属于本机 `benchmarks/results/step111/`，不加入 Git。


## 矩阵进行中的失败

8B TP4、16 请求混合输入的数值门禁失败：同一请求首个输出的 hidden state 在四个 rank 上最大绝对差为 1.125，超过预登记上限 1.0；相对 L2 为 0.02501，低于其上限 0.05302。输出也有分歧，但由于数值门禁未通过，独立平局检查拒绝将该组列为 near-tie 观察。原始结果和诊断保存在 `benchmarks/results/step111/full_matrix_runs/8b_tp4_b16_n1024_k0_eager_*`；不增大门限，不把该组放入性能主表，后续需定位差异。

14B TP2、16 请求混合输入也未通过：请求 11 的第 4 个输出 hidden state 最大绝对差 1.265625，超过预登记 1.125；请求 15 首个输出的 hidden 与 logits 相对 L2 分别为 0.11190 和 0.10123，超过 0.10038 和 0.09075。两 rank 均复现，平局检查因数值门禁失败而拒绝；原始产物保留于 `benchmarks/results/step111/full_matrix_runs/14b_tp2_b16_n1024_k0_eager_*`。

14B TP4、16 请求混合输入的 32 token 初始状态门禁归为平局观察，但正式 128 token 输出存在分歧；独立首分歧检查中请求 9 的第 58 个输出位置，参考 top-2 为 374/4588（30.875/30.75），mini 为 4588/374（31.0/30.875），两边均非精确 BF16 平局，因此状态为 `failed_output_divergence`，未进入正式性能主表。结果和重跑保留于 `benchmarks/results/step111/full_matrix_runs/14b_tp4_b16_n1024_k0_eager_*`。

## 完整性能矩阵（初次运行）

硬件为四张 RTX 4090，逻辑 rank 对应 `CUDA_VISIBLE_DEVICES=2,3,0,1`。表内为三次正式测量中位数；完整三次值、min/max、TPOT、inter-burst、排队和逐 rank 内存见 `benchmarks/results/step111/full_matrix.json`。输出吞吐只统计每个请求最终提交的 token，并以一次端到端执行的 wall time 为分母。`观察`表示经过首分歧平局复核但不严格逐 token 等价；失败组不列性能数值。

| case | 状态 | mini tok/s | vLLM tok/s | mini TTFT ms | vLLM TTFT ms |
|---|---|---:|---:|---:|---:|
| 8b_tp1_b1_n128_k0_eager | 严格通过 | 50.77 | 54.11 | 25.00 | 24.19 |
| 8b_tp1_b1_n1024_k0_eager | 严格通过 | 49.85 | 46.56 | 197.67 | 194.69 |
| 8b_tp1_b16_n1024_k0_eager | 观察 | 437.66 | 438.05 | 675.21 | 664.96 |
| 8b_tp1_b1_n128_k2_eager | 严格通过 | 51.19 | 51.34 | 24.57 | 24.04 |
| 8b_tp1_b1_n128_k0_graph | 严格通过 | 51.17 | 50.74 | 54.82 | 56.29 |
| 8b_tp1_b1_n128_k2_graph | 严格通过 | 49.57 | 48.94 | 55.28 | 57.42 |
| 8b_tp1_b4_n128_k0_eager | 严格通过 | 181.26 | 165.22 | 80.22 | 80.57 |
| 8b_tp1_b4_n128_k2_eager | 严格通过 | 165.76 | 185.16 | 81.02 | 77.85 |
| 8b_tp1_b4_n128_k0_graph | 严格通过 | 186.74 | 183.76 | 122.38 | 124.56 |
| 8b_tp1_b4_n128_k2_graph | 严格通过 | 186.65 | 188.39 | 111.23 | 120.74 |
| 8b_tp2_b1_n128_k0_eager | 严格通过 | 37.43 | 34.11 | 26.84 | 29.49 |
| 8b_tp2_b1_n1024_k0_eager | 严格通过 | 34.85 | 33.19 | 213.44 | 223.17 |
| 8b_tp2_b16_n1024_k0_eager | 观察 | 368.37 | 389.20 | 675.89 | 705.35 |
| 8b_tp2_b1_n128_k2_eager | 严格通过 | 50.38 | 40.34 | 21.46 | 32.04 |
| 8b_tp2_b1_n128_k0_graph | 严格通过 | 84.74 | 88.14 | 56.47 | 43.14 |
| 8b_tp2_b1_n128_k2_graph | 严格通过 | 80.28 | 79.78 | 55.12 | 55.08 |
| 8b_tp2_b4_n128_k0_eager | 严格通过 | 169.50 | 135.24 | 69.51 | 92.39 |
| 8b_tp2_b4_n128_k2_eager | 严格通过 | 181.55 | 153.96 | 73.55 | 86.88 |
| 8b_tp2_b4_n128_k0_graph | 严格通过 | 293.99 | 289.80 | 109.87 | 126.83 |
| 8b_tp2_b4_n128_k2_graph | 严格通过 | 272.24 | 274.44 | 110.81 | 109.46 |
| 8b_tp4_b1_n128_k0_eager | 严格通过 | 42.98 | 42.61 | 26.18 | 24.42 |
| 8b_tp4_b1_n1024_k0_eager | 严格通过 | 43.53 | 32.78 | 177.60 | 237.25 |
| 8b_tp4_b16_n1024_k0_eager | 数值门禁失败 | — | — | — | — |
| 8b_tp4_b1_n128_k2_eager | 严格通过 | 40.51 | 32.32 | 29.73 | 34.31 |
| 8b_tp4_b1_n128_k0_graph | 严格通过 | 122.17 | 130.51 | 53.24 | 29.76 |
| 8b_tp4_b1_n128_k2_graph | 严格通过 | 103.89 | 107.70 | 50.77 | 51.44 |
| 8b_tp4_b4_n128_k0_eager | 严格通过 | 215.62 | 196.40 | 60.96 | 66.09 |
| 8b_tp4_b4_n128_k2_eager | 严格通过 | 153.84 | 184.39 | 72.94 | 75.03 |
| 8b_tp4_b4_n128_k0_graph | 严格通过 | 397.56 | 410.73 | 109.18 | 108.52 |
| 8b_tp4_b4_n128_k2_graph | 严格通过 | 346.38 | 350.44 | 109.96 | 118.46 |
| 14b_tp2_b1_n128_k0_eager | 严格通过 | 43.65 | 45.93 | 31.86 | 31.51 |
| 14b_tp2_b1_n1024_k0_eager | 观察 | 33.56 | 32.68 | 261.52 | 255.15 |
| 14b_tp2_b16_n1024_k0_eager | 数值门禁失败 | — | — | — | — |
| 14b_tp2_b1_n128_k2_eager | 严格通过 | 61.71 | 53.78 | 31.97 | 32.20 |
| 14b_tp2_b1_n128_k0_graph | 严格通过 | 51.86 | 52.37 | 64.63 | 68.01 |
| 14b_tp2_b1_n128_k2_graph | 严格通过 | 64.71 | 64.53 | 65.52 | 70.68 |
| 14b_tp2_b4_n128_k0_eager | 严格通过 | 181.13 | 147.36 | 101.32 | 101.44 |
| 14b_tp2_b4_n128_k2_eager | 严格通过 | 215.27 | 179.85 | 102.78 | 107.06 |
| 14b_tp2_b4_n128_k0_graph | 严格通过 | 187.75 | 190.28 | 143.55 | 147.13 |
| 14b_tp2_b4_n128_k2_graph | 严格通过 | 222.65 | 222.85 | 139.13 | 141.44 |
| 14b_tp4_b1_n128_k0_eager | 严格通过 | 37.58 | 32.76 | 27.84 | 30.64 |
| 14b_tp4_b1_n1024_k0_eager | 观察 | 33.41 | 27.72 | 259.51 | 273.37 |
| 14b_tp4_b16_n1024_k0_eager | 输出分歧失败 | — | — | — | — |
| 14b_tp4_b1_n128_k2_eager | 严格通过 | 62.06 | 57.54 | 27.03 | 26.98 |
| 14b_tp4_b1_n128_k0_graph | 严格通过 | 78.69 | 79.99 | 62.42 | 62.64 |
| 14b_tp4_b1_n128_k2_graph | 严格通过 | 93.97 | 95.64 | 60.09 | 61.91 |
| 14b_tp4_b4_n128_k0_eager | 严格通过 | 141.33 | 167.65 | 90.35 | 85.26 |
| 14b_tp4_b4_n128_k2_eager | 严格通过 | 233.41 | 214.62 | 87.65 | 86.66 |
| 14b_tp4_b4_n128_k0_graph | 严格通过 | 272.69 | 274.35 | 127.92 | 133.89 |
| 14b_tp4_b4_n128_k2_graph | 严格通过 | 309.12 | 312.54 | 133.32 | 126.49 |

首次复现：`followup_v2/8b_tp4_b16_n1024_k0_eager_layers_comparison.json` 中请求 12、位置 1023 的 rank 0 第 0 层 input norm 和 QKV 完全相同，attention 首次出现最大约 1.19e-7 差异，经过 o_proj 的 BF16 舍入及后续层放大。该观察只能定位最早差异，不能单凭它认定错误属于某一实现；还需检查完整重跑和上游对应 attention 算子。

14B TP2 的请求 15、位置 1023 在第 0 层各观测输出一致，第 1 层 attention 第一次出现差异，最大 0.00048828125，后续逐层放大。两组原始首层有效 KV 也有微小差异，单行 QKV 一致不证明全部历史 KV 一致，因此这些数据还不能定位到 attention kernel 本身。逐层产物位于 `benchmarks/results/step111/followup_v2/`。

后续在相同负载重跑并保存完整 logits：14B TP4 四个请求首分歧处，每个请求此前的 token 历史一致，各 rank 的最大绝对差分别为 0.4375、0.25、0.375、0.375，相对 L2 分别约为 0.0186、0.0142、0.0067、0.0165。请求 9 两边各自领先 0.125；完整证据保存在 `benchmarks/results/step111/followup_v2/14b_tp4_b16_full_logits_comparison.json`。这些生成后期位置没有预登记的逐位置合同，观察不替代门禁，状态仍为失败。

历史 KV 差异位置 850 的首次 RoPE 钩子结果无效：冻结 vLLM 多层共享同一个 RoPE 模块，未按调用层筛选的钩子会被后续层覆盖。原错误产物保留在 `benchmarks/results/step111/kv_followup/`；修正探针后另存 `kv_followup_v2/`，不把旧差异当成模型结果。

修正层过滤后，8B TP4 请求 12、位置 850 的实际位置值一致，input norm、QKV、Q/K norm 完全一致；两套引擎持有的 RoPE 表在该位置最大差 0.0009765625，旋转后 key 差 0.00390625，与此前首层 KV 最大差一致。项目的 `_canonical_device(None)` 固定选择 CPU；模型构造后 `_apply` 按目标设备重新生成表，而该操作已离开 CUDA 默认设备上下文，因而用 CPU FP32 三角函数重新生成后搬到 GPU。冻结 vLLM 在 CUDA 默认设备上下文中生成表。独立同参 CPU/GPU 表复算在位置 850 也出现 BF16 差异。证据说明这个 RoPE 表差异是已观测 KV 差异的直接来源；是否解释所有失败还需修复后重跑。见 `benchmarks/results/step111/kv_followup_v3/`。

## 通信、拓扑和性能观察（修复 RoPE 前）

正式计时与 trace 分开。下面均为本次机器、输入和设备映射的观察；NUMA 使用同一 14B TP2 单请求 eager 输入，三次中位数：GPU `[0,1]` 时 mini 53.04 tok/s、冻结 vLLM 45.75 tok/s，GPU `[0,2]` 时分别为 46.10 和 45.77 tok/s。TTFT 分别为 mini 31.65/32.22 ms、上游 31.45/31.85 ms。它只能描述这两组测量；并未证明性能差完全由 NUMA 引起。

按 rank 统计，两个 NUMA trace 各有 10,368 次 all-reduce（输入 tensor 合计 211,507,200 字节）和 128 次 all-gather（19,447,808 字节）；这些是调用输入字节，不是链路传输字节。同 NUMA rank0/rank1 的 NCCL GPU interval 和分别为 105.1/828.6 ms，跨 NUMA 为 236.9/656.8 ms；这些区间包含等待、可能跨流重叠，不能相加作为端到端通信开销。同 NUMA 的两个 rank GEMM/GEMV 区间和均约 1.93 s，attention 约 13.5–13.7 ms。分类依据 kernel 名称，未识别项保留 other；首次分类遗漏 `gemvx` 的结果保留，修订后的八份分析位于 `benchmarks/results/step111/followup_v2/*_analysis_v2.json`。

14B TP2 EAGLE3 Graph 的独立 trace 中，每 rank 由图重放观察到 8,355 次 all-reduce（输入 250,705,920 字节）和 193 次 all-gather（输入 6,176,000 字节）；捕获期间的 host 发起记录单列，不能与 replay 调用重复计数。图节点通过捕获 NVTX 和 CUDA graph node 映射，映射不唯一时分析器报错。

在 B1、输入 128、输出 128 的 Graph 组，mini 8B TP1/2/4 的 K0 中位吞吐分别为 51.17/84.74/122.17 tok/s，K2 为 49.57/80.28/103.89 tok/s；同条件 14B TP2/4 的 K0 为 51.86/78.69 tok/s，K2 为 64.71/93.97 tok/s。此处 EAGLE3 对 14B 有收益，对该 8B 负载没有收益，不代表所有输入。14B BF16 能在 TP2 实际加载并运行，说明本机这一配置具备所需容量；没有 14B 单卡实测基线。16 请求混合吞吐的多个配置未过门禁，不能据此给出完整 TP1/2/4 吞吐曲线。这些数据来自 RoPE 修复前的源码，最终源码须重测。

对实际捕获的第 850 行另做 CPU 重算：mini 表逐位相等，冻结 vLLM 表有 1 项不同；源码归档 `model_executor/layers/rotary_embedding/base.py` 与已安装冻结版本该文件的 SHA256 均为 `c81804498f08f072356a26b41967f211475df2694dabf5da9f488c074d1fdef5`。

## 回归

`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmarks/check_step111_regressions.py` 串行执行：`tests/step111` 初版 23 passed，旧 `tests/step58`—`tests/step88` 为 1289 passed / 23 skipped，`tests/step089`—`tests/step110` 为 370 passed / 0 skipped；新增 trace 版本保护和诊断驱动测试后再次运行 `tests/step111` 为 25 passed。完整命令、退出码和日志见 `benchmarks/results/step111/regressions/`。需求/冻结资料静态检查 7911 项通过，不替代 GPU 证据。
