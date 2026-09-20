# step36：单卡基准与瓶颈定位（与 vLLM 同条件对照）

- 对应代码：**没有新增 `step36/` 包**，`step35/` 原样未改（包摘要见下），只新增 `benchmarks/` 下的基准与剖析工具
- 被对照的基线：`step35/`
- 包摘要 SHA256：`3f260b0caed96d4e7936ad1b3ec201a9db278d26b67671a4491890965201d3b0`
  （`sha256("\n".join(sorted("路径 sha256(文件)")) + "\n")`，只算 `step35/**/*.py`，共 15 个文件）

## 0. 需求大概

这一关是**诊断关**，不是实现关。要求回答三个问题：

1. 同一张卡、同一个真实模型、同样的输入输出工作量，我们和 vLLM 各要多久？
2. 差距随 prompt 长度、生成长度、并发数怎么变化？
3. 根据代码和 profiler，**下一次最值得改的一个地方**是什么？

约束：不新增生成策略、不预先指定优化算法；已知的容量问题单独记录、不塞进测速循环；最后只提**一个**具体改动，而不是一张功能清单。

环境与版本：

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 Laptop（24 GiB，compute capability 12.0，驱动 592.01） |
| 平台 | WSL2（Linux 6.6.87.2-microsoft-standard-WSL2） |
| torch | 2.13.0+cu130 |
| vLLM | 0.28.0（本机已装，未升级） |
| 模型 | Qwen3-0.6B（本地同一份权重，两边共用） |

## 1. 改动内容

只新增工具，不改引擎实现：

| 文件 | 作用 |
|---|---|
| `benchmarks/bench_step36_vllm_compare.py` | 两个引擎的统一入口 + 六个测点定义 + 输入生成 |
| `benchmarks/run_step36_matrix.py` | 跑 6 测点 × 2 引擎 × 3 独立进程，写原始 JSON |
| `benchmarks/summarize_step36.py` | 把原始 JSON 聚合成对照表 |
| `benchmarks/profile_step36.py` | 对代表性负载跑 profiler（两个引擎） |
| `benchmarks/parse_step36_trace.py` | 解析 vLLM 的 chrome trace |
| `benchmarks/step36_mem_breakdown.py` | 修正显存口径（见 §2.4） |
| `benchmarks/check_step36_capacity_gap.py` | 容量缺口的有界复现（见 §6） |
| `benchmarks/results/step36_*` | 原始数据、输入、trace |

复现命令：

```bash
python benchmarks/bench_step36_vllm_compare.py --make-inputs        # 只需一次
python benchmarks/run_step36_matrix.py                             # 主矩阵
python benchmarks/summarize_step36.py                              # 汇总表
python benchmarks/profile_step36.py --engine mine --case prefill_c8
python benchmarks/step36_mem_breakdown.py
python benchmarks/check_step36_capacity_gap.py
```

## 2. 设计要点：怎么保证两边可比

### 2.1 对齐了什么

| 项 | 值（两边相同） |
|---|---|
| 精度 / 采样 | BF16、greedy（temperature=0） |
| block_size | 16 |
| max_model_len | 1024 |
| 并发上限 `max_num_seqs` | 8 |
| token 预算 `max_num_batched_tokens` | 2048 |
| prefix caching | 关闭 |
| **KV 容量** | 都是 512 块 × 16 = **8192 token 槽位**（我 `num_kv_blocks=512`；vLLM `kv_cache_memory_bytes` 反推得 `num_gpu_blocks=512`） |
| 输入 | 同一份 `benchmarks/results/step36_inputs.json`，固定种子 20260920，各请求互不共享前缀 |

KV 槽位严格相等是刻意的：需求要求「主矩阵先保证双方有足够 KV 空间，不把缺容量造成的排队混进计算性能对照」。实际上六个测点里最重的 `decode_c8` 也只有 8×192 = 1536 token 的历史，远小于 8192。

### 2.2 固定输出工作量

测速走**固定长度模式**：两边都关掉 EOS 提前停止，生成到预算为止。

- vLLM：`SamplingParams(ignore_eos=True, max_tokens=gen)`。
- 我们的引擎：测试侧把 `engine.model.eos_token_ids = ()` 和 `engine.scheduler.eos_token_ids = set()` 清空。**这是测试侧适配，没有改实现**，只动两个公开属性；正常工作路径的 EOS 行为不受影响。

每次跑完都断言实际输出长度 == 预算、请求数 == 并发数，否则该次作废。

### 2.3 计时边界

从提交这一组请求到全部 token 就绪，**不含**模型加载、tokenizer、首次编译和 CUDA Graph 捕获。加载耗时单独记录（我 ~3.1 s，vLLM ~15 s）。

预热 2 次、测量 5 次、**每个引擎 3 个独立进程**，取进程内中位数后再取三进程的中位数。两个引擎不同时占卡。

### 2.4 一个必须说清的测量陷阱：显存

第一版矩阵里我用 `torch.cuda.mem_get_info()` 量显存，得出「vLLM 1.31 GiB vs 我 3.86 GiB，差 2.9 倍」。**这个数字是错的。**

实测确认：本机（WSL2）上 `torch.cuda.mem_get_info()` 只反映**调用进程自己**的 CUDA 上下文占用，看不到别的进程。用最小实验验证——子进程分配并持有 512 MiB 期间，父进程读到的值一动不动：

```text
父进程看到的已用显存 = 1.312 GiB
子进程分配 512 MiB 并持有期间，父进程看到的仍然是 1.312 GiB
```

vLLM 0.28 默认把 GPU 工作放在 EngineCore **子进程**里，所以 1.31 GiB 只是 WSL 的基线值，跟 vLLM 实际用了多少无关。

修正办法：`benchmarks/step36_mem_breakdown.py` 用 vLLM 自己的 `apply_model()` 把查询函数送进它真正跑 GPU 的 worker 进程，取那个进程的 torch 统计；我们的引擎在同一进程，直接读同一组统计。修正后见 §3.2。

顺带说明：vLLM 0.28 默认禁止序列化任意函数，`apply_model` 需要显式开 `VLLM_ALLOW_INSECURE_SERIALIZATION=1`。这是测量脚本的便利开关，不是产品配置。

### 2.5 两边「都开了 Graph」不等于「内部执行相同」

必须分开写清楚，不能拿「同样打开 Graph」蒙混：

| | 我们的引擎 | vLLM 0.28 |
|---|---|---|
| CUDA Graph | 整模型捕获，按 `(N, M, sample_rows_given)` 精确形状缓存 | `cudagraph_mode=FULL_AND_PIECEWISE`，捕获尺寸 `[1,2,4,8,16]`，另有 piecewise 分段图 |
| 编译 | 无 | `compilation_config.mode=VLLM_COMPILE`（inductor），带算子融合 |
| attention 后端 | 自写 Triton 分页注意力 | FlashAttention（`flash_fwd_splitkv_kernel`） |
| RMSNorm | 自写融合 Triton kernel | `kernel_config: rms_norm=['native']` + inductor 融合（`triton_per_fused_fused_add_rms_norm`） |
| 模型并行层 | Q/K/V 各自 `nn.Linear`，MLP 各自投影 | `QKVParallelLinear`、`gate_up_proj` + `SiluAndMul` |

另外，vLLM 在本机必须设 `VLLM_USE_V2_MODEL_RUNNER=0` 才能启动：V2 model runner 需要 pinned memory，而 WSL2 下 `is_pin_memory_available()` 返回 False，会报 `RuntimeError: UVA is not available`。这是**环境限制，不是性能调优**，两边都如实记录。

## 3. 六个测点结果

### 3.1 主表

每个测点 3 个独立进程，进程内 5 次取中位。长度按 token ID 数。

| 测点 | prompt | 生成 | 并发 | 我的引擎 | vLLM | 倍数 | 工作量 |
|---|---:|---:|---:|---:|---:|---:|---:|
| short_c1 | 64 | 32 | 1 | 0.1761 s | 0.0945 s | **1.86×** | 32 tok |
| short_c8 | 64 | 32 | 8 | 0.2680 s | 0.1226 s | **2.19×** | 256 tok |
| prefill_c1 | 512 | 1 | 1 | 0.0704 s | 0.0181 s | **3.89×** | 1 tok |
| **prefill_c8** | **512** | **1** | **8** | **0.5207 s** | **0.0755 s** | **6.90×** | 8 tok |
| decode_c1 | 64 | 128 | 1 | 0.6675 s | 0.3657 s | **1.83×** | 128 tok |
| decode_c8 | 64 | 128 | 8 | 0.9868 s | 0.4701 s | **2.10×** | 1024 tok |

吞吐（tok/s）：short_c1 182→339；short_c8 955→2088；prefill_c1 14→55；prefill_c8 15→106；decode_c1 192→350；decode_c8 1038→2178。

进程间离散度（三进程最大值与最小值之差 ÷ 中位）多数在 10% 以内，最大的两处是 `prefill_c1` 的 vLLM（37%，因为 1 个 token 的绝对时间太短，18 ms 量级）和 `short_c8` 的 vLLM（10%）。

**读法**：差距不是均匀的。五个点聚在 1.8–2.2×，`prefill_c8` 单独冲到 6.90×。而且注意 `prefill_c8` 只生成 8 个 token——它几乎全是 prefill 时间。差距**随 prefill 长度增长**，不随生成长度增长。

### 3.2 显存（修正后）

跑完 `short_c8`（8 请求 × 32 生成）之后：

| | 我的引擎 | vLLM | 差 |
|---|---:|---:|---:|
| torch allocated | 2.393 GiB | 2.069 GiB | +16% |
| torch reserved | 2.465 GiB | 2.164 GiB | +14% |
| 其中 KV cache | 0.875 GiB | 0.875 GiB | 一致 |
| 非 KV 开销 | ~0.42 GiB | ~0.09 GiB | |

KV 容量 512 块 × 16 slot × 28 层 × 2 × 8 kv_heads × 128 dim × 2 B = 0.875 GiB，两边完全相同。模型权重 BF16 约 1.10 GiB 是共同底。所以真实差距是**多出约 0.33 GiB 的非 KV 开销**（CUDA Graph 私有内存池 + 固定输入/输出/元数据缓冲），**不是 2.9 倍**。

## 4. 用 profiler 定位

对两个代表性负载各跑两个引擎的 profiler（`prefill_c8` 差距最大；`decode_c1` 生成最长）。**profiler 数字不是正式成绩**，只用来说明时间去向。

### 4.1 prefill_c8 —— 差距 6.90×

| 类别 | 我的引擎 | vLLM |
|---|---:|---:|
| attention | **430.38 ms (84.8%)**，56 次 | 7.18 ms (10.0%)，56 次 |
| GEMM/GEMV | 46.29 ms (9.1%)，394 次 | 58.37 ms (81.3%)，226 次 |
| elementwise/norm | 30.80 ms (6.1%)，3,959 次 | 6.24 ms (8.7%)，308 次 |
| **GPU 总** | **507.5 ms** | **71.8 ms** |
| 墙钟 | 0.5159 s | 0.0825 s |
| GPU 忙占比 | **98.4%** | 87% |

两件事同时成立，而且方向相反：

1. **GPU 忙占比 98.4%**——我们的 prefill 几乎没有空隙，不是 CPU 等待、不是启动开销，就是 GPU 上实打实地算得慢。
2. **同一批 56 次 attention 调用，430.38 ms vs 7.18 ms，差 60 倍。** 而 GEMM 我们反而更快（46.29 vs 58.37 ms，我们是 `cutlass_80_tensorop_bf16_s16816gemm`，用的是 tensor core）。

也就是说：**prefill 的差距几乎全部集中在一个 kernel 上**。有效 attention FLOPs 约 241 GFLOP（4096 token × 28 层 × 因果减半），据此换算实际算力：vLLM 33.6 TFLOPS，我们 0.56 TFLOPS。

代码位置 `step35/attention.py`：

```python
_paged_attention_kernel[(num_rows, num_q_heads)](...)     # attention.py:84
...
score = tl.sum(k * q[None, :], axis=1) * SM_SCALE          # attention.py:66
acc   = scale * acc + tl.sum(p[:, None] * v, axis=0)       # attention.py:76
```

两个结构原因：

- **没有用 tensor core。** 点积写成 `tl.sum(k * q[None, :], axis=1)`，是逐元素乘 + 归约，全程 CUDA core；PV 同理。没有一处 `tl.dot`。
- **没有 query 分块。** `grid = (打包 query 行数, q_heads)`，一个 program 只负责一行 query 的一个 head。prefill 时 grid = (2048, 16) = 32768 个 program，每个都独立从头遍历全部 KV 块。相邻 query 行读的 K/V 几乎相同，却各自重读一遍，**K/V 带宽按 query 行数放大**；而 `block_size=16` 使内层循环要转 32 圈，每圈只处理 16 个 key。

对照 vLLM 走的是 `flash_fwd_splitkv_kernel`：query 分块 + `tl.dot`/tensor core + split-KV。

### 4.2 decode_c1 —— 差距 1.83×

| 类别 | 我的引擎 | vLLM |
|---|---:|---:|
| elementwise/norm | **292.96 ms (49.8%)**，243,935 次 | 32.04 ms (11.0%)，23,324 次 |
| GEMM/GEMV | 258.45 ms (43.9%)，25,216 次 | 214.25 ms (73.6%)，14,464 次 |
| attention | 37.21 ms (6.3%)，3,584 次 | 44.94 ms (15.4%)，7,140 次 |
| **GPU 总** | **588.6 ms** | **291.2 ms** |
| 墙钟 | 0.832 s | 0.4362 s |
| GPU 忙占比 | 70.8% | 66.8% |
| kernel 发射总数 | **269,368** | **44,928** |
| 每步发射 | 2,104 | 351 |

decode 是**另一类问题**，不能用 prefill 的结论套：

- **GPU 空隙两边差不多**（我 70.8%，vLLM 66.8%）。所以「我们的 decode 慢是因为 CPU 串行等待」这个说法**不成立**——两边都有约 30% 空隙。
- 真正差别是**每步在 GPU 上做的活更多、发射的 kernel 多 6 倍**。每步 GPU 实际工作 4.95 ms vs 2.27 ms。
- 多出来的部分集中在 elementwise/norm 这 292.96 ms、243,935 次发射里（vLLM 只有 32.04 ms、23,324 次）。用同一个分母 28 层 × 128 步 = 3584 layer-step 摊，**我们每层约 68 次零碎 elementwise 发射，vLLM 约 6.5 次**，量级差 10 倍，说明一层被拆得太碎。（分母取的是我们的步数；vLLM 的 trace 里 `reshape_and_cache` 也是 3584 次，同一分母可比。）
- 反过来说，decode 里 **attention 我们更快**（37.21 vs 44.94 ms），GEMM/GEMV 也接近（258 vs 214 ms）。`gemvx` 在两边都占大头——batch=1 时 GEMM 退化成 GEMV，这是共性问题，不是我们的缺陷。

### 4.3 两个负载的结论正好互补

| | prefill_c8 | decode_c1 |
|---|---|---|
| 差距 | 6.90× | 1.83× |
| GPU 忙占比 | 98.4%（无空隙） | 70.8%（有空隙，但 vLLM 也一样） |
| 病灶 | attention kernel 本身 | 每层碎片太多 |
| 病灶占比 | 84.8% 的 GPU 时间 | 49.8% 的 GPU 时间 |
| vLLM 同项对照 | 慢 60 倍 | 慢 9 倍 |

## 5. 差距报告：事实 / 推测 / 功能缺口

### 5.1 测量事实

- 六个测点全部跑通，无失败点；五个点 1.8–2.2×，`prefill_c8` 6.90×。
- KV 容量严格对齐（都是 8192 slot），不是容量差异造成的。
- 差距**随 prefill 长度增长**，不随生成长度增长。
- `prefill_c8` 的 GPU 忙占比 98.4%，差距不在空隙。
- 同一次数（56 次）的 attention：430.38 ms vs 7.18 ms。
- 我们的 GEMM 在 prefill 上比 vLLM 快（46.29 vs 58.37 ms）。
- decode 的空隙比例两边接近（70.8% vs 66.8%），差距在每步发射次数（2,104 vs 351）。
- 修正后显存差距约 16%，不是 2.9 倍。

### 5.2 推测（尚未直接验证）

- attention 慢的主因是没用 tensor core，其次是没做 query 分块/ K/V 复用。**这个归因来自读 kernel 结构，不是来自对照实验**——要坐实还需要一个只换 attention 实现的小实验（见 §5.4）。
- decode 的 243,935 次 elementwise 里具体哪些算子占大头，目前只看到聚合数字，还没有逐个归因（`torch.profiler` 的 key_averages 按 kernel 名聚合，PyTorch 原生 elementwise kernel 名字是模板化的，区分度低）。
- vLLM 的 `cudagraph_capture_sizes=[1,2,4,8,16]` 与我们的精确形状缓存策略，谁在 decode 上更省，没有单独测。

### 5.3 功能缺口（与性能无关，单独记录）

**队首请求永远无法接纳，且会阻塞后面的请求。** 用 `benchmarks/check_step36_capacity_gap.py` 有界复现（只跑固定步数，不等它「永远结束」）：

```text
A. A 要 5 块、B 只要 1 块，池子 2 块
     step 1: running=0 waiting=2 用块=0 本轮计划=0
     step 2: running=0 waiting=2 用块=0 本轮计划=0
     step 3: running=0 waiting=2 用块=0 本轮计划=0
     队首卡住、无进展 = True   还有未完成请求 = True

C. 对照：只有小请求 S（1 块），池子 2 块
     step 1: running=0 waiting=0 用块=0 本轮计划=1
     还有未完成请求 = False          ← 小请求本身可行
```

C 的存在排除了「B 自己不可行」这个解释：池子 2 块够跑小请求。所以 A 里 B 永远进不来，只能归因于队首 A 把它挡住。三个问题：

1. **无错误、无拒绝**：A 需要的 5 块超过池子总量，是**无解配置**，但实现里不报错也不拒绝。
2. **永久空转**：外部状态不变时循环不取得进展，`has_unfinished_requests()` 一直为真。
3. **队头阻塞**：可行的小请求排在不可行请求后面，也永远得不到服务。

根因在两处：`step35/cache.py` 的 `allocate_block()` 按 `ceil((len(prompt) + max_new_tokens - 1) / block_size)` 一次性预留整个输出上限；`step35/scheduler.py` 的 `schedule()` 在分配失败时直接 `break`，不区分「暂时满」和「永远不可能满足」。

这一项**不放进测速循环**，也不为了好看而绕过，按需求保留为独立功能缺口。

### 5.4 证据强度自评

`prefill_c8` 的 attention 归因是本关最强的一条：同一个 kernel、同一次数、同样的输入、同一个 profiler 口径，60 倍差距，而且 GPU 忙占比 98.4% 排除了「时间花在等待上」。唯一没排除的是「这 60 倍是不是被 grid 配置（比如 num_warps）拖累的」——但那本身就是 attention 实现的一部分，改它仍然属于同一项改动。

decode 那条则较弱：只看到聚合的 elementwise 总量，没定位到具体算子，不足以开出「融合哪些算子」的清单。

## 6. 下一项实现需求

按需求要求的格式：**哪个负载暴露问题 → 什么证据支持 → 改哪些代码 → 如何证明正确 → 用哪个原负载复测。**

**唯一一项：重写 `step35/attention.py` 的 prefill 路径，改为 query 分块 + tensor core 的分页注意力。**

| 项 | 内容 |
|---|---|
| 哪个负载 | `prefill_c8`（512 prompt × 8 并发，生成 1），当前 **6.90×** |
| 什么证据 | GPU 忙占比 98.4%（不是等待）；56 次 attention 调用 430.38 ms vs vLLM 7.18 ms；占 GPU 时间 84.8%；同一份数据 GEMM 只花 46.29 ms 且比 vLLM 快 → 差距集中在这一个 kernel |
| 改哪些代码 | `step35/attention.py`：`_paged_attention_kernel` 的 grid 改为按 query 分块（如 `BLOCK_M=64/128`），内层用 `tl.dot` 吃 tensor core；KV 沿序列方向加大分块，让相邻 query 复用同一批 K/V 加载 |
| 如何证明正确 | 保留现有逐元素实现作为参考，对随机输入做逐元素数值对照（含 BF16 与 FP32）；跑通现有回归；因果掩码边界、变长序列、`block_size=16` 非 2 幂等都要覆盖 |
| 用哪个原负载复测 | `prefill_c8` 与 `prefill_c1`（原输入、原参数、原计时边界），并顺带看 `decode_c1` 是否有回退 |

**为什么是它，而不是继续优化 sampler，也不是先去融合 decode 的算子：**

- sampler 在 `prefill_c8` 里根本不占 GPU 时间（84.8% 在 attention），先优化它对这个最大差距没有帮助。
- decode 那条虽然也是真问题，但它是**分散**的：49.8% 的时间散在 243,935 次发射里，先得逐个归因才知道该融合什么，属于「再补一个小实验」的阶段，不足以现在开出改动。
- attention 这条是**集中且可验证**的：一个 kernel、56 次调用、60 倍差距，改完能直接用原负载复测。

**预期收益（粗略，改完必须实测）**：若 attention 达到 vLLM 量级，`prefill_c8` 的 GPU 时间从 507.5 ms 降到约 85 ms，墙钟约 0.086 s，对比 vLLM 的 0.0755 s，差距从 6.90× 收到约 **1.14×**。此时 `prefill_c8` 会变成 GEMM 主导，形状与 vLLM 一致。

**明确不做**：本关不新增 `step36/` 包；下一次实现任务再为那次改动保留独立版本。

## 7. 接口变化与遗留

**接口变化**：无。`step35/` 全部源文件未修改，包摘要与已存文档一致。

**遗留**：

1. **decode 的碎片问题没有归因到算子级**。需要补一个能区分 PyTorch 原生 elementwise kernel 的实验（比如按 trace 的时间轴而非按 kernel 名聚合，或者用 NVTX 标注层内各步骤）。这是 §5.4 里判定「证据不足」的那条。
2. **容量缺口未修**。§5.3 记录的三点（无解配置不报错、永久空转、队头阻塞）是独立功能缺口，按需求保留，不在本关处理。
3. **显存口径**：本机 `mem_get_info` 看不到子进程，跨进程显存比较只能用各自进程内的 torch 统计。如果以后要在别的机器上复现，需要先重新验证这一点是否仍然成立。
4. **`VLLM_USE_V2_MODEL_RUNNER=0`** 是本机 WSL2 的限制（UVA 不可用），记录在案；换机器应重新评估。
5. **本关没有验证 EOS 正常终止路径在固定长度模式之外的行为**——§2.2 只清了两个公开属性，正常路径未受影响，但没有专门跑一次 EOS 冒烟测试。
