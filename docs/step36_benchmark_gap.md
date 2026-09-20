# step36：单卡基准与瓶颈定位（与 vLLM 同条件对照）

- 对应代码：**没有新增 `step36/` 实现包**，本关只新增 `benchmarks/` 下的基准与剖析工具
- 被对照的基线：`step35/`
- 包摘要 SHA256：`a00a46d92b79d18a1af832b271e593bcd91ed34053af29438342826869cd93f1`
  （13 个 .py，删除 beam / Triton sampler 之后；`sha256("\n".join(sorted("路径 sha256(文件)")) + "\n")`）

> **本文有两个版本的数据，不要混用。**
>
> | 版本 | 文件前缀 | 配置 | 状态 |
> |---|---|---|---|
> | v1 | `step36_<engine>_…`（无 tag） | `norm_backend` **漏传**（实际 Torch）；`max_model_len` 记录含糊 | **已作废**，保留供对照 |
> | v2 | `step36_v2_<engine>_…` | `norm_backend=triton` 且运行时断言核对；记录真实 `max_seq_len` | 现行数据 |
>
> v1 的两处配置错误由验收方复核发现（复核 §3.1 / §3.2），v2 是修正后重跑的结果。
> 全部结论以 v2 为准。v1 的原始文件保留未删，因为验收方已记录它们的 sha256。

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

只新增基准工具，不改引擎实现：

| 文件 | 作用 |
|---|---|
| `benchmarks/bench_step36_vllm_compare.py` | 两个引擎的统一入口 + 六个测点定义 + 输入生成 |
| `benchmarks/run_step36_matrix.py` | 6 测点 × 2 引擎 × 3 独立进程，**交替两个引擎的顺序**，写带 tag 的原始 JSON |
| `benchmarks/summarize_step36.py` | 把原始 JSON 聚合成对照表（按 tag 分开两版） |
| `benchmarks/profile_step36.py` | 对代表性负载跑 profiler，**导出原始 trace** |
| `benchmarks/parse_step36_trace.py` | 解析 chrome trace，GPU 忙碌时间取**区间并集** |
| `benchmarks/step36_mem_breakdown.py` | 修正显存口径（见 §2.4） |
| `benchmarks/check_step36_capacity_gap.py` | 容量缺口的有界复现（见 §5.3） |
| `benchmarks/results/step36_*` | 原始数据、输入、trace |

复现命令：

```bash
python benchmarks/bench_step36_vllm_compare.py --make-inputs        # 只需一次
python benchmarks/run_step36_matrix.py --tag=v2                    # 主矩阵（默认 tag=v2）
python benchmarks/summarize_step36.py --tag=v2
python benchmarks/profile_step36.py --engine mine --case prefill_c8
python benchmarks/step36_mem_breakdown.py
python benchmarks/check_step36_capacity_gap.py
```

## 2. 测量口径

### 2.1 对齐了什么

| 项 | 值（两边相同） |
|---|---|
| 精度 / 采样 | BF16、greedy（temperature=0） |
| block_size | 16 |
| 并发上限 `max_num_seqs` | 8 |
| token 预算 `max_num_batched_tokens` | 2048 |
| prefix caching | 关闭 |
| **KV 容量** | 都是 512 块 × 16 = **8192 token 槽位** |
| 输入 | 同一份 `benchmarks/results/step36_inputs.json`，固定种子 20260920，各请求互不共享前缀 |

KV 槽位严格相等是刻意的：需求要求「主矩阵先保证双方有足够 KV 空间，不把缺容量造成的排队混进计算性能对照」。六个测点里最重的 `decode_c8` 也只有 8×192 = 1536 token 历史，远小于 8192。

### 2.2 固定输出工作量

测速走**固定长度模式**：两边都关掉 EOS 提前停止，生成到预算为止。

- vLLM：`SamplingParams(ignore_eos=True, max_tokens=gen)`。
- 我们的引擎：测试侧把 `engine.model.eos_token_ids = ()` 和 `engine.scheduler.eos_token_ids = set()` 清空。**测试侧适配，没有改实现。**

每次跑完都断言实际输出长度 == 预算、请求数 == 并发数，否则该次作废。v2 的 36 次运行全部通过该断言。

### 2.3 计时边界

从提交这一组请求到全部 token 就绪，**不含**模型加载、tokenizer、首次编译和 CUDA Graph 捕获。加载耗时单独记录（我 ~3.1 s，vLLM ~15 s）。预热 2 次、测量 5 次、**每个引擎 3 个独立进程**，取进程内中位数后再取三进程的中位数。两个引擎不同时占卡。

v2 起同一进程序号内**交替两个引擎**（p0 先 mine，p1 先 vllm），避免「同一引擎连跑 3 次」的顺序偏差（复核 §5 第 6 条）。

### 2.4 配置必须读运行时对象的真实值

**这是 v1 最主要的错误来源，单独列一节。**

v1 的 `build_mine()` 只传了 `attention_backend="triton"`，**没有传 `norm_backend`**。`attention` 后端和 `norm` 后端是两个独立开关，`Engine` 的 `norm_backend` 默认是 `"torch"`，所以 v1 实际跑的是 Torch RMSNorm，而报告 §2.5 却写成「自写融合 Triton kernel」——第三十三关的优化一直没启用。

修正做法（三层，缺一不可）：

1. `build_mine()` 显式传 `norm_backend`，并加 `--norm-backend` 开关便于 A/B。
2. 构建后**断言运行时对象的真实值等于请求的值**，不一致直接 `RuntimeError`：
   ```python
   if engine.model.norm_backend != args.norm_backend: raise RuntimeError(...)
   ```
   这条纪律来自验收方复核脚本里的 `assert engine.model.norm_backend == args.norm`。
3. 结果 JSON 里新增 `runtime` 字段，记录从**对象上读出的**实际配置，而不是把入参抄一遍。v2 的记录：
   ```json
   {"attention_backend":"triton","norm_backend":"triton","sampler":"torch",
    "max_seq_len":40960,"kv_blocks":512,"block_size":16,"use_cuda_graph":true}
   ```
   `--norm-backend torch` 时该字段如实记为 `torch`，不报错——那是合法的 A/B 配置，只有「请求值与实际不符」才是错误。

### 2.5 最大长度**没有**对齐（照实记录）

`MAX_MODEL_LEN=1024` 只传给了 vLLM。我们的 `from_model_dir()` 从模型配置读 `max_position_embeddings`，得到 `max_seq_len=40960`，没有公共覆盖接口。

```text
vLLM：max_model_len = 1024
我们：model.max_seq_len = 40960
```

六个测点的有效长度都没越界，KV 池容量也确实相同（都是 8192 槽位），**但不能声称所有容量配置一致**。较大的 RoPE 表、块表容量也会占缓冲，因此显存余量不能全部归因于 Graph 私有池。

按复核要求：没有公共覆盖接口时**明确记录差异**，不为了凑相同数字去改模型权重或加临时后门。

### 2.6 两边「都开了 Graph」不等于「内部执行相同」

| | 我们的引擎 | vLLM 0.28 |
|---|---|---|
| CUDA Graph | 整模型捕获，按 `(N, M, sample_rows_given)` 精确形状缓存 | `cudagraph_mode=FULL_AND_PIECEWISE`，捕获尺寸 `[1,2,4,8,16]` |
| 编译 | 无 | `mode=VLLM_COMPILE`（inductor），带算子融合 |
| attention 后端 | 自写 Triton 分页注意力 | FlashAttention（CUDA 模板 kernel） |
| RMSNorm | 自写融合 Triton kernel（v2 起真正启用） | `rms_norm=['native']` + inductor 融合 |
| 模型并行层 | Q/K/V 各自 `nn.Linear`，MLP 各自投影 | `QKVParallelLinear`、`gate_up_proj` + `SiluAndMul` |

vLLM 在本机必须设 `VLLM_USE_V2_MODEL_RUNNER=0` 才能启动：V2 model runner 需要 pinned memory，WSL2 下 `is_pin_memory_available()` 返回 False，报 `RuntimeError: UVA is not available`。这是**环境限制，不是性能调优**。

### 2.7 显存口径

v1 用父进程 `torch.cuda.mem_get_info()` 量显存，得出「vLLM 1.31 GiB vs 我 3.86 GiB，差 2.9 倍」。**这个数字是错的。**

实测确认：本机（WSL2）上 `mem_get_info()` 只反映**调用进程自己**的 CUDA 上下文占用。最小实验——子进程分配并持有 512 MiB 期间，父进程读到的值一动不动（都是 1.312 GiB）。vLLM 0.28 默认把 GPU 工作放在 EngineCore **子进程**里，所以 1.31 GiB 只是 WSL 基线值。

修正：`step36_mem_breakdown.py` 用 vLLM 的 `apply_model()` 把查询函数送进它真正跑 GPU 的 worker 进程。修正后见 §3.2。

## 3. 六个测点结果

### 3.1 主表（v2）

每个测点 3 个独立进程，进程内 5 次取中位。长度按 token ID 数。

| 测点 | prompt | 生成 | 并发 | 我的引擎 | vLLM | 倍数 | 工作量 |
|---|---:|---:|---:|---:|---:|---:|---:|
| short_c1 | 64 | 32 | 1 | 0.1281 s | 0.0904 s | **1.42×** | 32 tok |
| short_c8 | 64 | 32 | 8 | 0.2061 s | 0.1134 s | **1.82×** | 256 tok |
| prefill_c1 | 512 | 1 | 1 | 0.0650 s | 0.0157 s | **4.14×** | 1 tok |
| **prefill_c8** | **512** | **1** | **8** | **0.4972 s** | **0.0764 s** | **6.51×** | 8 tok |
| decode_c1 | 64 | 128 | 1 | 0.5346 s | 0.3757 s | **1.42×** | 128 tok |
| decode_c8 | 64 | 128 | 8 | 0.8575 s | 0.4646 s | **1.85×** | 1024 tok |

进程间离散度多在 12% 以内，两处偏大：`prefill_c1` 的 vLLM 29.3%（1 个 token 的绝对时间太短，15.7 ms 量级）、`prefill_c8` 的我方 13.5%。

**读法**：**4 个点在 1.4～1.9×，两个 prefill 点明显更慢**（4.14× 和 6.51×）。

> v1 的原文写的是「五个点聚在 1.8–2.2×」，这是错的：`prefill_c1` 的 3.89× 不在该区间，实际只有 4 个点。复核 §2 已指出。

差距**随 prefill 长度增长**，不随生成长度增长：`prefill_c8` 只生成 8 个 token，几乎全是 prefill 时间。

### 3.2 v1 → v2：漏开融合 RMSNorm 的影响

| 测点 | v1（norm=torch） | v2（norm=triton） | 变化 | v1 倍数 | v2 倍数 |
|---|---:|---:|---:|---:|---:|
| short_c1 | 0.1761 s | 0.1281 s | **−27.3%** | 1.86× | 1.42× |
| short_c8 | 0.2680 s | 0.2061 s | −23.1% | 2.19× | 1.82× |
| decode_c1 | 0.6675 s | 0.5346 s | −19.9% | 1.83× | 1.42× |
| decode_c8 | 0.9868 s | 0.8575 s | −13.1% | 2.10× | 1.85× |
| prefill_c1 | 0.0704 s | 0.0650 s | −7.7% | 3.89× | 4.14× |
| prefill_c8 | 0.5207 s | 0.4972 s | −4.5% | 6.90× | 6.51× |

两点必须同时说：

- **漏开融合 RMSNorm 让 v1 的所有点都偏慢**，decode 类最明显（−13%～−20%）。
  v1 里那部分 decode 碎片，**有一部分是「已有优化没打开」，不是「还没实现融合」**。
- **prefill 的结论没变。** `prefill_c8` 从 6.90× 只收到 6.51×，`prefill_c1` 甚至因 vLLM 那次跑得更快而变成 4.14×。长 prefill 的问题不是 norm 能解决的。

vLLM 两版的数字也有小幅波动（`short_c1` 0.0945→0.0904，`prefill_c1` 0.0181→0.0157），量级在进程噪声内。

### 3.3 显存（修正后）

跑完 `short_c8` 之后：

| | 我的引擎 | vLLM | 差 |
|---|---:|---:|---:|
| torch allocated | 2.393 GiB | 2.069 GiB | +16% |
| torch reserved | 2.465 GiB | 2.164 GiB | +14% |
| 其中 KV cache | 0.875 GiB | 0.875 GiB | 一致 |
| 非 KV 开销 | ~0.42 GiB | ~0.09 GiB | |

KV = 512 块 × 16 slot × 28 层 × 2 × 8 kv_heads × 128 dim × 2 B = 0.875 GiB，两边相同；模型权重 BF16 约 1.10 GiB 是共同底。真实差距是**多出约 0.33 GiB 非 KV 开销**，**不是 2.9 倍**。

但按 §2.5，我们的 `max_seq_len=40960` 比 vLLM 的 1024 大得多，RoPE 表等缓冲本来就更大，**这部分余量不能全部归因于 CUDA Graph 私有池**。

## 4. 用 profiler 定位

对两个代表性负载跑 profiler（`prefill_c8` 差距最大，`decode_c1` 生成最长）。**profiler 数字不是正式成绩**，只用来说明时间去向。本次已导出原始 chrome trace，可独立复核：

- `benchmarks/results/step36_traces/mine_prefill_c8.pt.trace.json.gz`
- `benchmarks/results/step36_traces/mine_decode_c1.pt.trace.json.gz`
- vLLM 两份同目录下 `rank0.*.pt.trace.json.gz`

### 4.1 prefill_c8

| 类别 | 我的引擎（v2） | vLLM |
|---|---:|---:|
| attention | **421.81 ms (86.2%)**，56 次 | 7.18 ms (10.0%)，56 次 |
| GEMM/GEMV | 46.62 ms (9.5%)，394 次 | 58.37 ms (81.3%)，226 次 |
| elementwise/norm/其他 | 20.90 ms (4.3%)，2,151 次 | 6.24 ms (8.7%)，308 次 |
| **GPU 忙碌（区间并集）** | **489.1 ms** | **71.8 ms** |
| 该次 profiling 墙钟 | 495.1 ms | 82.5 ms |
| 忙碌占比 | **98.8%** | 87.0% |

两件事同时成立，方向相反：

1. **忙碌占比 98.8%**——几乎无空隙，不是 CPU 等待、不是启动开销，是在 GPU 上实打实算得慢。
2. **同一批 56 次 attention 调用，421.81 ms vs 7.18 ms，差约 59 倍。** 而 GEMM 我们反而更快（46.62 vs 58.37 ms）。

也就是说：**prefill 的差距几乎全部集中在一个 kernel 上**。有效 attention FLOPs 约 241 GFLOP，据此换算实际算力：vLLM 约 33.6 TFLOPS，我们约 0.57 TFLOPS。

代码位置 `step35/attention.py`：

```python
_paged_attention_kernel[(num_rows, num_q_heads)](...)     # attention.py:84
score = tl.sum(k * q[None, :], axis=1) * SM_SCALE          # attention.py:66
acc   = scale * acc + tl.sum(p[:, None] * v, axis=0)       # attention.py:76
```

两个结构原因：

- **没有用 tensor core。** 点积写成 `tl.sum(k * q[None, :], axis=1)`，逐元素乘 + 归约，全程 CUDA core；PV 同理。没有一处 `tl.dot`。
- **没有 query 分块。** `grid = (打包 query 行数, q_heads)`，一个 program 只负责一行 query 的一个 head。prefill 时 grid = (2048, 16) = 32768 个 program，各自独立遍历全部 KV 块。相邻 query 行读的 K/V 几乎相同却各自重读，**K/V 带宽按 query 行数放大**；`block_size=16` 又让内层循环转 32 圈、每圈只处理 16 个 key。

> **一处措辞更正**：v1 的报告写「vLLM 走的是 `flash_fwd_splitkv_kernel`：query 分块 + `tl.dot`/tensor core」。
> `tl.dot` 是 Triton 语法，而 vLLM 那份 trace 里是 **FlashAttention 的 CUDA 模板 kernel**，不是 Triton 写的（复核 §5 第 5 条）。
> 正确的说法是：**vLLM 的 attention 做了 query 分块并使用矩阵乘法单元**；我们借鉴的是分块思想，不是「对方也用 Triton」。

### 4.2 decode_c1

| 类别 | 我的引擎（v2） | vLLM |
|---|---:|---:|
| GEMM/GEMV | 268.54 ms (57.1%)，25,216 次 | 214.25 ms (73.6%)，14,464 次 |
| elementwise/norm/其他 | 162.65 ms (34.6%)，**128,202 次** | 32.04 ms (11.0%)，23,324 次 |
| attention | 38.77 ms (8.2%)，3,584 次 | 44.94 ms (15.4%)，7,140 次 |
| **GPU 忙碌（区间并集）** | **465.3 ms** | **291.2 ms** |
| 该次 profiling 墙钟 | 644.9 ms | 436.2 ms |
| 忙碌占比 | 72.2% | 66.8% |

decode 是**另一类问题**：

- **GPU 空闲比例我们反而更低**（27.8% vs 33.2%）。所以**「我们的 decode 慢是因为 CPU 串行等待」这个说法不成立**。
  但按复核 §5 第 1 条，**这也不能反过来证明 CPU 完全没影响**——墙钟减 kernel 时间的差值（我们约 180 ms、vLLM 约 145 ms）需要时间轴定位才能归因，只能说**GPU 工作量是重要差距之一**。
- 真正的差别是**每步在 GPU 上做的活更多**：每步 GPU 工作 3.63 ms vs 2.27 ms。
- 多出来的部分集中在 elementwise 这 162.65 ms / 128,202 次里（vLLM 32.04 ms / 23,324 次）。用同一分母 28 层 × 128 步 = 3584 layer-step 摊，**我们每层约 36 次，vLLM 约 6.5 次**。
- 反过来说，decode 里 **attention 我们更快**（38.77 vs 44.94 ms），GEMM/GEMV 也接近（268.54 vs 214.25 ms）。`gemvx` 两边都占大头——batch=1 时 GEMM 退化成 GEMV，是共性问题。

> **三处口径更正**（复核 §5 第 2、3、4 条）：
> 1. **kernel 执行次数 ≠ CPU 发射次数。** 一次 CUDA Graph replay 会跑很多 kernel；`128,202 次`是 GPU 侧执行记录，不代表 Python 调了同样多次 launch。v1 报告把它写成「每步发射 2104 次」是不准确的。
> 2. **累计 kernel 时长不总等于 GPU 忙碌时长**，多流并行时要取区间并集。本次四份 trace 的「累加/并集」都恰好是 `1.000`（无多流重叠），但解析脚本已改为取并集，不再依赖这个巧合。
> 3. **`elementwise/norm/其他` 不是纯 norm。** 这是排除 attention / GEMM 之后的**剩余项**，包含 KV 写入（`reshape_and_cache`）、RoPE、各种 copy 等，不能认定全部能用「一个融合算子」消除。

### 4.3 融合 RMSNorm 打开的实证

同样两份 trace，v2 比 v1 的 elementwise 执行次数接近腰斩：

| | v1（norm=torch） | v2（norm=triton） |
|---|---:|---:|
| prefill_c8 elementwise | 3,959 次 / 30.80 ms | **2,151 次 / 20.90 ms** |
| decode_c1 elementwise | 243,935 次 / 292.96 ms | **128,202 次 / 162.65 ms** |

## 5. 差距报告：事实 / 推测 / 功能缺口

### 5.1 测量事实

- v2 六个测点全部跑通，36 次运行零失败，输出长度断言全过。
- 4 个点在 1.4～1.9×，两个 prefill 点在 4.1× 与 6.5×。
- 差距**随 prefill 长度增长**，不随生成长度增长。
- `prefill_c8` 的 GPU 忙碌占比 98.8%，差距不在空隙。
- 同一次数（56 次）attention：421.81 ms vs 7.18 ms。
- 我们的 GEMM 在 prefill 上比 vLLM 快（46.62 vs 58.37 ms）。
- 漏开融合 RMSNorm 让 v1 全部测点偏慢 4.5%～27%。
- 修正后显存差距约 16%，不是 2.9 倍。

### 5.2 推测（尚未直接验证）

- attention 慢的主因是没用 tensor core，其次是没做 query 分块 / K/V 复用。**归因来自读 kernel 结构，不是对照实验**——要坐实需要只换 attention 实现的小实验（下一关要做的正是这件事）。
- decode 的 128,202 次 elementwise 里具体哪些算子占大头，目前只有聚合数字，**没有算子级归因**。PyTorch 原生 elementwise kernel 名字是模板化的，按 kernel 名聚合区分度低。
- 我们 `max_seq_len=40960` 对显存余量的贡献没有单独测。

### 5.3 功能缺口（与性能无关，单独记录）

**队首请求永远无法接纳，且会阻塞后面的请求。** `benchmarks/check_step36_capacity_gap.py` 有界复现（只跑固定步数，不等它「永远结束」）：

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

C 排除了「B 自己不可行」这个解释：池子 2 块够跑小请求。所以 A 里 B 永远进不来，只能归因于队首 A 把它挡住。三个问题：

1. **无错误、无拒绝**：A 需要的 5 块超过池子总量，是**无解配置**，但实现里不报错也不拒绝。
2. **永久空转**：外部状态不变时循环不取得进展，`has_unfinished_requests()` 一直为真。
3. **队头阻塞**：可行的小请求排在不可行请求后面，也永远得不到服务。

根因两处：`step35/cache.py` 的 `allocate_block()` 按 `ceil((len(prompt) + max_new_tokens - 1) / block_size)` 一次性预留整个输出上限；`step35/scheduler.py` 的 `schedule()` 在分配失败时直接 `break`，不区分「暂时满」和「永远不可能满足」。

这一项**不放进测速循环**，也不为了好看而绕过。

### 5.4 证据强度自评

`prefill_c8` 的 attention 归因是本关最强的一条：同一个 kernel、同一次数、同样输入、同一个 profiler 口径、可导出的原始 trace，约 59 倍差距，且忙碌占比 98.8% 排除了「时间花在等待上」。唯一没排除的是「这 59 倍有多少来自 grid / `num_warps` 配置」——但那本身就是 attention 实现的一部分。

decode 那条较弱：只有聚合总量，没有算子级定位，不足以开出「融合哪些算子」的清单。

## 6. 下一项实现需求

按需求要求的格式：**哪个负载暴露问题 → 什么证据支持 → 改哪些代码 → 如何证明正确 → 用哪个原负载复测。**

**唯一一项：把 `step35/attention.py` 的 prefill 路径改为 query 分块 + 矩阵乘法的分页注意力。**

| 项 | 内容 |
|---|---|
| 哪个负载 | `prefill_c8`（512 prompt × 8 并发，生成 1），当前 **6.51×** |
| 什么证据 | 忙碌占比 98.8%（不是等待）；56 次 attention 调用 421.81 ms vs 7.18 ms；占 GPU 时间 86.2%；同一份数据 GEMM 只花 46.62 ms 且比 vLLM 快 → 差距集中在这一个 kernel |
| 改哪些代码 | `step35/attention.py`：grid 改为按 query 分块，内层用 `tl.dot` 吃矩阵乘法单元；KV 沿序列方向加大分块，让相邻 query 复用同一批 K/V 加载 |
| 如何证明正确 | 保留现有逐元素实现作参考，做逐元素数值对照（BF16 与 FP32 都测）；覆盖因果掩码边界、变长序列、尾块、非连续物理块、非 2 的幂 `block_size`；跑通现有回归 |
| 用哪个原负载复测 | `prefill_c8` 与 `prefill_c1`（原输入、原参数、原计时边界），并看 `decode_c1` 是否回退。**两版 norm 都要明确开 Triton**，清理带来的收益不能算给 attention |

**为什么是它，而不是继续优化 sampler，也不是先去融合 decode 的算子：**

- sampler 在 `prefill_c8` 里根本不占 GPU 时间（86.2% 在 attention），改它对这个最大差距没有帮助。
- decode 那条也是真问题，但**分散**：34.6% 的时间散在 128,202 次执行里，且没有算子级归因，属于「再补一个小实验」的阶段。
- attention 这条**集中且可验证**：一个 kernel、56 次调用、约 59 倍差距，改完能直接用原负载复测。

**预期收益（假设，不是承诺）**：若 attention 达到 vLLM 量级，`prefill_c8` 的 GPU 时间从 489.1 ms 降到约 74 ms，差距从 6.51× 收到约 1.2× 量级。这只是「其他时间不变、attention 达到指定水平」的算术推演，**不设成硬指标**。

**明确不做**：本关不新增 `step36/` 实现包；下一次实现任务（第三十七关）再为那次改动保留独立版本。

## 7. 接口变化与遗留

**引擎接口变化**：本关基准工具不改 `step35/`。另有一次独立改动删除了 beam 与 Triton sampler，见 [step36_remove_beam_triton_sampler.md](step36_remove_beam_triton_sampler.md)。

**遗留**：

1. **decode 的碎片没有算子级归因**。需要能区分 PyTorch 原生 elementwise kernel 的实验（按 trace 时间轴而非 kernel 名聚合，或用 NVTX 标注层内步骤）。
2. **容量缺口未修**（§5.3 的三点）。
3. **自然 EOS 路径没有在基准流程里验证。** §2.2 只清了两个公开属性，正常路径未受影响，但没有专门的 EOS 冒烟测试。CLI 侧单独验证过（`--max-new-tokens 12` 实际生成 9 个即停）。
4. **跨后端数值对照未做。** 复核发现 norm=torch 与 norm=triton 在同一输入下输出不同（prefill 8 个输出有 1 个不同；decode 128 个位置都不同）。工作量相同，但**这不是正确性证明，也不能全部归因于 BF16**。需要相同历史下的 logits/attention 数值检查来区分原因。
5. **`max_seq_len` 仍无公共覆盖接口**（§2.5），两边最大长度不一致。已照实记录，未强行对齐。
6. **显存口径**：本机 `mem_get_info` 看不到子进程，跨进程比较只能用各自进程内的 torch 统计；换机器需重新验证这一前提。
7. **`VLLM_USE_V2_MODEL_RUNNER=0`** 是本机 WSL2 限制，换机器应重新评估。
