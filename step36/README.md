# step36：单卡基准与瓶颈定位 —— 测试结果归档

**这个目录不是代码包，是第三十六关的测试结果归档。**

第三十六关是诊断关，没有实现产物：不改引擎，只做「同条件基准 → 找差距最大的负载 →
profiler 定位 → 提出一处改动」。所以这里放的是那一关跑出来的数据，不是一份代码。

> 验收方在 [第三十七关需求](../../vllm-omni/learning_notes/14_vllm_from_scratch/138_第三十七关_query分块的分页prefill_attention.md)
> §5 里写过「第三十六关没有实现包……不需要补一个空的 step36」。
> 本目录**不是**那种空的实现包，也不参与版本谱系（`step37/` 应从 `step35/` 复制，
> 不从 `step36/` 复制）。它只是把该关的测量结果就地归档。

## 1. 这里有什么

```text
step36/
  README.md            本文件：结论摘要与数据索引
  results/             修正后（v2）的原始测量数据
    step36_v2_<engine>_<case>_p<n>.json    36 份单次运行结果
    step36_v2_matrix_raw.json              36 条汇总
    step36_v2_summary.json                 按测点聚合的对照表
    step36_v2_trace_summary.json           四份 trace 的区间并集口径汇总
    step36_inputs.json                     六个测点的固定输入 token IDs（种子 20260920）
  traces/
    mine_prefill_c8.pt.trace.json.gz       代表性负载，我们的引擎
    vllm_prefill_c8.pt.trace.json.gz       同一负载，vLLM
```

`decode_c1` 的 trace 有 8 MB（15 万次 kernel 执行），不便入库，未收录。
一条命令可重新生成：

```bash
python benchmarks/profile_step36.py --engine mine --case decode_c1
```

基准工具本身在 `benchmarks/`，完整分析写在 [`docs/step36_benchmark_gap.md`](../docs/step36_benchmark_gap.md)。

## 2. 结论摘要

六个测点，每个 3 个独立进程、进程内 5 次取中位。**下表是修正后的 v2 数据。**

| 测点 | prompt | 生成 | 并发 | 我的引擎 | vLLM | 倍数 |
|---|---:|---:|---:|---:|---:|---:|
| short_c1 | 64 | 32 | 1 | 0.1281 s | 0.0904 s | 1.42× |
| short_c8 | 64 | 32 | 8 | 0.2061 s | 0.1134 s | 1.82× |
| prefill_c1 | 512 | 1 | 1 | 0.0650 s | 0.0157 s | 4.14× |
| **prefill_c8** | **512** | **1** | **8** | **0.4972 s** | **0.0764 s** | **6.51×** |
| decode_c1 | 64 | 128 | 1 | 0.5346 s | 0.3757 s | 1.42× |
| decode_c8 | 64 | 128 | 8 | 0.8575 s | 0.4646 s | 1.85× |

**4 个点在 1.4～1.9×，两个 prefill 点在 4.1× 与 6.5×。**

速度瓶颈的定位：`prefill_c8` 里 attention 占 GPU 时间 **86.2%**（421.81 ms / 56 次），
而 vLLM 同一批 56 次调用只要 7.18 ms；同一份数据里我们的 GEMM 反而比 vLLM 快
（46.62 vs 58.37 ms），且该次 GPU 忙碌占比 98.8%，没有空隙。**差距集中在一个 kernel 上。**

## 3. 数据分两版，不要混用

| 版本 | 文件前缀 | 配置 | 状态 |
|---|---|---|---|
| v1 | `benchmarks/results/step36_<engine>_…`（无 tag） | `norm_backend` **漏传**（实际跑 Torch RMSNorm） | **已作废**，保留供对照 |
| v2 | `results/step36_v2_<engine>_…` | `norm_backend=triton`，且有运行时断言核对 | 现行数据 |

v1 的两处配置错误由验收方复核发现：

1. **融合 RMSNorm 没打开。** `build_mine()` 只传了 `attention_backend="triton"`，
   没传 `norm_backend`，而 `Engine` 的默认值是 `"torch"`。两个是独立开关。
   第三十三关的优化一直没启用，v1 的所有测点因此偏慢 4.5%～27%。
2. **最大长度实际不一致。** `MAX_MODEL_LEN=1024` 只传给了 vLLM；
   我们从模型配置读到 `max_seq_len=40960`，没有公共覆盖接口。

修正后：

| 测点 | v1（norm=torch） | v2（norm=triton） | 变化 |
|---|---:|---:|---:|
| short_c1 | 0.1761 s | 0.1281 s | −27.3% |
| short_c8 | 0.2680 s | 0.2061 s | −23.1% |
| decode_c1 | 0.6675 s | 0.5346 s | −19.9% |
| decode_c8 | 0.9868 s | 0.8575 s | −13.1% |
| prefill_c1 | 0.0704 s | 0.0650 s | −7.7% |
| prefill_c8 | 0.5207 s | 0.4972 s | −4.5% |

**prefill 的结论没变**：`prefill_c8` 从 6.90× 只收到 6.51×。长 prefill 的问题不是 norm 能解决的。

为防止同类错误，v2 的基准脚本加了三层保护：显式传 `norm_backend`、
**构建后断言运行时对象的真实值等于请求的值**、结果里记录从对象读出的实际配置。

## 4. 复现方式

```bash
python benchmarks/bench_step36_vllm_compare.py --make-inputs   # 只需一次
python benchmarks/run_step36_matrix.py --tag=v2                # 36 次运行，约 12 分钟
python benchmarks/summarize_step36.py --tag=v2
python benchmarks/profile_step36.py --engine mine --case prefill_c8
python benchmarks/step36_mem_breakdown.py
python benchmarks/check_step36_capacity_gap.py
```

环境：RTX 5090 Laptop（24 GiB，sm_120）、WSL2、torch 2.13.0+cu130、vLLM 0.28.0、
Qwen3-0.6B、BF16、greedy、`block_size=16`、两边 KV 都是 512 块 = 8192 token 槽位。
vLLM 需 `VLLM_USE_V2_MODEL_RUNNER=0`（WSL2 下 UVA 不可用）。

## 5. 本关没有解决的事

1. **队首请求永远无法接纳，并阻塞后面的请求**（独立功能缺口，未修）。
   复现：`python benchmarks/check_step36_capacity_gap.py`。
2. **decode 的碎片没有算子级归因** —— 只有聚合总量，不足以开出「融合哪些算子」的清单。
3. **跨后端数值对照未做** —— norm=torch 与 norm=triton 同一输入下输出不同，
   需要相同历史下的 logits/attention 数值检查来区分原因，不能只归因于 BF16。
4. **自然 EOS 路径未纳入基准流程**（CLI 侧单独验证过）。
5. `max_seq_len` 仍无公共覆盖接口，两边最大长度不一致，已照实记录。

下一步是第三十七关：把 prefill 的 attention 改成 query 分块 + 矩阵乘法。
