# 第 82 关：真实 Gemma4 单卡验收与性能测量

> 2026-10-29。目标是用固定的真实 target/assistant 回答：**同一输入/输出规则下，投机能否减少总时间，成本在哪。**
> 不以达到某个加速倍数为通过标准；慢也是结果。

## 1. 实验身份

| 角色 | 仓库 | revision | 本地路径 |
|---|---|---|---|
| target | `google/gemma-4-E2B-it` | `3e22461f65e89153144f8adb70e3b8c2cc9845a7` | `models/gemma-4-E2B-it` |
| assistant | `google/gemma-4-E2B-it-assistant` | `2d874ef7d29f9a30599a1e4b3c1cbc9595f005df` | `models/gemma-4-E2B-it-assistant` |

- 只加载**文本路径**；checkpoint 里的 vision/audio 权重不参与，也不代表支持多模态。
- target `model.safetensors` 10,246,357,958 B（sha256 `2db548…`），assistant 157,559,816 B（sha256 `93682e…`）。
- GPU：RTX 5090 Laptop，24,463 MiB；torch 2.13.0+cu130 / triton 3.7.1 / Python 3.12.14。
- 运行：V2、同步、Torch backend、BF16、单卡、纯文本；prefix 默认关（补充实验单独开）。
- 初始预算：`max_model_len=2048`、`block_size=16`、`num_gpu_blocks=512`、`max_num_batched_tokens=128`。
- 包摘要：173 个 Python 文件，sha256 `ffa7f33c…2806f0`。

manifest、原始结果和 trace 在 `benchmarks/results/step82/`（按仓库规则不跟踪）。

## 2. 正确性门槛

四个固定文本 prompt：

```text
zh    中文解释 KV cache
en    英文解释投机延迟
code  Python fib 补全
long  662-token 长公共前缀（> window=512）
```

真实 tokenizer / chat template；每条生成 8 token；EOS=1，stop=[106,50]。

对照：

```text
target-only eager
speculative K=2 eager
speculative K=2 FULL_DECODE_ONLY graph
```

结果：

```text
target_only_eager == spec_k2_eager   True
target_only_eager == spec_k2_graph   True
spec_k2_eager     == spec_k2_graph   True
mismatches = {}
```

交付 token（四个 prompt 完全相同）：

```text
zh   [1018, 57137, 47910, 236743, 55118, 237184, 1018, 108]
en   [9798, 20852, 53255, 17707, 51534, 684, 1699, 496]
code [8291, 659, 496, 2321, 4861, 531, 4133, 506]
long [8939, 20748, 24794, 13687, 236924, 106]
```

本次 BF16 没有触发 near-tie 翻转；这不代表所有 batch/上下文组合都逐 token 等价。已有 FP32 严格对照仍保留。

## 3. 主表：B=1/4 × K=0/2/4

配置：

```text
128 output tokens / request
ignore_eos=True（固定长度吞吐轨）
warmup 1 次 + 3 次独立测量；报告 median
graph = none；prefix = off
```

| 并发 | 投机 | tok/s | wall (s) | 接受率 | 平均接受长度 | TTFT mean (s) | TPOT (s) | 加速比 vs K=0 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| B=1 | K=0 | 12.07 | 10.60 | — | — | 0.133 | 0.081 | 1.00× |
| B=1 | K=2 | 6.79 | 18.84 | 35.33% | 1.71 | 0.540 | 0.246 | 0.56× |
| B=1 | K=4 | 5.21 | 24.58 | 21.69% | 1.87 | 0.537 | 0.352 | 0.43× |
| B=4 | K=0 | 6.93 | 73.83 | — | — | 3.458 | 0.543 | 1.00× |
| B=4 | K=2 | 13.30 | 38.49 | 59.40% | 2.19 | 2.059 | 0.523 | **1.92×** |
| B=4 | K=4 | 11.52 | 44.45 | 44.57% | 2.78 | 3.095 | 0.684 | **1.66×** |

位置接受率（accepted[j]/num_drafts）：

```text
B=1 K=2: [0.493, 0.213]
B=1 K=4: [0.500, 0.235, 0.088, 0.044]
B=4 K=2: [0.675, 0.513]
B=4 K=4: [0.707, 0.500, 0.337, 0.239]
```

解释：

- **B=1 投机更慢**：target 单请求已经很快，assistant 4 层 + 验证 + Python/调度开销没有摊薄；
- **B=4 投机变快**：K=2 1.92×，K=4 1.66×；batch 更大时 assistant 的固定开销被摊薄；
- **B=4 K=4 慢于 K=2**：第 3/4 个位置接受率下降，额外草稿计算大于收益；
- B=4 的 K=0 只有 6.93 tok/s，是因为 4 条请求包含 662-token 长 prompt 的 chunked prefill
  与 target-only batch 调度/Torch attention 开销；不同 B 之间不能直接横向比较单请求速度。

峰值显存（同一进程顺序建引擎；`max_memory_allocated`）：

```text
约 14.0–14.8 GiB allocated
```

`max_memory_reserved` 在顺序建多个引擎时受 allocator 累计影响，出现大于物理显存的计数，
不作为实验预算；以 allocated + nvidia-smi 为准。

## 4. 定点补充

### 4.1 Graph（B=1，16 token，eager vs FULL_DECODE_ONLY）

| K | eager tok/s | graph tok/s | 真实 capture |
|---:|---:|---:|---|
| 0 | 11.88 | 8.44 | 是 |
| 2 | 14.30 | 12.57 | 是 |

本次小规模测量里 graph **没有变快**（捕获/重放开销 + 图 bucket 与 workload 不匹配）。
结论只能写“图路径可用且输出一致”，不能写“图带来加速”。

### 4.2 Prefix + chunked prefill（B=4，16 token）

| K | 冷命中 | 热命中 | 冷 tok/s | 热 tok/s | 冷 TTFT | 热 TTFT |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 704 | 13.39 | 21.14 | 0.898 | 0.233 |
| 2 | 0 | 704 | 6.52 | 19.26 | 3.309 | 0.690 |

公共前缀为 662-token long prompt（> window=512，budget=128，按配置会走 chunked prefill）。
热启动确实命中（704 token），TTFT 明显下降；这只证明 prefix 路径有效，不是生产服务压测。

### 4.3 动态请求（K=2）

```text
起始 4 条 -> 第一条（max_tokens=4）完成后补入第 5 条 -> 共交付 68 token
tok/s = 14.29
num_drafts=31, num_draft_tokens=62, num_accepted_tokens=39
接受率=62.9%，平均接受长度=2.26
batch 4->3->4 发生 = True
```

## 5. 结论

1. 固定真实 Gemma4 BF16，四个 prompt 的 target-only eager、K=2 eager、K=2 graph
   **逐 token 相同**；本次没有 near-tie 翻转。
2. B=1 投机**更慢**（0.56×/0.43×）；B=4 投机**更快**（K=2 1.92×、K=4 1.66×）。
3. 接受率随位置快速衰减：K=2 第二个位置已明显低于第一个；K=4 后两位收益很低。
4. prefix 热命中和 TTFT 改善明显；graph 在当前小规模下没有加速。
5. 主瓶颈不是“草稿完全没用”，而是：**单请求固定开销 + 位置接受率衰减 + target batch 调度/attention 开销**。
   B=4 的 K=2 是当前这组实验里更合适的点。

## 6. vLLM 0.28.0 单场景对照

同一 target/assistant、同一 token IDs、BF16、eager、单卡，B=1、K=2：

| 框架 | target-only | 投机 K=2 | 本框架内加速比 |
|---|---:|---:|---:|
| minivllm | 12.07 tok/s | 6.79 tok/s | 0.56× |
| vLLM 0.28.0 | 28.53 tok/s | 37.36 tok/s | **1.31×** |

- vLLM 用 `TRITON_ATTN` / FlashInfer / 自身调度与 kernel 栈；这不是“只比较投机算法”的数字。
- vLLM 的 target-only 与 K=2 输出**不是逐 token 相同**：第 90 个 token 开始分歧
  （target `13984` vs spec `119519`）。本报告不把它写成“等价”，也未把差异归因于投机。
- vLLM 在本机 WSL2 必须设 `VLLM_WSL2_ENABLE_PIN_MEMORY=1`；否则 EngineCore 报
  `UVA is not available`，这是 82 之外的环境门槛。

这组对照说明：同一个 B=1 场景，minivllm 的投机更慢、vLLM 的投机更快，差距主要来自
实现栈（attention/kernel/调度），不能拿单框架加速比直接外推算法优劣。

## 7. 未完成/边界

- **没有新增“真实 assistant 每轮 hidden/logits 对冻结 assistant 的逐值差分”**。现有证据是
  80E.E1 的 tiny 模块差分、80E.E2 的真实 cache/源层/生成对照；本报告只记录真实候选/接受率。
  这是 82 要求里唯一没有在本轮补齐的诊断项。
  原因不是“做不到”，而是需要把 target paged KV、Q-only 源层和 assistant 输入序列从
  minivllm 捕获后，注入冻结 vLLM `Gemma4MTP` 的同一缓存布局；E1 的模型层差分不接真实 attention，不能替代真实 Q-only 读路径。
- vLLM 对照只做了 B=1/K=2 一个场景；vLLM 的 target/spec 输出在第 90 token 分歧，
  未进一步定位（可能是 backend/精度/调度差异），不声称两者数值等价。
- 没有量化、TP/DP/PP/CP/EP、多模态、PD、Mamba、torch.compile、异步端到端。
- graph/prefix/dynamic 是定点补充，不是全排列；没有生产级压测。
- 接受率是草稿接受率，不是回答正确率；tok/s 不含 prompt 和被拒绝草稿。

## 7. 复跑

```bash
PY=/home/user/anaconda3/envs/vllm-omni-dev/bin/python
VLLM_WSL2_ENABLE_PIN_MEMORY=1 $PY ...  # vLLM 对照
$PY benchmarks/check_step82_real_spec.py --preflight
$PY benchmarks/check_step82_real_spec.py --correctness --max-new-tokens 8
$PY benchmarks/check_step82_real_spec.py --perf --max-new-tokens 128 --repeats 3 \
    --batch-sizes 1 4 --ks 0 2 4
$PY benchmarks/check_step82_real_spec.py --supplement --max-new-tokens 16
$PY -m pytest tests/step82 -q   # 3 passed
```
