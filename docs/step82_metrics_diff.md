# minivllm vs vLLM 0.28 指标差异（B=1 单场景）

> 目的：把 82 实测到的**指标差异**固定下来，避免后续只记“vLLM 更快/更慢”而没有条件。
> 这不是算法归因报告，也不能外推到 B=4、其他 prompt 或其他模型。

## 1. 实验条件

| 项 | minivllm | vLLM 0.28.0 |
|---|---|---|
| target | Gemma4 E2B 本地固定 revision | 同左 |
| assistant | gemma4 assistant 本地固定 revision | 同左 |
| dtype / GPU | BF16 / RTX 5090 Laptop 24GB | 同左 |
| runner | V2、同步、单卡 | V1 engine / V2 model runner |
| attention | Torch SDPA | TRITON_ATTN（FA4 不可用） |
| sampler | minivllm Sampler | FlashInfer |
| 执行模式 | eager | `enforce_eager=True` |
| prefix cache | 关闭 | **默认开启**（见 §5 边界） |
| prompt | 同一中文 prompt，同 token IDs | 同左 |
| 输出 | 128 token，ignore_eos | 同左 |
| 计时 | 包请求 + prefill + decode + 提交 | warmup 后 `generate`，含 prompt/prefill |

## 2. 指标表

| 场景 | minivllm tok/s | vLLM tok/s | minivllm wall | vLLM wall |
|---|---:|---:|---:|---:|
| target-only（K=0） | 12.07 | 28.53 | 10.60 s | 4.49 s |
| 投机 K=2 | 6.79 | 37.36 | 18.84 s | 3.43 s |
| 框架内加速比 K=2/K=0 | **0.56×** | **1.31×** | — | — |

差值：

```text
target-only: minivllm 比 vLLM 少 16.46 tok/s（约为 vLLM 的 42.3%）
K=2:          minivllm 比 vLLM 少 30.57 tok/s（约为 vLLM 的 18.2%）
```

minivllm 的 B=1 K=2 绝对时间：

```text
10.60 s -> 18.84 s，慢 8.24 s（+77.7%）
```

vLLM 的 B=1 K=2 绝对时间：

```text
4.49 s -> 3.43 s，快 1.06 s（-23.6%）
```

## 3. 接受率/草稿指标

minivllm：

```text
K=2 接受率 35.33%
平均接受长度 1.71
位置接受率 [0.493, 0.213]（accepted[j]/num_drafts）
```

vLLM：

```text
本轮没有导出 acceptance stats（disable_log_stats=True）
因此不能把 vLLM 的 1.31× 归因于更高接受率；
两者的接受率差异未测。
```

## 4. 输出一致性

同一 prompt、128 token：

```text
minivllm target-only vs K=2：第 45 个 token 开始不同
vLLM     target-only vs K=2：第 90 个 token 开始不同
```

两者都不是 128-token 逐 token 等价；8-token correctness 阶段是逐 token 相同的。
因此：

- 这份指标表只比较**吞吐/时延**；
- 不声称任何一边“输出等价”；
- 第 45/90 token 的分歧未定位到具体 kernel/precision/调度层。

## 5. 为什么不能直接归因于投机算法

至少四个混杂因素：

1. **attention 后端不同**：Torch SDPA vs TRITON_ATTN；
2. **采样不同**：minivllm Sampler vs FlashInfer；
3. **调度/输入组装不同**：两个独立引擎；
4. **prefix cache 状态不同**：minivllm 关，vLLM 默认开，且 vLLM 有 warmup。

所以：

```text
vLLM 1.31× ≠ “vLLM 的投机算法比 minivllm 好 1.31 倍”
minivllm 0.56× ≠ “投机在 B=1 一定变慢”
```

只能写：

> 在这两个实现栈、这组固定条件下，B=1 K=2 的实测总时间差异如上。

## 6. 已知缺口

- vLLM 只跑了 B=1/K=2 一个场景；没有 B=4、没有 K=4、没有重复三次；
- vLLM 没导出 acceptance stats；
- prefix cache 条件不一致，vLLM 数字可能受 warmup/prefix 命中影响；
- 两边输出都在 128-token 内出现分歧，未定位；
- 没有 profiler，不能说明差异具体花在 target/draft/verify/CPU 提交哪一段。

## 7. 复跑

```bash
# minivllm
PY=/home/user/anaconda3/envs/vllm-omni-dev/bin/python
$PY benchmarks/check_step82_real_spec.py --perf --max-new-tokens 128 \
    --repeats 3 --batch-sizes 1 --ks 0 2

# vLLM（必须设置 WSL2 pin memory，否则 UVA 报错）
VLLM_WSL2_ENABLE_PIN_MEMORY=1 $PY benchmarks/compare_step82_vllm.py
```

原始结果在 `benchmarks/results/step82/`（不跟踪）。

## 8. 一句话

**固定条件下，B=1/K=2：minivllm 12.07/6.79 tok/s（0.56×），vLLM 28.53/37.36 tok/s（1.31×）；差异是整个实现栈的差异，不是投机算法单项归因；两边 128-token 输出都出现分歧，指标表只用于固化事实，不用于宣称等价或算法优劣。**
