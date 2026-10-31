# 第八十八关主报告：现在距离 vLLM 还有多远？

> 需求：`单卡执行优化需求/088_单卡执行公平对照与收尾.md`
> 对照实现、配置映射与边界见 `docs/step88_alignment.md`
> 原始结果：`benchmarks/results/step88/full/summary.json`（gitignored）；
> 机器可读汇总见 `docs/results.json` → `step88.results`

## 一页结论

**20/20 case 完成。mini 的 eager/Graph 输出与接受计数全部一致；vLLM B=1 也逐 token 一致。
vLLM B=4 出现 run-to-run 非确定性：同一 case 的 3 次重复输出 hash 就不同，K=0 控制项也
复现，因此不是 assistant/投机路径，也不是稳定的“Graph 数值差”。主表因此分成“严格等价”
和“同模式性能观察”，不把 vLLM B=4 写成严格加速比。**

当前固定 Gemma4 target/assistant 上最值得记住的数（median tok/s）：

| 场景 | mini | vLLM | mini/vLLM |
|---|---:|---:|---:|
| B=1 K=0 eager | 30.07 | 29.99 | 1.003 |
| B=1 K=0 FULL Graph | 119.89 | 126.78 | 0.946 |
| B=1 K=2 eager | 45.43 | 43.46 | 1.045 |
| B=1 K=2 FULL Graph | 171.28 | 199.96 | **0.857** |
| B=4 K=0 eager | 123.56 | 113.78 | 1.086 |
| B=4 K=0 FULL Graph | 460.71 | 475.27 | 0.969 |
| B=4 K=2 eager | 174.91 | 161.58 | 1.082 |
| B=4 K=2 FULL Graph | 699.61 | 722.01 | 0.969 |
| 1024 ctx B=1 K=2 eager | 62.30 | 58.33 | 1.068 |
| 1024 ctx B=1 K=2 FULL Graph | 125.63 | 169.50 | **0.741** |

解读：

1. **82 的旧数字已经不能代表当前路径**：K=0 从 12.07 提到 30.07 tok/s，K=2 eager 从
   6.79 提到 45.43 tok/s；84–87 的 RoPE/融合/分页 attention/Graph 改变了执行路径。
2. **eager B=1/B=4 已经基本同量级**：K=0 1.003×、K=2 1.045×（B=1）；B=4 eager 分别
   1.086×/1.082×。这是两套实现栈在相同 workload 下的总时间，不是“投机算法更优”。
3. **Graph 的剩余差距集中在 B=1 K=2 与 1024 上下文**：0.857× 与 0.741×；B=4 Graph
   已接近（0.969×，但 B=4 是非确定性观察）。后续 microbenchmark 定位到 2D kernel 的
   KV 串行扫描：q_len=3/context=1024 时每 program 约 33 个 tile，只有 2 个 program；
   强制 3D 分段原型在 1024/2048 约 2.7–4.6×，128/256 反而更慢。
4. **启动成本不是稳态吞吐**：mini build 4–6 s，vLLM build 63–72 s；不把模型加载算进 tok/s。
5. **严格等价边界**：B=1 和跨引擎 B=1 K=0 是强对照；vLLM B=4 的 eager/Graph 与重复
   测量都不稳定，B=4 性能只作同模式观察。

## 1. 这次到底比了什么

### 1.1 固定条件

- 本机 RTX 5090 Laptop 24,463 MiB，单卡、BF16、无量化、TP=1、同步、纯文本；
- 固定 Gemma4 target/assistant revision：
  - target `models/gemma-4-E2B-it`，`3e22461f65e89153144f8adb70e3b8c2cc9845a7`
  - assistant `models/gemma-4-E2B-it-assistant`，`2d874ef7d29f9a30599a1e4b3c1cbc9595f005df`
- 同一 chat template、同一批 prompt token IDs、greedy、`ignore_eos=True`、输出 128 token；
- prefix caching 两边关闭；
- 两边 attention 都是 `TRITON_ATTN`；
- 主对照两边都不使用 torch.compile，Graph 只使用已有 FULL decode Graph；
- 每 case 一个全新子进程；预热后测 3 次；GPU case 串行，配对启动顺序交替。

### 1.2 矩阵

```text
主表：2 引擎 × B{1,4} × K{0,2} × {eager, FULL decode Graph} = 16 case
上下文：B=1、K=2、1024-token 输入 × 2 引擎 × 2 模式 = 4 case
```

B=4 使用同一组短 prompt 的 4 条请求，避免 82 的“长 prompt + chunked prefill”把 prefill
差异混入 B=4 对照。

### 1.3 为什么不直接复用旧数字

| 来源 | 条件 | 为什么不能直接进 88 主表 |
|---|---|---|
| 82 | 同进程顺序跑、Torch SDPA、vLLM prefix cache 默认开 | allocator/对象污染，attention/prefix 不一致 |
| 83 | 一配置一子进程，但只跑 B=1 单样本，vLLM 仍单样本/prefix 不一致 | 解决尺子，不是完整矩阵 |
| 84–87 | 改了 RoPE/融合算子/paged attention/Graph | 没有重新跑完整双方矩阵 |

88 的三栏历史对照只用于说明“旧数字为什么不能混在一起”，不是把三栏拼成一条加速曲线。

## 2. 主表

短 prompt 26 token，输出 128 token；每格 3 次，表中为 **min / median / max tok/s**。
GPU case 串行；配对顺序交替。

| pair | mini tok/s (min/med/max) | vLLM tok/s (min/med/max) | mini/vLLM (median) |
|---|---:|---:|---:|
| B1 K0 eager | 28.99 / **30.07** / 33.51 | 28.20 / **29.99** / 30.69 | 1.003 |
| B1 K0 FULL Graph | 118.06 / **119.89** / 120.16 | 126.63 / **126.78** / 126.80 | 0.946 |
| B1 K2 eager | 39.68 / **45.43** / 45.92 | 41.99 / **43.46** / 45.95 | 1.045 |
| B1 K2 FULL Graph | 170.97 / **171.28** / 171.51 | 199.04 / **199.96** / 200.65 | 0.857 |
| B4 K0 eager | 122.62 / **123.56** / 125.58 | 106.20 / **113.78** / 114.47 | 1.086 |
| B4 K0 FULL Graph | 454.85 / **460.71** / 461.40 | 474.96 / **475.27** / 484.65 | 0.969 |
| B4 K2 eager | 168.29 / **174.91** / 190.76 | 152.43 / **161.58** / 170.30 | 1.082 |
| B4 K2 FULL Graph | 696.63 / **699.61** / 714.18 | 719.72 / **722.01** / 725.32 | 0.969 |

补充指标（median，配对同一次测量）：TTFT 与 TPOT 的单位为秒；接受率只对 K=2 有定义。
vLLM 的 `TTFT` 包含 tokenizer/detokenizer 前端；mini 的 tokenizer 在 engine 外，**这部分
不是纯 GPU 时间**。

| pair | TTFT mini/vLLM | TPOT mini/vLLM | 接受率 mini/vLLM |
|---|---:|---:|---:|
| B1 K0 eager | 0.032 / 0.065 | 0.0333 / 0.0331 | — |
| B1 K0 Graph | 0.035 / 0.043 | 0.0082 / 0.0076 | — |
| B1 K2 eager | 0.040 / 0.088 | 0.0219 / 0.0225 | 0.3533 / 0.3819 |
| B1 K2 Graph | 0.045 / 0.040 | 0.0055 / 0.0047 | 0.3533 / 0.3819 |
| B4 K0 eager | 0.030 / 0.089 | 0.0324 / 0.0347 | — |
| B4 K0 Graph | 0.039 / 0.079 | 0.0084 / 0.0079 | — |
| B4 K2 eager | 0.039 / 0.097 | 0.0227 / 0.0237 | 0.3767 / 0.3733 |
| B4 K2 Graph | 0.037 / 0.095 | 0.0054 / 0.0048 | 0.3767 / 0.3627（vLLM 非确定性） |

TTFT 的绝对值受前端/预热影响，尤其 vLLM 有独立 EngineCore 和 detokenizer；这里只用来
解释“同一 case 的启动段占比”，不作为框架排名。

## 3. 1024-token 上下文补充

只做 B=1、K=2。prompt 固定 1024 个 token IDs；1024 + 128 输出 + 2 草稿预留 < 2048。

| 模式 | mini tok/s (min/med/max) | vLLM tok/s (min/med/max) | mini/vLLM | TTFT mini/vLLM | 接受率 mini/vLLM |
|---|---:|---:|---:|---:|---:|
| eager | 55.59 / **62.30** / 63.14 | 57.22 / **58.33** / 59.69 | 1.068 | 0.317 / 0.403 | 0.8617 / 0.8438 |
| FULL Graph | 124.73 / **125.63** / 128.95 | 168.55 / **169.50** / 175.02 | **0.741** | 0.333 / 0.287 | 0.8617 / 0.8438 |

上下文把“短 prompt 的启动开销占比”压低后，Graph 差距从 B=1 短 prompt 的 0.857× 扩大到
0.741×；eager 反而 1.068×。这说明剩余差距主要在 Graph 下的长上下文 decode/attention
路径，而不是 eager 的通用 Python 开销。具体见 §6。

后续 microbenchmark 已把这一格拆到 kernel 层：q_len=3、context=1024 时 2D 约 0.19 ms，
强制 3D 分段约 0.06–0.07 ms；context=2048 为 4.4–4.6×；context 128/256 反而是 0.53–0.90×。
完整方法与数据见 `docs/step88_attention_micro.md`。

## 4. 输出身份与数值边界

### 4.1 同引擎 eager vs Graph

| 引擎 | B=1 K=0 | B=1 K=2 | B=4 K=0 | B=4 K=2 | 1024 ctx K=2 |
|---|---|---|---|---|---|
| mini | 逐 token 相同 | 逐 token 相同 | 逐 token 相同 | 逐 token 相同 | 逐 token 相同 |
| vLLM | 逐 token 相同 | 逐 token 相同 | **第 90 token 分叉** | **第 90 token 分叉** | 逐 token 相同 |

mini 的 5 组 eager/Graph 输出和 K=2 接受计数全部一致。vLLM B=4 两个配置都在 token 90
分叉；**K=0 没有草稿/assistant，仍分叉**，所以不能归因于投机路径或 assistant KV 路由。

更关键的是**重复稳定性**：同一 case 预热后的 3 次测量，vLLM B=4 的输出 sha256 不是同一个：

```text
vllm_B4_K0_eager  ['8065e1…', 'c2a6d4…', '226de0…']
vllm_B4_K0_graph  ['8065e1…', '8065e1…', 'c2a6d4…']
vllm_B4_K2_eager  ['947242…', '947242…', 'c2a6d4…']
vllm_B4_K2_graph  ['8065e1…', '8065e1…', '947242…']
```

mini B=1/B=4 三次输出完全一致（B=4 K=0 `c3d56a…`×3，K=2 `85f651…`×3）；vLLM B=1
三次也完全一致。因此“vLLM B=4 eager vs Graph 第 90 token 分叉”是 **vLLM B=4 本身
run-to-run 非确定性**，不是单一 Graph 机制差异；主矩阵的 B=4 因此标记为
`strict_equivalent=False`，只作同模式性能观察，B=4 接受计数也不用于草稿质量归因。

为了取 margin，本关另跑 `benchmarks/check_step88_numerics.py`（B=4 K=0，eager/Graph，
`logprobs=5`）。**单次诊断这一对恰好 128/128 token 相同**（`first_diff=None`），这反而
说明“单次相同”不能证明确定性；真正证据是上面的 3 次重复 hash。由于没有稳定的首分歧，
本报告不编造 margin，只记录 vLLM B=4 的非确定性边界。

### 4.2 跨引擎输出身份

| 配置 | eager 首分歧 | Graph 首分歧 | 说明 |
|---|---:|---:|---|
| B1 K0 | 无 | 无 | **逐 token 相同，最强对照** |
| B1 K2 | 98 | 98 | BF16 分叉；接受计数不同 |
| B4 K0 | 92 | 90 | 批布局差异 + vLLM 非确定性 |
| B4 K2 | 90 | 92 | BF16 + vLLM 非确定性 |
| 1024 ctx B1 K2 | 108 | 108 | 分页 attention/长上下文差异 |

“首分歧”是生成 token 的下标（0-based）。跨引擎不要求逐 token 相等；B1 K0 的完全相同是
本轮最强等价证据。K=2 的接受计数不同（例如 B1 K2：mini 0.3533、vLLM 0.3819）不能单独
解释为草稿质量差异，因为两边 target 的 BF16 轨迹已经分叉。

### 4.3 83/85 的固定历史结论

83 用固定中文 prompt、64 token、K=0/K=2，把 BF16 第一分歧定位到 target near-tie
（position 70，K=0 margin 0.0625），完整 FP32 前向可复现同一 top1；85 改变归约顺序后该次
固定复放不再分叉，但 position 70 的 margin 仍可为 0。088 不重新宣称“BF16 位等价”，
只把当前矩阵的同引擎/跨引擎首分歧放在同一张表里。真实 assistant 拒绝后逐轮差分由
`tests/step88/test_real_assistant_rejection.py` 单独覆盖（见 §6.4）。

## 5. Graph 与机制证据

### 5.1 Graph 实际捕获

两边主对照都是 `CompilationMode.NONE` + `FULL_DECODE_ONLY`，prefill eager；vLLM 不启用
torch.compile。测量前后图数不变，没有“稳态再捕获”。

| 配置 | mini target 图（token 档位） | mini assistant 图 | vLLM target | vLLM assistant | 测量期间新捕获 |
|---|---|---|---|---|---:|
| B1 K0 Graph | 1（[1]） | — | 1（[1]） | — | 否 |
| B1 K2 Graph | 1（[3]） | prefill 1（[3]）+ decode 1（[1]） | 1（[3]） | prefill 1 + decode 1 | 否 |
| B4 K0 Graph | 4（[1,2,3,4]） | — | 4（[1,2,3,4]） | — | 否 |
| B4 K2 Graph | 4（[3,6,9,12]） | prefill 4 + decode 4 | 4（[3,6,9,12]） | prefill 4 + decode 4 | 否 |
| 1024 ctx B1 K2 Graph | 1（[3]） | prefill 1 + decode 1 | 1（[3]） | prefill 1 + decode 1 | 否 |

mini 的测量期 `num_replays` 增量是正数（例如 B1 K2 678 次、B4 K2 660 次、上下文 447 次）；
vLLM 不暴露 replay 计数，实际 `cudaGraphLaunch` 由 `trace_step88_execution.py` 的 profiler
证明（见 §6），主表只记录 `compilation_counter` 的捕获数。

### 5.2 KV 物理布局

两边都用 `kv_cache_memory_bytes=144 MiB`：

```text
mini: 512 blocks × 16 slots = 8192 token 容量，15 个源层，144 MiB
vLLM: 1536 个逻辑块，sliding/full 组 block_size 32/16，13,797 token 容量，144 MiB
```

vLLM 的 `kv_caches` 有 35 个 target tensor（K=2 再加 4 个 assistant），但只有 **3 个唯一
底层 storage**；YOCO Q-only 层是源层 KV 的别名。如果逐 tensor 相加，会把 144 MiB 误读成
1.76 GiB；`step88_runtime.py` 已按 storage 去重。mini 的 `kv_caches` 只有 15 个源层，
没有别名。两边的 block 数/可容纳 token 数不同是正确布局差异，不把它们当同一容量。

### 5.3 启动与常驻显存

| 项 | mini | vLLM |
|---|---:|---:|
| build（含图捕获） | 3.98–6.00 s | 63.23–72.32 s |
| mini graph capture | 0.55–1.17 s | vLLM build 内，未单列 |
| measure peak allocated（B1 K2 Graph） | 9.21 GiB | 9.92 GiB |
| measure peak allocated（B4 K2 Graph） | 9.23 GiB | 9.94 GiB |

vLLM 的 63–72 s 包含独立 EngineCore 启动、权重加载、KV/cudagraph 初始化；它不进入 tok/s
分子/分母，但说明“短请求端到端”与“稳态吞吐”必须分开看。

### 5.4 机制证据的边界

- 两边 attention 都是 `TRITON_ATTN`；mini trace 中 `kernel_unified_attention` 每步覆盖
  target+assistant，`reshape_and_cache_kernel` 只写 15 个非共享源层；没有旧
  `_paged_attention_kernel`/`fixed_paged_attention` 事件。
- RoPE 已按 84 的共享紧凑表 + 原地 Q/K kernel 执行；84 的 192 MiB 唯一表结论不被 88
  改变，88 只记录当前 trace 里的 kernel。
- 85 的 RMSNorm/激活融合、Gemma4 packed QKV 已进入当前路径；88 不重新做单算子消融，
  因此不能把 84/85/86/87 的收益相乘。
- 87 的草稿层 backend 跟随 target 已在 V2 生效；V1 仍是独立 Torch 路径，不在本矩阵。

## 6. profiler：相同边界下看剩余瓶颈

`benchmarks/trace_step88_execution.py` 在 prefill 完成后才开 profiler，只录接下来 8 个
稳态 decode step；两边同一 prefill/decode 边界。原始 trace 在
`benchmarks/results/step88/trace_full/raw/`（gitignored）。

| case | graph launch | CPU kernel launch | GPU kernel | GPU time | D2H memcpy | profile wall |
|---|---:|---:|---:|---:|---:|---:|
| mini B1 K2 eager | 0 | 2584 | 8090 | 63.79 ms | 16 | 575.6 ms |
| mini B1 K2 Graph | 24 | 88 | 8098 | 63.38 ms | 16 | 154.8 ms |
| vLLM B1 K2 eager | 0 | 3505 | 5041 | 39.05 ms | 10 | 438.7 ms |
| vLLM B1 K2 Graph | 3 | 24 | 1012 | 7.86 ms | 2 | 45.9 ms |
| mini B1 K2 Graph @1024 | 24 | 88 | 8098 | 110.00 ms | 16 | 188.4 ms |
| vLLM B1 K2 Graph @1024 | 3 | 24 | 1013 | 8.94 ms | 2 | 48.4 ms |

表里的 D2H 是 profiler 观察到的 `Memcpy DtoH` 事件数（8 个 decode step 的窗口），不是
所有同步调用；vLLM 的 graph trace 覆盖不完整，所以 2 不能当作“只剩 2 次 D2H”。

### 6.1 先看 mini 自己的 eager → Graph

```text
CPU kernel launch：2584 → 88（8 个 decode step，约 29× 减少）
cudaGraphLaunch：0 → 24
GPU kernel events：8090 → 8098（几乎不变）
GPU kernel time：63.79 → 63.38 ms（几乎不变）
profile wall：575.6 → 154.8 ms
```

这是一条很重要的边界：**mini 的图减少的是 CPU 发起/等待，不是 GPU 数学**。同一组 kernel
在重放时仍然全跑；实际 B=1 K2 吞吐 45.43 → 171.28 tok/s（3.77×），和 CPU launch 的下降
方向一致，但不能说“图把 GPU 计算加速了 3.8×”。

CPU 阶段标注（8-step 窗口，`record_function` 的 count 可能因嵌套/线程大于 8，不能把 count
当请求数）：

| 阶段 | CPU span | device span | 解释 |
|---|---:|---:|---|
| `mini/engine_step` | 5.36 ms | 4.74 ms | scheduler/提交最外层，包含子阶段 |
| `mini/input_prepare` | 4.23 ms | 5.03 ms | 输入组装 |
| `mini/target_execute` | 2.76 ms | 56.66 ms | target 前向（含 1+K 行验证） |
| `mini/target_sample` | 17.94 ms | 16.77 ms | logits/采样/提交 |
| `mini/draft_propose` | 1.91 ms | 4.42 ms | assistant 提议；图内重放 |

`target_execute` 的 CPU span 从 eager 的 169.4 ms 降到 2.8 ms、`draft_propose` 从 43.8 ms
降到 1.9 ms；`target_sample` 仍有 17.9 ms CPU span，说明**采样/提交没有被图消掉**。这些
span 有嵌套，不能相加当总时间。

### 6.2 vLLM 的图与 trace 口径

vLLM B1 K2 Graph 的 profile wall 从 438.7 ms 降到 45.9 ms，CPU kernel launch 3505→24；
但它的 trace 里只有 **3 次 `cudaGraphLaunch`**、1012 个 kernel 事件，而 mini 同一窗口是
24 次 graph launch、8098 个 kernel 事件。这个差异不能解释成“vLLM 只重放 3 次”：Kineto 对
graph replay 的 kernel node 覆盖并不保证每个 replay 都逐节点展开；vLLM 的 EngineCore 又在
独立进程。**因此 §6 只把 vLLM 的 launch 下降和 wall 变化当证据，不把 1012 vs 8098 当
GPU 工作量比，也不拿 3 次 graph launch 当重放次数。**

vLLM eager 的 top kernel 仍是 cutlass GEMM / RMSNorm / `kernel_unified_attention`；Graph
后同一组名字仍在，说明不是换了一套数学。

### 6.3 1024 上下文：attention 变成主项

mini 同一张 B1 K2 Graph：

| 项目 | 短 prompt（26 token） | 1024 上下文 |
|---|---:|---:|
| GPU kernel time / 8 steps | 63.38 ms | 110.00 ms |
| `kernel_unified_attention` | 4.17 ms | **50.52 ms** |
| cutlass GEMM（128x2） | 39.80 ms | 39.92 ms |
| `_rms_norm_kernel` | 2.58 ms | 2.59 ms |
| `_rope_qk_kernel` | 0.46 ms | 0.46 ms |

短 prompt 下 GEMM 占大头；1024 上下文下 attention 从 4.17 ms 涨到 50.52 ms（约 12×），
GEMM 几乎不变。实际上下文 Graph 吞吐 mini 125.63 vs vLLM 169.50（0.741×），eager 反而
62.30 vs 58.33（1.068×）；这与“Graph 下 attention/长上下文路径是剩余差距”一致。

vLLM 的 Graph trace 在上下文下 attention 只有 1.39 ms（短 0.30 ms），同样受 graph trace
覆盖影响，不能与 mini 的 50.52 ms 直接相减；但实际 wall/吞吐比和 mini 自己的 attention
增长方向一致，足以把下一项指向长上下文 attention，而不是 GEMM。

### 6.4 真实 assistant 拒绝后差分

`tests/step88/test_real_assistant_rejection.py` 用真实 target/assistant、K=2、随机 seed=0，
在 eager 与 FULL Graph 下记录：

- target 每个 sample 入口的 hidden/logits；
- assistant 每轮 propose 的 input hidden/positions/input IDs；
- 每轮 `draft_tokens` 与 `num_rejected`。

要求至少出现一次拒绝，并逐轮比较。该测试是 82 遗留“真实 assistant 逐轮 hidden/logits
差分”的收口；单跑实测 **1 passed, 14 warnings in 17.12s**，拒绝后 assistant 输入/草稿/
`num_rejected` 与 target hidden/logits 均逐轮一致。

### 6.5 四个问题的直接回答

1. **Scheduler/输入准备/target/draft/verify/提交在哪？**
   - mini：`engine.step` 外层（scheduler/提交）→ `input_prepare` → `target_execute`（target
     的 1+K 行验证）→ `target_sample`（logits/采样）→ `draft_propose`（assistant）；trace
     labels 见 §6.1。
   - vLLM：前端 `llm_engine step` 只在父进程；scheduler/target/draft/verify 在 EngineCore；
     trace 里能稳定识别的是 CPU launch/`cudaGraphLaunch` 与 kernel 类别，不能像 mini 一样
     给 Python 阶段精确切片。
2. **kernel/launch/D2H 变了多少？** mini Graph 把 CPU launch 2584→88；vLLM 3505→24。
   mini 的 D2H memcpy 16→16（8 step，约 2/step 不变），说明**图没有消掉采样提交的 D2H**；
   `cudaEventSynchronize` 8 次、`cudaDeviceSynchronize` 1 次也都在。vLLM graph trace 只看到
   2 次 D2H，受 graph node 覆盖影响，不能宣称 vLLM 只剩 2 次。标量 `.item()` 本身没有单独
   计数；本报告用 D2H/同步事件作代理，不把未计数说成 0。
3. **KV gather/GQA repeat 是否消失？RoPE 是否共享？** 两边当前 trace 的 top kernel 里没有
   旧的 KV gather/repeat kernel；`kernel_unified_attention` 直接读分页 KV；RoPE 表按 84 的
   共享紧凑表执行（短/长上下文 `_rope_qk_kernel` 都是 0.46 ms/8 steps）。本关不做单算子
   消融，不能把这三项收益相加。
4. **GPU kernel 与 CPU 发起/等待谁是主因？** mini B1 K2 Graph：GPU 63.4 ms / profile wall
   154.8 ms，剩余约 91 ms 是 CPU/提交/等待；1024 上下文：GPU 110 ms / wall 188 ms，GPU
   attention 明显变重。vLLM 的 trace wall 只覆盖 EngineCore 的 45.9 ms，不能和端到端 wall
   直接比。

## 7. 三栏历史对照

三栏的**测量协议不同**，只能看“旧数字为什么不能直接用”，不能拼成一条加速曲线。

| 场景 | 82 旧（同进程、Torch SDPA） | 83 隔离（一配置一进程、单样本） | 88 当前（同模式、3 次、Triton/Graph） |
|---|---:|---:|---:|
| B1 K0 mini | 12.07 | 12.31 / 13.00（正反顺序） | **30.07** |
| B1 K0 vLLM | 28.53 | 28.08 / 31.33 | **29.99** |
| B1 K2 mini | 6.79 | 18.46 / 17.77 | **45.43 eager / 171.28 Graph** |
| B1 K2 vLLM | 37.36 | 45.44 / 44.32 | **43.46 eager / 199.96 Graph** |

82 的 mini K=2 甚至比 K=0 慢（0.56×）；83 隔离后 K=2 已变成 1.4–1.5×；88 的当前
eager 为 1.51×（45.43/30.07），Graph 为 5.70×（171.28/30.07）。这些比值是**同引擎内**
的，不是跨框架；82/83/88 的绝对值不可直接相减。

82 的 B=4 K=2 1.92× 是旧的 Torch/eager 基线；88 的 B=4 K=2 eager 1.42×
（174.91/123.56）、Graph 5.66×（699.61/123.56）。数值变化来自执行路径和测量协议，
不是“投机算法突然变好”。

## 8. 下一步只选一个瓶颈

**选 1024 上下文、B=1、K=2 的 2D 分页 attention 并行度**，而不是 GEMM 或再开 compile/异步。

证据链：

1. 短 prompt B=1 K=2 Graph：mini 171.28 vs vLLM 199.96（0.857×）；1024 上下文 Graph：
   125.63 vs 169.50（0.741×），差距随上下文增长；
2. mini 自己的 trace：attention 从 4.17 ms 涨到 50.52 ms（12×），GEMM 基本不变
   （39.80→39.92 ms），RMSNorm/RoPE 也不变；
3. mini Graph 的 CPU launch 已经降到 88/8 steps，继续只砍 launch 不能解释长上下文差距；
4. vLLM 的 graph trace 对 kernel node 覆盖不完整，不能拿它的绝对 kernel 数做下一项依据，
   但实际吞吐和 mini 的 attention 增长方向一致。

这一步已经做了（`docs/step88_attention_micro.md`）：结论不是“query 行数/launch 不够”，
而是 **2D kernel 每个 program 串行扫完整 KV**。q_len=3、B=1 时只有 2 个 program，每个
扫约 33 个 KV tile；强制 3D 分段后 32 个 program、每 program 约 5 个 tile，单 kernel 在
1024/2048 上下文约 2.7–4.6×，且 3 个 seed 的 synthetic 输出与 2D 一致。

但 128/256 上下文 3D 更慢（segment/reduce 固定开销），且原型没有经过真实 engine/Graph。
因此下一步是**受控的真实 engine 原型**，不是直接改生产：

1. 保留 2D 默认，只在 `seq_len >= 512` 且 q_len>1 时尝试 3D；
2. 复用 `TritonAttentionWorkspacePool`，确认 metadata/Graph capture/reduce 都能重放；
3. 真实 Gemma4 B=1 K=2 在短/1024/2048 下比 eager/Graph 的 2D/3D；
4. 先看 attention 是否从 50 ms 降到约 15–20 ms，再决定是否合入。

不要先开 torch.compile、异步调度或重写 GEMM；K=2 target 的 3 行 query 仍走 2D，不能把
“换成 3D”当作已证明的生产修复。

## 9. 复现命令

```bash
PY=/home/user/anaconda3/envs/vllm-omni-dev/bin/python
export PATH=/home/user/anaconda3/envs/vllm-omni-dev/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

$PY benchmarks/compare_step88_execution.py --all --repeats 3
$PY benchmarks/trace_step88_execution.py --decode-steps 8
$PY benchmarks/check_step88_numerics.py
$PY benchmarks/check_step88_attention_micro.py --warmup 50 --repeats 200 --rounds 10
$PY -m pytest tests/step88 -q

# 全量
$PY -m pytest tests/step58 … tests/step88 -q
```

## 10. 已知边界

- 主表固定一个中文短 prompt；B=4 是其 4 条重复请求，不外推到其他 prompt 分布；
- 上下文补充只到 ~1024 token，不外推到 8k/32k；
- 跨引擎不要求 BF16 逐 token 相等；mini 的 eager/Graph 才是硬门禁；
- vLLM B=4 在本机 run-to-run 非确定；B=4 性能只作同模式观察；
- 真实 DFlash/DSpark/DFlash2/多模块 MTP 权重不在本机，不在本报告声称；
- vLLM 的 Graph 实际重放由 profiler 证明，主表只记录图数/捕获计数；graph trace 的 kernel
  node 覆盖不完整，不把 1012 vs 8098 当 GPU 工作量比；
- 不把 kernel 间隙当未插桩利用率，不把 inclusive 范围重复相加；
- `limit_mm_per_prompt={"image":0,"video":0}` 只跳过纯文本 case 的多模态 warmup；
- 不把 84/85/86/87 的单算子收益相乘，本关不做新的 kernel 消融；
- `check_step88_attention_micro.py` 的 3D q_len>1 是**诊断原型**，生产 `can_use_3d()` 仍
  拒绝 q_len>1；短上下文 3D 更慢，未做真实 engine/Graph 验证。
