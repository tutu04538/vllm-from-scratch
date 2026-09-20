# step35：生成策略与采样加速（A 采样策略 / B beam search / C Triton 采样 kernel）

> **后续变更说明（2026-09-20）**：B 部分（`beam.py`）与 C 部分（`triton_sampling.py`）
> 已**从 `step35/` 中移除**，这两项目前不列入研究主线（见
> [step36_remove_beam_triton_sampler.md](step36_remove_beam_triton_sampler.md)）。
> 本文保留原文不加改写，作为当时实现与验证过程的记录——其中的设计与验证结论
> 在被删除时都是成立的。A 部分（采样策略、惩罚、请求随机状态）保持原样。
>
> 因此本文下方描述的包摘要 `3f260b0c…` 对应的是**删除前**的版本；
> 基于该版本跑出的基准与验收结果同样属于删除前，两者不要混用。

- 对应代码：`step35/`（新增，未提交）
- 包摘要 SHA256：`3f260b0caed96d4e7936ad1b3ec201a9db278d26b67671a4491890965201d3b0`
  （`sha256("\n".join(sorted("路径 sha256(文件)")) + "\n")`，只算 `step35/**/*.py`）
- 基线：`step34/` 原样保留、未修改

## 0. 需求大概

三件事一起做，但不是一回事：

```text
模型输出选定行的 logits
    → 根据请求历史施加惩罚
    ├─ greedy：取最高分 token
    ├─ random：temperature → top-k → softmax → top-p → 抽样
    └─ beam ：log_softmax → 合并多个父分支的候选 → 保留若干续写
```

Torch / Triton 是**执行后端**，不是第四种生成策略。

## 1. 改动内容

| 文件 | 内容 |
|---|---|
| `step35/sampling.py` | **新增**：`SamplingParams`（校验）、`SamplingState`（按请求隔离的计数与随机数流）、`apply_penalties`、`TorchSampler` |
| `step35/beam.py` | **新增**：`BeamSearch`，含分支剪枝、KV 深复制、最终排序 |
| `step35/triton_sampling.py` | **新增**：分块 argmax / Gumbel 的 Triton kernel + `TritonSampler` |
| `step35/cache.py` | `SequenceConfig` 带上 `sampling_params` / `sampling_state` |
| `step35/scheduler.py` | `add_request` 先校验参数再建请求；拒绝不认识的字段 |
| `step35/engine.py` | `_sample` 走批量接口；新增 `sampler_backend` 与 `beam_search()` |
| `step35/step35.py` | `--mode {greedy,random,beam}` 与全部采样参数、`--sampler-backend` |

## 2. 设计要点（A：采样策略）

### 2.1 顺序是语义的一部分

固定顺序，不能各写各的：

```text
FP32 工作副本
    → repetition penalty
    → frequency penalty + presence penalty
    → 除以 temperature
    → top-k：其余设为 -inf
    → 对剩余候选算概率，再做 top-p
    → 在最终保留集合里抽样
```

**顺序换一下结果就变。** 需求给的例子 `logit=4、r=2、p=0.5、f=0.2、count=3` 算的是 `4/2 − 0.5 − 0.2×3 = 0.9`；我第一版把 frequency/presence 放在 repetition 前面，得到 **1.45**。这个例子就是为了钉死顺序而给的。

Greedy 路径只施加惩罚再 argmax——不算 softmax、不缩放温度、不筛 top-k/top-p。

### 2.2 三种惩罚数的是不同的东西

| 参数 | 数什么 | 操作 |
|---|---|---|
| `repetition_penalty = r` | **prompt + 已生成** | 出现过的 token：正 logit 除以 r、负 logit 乘以 r、零不变 |
| `presence_penalty = p` | **只看已生成** | 出现过就减 p，一次 |
| `frequency_penalty = f` | 只看已生成，按次数 c | 减 `f × c` |

repetition 那个「正负方向」最容易写反：原 logit `-4`、`r=2` 应该是 `-4 × 2 = -8`。写成除以 2 的话负分反而更接近零，**惩罚变成了奖励**。

prompt 那部分在请求创建时就冻结成 `prompt_token_ids`，不在每步重扫——prefix 命中跳过了一些 prompt 块，但 repetition 仍然要看整个逻辑 prompt。

### 2.3 计数只在真实提交后更新

`state.note_output_token()` 只在 `_sample` 里、token 真的写进 `output_ids` 之后调用。M=0 的步、预填充的中间 chunk 都不经过那里，所以不会重复计数。

### 2.4 随机状态绑请求，不绑 batch 行号

每个请求一个 `SamplingState`：

- Torch 后端：自己的 `torch.Generator`（`seed` 给定时 `manual_seed`）；
- Triton 后端：自己的内核种子 + **已消耗的随机数个数** `rng_offset`。

同一个请求这一轮在第 0 行、下一轮在第 2 行，随机流不能因此换一条；别的请求插进来也不能扰动它。实测：三条请求一起跑和单独跑，第一条的 token 序列完全相同。

### 2.5 不原地破坏模型返回的 logits

Graph 的输出缓冲会被下次 replay 复用，而且 FP32 张量的 `.float()` **不产生副本**。所以每个请求都取一份明确的副本：

```python
row = logits[i].to(torch.float32, copy=True)      # copy=True 是必须的
row = apply_penalties(row, seq.sampling_params, seq.sampling_state)
```

`apply_penalties` 内部也只用 out-of-place 操作。

## 3. 设计要点（B：beam search）

### 3.1 剪枝是全局的，不是每个父分支各留一个

每轮把所有父分支的子候选**和已经结束的候选**放在一起比，只留累计分数最高的 `beam_width` 条。需求给的例子：A 的两个孩子 `-1.7、-1.8` 好过 B 的最佳孩子 `-2.1`，那就留下 A 的两个孩子。这正是需要「同一父 KV 分叉」的原因。

中途剪枝按**原始累计分数**，并列时按完整输出 token 序列的字典序升序；最终排序才用

```text
final_score = sum_logprob / max(1, generated_length) ** length_penalty
```

`generated_length` 不含 prompt、包含已生成的 EOS。

### 3.2 KV 不能串分支

从同一父候选得到两个孩子时，**不能只复制 Python 的 block_table 列表却让两条分支写同一个物理尾块**。首版用完整深复制：

```python
def _copy_kv(pool, src_cache, dst_cache):
    # 整块地拷，最后那个不满的块按有效长度切片
```

每轮剪枝后，给留下的每条分支新建一条 KV 并从父分支深复制，**复制完才释放上一轮的全部块**。实测两个孩子拿到的物理块互不相交，改写其中一个的块不影响另一个和父分支。

### 3.3 刚选出的 token 还没有 KV

候选的状态是「cache 覆盖 `prompt + tokens`，tokens 是已选出的部分」。下一轮 forward 输入的是 `tokens[-1]`，正好在这一轮补上它的 KV，同时给出「下一个 token」的 logits。所以：

- 孩子的 cache 此刻**等于父的 cache**（直接引用），KV 深复制发生在下一轮开始前；
- 不能提前把 `cache.length` 加一。

根请求只做一次 prefill（按 `max_num_batched_tokens` 分块，中间块用 `sample_rows=[]`——就是上一关的 M=0），之后每条分支各自前进一个 token。

### 3.4 容量与释放

按「父 + 子」的最坏复制峰值保守检查：

```text
2 × beam_width × ceil((prompt_len + max_new_tokens - 1) / block_size)
```

不够就在开跑前明确报错。跑完（或中途出错）在 `finally` 里把持有的块全部还回去——不能占着块空转。实测跑完 `block_usage` 全 0。

首版限制都明确报错而不是偷偷改全局开关：要求 Engine 空闲、要求 `enable_prefix_caching=False`、要求 `beam_width ≤ max_num_seqs` 且 `≤ max_num_batched_tokens`。

## 4. 设计要点（C：Triton 采样 kernel）

### 4.1 什么留在 Torch，什么必须进 Triton

- **留在 Torch**：惩罚、temperature、top-k / top-p 的排序与累计概率（需求明确允许）。
- **必须在 Triton**：最终那一步的 token 选择——greedy 的最大值/索引归约，random 的按分布选择。

profiler 实测 Triton 后端的 forward 里出现 `_argmax_partial_kernel` / `_argmax_merge_kernel`，且**没有** torch 的 `multinomial`。

### 4.2 分块 + 二级合并，不给全词表开一个大 `tl.arange`

真实词表 151936，按 `BLOCK=1024` 分成 149 个 program：

```text
第一次发射：网格 (行, 块)，每块算出局部最优（值 + 块内最小下标）
第二次发射：网格 (行,)，每行合并 149 个局部结果
```

**发射次数与请求数无关**——第一版逐请求发射，M=3 时 greedy 慢了 95%，改成批量后回到持平。`BLOCK` 也调过：1024 有 149 个 program、够铺满 82 个 SM；4096 只有 38 个，喂不饱机器。`num_warps=4` 实测最好。

### 4.3 Gumbel-max

对最终保留的 logit `z_i`，取 `argmax_i [z_i − log(−log(u_i))]`，`u_i ~ U(0,1)`。精确运算下与按 `softmax(z)` 抽样等价。

**被过滤掉的 token 仍是 `-inf`，减去有限噪声还是 `-inf`，不会因为加噪声复活**。实测 `top_k=3` 下抽 500 次只落在 `{0,1,2}`。

### 4.4 随机数的位置绑请求

内核吃 `(seed, offset)`。每个请求带自己的 `rng_seed` 和 `rng_offset`，这一轮用 `[offset, offset + V)`，下一轮 `offset += V`——不换流、也不每次从种子的第一个随机数开始。M=0 或本轮没被采样时 `rng_offset` 不动。

`seed` 允许到 `2^63-1`，但内核只吃 32 位，所以折成 `(seed ^ (seed >> 32)) & 0x7FFFFFFF`——仍是种子的确定函数，写进文档以免误会。

## 5. 验证

conda `vllm-omni-dev`，RTX 5090 Laptop，torch 2.13.0+cu130，Triton 3.7.1，TF32 关闭。

### 5.1 A：参数、惩罚、抽样

| 检查 | 结果 |
|---|---|
| 需求给的两个惩罚例子 | `4 → 0.9000`、`-4 → -8.0000`（逐位符合） |
| 只施加单个惩罚 | rep=2 → 2.0；presence=0.5 → 3.5；frequency=0.2 → 3.4；负系数方向也对 |
| top-p 边界（需求 §1 的 `0.5/0.3/0.15/0.05`） | `p=0.7` 保留 2 个、`p=0.95` 保留 3 个、`p=0.5` 保留 1 个、`p=1` 全留、`p=1e-9` 仍保留 1 个 |
| top-k 之后再 top-p（需求 §2） | `k=2,p=0.6` → 只剩第一个 ✓ |
| 并列分取小 token id | `[1,5,5,5,2]` + `top_k=2` → 保留 `[1,2]` |
| 4000 次抽样的频率 | 最大偏差 0.0040（理论 `[0.5,0.3,0.15,0.05]`） |
| 同 seed 可复现 | 是；换 seed 不同 |
| 请求隔离 | 三条一起跑与单独跑，A 的序列相同 |
| M=0 的步 | 随机状态未被推进、计数只等于真实输出数 |
| 非法参数 | 12 类越界/NaN/bool/写错字段名，全部在**入队之前**报错且不占 KV |

真实模型上 `repetition_penalty=1.5` 会改变文本；`temperature=0.7, seed=3` 两次一致。

### 5.2 B：beam 与独立参考对照

写了一份**不共享 KV、每次整段重算**的 Python 参考搜索，对照 40 组组合（`beam_width` 1–4 × `length_penalty` 0/0.5/1 × 三种惩罚 × `max_new_tokens` 1/3/6/10）：

```
40 组组合，不一致 = 0
```

逐候选核对「搜索记录的 `sum_logprob`」与「从头重算的累加 log 概率」：**27 条候选全部一致**——这同时证明了兄弟分支的 KV 没有互相污染。

其他：

- `beam_width=1`、无惩罚、`lp=0` → 与普通 greedy **逐位相同**；
- `beam_width=2` 的两条候选有 4 个 token 的共同前缀（确实是同源分叉）；
- 兄弟分支物理块互不相交，改写一个不影响另一个；
- 预算 0 → 返回一条空续写、分数 0、不跑模型；
- 跑完（含异常路径）块全部归还。

### 5.3 C：Triton 采样

| 检查 | 结果 |
|---|---|
| greedy vs Torch argmax：200 组随机词表（含人为并列） | 不一致 0 |
| 并列判据 | 值为 7.0 在 token 3 与 97 → 取 3 |
| 词表大小 1 / 11 / 13 / 1023 / 4096 / 4097 / 20000 / **151936** | 全部与 Torch 一致 |
| Gumbel 分布：20000 次 | 最大偏差 0.0057 |
| 过滤不复活 | `top_k=3` 抽 500 次只落在 `{0,1,2}` |
| 同 seed 可复现、请求间隔离 | 都通过 |
| profiler | 出现 `_argmax_partial_kernel` / `_argmax_merge_kernel`，**没有** `multinomial` |
| CPU + triton | `ValueError`，不静默回退 |

**端到端**：真实模型上两个后端的 greedy **逐位一致**（31 个 token，且与官方参考一致）；小模型 6 种组合（norm × attention × graph）两后端也全部一致。随机路径两个后端各自可复现（不要求两者逐 token 相同）。

### 5.4 性能（真实 Qwen3-0.6B）

**引擎整体**（3 条请求 × 24 token，9 组取中位）：

| 负载 | torch 后端 | triton 后端 | 相对 |
|---|---:|---:|---:|
| 1 请求 greedy | 0.942 s | 0.791 s | **−16.0%** |
| 1 请求 random | 0.876 s | 0.884 s | +0.9% |
| 3 请求 greedy | 0.803 s | 0.796 s | −0.9% |
| 3 请求 random | 0.961 s | 0.935 s | −2.7% |

**单独测「选 token」这一步**（V=151936，交错测量 15 轮取中位）：

| 模式 | M | torch | triton | 相对 |
|---|---:|---:|---:|---:|
| greedy | 1 | 52.0 µs | 138.1 µs | **+165%** |
| greedy | 3 | 75.3 µs | 72.1 µs | −4.2% |
| random | 1 | 1119.3 µs | 1417.1 µs | +26.6% |
| random | 3 | 3097.3 µs | 3253.9 µs | +5.1% |

**本关不预设加速倍数，这就是测量结果：Triton 后端总体上不比 Torch 快。**

- 引擎整体上两者差距在 ±16% 以内，因为每一步的模型 forward 是几十毫秒量级，而选 token 只有几十到几千微秒——**采样不是瓶颈**。
- 单看选 token：greedy 在 M=1 时慢 165%，因为 torch 的 `argmax` 是一次高度调优的归约，而我要两次发射；M=3 时追平（批量的收益）。
- random 路径的大头是 torch 的排序（top-k/top-p 按需求允许留在 Torch），Triton 只做了末尾一小段，还要生成 151936 个随机数。
- 第一版**逐请求发射**（每次 2 个 kernel）在 M=3 greedy 上慢 95%，改成按 `(行, 块)` 的二维网格一次覆盖整批后回到持平——这是本关真正修掉的一个性能问题。

### 5.5 首次验收后补的四处边界

验收方报了 5 个失败用例，归成四类。**都不是 beam 算法的问题，是状态与参数边界。**

#### （1）`seed=None` 没有真正初始化 Torch 的随机流

`SamplingState` 原来只对 Triton 的 `rng_seed` 随机赋值，Torch 那个 `torch.Generator` 建出来就不管了。**新建 Generator 不等于给了随机种子**——所有新 Generator 的初始种子都是同一个常量（本机 `67280421310721`），于是两个没写 seed 的请求像两台从同一页开始读的机器，输出完全相同的随机序列。

修法：没有显式 seed 时也从这个请求自己的随机源播一次种，只在创建时播，不每步重置。

```
两个 generator 的 initial_seed 相同吗 = False
两个 rng_seed 相同吗 = False
64 次抽样完全相同吗 = False      ← 修之前是完全相同
显式同 seed 仍可复现 = True
```

#### （2）Triton 的 offset 装不进 int32

`rng_offset` 表示「这个请求累计消耗了多少个随机数」= 抽样次数 × 词表大小。Gumbel 每输出一个 token 就要为整个词表生成随机数，所以 `151936 × 14135 = 2147615360` 就越过了 int32 上限（`2147483647`）。

修法：整条链改成 int64。Triton 的 `randint4x` 本来就处理大于 32 位的 offset（`language/random.py` 里显式取 `offset >> 32` 送进 Philox），所以这是它认的类型。**不能取模归零**——那会让随机流从头重复。

```
offset = 0                    -> token 150676
offset = 2147615360（14135 次后） -> token 103900
offset = 2**31                -> token 4692
offset = 2**32                -> token 130000
offset = 2**33 + 12345        -> token 68675
```

#### （3）CLI 两个分支吞掉了非法参数

- **beam 的惩罚值**：原来写的是 `args.repetition_penalty or 1.0`，`0 or 1.0` 得 `1.0`——用户明明传了非法的 0，程序悄悄跑了默认配置。改成用 `None` 判断「没传」。
- **greedy 下的 top-k / top-p**：原来只有 random 分支才把这些参数写进请求，于是 `--mode greedy --top-p 0` 绕过了校验。需求 §3 的约定是「temperature=0 时忽略 top-k/top-p 的**筛选**，但参数本身仍要合法」，所以显式传进来的照样送进校验。另外 `--mode greedy` 配非 0 的 `--temperature` 属于模式冲突，明确报错。

```
--mode beam --repetition-penalty 0  -> ValueError: repetition_penalty 必须在 [0.01, 100.0] 内，收到 0.0
--mode greedy --top-p 0             -> ValueError: top_p 必须在 (0.0, 1.0] 内，收到 0.0
--mode greedy --top-k -1            -> ValueError: top_k 必须在 [0, 151936] 内，收到 -1
--mode greedy --temperature 0.8     -> 退出码 2，提示模式冲突
```

#### （4）top-p 下界被额外缩小了

约定是 `0 < top_p <= 1`，我原来写成 `[1e-12, 1]`，把合法的 `top_p=1e-13` 拒了。改成开区间下界。

顺带查出一个**真问题**：`top_p=1e-300` 会保留 0 个 token、除出 NaN。原因是 `1e-300` 与 float32 张量比较时**下溢成 0**，`0.0 < 0.0` 不成立，于是判据一个都没通过。需求 §1 本来就写着「**至少保留一个**」，所以显式加上这条硬保证：

```python
keep_sorted = cumulative < params.top_p
keep_sorted[0] = True    # 至少保留一个
```

```
top_p=1e-300 / 1e-45 / 1e-13 / 1e-6 -> 都保留 1 个，和为 1.000000
top_p=0.5 / 0.7 / 0.95 / 1.0        -> 保留 159 / 326 / 741 / 1000 个（常规行为未变）
```

#### 复验

验收方的 `verify_step35_sampling_beam.py`：**81/81**（原来 76/81）。五个失败用例对应的检查全部通过：`unseeded_torch_independent_initialization`、`triton_long_offset_boundary`、`cli_invalid_beam`、`cli_invalid_greedy`、`valid_tiny_top_p`。

### 5.6 前几关没有回归

- 引擎默认行为不变：不加采样参数的请求仍是 greedy，小模型逐位结果与 step34 相同。
- 上一版验收脚本的失败数与 step34 **完全一样**（见下），说明本关没有新增破坏。

| 脚本 | 对 step34 | 对 step35 |
|---|---:|---:|
| `verify_step29_contract.py` | 61/96 | 61/96 |
| `verify_step29_io_contract.py` | 36/45 | 36/45 |
| `verify_step29_qwen3.py` | 8/21 | 8/21 |
| `verify_step33_norm.py` | 38/56 | 38/56 |

四处边界修复之后又跑了一遍，四个数字一个没变。

失败原因仍是 step34 就存在的两个测试侧签名问题（`wrapped()` 不收 `sample_rows`、`model.graphs` 的键变成 `(N, M, 是否挑选)`），不是行为回归。

## 6. 接口变化与遗留

- 请求字典新增可选字段：`temperature`、`top_k`、`top_p`、`repetition_penalty`、`presence_penalty`、`frequency_penalty`、`seed`；**不接受不认识的字段**（写错参数名会明确报错）。
- 新增 `Engine.beam_search(prompt_ids, max_new_tokens, beam_width=1, length_penalty=0.0, repetition_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0)`。
- `Engine(...)` / `Engine.from_model_dir(...)` 新增 `sampler_backend="torch"`；`engine.sampler_backend` 与 `engine.sampler.name` 如实反映实际后端。
- 新增模块 `sampling.py` / `beam.py` / `triton_sampling.py`；`SequenceConfig` 新增 `sampling_params` / `sampling_state`。
- `step35/step35.py` 新增 `--mode {greedy,random,beam}`、`--sampler-backend`、六个采样参数、`--beam-width`、`--length-penalty`；beam 模式下显式传抽样参数会被拒绝。
- 采样/搜索仍在模型 Graph 外，没有捕获随机数过程；模型 checkpoint 不保存任何请求设置。
- 遗留：beam 是**单个请求独占、要求在空闲 Engine 上跑**，不与其他请求混排、不流式输出、不做 copy-on-write、不做抢占/swap/共享尾块；图缓存也没有淘汰策略，`(N, M)` 组合变多会多捕获几张图。
- 遗留：Triton 后端只融合了最终选择那一步；惩罚与 top-k/top-p 仍在 Torch。按测量，这条路在当前规模下不比 Torch 快。
- 未提交。
