# step57D：普通采样与停止

- 对应代码：`step57/sample/{metadata,sampler}.py`、`step57/sample/ops/{penalties,topk_topp_sampler}.py`
  （新增），`step57/worker/gpu_input_batch.py`（按行采样状态）、`gpu_model_runner.py`（接线）、
  `step57/sampling_params.py`（`all_stop_token_ids` + 参数校验）、
  `step57/outputs.py`（`SamplerOutput`、`RequestOutput.stop_reason`）、
  `step57/engine/output_processor.py`（透传 `stop_reason`）
- 包摘要 SHA256：`69bb04afb5d8f1e5…`（56 个 .py / 5922 行；口径 = 包内 `*.py` 按相对路径排序，
  每个文件取自身 sha256，拼成 `名字\0哈希\n` 再取 sha256）
- 验收脚本：`benchmarks/check_step57_{sampler,stop_and_outputs}.py`
  （对应需求里点名的 `test_sampler.py` / `test_stop_and_outputs.py`）
- 参考实现：本机 `vllm 0.28.0`（`v1/sample/{sampler,metadata}.py`、
  `v1/sample/ops/{penalties,topk_topp_sampler}.py`、`v1/sample/logits_processor/builtin.py`
  的 `MinTokensLogitsProcessor`、`model_executor/layers/utils.py` 的惩罚实现）

## 0. 需求大概

199（采样与投机时序对齐）+ 200 §57D：**普通采样与停止**。交付两个测试，覆盖
"greedy/random 混批、筛行后映射、惩罚历史、用户增量输出"，随机正确性用**固定分布 + 概率/统计
检查**（不要求和旧代码同 seed 同 token）。199 §2 特别划线：

- `Sampler` **不负责请求生命周期**——不接 Request、不追加输出、不释放块、不决定 finished；
- `min_tokens` 的"暂不允许采到什么"（采样侧）与"已提交的 token 是否结束请求"（Scheduler 侧）
  **不是重复的同一职责**。

## 1. 改动内容

| 文件 | 对应 vLLM | 它回答的问题 |
|---|---|---|
| `sample/metadata.py` | `v1/sample/metadata.py` | 采样器只认"第 i 行 logits + 第 i 行配置"；把请求的参数/历史/随机源翻译成按行的东西 |
| `sample/sampler.py` | `v1/sample/sampler.py` | **顺序**：约束 → 惩罚 → greedy/random 分流 → 合成 `SamplerOutput` |
| `sample/ops/penalties.py` | `v1/sample/ops/penalties.py` + `layers/utils.py` | 三种惩罚（会改变 argmax，所以必须在 greedy 之前） |
| `sample/ops/topk_topp_sampler.py` | `v1/sample/ops/topk_topp_sampler.py` | top-k/top-p 掩码（含边界与"至少一个候选"）与指数竞赛抽样 |
| `worker/gpu_input_batch.py` | `v1/worker/gpu_input_batch.py` | 按行的采样定长张量（temperature/top-k/top-p/三种惩罚）、`num_prompt_tokens`、输出历史引用 |
| `worker/gpu_model_runner.py` | `v1/worker/gpu_model_runner.py` | 只对**要采样的行**建元数据、调用采样器、把 `[B,1]` 张量摊回请求 |
| `outputs.py` / `engine/output_processor.py` | 同名 | `SamplerOutput`；`RequestOutput.stop_reason` 透传 |

## 2. 设计要点

### 2.1 顺序就是语义

```text
logits → fp32
→ 会改变 argmax 的约束（min_tokens 屏蔽停止 token）
→ 惩罚（repetition / frequency / presence）
→ greedy 行：argmax（**不除温度**）
→ random 行：除温度 → top-k/top-p → 指数竞赛抽样
→ torch.where(temperature < eps, greedy, random) 合成
```

两处最容易踩的坑：

- **贪心行不能除温度**：温度 0 除下去是 inf/nan。vLLM 的做法是 `where(temp < eps, 1.0, temp)`
  先把贪心行的温度换成 1.0；本关照做，而且全贪心批直接走 `temperature is None` 的快路径。
- **约束必须在 greedy 之前**：`min_tokens` 的意义是"还不许吐出停止 token"——贪心行当然也算。
  用例里专门有"eos 是最大 logits，但 min_tokens 未到 → 贪心行也不能吐 eos"。

### 2.2 min_tokens：两处判断，各管一段

    采样侧（`Sampler.apply_logits_processors`）  还没生成够 → 把这一行的停止 token 打成 -inf
    调度侧（`Scheduler.check_stop`）            已提交的 token 是停止 token → 结束 + 截断本轮候选

端到端用例把这条钉死了：**把 eos 设成唯一的高 logits，再要求 `min_tokens=3`**——

```text
min_tokens=3 → tokens=[0, 0, 0, 4]（eos=4）：前 3 个不可能是 eos，第 4 个才是
min_tokens=0 → tokens=[4]（第一个就是 eos，只生成 1 个）
```

只有采样侧屏蔽而没有调度侧结束，第 4 个 token 不会让请求停下来；只有调度侧而没有屏蔽，
第 1 个 token 就已经结束了。缺一个这条用例都过不去。

### 2.3 top-p 的两个边界

vLLM 的参考实现是**升序排序**后在排序空间做掩码（本关照抄）：

```python
probs_sum = cumsum(softmax(sorted, dim=-1))
top_p_mask = probs_sum <= 1 - p      # 从概率小的那头切
top_p_mask[:, -1] = False            # 最大的那个永远保留 → 至少一个候选
```

两个边界各有用例：**边界 token**（累积和刚好跨过阈值的那一个）必须保留——用 `<= 1 - p`
且掩码作用在小的那头，它正好落在保留侧；`p` 极小时也必须留一个，否则全 -inf、softmax 出 NaN。

### 2.4 抽样用指数竞赛，不用 `torch.multinomial`

```text
q_i ~ Exp(1)  →  argmax(probs_i / q_i)
```

这个 argmax 恰好以 `probs` 为分布（Gumbel-max 的等价形式），**整批一次算完**，不需要把概率
拉回 CPU。vLLM 的注释说这么写是为了避开 `torch.multinomial` 的同步；本机 torch 2.13 没有复现出
同步（GPU 忙时的 CPU 侧耗时并不变大），但开销差距是真的：每次调用 CPU 侧 167 μs vs 51 μs、
GPU 侧 204 μs vs 76 μs（`[8, 151936]`），多出来的部分是 `torch.multinomial` 额外挂的校验
（profile 里能看到 `aminmax` + `sum` + 两次 `_assert_async`）。用例里的证据：

- 20000 次抽样的频率与给定分布的最大偏差 < 0.02；
- 同一个 generator + 同一个种子 → 结果可复现；
- top-p 之后的不可采 token 一次都抽不到。

顺带一句对照：第 56 关的拒绝采样用的是同一套"指数竞赛"，那里写的是 `-log(u)/w` 取 argmin——
两者是同一件事的两种写法。

### 2.5 随机流归请求，不归行号

generator 存在 Worker 的 `CachedRequestState` 上，每轮按"行 → 请求"重新映射进
`SamplingMetadata.generators`（键是**紧凑行号**）。行会因 `condense()`/抢占而搬家，用例专门查了
"删掉一行、压实之后，剩下两条请求拿到的仍是自己的 generator"。

`seed=None` 的行不在字典里，用全局 RNG（与 vLLM 的 `SamplingType.RANDOM` 一致）：不可复现，
除非调用方先 `torch.manual_seed`。

### 2.6 惩罚的历史 = prompt + 已提交输出

`SamplingMetadata.output_token_ids` 里的每个 list 是**请求镜像那个 list 的引用**（vLLM 同款）：
`_bookkeeping_sync` 往里 append 之后，下一轮建元数据时立刻能看到，不需要另存一份历史。用例里
"往镜像 append 一个 token，元数据立刻看到"就是查这条。

实现上照抄 vLLM 的两步：先把 token 打成 bin counts（`scatter_add_` 到一个多一列的缓冲，
`vocab_size` 作 padding 值落进那一列），再按公式改 logits。`repetition` 的写法与 vLLM 的
`apply_repetition_penalties_torch`（CUDA 算子的参考实现）逐行一致。

### 2.7 用户增量输出

`OutputProcessor` 交付的是**累计**（不是增量）快照：每轮的 `RequestOutput.token_ids` 是
"到目前为止的全部输出"，相邻两轮互为前缀（用例断言）。三条边界：

- **只在结束结果交付之后**才删用户侧状态，所以同一个 ID 在清理前不能复用；
- 交付的是**新 list**，用户改自己那份不影响引擎；
- abort 之后不再交付，状态立刻清掉（ID 可复用）。

`stop_reason` 只在结束那一条上有值（显式 stop token 命中时是那个 token id）。

## 3. 与 vLLM 的差异账本

| 差异 | 原因 / 后续 |
|---|---|
| 元数据按行**现扫**生成（vLLM 在增删请求时维护 `all_greedy`/`no_top_p`/`top_k_reqs` 这类增量集合） | 本关批量小、可读优先；每轮扫一遍不会漏，代价是 O(批) 的常数 |
| `top_k` 的"不筛"归一化写在 `InputBatch._write_sampling_params`（vLLM 写在同一处的 `add_request` 里） | **同一规则、同一位置**：`0 < top_k < vocab_size` 才算要筛，其余一律写成 `vocab_size`——算子因此不需要任何分支或夹取（`V - k = 0` → 阈值取最小值 → 什么都不屏蔽）。`SamplingParams` 的 `top_k` 取值校验也与 vLLM 一致（`-1`/`0` 表示关，`>= 1` 才有效） |
| 惩罚在算子内部把 list 拼成 padded 张量（vLLM 在 InputBatch 里维护 CPU 张量、按需上传） | 便于逐值对照；批量小 |
| `min_tokens` 每轮现算屏蔽掩码（vLLM 用 `MinTokensLogitsProcessor` 维护状态字典） | 语义相同（`len(已提交输出) < min_tokens` 就屏蔽），少一套增量状态 |
| 没有 logprobs（`max_num_logprobs`/`logprob_token_ids`）、白名单、bad words、`logitsprocs` 插件框架 | 57D 不要求；采样器里那条"会改变 argmax 的约束"只剩 min_tokens 一个 |
| 没有 `min_p`、没有 `SamplerOutput.logprobs_tensors` | 同上 |
| `all_greedy`/`all_random` 同时为真的唯一情况是"批为空"（采样器里有断言） | 与 vLLM 一致 |
| 采样器是 `Sampler.forward`（vLLM 是 `nn.Module`，为了 `torch.compile` 与图捕获） | 本关不捕获图；接口形状保持一致 |
| `Sampler` 逐行 Python 循环？没有——是一个批处理调用（指数竞赛整批一次） | 性能不在本关范围，但也没有退化成逐请求 Python 采样 |

## 4. 验证

| 脚本 | 项数 | 覆盖 |
|---|---:|---|
| `check_step57_sampler.py` | 27 | 混批与分流（含"温度 0 不除法"）、min_tokens 屏蔽（贪心行同样受约束）、三种惩罚逐值对照手写公式、不同 prompt 长度混批的 padding 不污染、top-k、**top-k 归一化（`-1`/`0`/`>= V` 都当不筛，混批时不筛的行原样保留）**、**top-p 边界 token 保留 + 至少一个候选**、20000 次抽样的分布统计、同 seed 可复现、行重排后 generator 跟着请求、输出历史是引用、签名里没有 Request |
| `check_step57_stop_and_outputs.py` | 20 | 五条停止规则（max_tokens / eos / ignore_eos / 显式 stop token / 上下文上限）与"多候选只提交到停止位置"、**min_tokens 的采样侧屏蔽 + 调度侧结束**（真模型，eos 设成唯一高 logits）、清理轮 0 输出、增量→累计互为前缀、快照隔离、abort 与 ID 复用、带 tokenizer 的 text、同 seed 端到端可复现 |

十一个脚本全部通过（共 272 项）。

真实模型演示（本机 Qwen3-1.7B，bf16，CUDA）：

```bash
python step57/step57.py --max-new-tokens 6 --temperature 1.0 --top-k 20 \
    --repetition-penalty 1.2 --min-tokens 2 --seed 3 "你好"
# 答: 你好！有什么我可以帮助你的
```

**没有跑**性能测试（本关只做功能与概率正确性）。

## 5. 接口变化与遗留

- 新增可导入名字：`step57.sample.{Sampler, SamplingMetadata, SAMPLING_EPS}`、
  `step57.sample.ops.{apply_all_penalties, apply_top_k_top_p, random_sample, TopKTopPSampler}`、
  `step57.outputs.SamplerOutput`。
- `Sampler.sample(logits, sampling_params, generators)` 换成 `Sampler.forward(logits,
  sampling_metadata)`（跑批、按行、返回 `SamplerOutput`）；旧的"逐行 Python 采样 + 参数白名单"
  删掉了——57B 那版遇到 top-k/top-p/惩罚会明确报错，现在是真的实现了。
- `SamplingParams` 新增 `all_stop_token_ids` 属性；`top_k`/`top_p`/`repetition_penalty`
  加了取值校验；`is_greedy` 的阈值与采样侧统一为 `1e-5`。
- `InputBatch` 新增 `vocab_size`（来自 `ModelConfig.hf_config`，不是等模型加载完再问——
  `top_k` 的归一化在建批时就要用；假执行路径没有 hf_config 时为 `None`，此时不归类）。
- `RequestOutput` 新增 `stop_reason`（只在结束时非空）。
- `InputBatch` 新增 `num_prompt_tokens`、`req_output_token_ids`（引用请求镜像）与六个按行 CPU 张量；
  `_move_row` 一并搬运（漏一个就会出现"token 是 A 的、惩罚是 B 的"）。
- **遗留**：logprobs、白名单/bad words、`min_p`、字符串 stop（属解码层）；
  58 段的投机验证（57E）会复用这套 metadata、惩罚算子与 `[B, 1]` 的 `SamplerOutput` 形状。
