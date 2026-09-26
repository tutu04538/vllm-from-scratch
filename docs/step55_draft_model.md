# step55：draft model、双 KV 与一般拒绝采样

需求：[185（本关需求）](../../vllm-omni/learning_notes/14_vllm_from_scratch/185_第五十五关_draft_model双KV与一般拒绝采样.md)、
[187（真实模型方案）](../../vllm-omni/learning_notes/14_vllm_from_scratch/187_第五十五关改用真实Qwen3模型.md)。
代码：`step55/`（从 `step54/` 复制，旧包不改）。包指纹 `b4a44645272de161`（19 个 .py / 4774 行）。

## 0. 需求大概

前四关的投机是 **n-gram 提议**：草稿来自重复历史，`q(d)=1`，验证只要 `p[d]`。这一关换成
**真正的第二个模型**：

- 独立的小模型提议（Qwen3-0.6B 提，Qwen3-1.7B 验），不再依赖重复历史；
- **两套 KV**：target 与 draft 各按自己的层数/头数/精度建池子，物理块绝不互换；
- **一般 q 分布**的 `min(1, p/q)` 接受与 `max(p-q, 0)` 纠正；
- 双目录加载（1.7B 是**分片** safetensors）；
- 多请求批量提议、双池容量不足时缩 K 或退回普通 target、抢占后两套 KV 都能恢复。

不承诺加速：本关只把机制做完整、做对。

## 1. 一轮到底发生了什么

一轮的顺序（实现在 `engine.py:step()` 与 `draft.py:run_round()`）：

```text
Scheduler.schedule()      真实 token 预算 + max_draft_k 预留 + 两池只读容量检查
  → DraftModelProposer.run_round()
        补算：把已提交历史补进 draft 的 KV（按 chunk，受 draft 自己的预算限制）
        提议：第 j 步把「还想要第 j 枚」的请求合成一个 batch 跑一次 draft
  → 组装 target 输入（此时才知道最终草稿数）→ plan_sample_rows
  → target 一次 forward
  → 采样层：一般拒绝采样 + 两套 KV 的回滚/对齐（唯一提交入口）
  → post_step（发布前缀、判停、回收）
```

`run_round` 必须在**组装输入之前**：它改的正是本轮的输入行与计数。这一条踩过坑——
先组装再提议，target 会拿旧的输入配新的计数，直接 `IndexError`。

### 两套 KV 的具体轨迹（脚本模型，K=2，prompt 长 6，两个模型都「整体 +1」）

```text
进轮前：target.length = 6，draft.length = 6，真实历史 [1,2,3,1,2,3,4]，x = 4（位置 6）

提议：draft 喂 x=4 → d0=5（draft.length 6→7）
      draft 喂 d0=5 → d1=6（draft.length 7→8）
      ⚠ 此时 d1 还没进过 draft 模型
target：一次输入 [4,5,6]，写 KV 到 9
验证：  全接受 → 提交 [5,6,7]（d0、d1、bonus=7）
        target 保留 1+K = 3 个输入位置 → target.length = 9
        draft 只看得到 x、d0 → draft.length = 8，**比 target 少 1**
下一轮：补算那 1 枚（d1=6）→ draft.length = 9 追平 → 再从 x=7 提议
```

三种验证结果对应的对齐（需求 §3 的表，这里给实测数字）：

| 目标验证结果 | 提交 | target 保留 | draft 处理 |
|---|---|---:|---|
| 首拒绝（第 1 枚被拒） | `[d0, c]` | `6+2 = 8` | 提议后 draft 正好 8，**齐平** |
| 部分接受 / 首枚即拒（跨块例） | `[x]` | `8+1 = 9` | draft 从 12 夹到 9（丢掉 3 个位置） |
| 全接受 | `[d0,d1,b]` | `6+3 = 9` | draft 8 → 下一轮补算 1 枚 |

**对齐规则只有两条**（`draft.py:align()`）：draft 比 target 长就夹到 `seq.cache.length`
（多出来的一定是被拒或未验证的草稿），比 target 短就什么都不做、下一轮提议前统一补算。
时机与 target 的回滚一样，必须在 `post_step()` 之前。

## 2. 一般 p/q 拒绝采样

```
接受概率 = min(1, p_i[d_i] / q_i[d_i])
纠正分布 = normalize(max(p_i - q_i, 0))
```

为什么必须是 `max(p-q, 0)`：按 q 抽到 y 的概率是 `q[y]`、以 `min(1,p[y]/q[y])` 被接受，
被拒的质量 `Σ_x q[x](1-min(1,p[x]/q[x]))` 全部落到纠正分布上，于是

```
提交 y 的概率 = min(q[y], p[y]) + max(p[y] - q[y], 0) = p[y]
```

两个 min/max 刚好互补。**换成别的纠正分布这条等式就断了**——第 52~54 关的 n-gram 是
确定性提议（q 是 one-hot），那时 `max(p-q,0)` 退化成「把该 token 挖掉再归一化」；对一般 q
继续只挖掉一个 token，得到的就不是 p：需求 §4 的例子 `p=[0.6,0.3,0.1]`、`q=[0.2,0.5,0.3]`、
`d=1` 时正确值是 `[1,0,0]`，只挖 token 1 会得到 `[6/7,0,1/7]`。

`draft_probs=None` 表示确定性提议，两条路径共用同一个函数、同一份 EOS 与「保留多少 KV」
规则（`_finish_candidates`）。边界约定（判的是**接受概率**）：

- 接受概率 `>= 1`（`p[d] >= q[d]`，含 `p[d]=1`）：必接受，不抽 uniform；
- 接受概率 `<= 0`（`p[d]=0`）：必拒，不抽 uniform；
- `p == q` 时每一枚都落在必接受上：整轮全接受、一次接受随机数都不抽，也不会走到 residual；
- `q_i[d_i] = 0` 是**非法输入**（从 q 里抽不出质量为零的 token），明确报错、不除零；
- 接受的草稿若是终止 token：**在接受的当下就停**（第五十四关复验的结论，见
  `docs/step54_random_speculative.md` §8.10），后面的草稿不验证、bonus 不抽。

## 3. 提议层（`draft.py`）

`DraftModelProposer` 持有 draft 模型、**第二个** KV 池、采样后端与 draft 自己的 token 预算。
它不认识 `Engine`，也不认识 `Scheduler`。五个入口：

| 方法 | 做什么 |
|---|---|
| `backlog(seq)` | draft 还差多少个 token 才追上 target 的已计算前缀 |
| `fits(seq, k)` | 只读地问：draft 池容不容得下「补算缺口 + k 枚草稿」（给计划阶段用） |
| `catch_up(seq, budget)` | 把已提交历史 `[draft.length, target.length)` 按 chunk 补进 draft 的 KV |
| `run_round(items)` | 补算 + 逐位置批量提议，把草稿写回本轮计划（含 q） |
| `align(seq)` / `release(seq)` | 验证后夹回边界 / 抢占、完成、失败时释放 |

**补算只有一条路径**：draft 落后只会发生在（a）刚准入或命中前缀，（b）被抢占后重算，
（c）上一轮没排上、草稿池不够或走了 fallback。三种情形都走 `catch_up()`——不维护第二份
可分叉的真实 token 列表，draft 也**从不读 target 的 KV**（测试用「draft 的 KV 必须等于它
自己对已提交前缀重算的结果」钉死这一点）。

**提议是批量的**：第 j 步把所有还想要第 j 枚的请求合成一个 batch（每请求一行），
不套「请求循环 × K 次单请求前向」。真实模型上实测：提议前向 13 次提出 19 枚（平均
1.46 枚/次），补算另计 13 次 / 54 个 token。请求的 K 可以不同（终止 token、池子不够、
预算见底），batch 因此是 ragged 的。

**终止 token 之后不再提议**：它要么被接受（验证在那里就结束）、要么被拒（首个拒绝就是它），
后面的草稿永远读不到。

**预算分开计数**：`draft_max_num_batched_tokens` 是 draft 自己的预算（默认与 target 相同），
补算的 chunk 与提议的位置都从这里扣，报告里 `num_catchup_forwards` / `num_proposal_forwards`
分开统计——小模型的前向不能偷偷记成 0。

## 4. 双池容量与失败原子性

- 计划阶段（`Scheduler._plan_tokens`）只给 `max_draft_k` 预留，草稿要跑过 draft 才知道；
- `_reserve_blocks()` 缩草稿时**两个池子都问**：`can_grow(seq, num_scheduled_tokens)`
  与 `draft.fits(seq, k)`，任一不满足就砍一枚，砍到 0 就是普通的 1-token 路径；
- 所有池操作都是**先计划后提交**（`ensure_blocks_for` 失败时一个字节都不改），
  不会出现「target 成功分配、draft 失败」的泄漏；
- draft 池不够时**只缩草稿**，绝不为可选草稿去抢占 target 的其他请求；
- 实际草稿比预留少时，多预留的 **target** 块由验证后的回滚（`truncate` 到保留长度）
  还回，draft 那边由 `align()` 夹回边界——两个池子各有一条归还路径，不重复还；
- 请求完成 / 失败 / 被抢占：两套活动 KV 一起释放（`_preempt` / `_finish_completed_requests`
  / `_fail` 三处），真实历史、优先级、两条随机流全部保留。

测试构造了两种「draft 资源不够」：**计划阶段**就放不下历史（K 直接缩到 0，全程走普通
target 路径）与**运行时**才不够（两条请求的计划都通过了只读检查，第一条补算+提议吃掉
大部分池子，第二条当场补不动）。两种情况下 target 的输出都与「不开投机」逐 token 相同。

## 5. RNG 与 seed 派生

- **两条独立的流**：draft 抽提议用 `seq.draft_generator`，target 抽接受/纠正/bonus 用
  `seq.sampling_state.generator`。
- draft 的种子由请求的 `seed` **稳定派生**（`derive_draft_seed()`：一次线性同余混合，
  常数写在代码里）。**不用 Python `hash()`**——它被 PYTHONHASHSEED 打乱，会让
  「同 seed 同工作序列可复现」失效。`seed=None` 时从全局随机源取一个（和 target 侧一致，
  那时本来就不承诺复现）。
- 抢占、重算、缩草稿、fallback 都**不重置也不消耗**任何一条流。
- 草稿被拒后**不回退** draft 的随机流：KV 回滚与 RNG 回滚是两件事。
- 同 seed 同工作序列可复现；不要求与「普通随机解码」的文本相同（两条流的抽样次数不同）。

## 6. 加载：分片 safetensors 与双目录

- `formats.read_raw_weights()` 现在同时支持单文件与分片目录：按
  `model.safetensors.index.json` 的 `weight_map` 读，每个唯一分片只 `load_file` 一次，
  合并成一份字典（本关不做流式低峰值加载）。
- 四类坏索引明确报错：分片缺失、索引声明的参数在分片里不存在、分片里有索引未声明的
  参数、参数被声明到别的分片（重复）。分片名必须是**目录内的文件名**，挡 `../` 穿越。
- `Engine.from_model_dir(target_dir, draft_model_dir=...)`：两个模型各读自己的目录，
  结构可以完全不同（本关的 0.6B 是 28 层 hidden 1024、1.7B 是 28 层 hidden 2048），
  draft 池按 draft 自己的层数/头数/精度建。
- `check_draft_model()` 校验：同设备、**同词表**、**同 EOS 集合**、draft 的 `max_seq_len`
  不短于 target、输入缓冲不小于 draft 预算、必须显式给 `draft_num_kv_blocks`。
  词表大小相同不等于 tokenizer 相同——本关只接受调用方提供的同词表模型，不做转换。

## 7. 模块划分

```text
speculative.py  算法层（纯函数）：n-gram 提议 + 一般拒绝采样（p/q）+ 纠正分布
      ↓
sampling.py     采样原语：参数/状态/惩罚/分布(TorchSampler.distribution)/取 token(draw)
      ↓
draft.py        draft 提议层：补算 → 批量提议 → 对齐 → 释放（第二套 KV，不认 Engine/Scheduler）
      ↓
sample_runtime.py  采样执行层：行映射 → 三路径 → 验证与两套 KV 回滚 → 唯一提交入口
      ↓
scheduler.py    本轮跑谁、跑几个 token、两个池子的容量取舍
      ↓
engine.py       装配（两个模型、两个池子、提议层）与一轮的编排
```

两处低层接口是本关新加的：`KVCachePool` 的 `ensure_blocks_for / can_grow_for /
truncate_cache / release_cache`（按显式 `CacheConfig` 操作，target 池与 draft 池共用
同一份块管理，不复制第二份），`TorchSampler.draw()`（「分布 → token」的唯一实现，
提议与采样共用）。

## 8. 验证

### 8.1 脚本清单（全部通过，共 293 项）

| 脚本 | 项数 | 覆盖 |
|---|---:|---|
| `check_step55_speculative.py` | 57 | 第五十二关的纯函数 + 本关的 draft_model 配置校验 |
| `check_step55_batch.py` | 35 | 第五十三关的行映射 / 混批 / 预算 / 容量 / 恢复 |
| `check_step55_rejection.py` | 48 | 拒绝采样验证层（含一般 p/q 与统计检验） |
| `check_step55_random.py` | 22 | 第五十四关的引擎状态（临时计数、RNG、重算、混批） |
| `check_step55_engine.py` | 55 | 单/多请求等价性、目录加载入口 |
| `check_step55_combinations.py` | 17 | priority / 前缀缓存与投机的组合 |
| `check_step55_draft_kv.py` | 39 | **本关的双 KV**（轨迹、对齐、边界、回退、抢占、前缀命中） |
| `check_step55_loading.py` | 12 | 分片权重 + 双目录加载 |
| `check_step55_real_qwen3.py` | 9 | 真实 1.7B + 0.6B 端到端（CUDA BF16） |
| `diff_step54_step55.py` | 88 | `speculative_mode=None` 下与 step54 **逐步逐字节一致** |

### 8.2 有判别力的几条

- **一般 q 的统计检验**：草稿每次从 q 抽，最终首 token 的经验分布回到 p
  （`[0.5998, 0.2998, 0.1004]`，20 万次，5σ 内）；条件两 token 的联合分布与「接受部分
  `q_y·min(1,p_y/q_y)·p_row1[z]` + 拒绝部分 `(q_y - accept)·residual_y`」的理论值一致。
  **反证**：把接受概率里的 q 去掉，确定性用例立刻 FAIL；把纠正分布退回「只挖掉草稿」，
  4 条 FAIL，统计用例实测 `[0.5051, 0.3659, 0.129]`——正落在预期的错误分布上。
- **两套 KV 与单独重算一致**（本关最重要）：跑完之后取每套 KV 的**有效部分**，与该模型
  对已提交前缀单独重算一遍的结果逐个张量比对。同一个模型在**不同 batch 形状**下前向会有
  FP32 级舍入差（实测 ~3e-7），所以按 `atol=1e-5` 比；**并配了「故意改坏一个有效位置」
  的对照**（改坏后差 5e-1，立刻 FAIL），证明这条比对真的在看东西。
- **对齐契约**：全接受时 draft 恰好比 target 少 1（最后一枚被接受的草稿还没进 draft 的
  KV），下一轮补算后追平；第二轮从**上一轮最后一个提交的 token**（bonus）继续提议，
  不重复喂 x；部分接受时两边齐平；跨块回滚真的把多占的整块还回池子（4 块 → 3 块）。
- **真实模型 greedy 逐 token 相同**：1.7B target + 0.6B draft，两条请求各 16 个 token，
  与不开投机的贪心**逐个 ID 相等**。贪心时目标分布是 one-hot，投机只能改「怎么算」。
- **target 每轮最多一次 forward**（16 次 / 16 步）、提议是批量的（1.46 枚/次）。

### 8.3 反证清单

| 断言 | 怎么反证的 |
|---|---|
| `max(p-q,0)` 纠正分布 | 退回「只挖掉草稿」→ 4 条 FAIL，统计值落在错误分布上 |
| `min(1,p/q)` 接受概率 | 去掉 q → 确定性用例 FAIL |
| 两套 KV 与重算一致 | 故意改坏一个位置 → 差 5e-1 立刻 FAIL |
| 双 KV 对齐（draft ≤ target） | 每一步的提交时刻都断言 `draft.length <= target.length` |
| 分片加载的错误处理 | 五种坏索引逐个造出来，确认每一种都报错 |

## 9. 接口变化与遗留

### 9.1 接口变化

**新增**：`speculative_mode="draft_model"`；`Engine(..., draft_model=, draft_num_kv_blocks=,
draft_max_num_batched_tokens=)`（全部 keyword-only，旧位置参数一个都没挪）；
`Engine.from_model_dir(..., draft_model_dir=...)`；`draft.DraftModelProposer` /
`DraftProposal` / `derive_draft_seed()` / `make_draft_generator()`；
`speculative.verify_drafts_random(..., draft_probs=)`；`residual_probs(p, q, token_id)`；
`sampling.TorchSampler.draw()`；`KVCachePool` 的四个 `*_for` / `*_cache` 方法；
`SequenceConfig.draft_cache` / `draft_generator` / `draft_seed`。

**不变**：`speculative_mode=None` 与 `"ngram"` 的行为（`diff_step54_step55.py` 88 项
逐步逐字节一致）；旧包的 import 路径；`TorchSampler.select()` 的语义（仍收**原始**行）。

### 9.2 遗留

1. **不做 GPU rejection kernel**：验证仍是逐请求的 Torch 循环，允许必要的设备同步
   （与第五十四关同一条约束）。
2. **不做异步调度下的采样**：`draft` 与 `target` 的前向都在本轮内同步完成。
3. **draft 池不做前缀共享**：恢复、命中、fallback 全部靠 `catch_up()` 从已提交历史补算。
   好处是两套缓存的交互只有一条路径，代价是重复计算。
4. **不做流式低峰值加载**：分片先合并成一份字典（1.7B 峰值约 4 GB），不做逐片装入。
5. **不做自适应 K、不重发缩 K 后的闲置预算**（需求明确不要求）。
6. **不承诺加速**：本关没有任何吞吐结论。真实模型上小 K、短输出的场景里草稿前向本身
   也是成本，收益取决于接受率与 batch 形状——那是后续专题的事。
7. **同词表假设**：只校验 `vocab_size` 与 EOS 集合，不做 tokenizer 转换（需求允许）。
8. **`speculative_mode="ngram"` 与 `"draft_model"` 不能同时开**：本关只做二选一。
