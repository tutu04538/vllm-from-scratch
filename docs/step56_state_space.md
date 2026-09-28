# step56 附篇：这一关的状态空间地图

这篇不是改动记录，是**读代码时用的地图**：这一关的验证/抽样路径上，到底有哪些轴、哪些组合
不可能出现、每个函数真正要在脑子里装几个量。

之所以要单独写：这条路径把「全接受 / 首拒绝 / 接受 EOS / K=0 / 没有 q / 贪心 / 有错」这些情况
融合进了少数几行（GPU 上按项分支要么做不到、要么会引入同步，见
[附篇·设计思路](step56_triton_kernel_design.md) §四）。融合**不会**让状态空间变小，只是把复杂度
从「很多函数」搬到「少数函数 + 掩码」；补偿手段就是这篇里的东西：命名、不变量、以及把「哪些
情况被排除在外」写清楚。

## 1. 五层，逐层看

### 第 1 层：配置（引擎一生不变）

| 轴 | 取值 | 影响 |
|---|---|---|
| `rejection_backend` | `torch` / `triton` | 路由、用哪条随机流、是否走内核 |
| `speculative_mode` | `None` / `ngram` / `draft_model` | 有没有草稿、草稿怎么来 |
| `is_greedy`（每请求） | 真 / 假 | 要不要抽随机数、纠正/bonus 怎么取 |
| `has_penalty`（每请求） | 真 / 假 | 能否走 `greedy_fast`；逐行惩罚历史 |
| `enable_prefix_caching`、`scheduling_policy` | — | 会不会抢占、恢复时重算多少 |
| 预算类（`max_num_batched_tokens`、两个池子、`num_speculative_tokens`） | — | K 最多能批几枚、会被缩到几 |

这一层**不用背**：非法组合在构造时就被拒了——`triton` + 不开投机、`triton` + 非 CUDA、
`greedy_fast` 只有 torch 后端会构造（`uses_greedy_fast_path`）。

### 第 2 层：计划（每一项、每一轮都在变）

| 轴 | 取值 | 备注 |
|---|---|---|
| 来源 | ready-decode 行 / prefill 块 / 重算块 / **重算末块** | 描述「KV 怎么算」 |
| `can_sample` | 真 / 假 | **派生**：本轮算到历史末尾了吗 |
| 已有输出 | `len(output_ids) == 0` / `> 0` | 区分「首次 prefill」与「已经在生成」 |
| **`speculative`** | 真 / 假 | **派生**：投机开 ∧ `can_sample` ∧ 已有输出 |
| `num_real_inputs` | 1 / >1 | 回滚时保留几个输入位置 |
| `draft_ids` | 0 枚 / K 枚 | 实际草稿（ngram 计划期就有，draft_model 提议后填） |
| `draft_probs` | `None` / K 行 | **派生**：`None` ⟺ ngram 项 或 K=0 项 |
| `num_reserved_drafts` | 0 / >0 | `>0` ⟹ 一定是 draft_model（ngram 恒为 0） |

这一层最容易混的两组，也是验收方两次打回的地方：

```text
「有没有草稿」      ≠  「是不是投机行」        ← 191 号：K=0 的投机行切回了旧随机流
「走 decode 还是重算」 ≠  「用哪条随机流」      ← 192 号：重算末块切回了旧随机流
```

### 第 3 层：路由（`_needs_verification`，每项一个布尔）

```text
backend(2) × speculative(2) × 有草稿(2) = 8 种，实际可达 3 种：
    triton + speculative     → 走验证（不论有没有草稿，也不论 decode 还是重算末块）
    torch  + 有草稿          → 走验证
    其余                     → 走普通采样后端
```

两个后端判据不同是有意的：`torch` 后端两条路**共用** `sampling_state.generator`，所以它的路由
不影响随机流（K=0 退回普通采样没有副作用），行为与第五十五关一字不改；`triton` 后端两条路用的
是**两套**随机机制，所以路由必须严格按配置。

### 第 4 层：验证内核（每项，逐位置）

**每个位置**只有四种比值档 × 「是不是终止 token」：

| 比值 `p[d]/q[d]` | 动作 | 消费随机事件 |
|---|---|---|
| 非法（`q[d] = 0`） | `error = 1`，整项作废 | 0 |
| ≥ 1 | 必接受 | 0 |
| ≤ 0 | 必拒绝，停止 | 0 |
| 落在 (0,1) | 抽一次 uniform 决定 | 1 |
| （贪心却落进 (0,1)） | `error = 3`（不可达；前提被破坏时才响） | 0 |

**每项**把上面 8 种情况折叠成四个量——这就是「融合」发生的地方：

| 量 | 取值 | 不变量 |
|---|---|---|
| `kind` | `ALL_ACCEPTED` / `FIRST_REJECT` / `ACCEPTED_EOS` | 见 §2 |
| `accepted` | 0..K | `ALL_ACCEPTED ⟺ accepted == K` |
| `consumed` | 0..K | 只数「真的抽了」的位置 |
| `error` | 0 / 1 / 2 / 3 / 4 | `≠ 0` ⟹ **整项作废，另外三个量不必满足不变量** |

于是 `_weight_rows()` 里只剩 **4 个有意义的组合**：

| `kind` | 有没有 q | 取哪一行 | 用什么分布 |
|---|---|---|---|
| `ALL_ACCEPTED` | — | 收尾行（起点 + K） | 原分布（这就是 bonus） |
| `FIRST_REJECT` | 有 q | 拒绝点行（起点 + `accepted`） | `max(p − q, 0)` |
| `FIRST_REJECT` | 没 q（ngram） | 同上 | 挖掉被拒那一枚（`dig_out`） |
| `ACCEPTED_EOS` | — | — | **不抽样** |

### 第 5 层：打包与随机流

| 量 | 公式 | 依赖 |
|---|---|---|
| `lengths` | `accepted + (kind != ACCEPTED_EOS)` | kind |
| `kept` | `1 + accepted − (kind == ACCEPTED_EOS)`，调用方再 `+ (num_real_inputs − 1)` | kind + 计划层 |
| `rng`（消费的事件数） | `consumed + (drew & ~greedy & errors == 0)` | kind + greedy + error |
| 列布局 | `pad = max(kmax, 1)` | 批内最大 K（含整批 K=0 的退化形状） |

随机流那侧还有三个轴：**事件编号**（本请求累计 + 本轮第几次）、**token 下标**、**32 位边界**
（起始越界在主机侧查，抽着抽着跨过由内核查，错误码 4）。

## 2. 不变量：它们的作用是**删掉不可能的组合**

| 不变量 | 删掉了什么 |
|---|---|
| `kind == ALL_ACCEPTED ⟺ accepted == K`（K=0 时 0 == 0 也算） | 「全接受但 accepted < K」这个组合；`_weight_rows()` 因此不必写 `where(kind==0, K, accepted)` |
| `kind == ACCEPTED_EOS ⟹ accepted ≥ 1` | 「0 枚接受却停在 EOS」 |
| `kind == FIRST_REJECT ⟹ K ≥ 1` | **K=0 的项永远不碰纠正分布**——它那份 `dig_out` 是垃圾但一定被 `where` 丢掉 |
| `draft_probs is None ⟺ ngram 项 或 K=0 项` | 「draft_model 的 K>0 项没有 q」这个组合 |
| `has_q_item ⟺ 有 q`（按项；位置级别那一份由 `proposal_row_index >= 0` 就地算） | 「同一项里有的位置有 q、有的没有」 |
| 贪心 ⟹ `ratio ∈ {0, 1/q[d]}`（p 是 one-hot） | 「贪心落进 (0,1) 需要抽随机数」——不可达，真落进去报错误码 3 |
| `num_reserved_drafts > 0 ⟹ draft_model` | 「ngram 却预留了名额」 |
| `mode == greedy_fast ⟹ backend == torch` | 「triton 走 argmax 快路径」 |
| `speculative ⟹ 本轮要采样`（`can_sample`） | 「中间的重算 chunk 进验证批」 |
| 报错的项 ⟹ 整项作废 | 「报错但字段还得自洽」——不用管 |

**读代码时的用法**：看到一个 `where` / 掩码，先找出它在用哪条不变量把某几种组合排除掉。找得到，
这段代码就是可读的；找不到，多半是我漏写了注释（那就问，或者补上）。

## 3. 每个函数真正要装几个量（比看上去小得多）

| 函数 | 表面组合 | 实际要同时装的 |
|---|---|---|
| `verify_prefix_kernel` 循环体 | 4 档比值 × EOS × 贪心 = 12 | **4 个分支**，输出折叠成 `kind`(3) + `consumed`(计数) + `error` |
| `_weight_rows()` | `kind`(3) × 有没有 q(2) | **4 个组合**（§1 那张表） |
| `_pack()` 的三个公式 | `kind`(3) × greedy(2) × 有无错(2) = 12 | **1 个布尔**（`drew`）+ 3 个公式 |
| `_needs_verification()` | backend(2) × speculative(2) × 有草稿(2) = 8 | **3 种可达** |
| `_commit_verified()` | — | 1 个公式 + 计划层的 `num_real_inputs` |

## 4. 容易混的轴（这一关真实翻过的车）

| 翻车 | 混掉的两个轴 |
|---|---|
| 布局里少了收尾行，读到别人的分布 | 「草稿行」与「分布行」两套编号 |
| 被拒位置没夹住 → CUDA 设备断言 | 「本项内的位置」与「全局位置」 |
| K=0 的投机行切回旧随机流 | 「有没有草稿」与「是不是投机行」 |
| 重算末块切回旧随机流 | 「KV 怎么算」与「用哪条随机流」 |
| 抢占恢复把 KV 裁短（10 → 9） | 「本轮算了几个真实 token」与「隐含的 1 枚 pending」 |
| `has_q` 与 `has_q_item` 两张表 | 「按位置」与「按项」的粒度 |
| 溢出判据按「计数器指向的下一位」 | 「已经用到的编号」与「下一个编号」 |
| 混合批里普通采样与验证两份名单重叠 | 「要不要验证」与「有没有草稿」（两个表达式各写了一遍） |
| 贪心项白算一次抽样（已回退） | 「计算了」与「用到了」（counter RNG 无状态，算了不用没有副作用） |

规律很一致：**每一次都不是算错，而是两个轴被当成了一个轴。**

## 5. 读这段代码的配方

1. **先问它在哪个空间里工作**：位置空间（草稿位置，ΣK）、行空间（分布行，Σ(K+1)）、
   请求空间（每项一行）——三套编号，别混。
2. **再问它依赖哪几个轴**：拿 §1 的表对一遍；派生轴（`can_sample`/`speculative`/
   `draft_probs`/`has_q_item`）不必记，记它的来源。
3. **找它的不变量**：§2 那张表。找不到就是注释的缺口。
4. **看它丢掉了什么**：掩码写给谁看？哪些组合是「算出来但会被丢掉」的？
   （本关的所有掩码都是这个模式——代价是少量白算，换来没有分支、没有数据依赖形状、
   没有 GPU 标量读取。）
