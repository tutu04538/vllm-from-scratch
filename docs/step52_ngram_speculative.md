# step52：单请求贪心 n-gram 投机解码

- 对应代码：`step52/`（新增，从 `step51/` 复制，入口改名 `step52.py`）
- 包摘要 SHA256：`a104e65f512950ea…`（16 个 .py / 3747 行，验收方 `source_digest()` 口径）
- 基线：`step51/`，指纹 `7115f26936cf9472…`（15 个 .py / 3425 行），原样保留未改
- **改动 5 个文件 + 新增 1 个**：

| 文件 | 改动 |
|---|---|
| `speculative.py`（新增） | 两个纯函数：`propose_ngram()` 提议、`verify_drafts()` 验证 + `DraftVerification` |
| `scheduler.py` | `_plan_drafts()` 按四个上限缩短 K；`_reserve_blocks()` 容量不够时逐枚缩草稿；计划项新增 `draft_ids` / `start_cache_length`；重算统计改按真实起点；**新增 `_check_request_ids()` 入口校验**（§1.8）；**删掉 `preemption_mode` 参数、`num_uncomputed == 0` 兜底与 `running_snapshot`**（§1.8~§1.10） |
| `cache.py` | 新增 `truncate()`（回滚 KV）与 `can_grow()`（只读容量查询）；**删掉 `over_subscribe`、`_available_blocks()` 与承诺额度记账**（§1.9） |
| `engine.py` | 投机配置与组合校验；`_sample_plan()` 改为「每个请求取几行」；`_commit_tokens()` / `_commit_drafts()` 拆出提交点；**删掉 `preemption_mode` 与相关的两条组合校验**（§1.9） |
| `request.py` | 删掉 `promised_blocks` 字段（§1.9） |
| `__init__.py`、`step52.py` | 包说明、入口改名 |

`model.py`、`attention.py`、`norm.py`、`rope.py`、`sampling.py`、`sampler.py`、`formats/` 未改。

## 0. 需求大概

投机解码的第一个闭环，范围**只有单请求、贪心、n-gram 提议**：

已有历史 `[..., x]`（`x` 是还没写进 KV 的最后一个真实 token），n-gram 猜 `[d0, d1]`。
本轮目标模型输入 **`[x, d0, d1]`**，一次返回三行 logits，对应贪心 token `[t0, t1, t2]`：

| 验证结果 | 真正提交的输出 | 保留哪些本轮输入的 KV |
|---|---|---|
| `t0 != d0` | `[t0]` | `[x]` |
| `t0 == d0, t1 != d1` | `[d0, t1]` | `[x, d0]` |
| `t0 == d0, t1 == d1` | `[d0, d1, t2]` | `[x, d0, d1]` |

`t2` 是全部接受时的 **bonus token**，还没进过模型。被拒绝的草稿绝不能写进请求状态、
不能触发惩罚计数或回调。

这不是「让模型变快」的优化，而是**把闭环做对**：草稿 → 目标验证 → 只提交认可的 →
回滚 KV。加速与否取决于接受率与 K+1 行验证的额外计算，本关不承诺。

## 1. 实现

### 1.1 提议：一个纯函数

```python
def propose_ngram(token_ids, n, k):
    length = len(token_ids)
    if n <= 0 or k <= 0 or length <= n:
        return []
    pattern = token_ids[length - n:length]
    for start in range(length - n - 1, -1, -1):
        ...                                  # 逐位比较 n 个 token
        return token_ids[start + n:min(start + n + k, length)]
    return []
```

三个刻意的选择：

- **从靠近末尾处往前找**，第一个命中的就是「最近一次出现」；
- **匹配片段绝不与末尾的 n 个 token 重叠**（起点上界 `length - n - 1`），否则会自己匹配自己；
- **只按下标访问，不复制整段历史**。`token_ids` 可以是第五十一关的只读视图，
  全程只有末尾的 pattern 与命中的续写会被切片出来（都不超过 `max(n, k)` 个）。
  写成 `list(history)` 会把第五十一关刚消掉的开销又加回来。

复杂度 `O(len(history) * n)`，命中位置越靠后越快：末尾附近就有匹配时是微秒级，
最坏情况（一路无匹配、8192 长历史走满全程）实测约 511 μs/次。本关先要正确性，
这是唯一明显更慢的一处；真要上规模应当换成哈希索引（见 §4.2 遗留 4）。

### 1.2 验证：另一个纯函数

```python
@dataclass
class DraftVerification:
    num_accepted: int      # 前几枚草稿与目标贪心相同
    committed_ids: list    # 真正要提交的输出（遇到 EOS 就到此为止）
    kept_inputs: int       # 本轮输入的 KV 要保留几个
    stopped: bool          # 是否因 EOS 提前停下
```

四条规则，逐条对应需求 §1 与 §3.D：

1. 逐枚比较 `greedy_ids[i] == draft_ids[i]`，第一个不同就停；
2. 候选 = `draft_ids[:accepted] + [greedy_ids[accepted]]`；
3. 逐枚提交，**遇 EOS 立即停**，后面的草稿与 bonus 都不提交；
4. 保留的输入 = `1 + 还需要当下一轮输入的草稿数`：
   - 被拒绝的草稿一律不留（就是需求表里的「保留哪些 KV」）；
   - **被接受的草稿本身若是终止 token，它也不留**——它结束了这条请求，
     不会再作为下一轮输入，所以要留的是它**之前**那些草稿（`kept_drafts = index`）。

`kept_inputs` 是「回滚到哪儿」的唯一依据，所以把它算成一个显式的返回值，而不是
在提交循环里边走边推。

#### 1.2.1 `greedy_ids[a]` 不是「两种情况」

第 2 条那个式子，全部接受（`a == K`）与部分接受（`a < K`）走的是**同一行代码**。
别读成「全接受时多给一枚 bonus、部分接受时换成纠正 token」两条分支——`greedy_ids[a]`
两种情况是同一枚东西：目标模型对位置 `p+a+1` 的预测，而且**都还没进过模型**
（`a == K` 时那个位置压根没有输入行；`a < K` 时那个位置是刚被否掉的 `d_a`，
它的 KV 马上要被回滚）。两者都会成为下一轮的「最后一个真实 token」。

要分情况的是**回滚到哪儿**：

| | 位置 `p+a+1` 上有没有输入行 | 回滚 |
|---|---|---|
| `a == K` | 没有（模型没被问过那里） | 不用回滚，`kept_inputs = 1+K`，整段输入都留 |
| `a < K` | 有，就是 `d_a`（KV 已经写好了） | 从 `d_a` 起丢掉，`kept_inputs = 1+a` |

「bonus」这个词只描述全接受那一支，代码里没有对应的分支。

#### 1.2.2 输出上限是前置条件，不是截断阈值

函数要求 `K <= R-1`（`R = remaining_outputs`，`_plan_drafts()` 正是这么缩 K 的），
**不满足直接 `ValueError`**。之前这里写的是「提交到 R 枚就停」的截断分支，
但它是**死代码**：`K+1 <= R` 是构造保证的，循环最多提交 `K+1 <= R` 枚，
永远够不到那个阈值（实测引擎路径 400 组 + 引擎测试 11 次调用，触发 0 次）。

删掉它而不是留着，是因为那段截断产出的状态**自相矛盾**：`kept_inputs` 按接受的
草稿数算，会比实际提交的 token 数还多，于是 `cache.length` 要么超过
`len(all_token_ids)`（KV 进度比历史还长），要么正好相等（就是「历史已算完却还不是
ready」那个状态，§1.8 刚把它的兜底删掉）。与其默默产出这种状态，不如把前置条件
钉死、直接报错。

顺带一说，**草稿里是可以出现 EOS 的**：草稿来自 `all_token_ids` = prompt + 已生成，
已生成的 EOS 不可能出现（有 EOS 请求当场就结束了），但 **prompt 里可以有**——
Qwen3 的 chat template 每一轮用户消息都以 `<|im_end|>` 结尾，而它就是
`eos_token_ids` 里的一员（实测单轮 prompt 出现 1 次、三轮 3 次）。所以规则 4 里
最后那条「被接受的草稿本身是 EOS」不是构造出来的边角情况，实际跑得到。

### 1.3 草稿是临时计划，不是已提交历史

草稿只出现在本轮 `scheduled_item` 里：

```python
item["input_ids"]           = 历史片段 + draft_ids     # 传给模型的真实输入
item["draft_ids"]           = draft_ids                # 供验证与回滚使用
item["num_scheduled_tokens"] = 1 + len(draft_ids)      # K+1，不是 1
item["start_cache_length"]   = start                   # 本轮从哪儿开始算
```

`seq._all_token_ids` / `_output_ids` 直到 `_commit_tokens()` 被调用之前**一个草稿都没有**。
提交点只有一个（`_commit_tokens`），普通路径与投机路径共用它，所以「历史、惩罚计数、
`on_token` 回调」三者同进同退这件事没有被复制成两份。

### 1.4 K 的四个上限（全部在计划阶段，只读）

```python
k = min(num_speculative_tokens, spare_budget, remaining_outputs - 1)
if max_seq_len is not None:
    k = min(k, max_seq_len - cache.length - 1)
```

- **剩余 token 预算**：`1+K` 个 token 同样占 `max_num_batched_tokens`；
- **剩余输出上限**：全部接受还要多一个 bonus，所以 `K <= R - 1`；
- **剩余上下文长度**：本轮要算 `[cache.length, cache.length + 1 + K)`。

`K <= 0` 就返回空列表——**普通的 1-token 路径不是特例，是 K=0 的退化情形**，
不需要另写一条分支。

第四个上限是**容量**（物理块），它不在计划里判，而在 `_reserve_blocks()` 里：

```python
while item["draft_ids"] and not self.kv_cache_pool.can_grow(seq, item["num_scheduled_tokens"]):
    self._shrink_draft(item)          # 逐枚缩，最终退回普通 1-token 路径
```

`can_grow()` 是只读的（只沿可分配链看前 k 个够不够，不摘链、不清 hash），
真正补块仍然只有 `ensure_blocks()` 一处——这和第四十八关以来的「先计划、后提交」
是同一条规矩。

**要说实话：这个缩短循环在当前允许的配置下走不到。** 准入已经按最坏情况
`ceil((len(prompt) + max_new_tokens - 1) / block_size)` 把额度承诺给了这条请求，
而 `K <= R-1` 保证本轮要的块数不超过那个最坏值，池子里又只有这一条请求
（`max_num_seqs=1`），所以 `ensure_blocks()` 必然成功。它是安全网，不是热路径；
单测里直接验 `can_grow()` 的判断本身，端到端跑不到它。

### 1.5 一次 forward，取 K+1 行

`_sample_plan()` 原来给每个就绪请求取「片段末行」，现在：

```python
if item["draft_ids"]:
    item["num_sample_rows"] = item["num_scheduled_tokens"]   # 整段都要
    rows.extend(range(offset - n, offset))
else:
    item["num_sample_rows"] = 1
    rows.append(offset - 1)
```

投机项的整段输入就是 `[x, d0, d1]`，K+1 行全要：前 K 行验证草稿，最后一行给 bonus。
普通项一个字节都没变（仍只取末行），既有路径的批量选 token（一次 `.tolist()`，
不逐请求 `.item()`）也原样保留。

验证就是逐行 `argmax`：本关把投机限制在**贪心且无惩罚项**（构造时与 `add_request()`
时都校验），所以不需要走 `apply_penalties`，批量算一次即可。并列时 `argmax` 返回
下标最小的那个，与 `TorchSampler` 的贪心路径一致。

### 1.6 回滚：`KVCachePool.truncate()`

```python
keep_blocks = math.ceil(new_length / block_size)
for block_idx in seq.cache.block_table[keep_blocks:]:
    self.block_usage[block_idx] -= 1
    if self.block_usage[block_idx] == 0:
        self._release_block(block_idx)     # 无 hash 回队首，带 hash 留作闲置缓存
del seq.cache.block_table[keep_blocks:]
seq.cache.length = new_length
```

只动两样：`cache.length` 与多占的**整块**。仍在用的完整块不动；不完整的尾块以后
会被新内容覆盖，所以既不写 KV、也不碰 hash 条目。

**回滚必须在 `post_step()` 之前。** 顺序是：模型 forward（`cache.length` 已经推进到
`start+1+K`）→ `_commit_drafts()` 里先 `truncate()` 再提交 → `post_step()`。
这样 `publish_computed_blocks()` 读到的才是回滚后的真实进度，被拒绝的草稿不会被
登记成可复用前缀。

两条让它安全的结构性事实（不是巧合，都值得盯住）：

- **回滚永远不会退到已发布的块**：本步的 `new_length >= start + 1`，而发布过的块是
  `[0, floor(start / block_size))`，两者不相交。所以不会把已登记的前缀块退掉，
  更不会让别的请求命中被拒绝草稿的内容；
- **被退掉的整块没有 hash**：它从没被发布过（发布要求 `cache.length` 覆盖整块），
  所以它是「真正空闲」块，回队首优先复用，不会留下脏的缓存条目。

### 1.7 调度记录：进模型的输入数 vs 验证后保留的输入数

`num_scheduled_tokens` **保持 `K+1`**——那是 GPU 真的算过的 token 数，改小它会让
「本轮算了多少」这件事对不上账。但 `_update_recompute_metrics()` 原来用
`end - num_scheduled_tokens` 反推起点，回滚之后那个 `end` 是**最终保留**的长度，
相减得到的位置是错的（本步会凭空记出 K 个「重算」）。

所以计划里显式记下 `start_cache_length`，统计按真实起点算：

```python
start = item["start_cache_length"]
end = seq.cache.length
seq.recomputed_tokens += max(0, min(end, seq.high_water) - start)
seq.high_water = max(seq.high_water, end)
```

端到端用例直接盯这一点：首枚拒绝的那一步进了 3 个 token、只保留 1 个，
`recomputed_tokens` 必须是 0（按旧公式会是 2）。

### 1.8 顺带：请求内容的入口校验，以及随之删掉的一句兜底

`add_request()` 一直只校验三样东西：**请求字典里的字段名**、**priority 是不是整数**、
**采样参数的范围**。`prompt_ids` 的内容与 `max_new_tokens` 的符号没人查过，于是坏输入
会一路走到很深的地方才炸——或者更糟，静默算错：

| 输入 | 补校验之前 | 补校验之后 |
|---|---|---|
| `max_new_tokens = -1` | **请求静默消失**：`_finish_zero_budget_waiting()` 只给 `== 0` 补 `on_finished` 记录，紧接着的过滤器写的却是 `max_new_tokens > 0`，负数被从 waiting 里直接滤掉，既不回调也不报错 | `ValueError`：「不能为负；0 表示只算 prompt、不生成」 |
| `prompt_ids` 里有浮点 | **静默截断**：`seq.prompt_ids` 保留 `2.5`（prefix hash 按它算），`torch.tensor(..., dtype=torch.long)` 交给模型的是 `2`。跑得完、有输出、零提示 | `ValueError`：「prompt_ids[1] 必须是整数（bool 不算），收到 2.5」 |
| `prompt_ids` 越界 / 为负 | `IndexError: index out of range in self`，从 embedding 里冒出来 | `ValueError`：「超出词表范围 [0, vocab_size)」 |
| `prompt_ids = []` | 调度器既排不出 token 也无法判停，靠零进展守卫兜底报错 | `ValueError`：「不能为空：至少要有一个 token 才能预测下一个」 |
| `max_new_tokens = 0` | 合法：只算 prompt、不生成（有专门路径） | **不变**，仍然合法 |

校验放在 `add_request()` 里、和现有那三样同一处，所以在线程入队、分配 KV **之前**
就报出来。空 `prompt_ids` 与 `max_new_tokens=0` 的组合也拒绝——空历史本身不合法，
与「要不要生成」无关。

**顺带删掉 `_plan_tokens()` 里的半句兜底**：

```python
if num_uncomputed == 0 or prefill_token_budget == 0:   # 改之前
if prefill_token_budget == 0:                          # 改之后
```

`num_uncomputed == 0` 那一半是「历史已算完、却又不是 ready」这个状态的兜底，而那个
状态**只有空 prompt 能造出来**（其余路径要么准入时只借到 `len(all)-1` 之前的块，
要么本轮算到历史末尾就在同一步采样、历史立刻长回 1 个）。入口拒绝空 prompt 之后
它就成了死代码——而且是个**有害**的死代码：万一哪天记账真错了，它会把「这条请求
永远排不动」伪装成「本轮没额度」，把零进展守卫的报错一起吞掉。

**这不是「防御性检查冗余所以删掉」，是「不变量现在由入口保证，兜底反而掩盖错误」。**
删的同时留了注释说明为什么不变量成立，并加了回归用例：`max_num_batched_tokens=1/2`
的预算压力下，计划里不允许出现 0 token 的项。

### 1.9 抢占不再有模式开关：删掉 `preemption_mode` 与 `over_subscribe`

**起因**：`preemption_mode` 在调度器里其实是**死状态**——`self.preemption_mode` 写了
从来没人读，真正决定行为的是池子的 `over_subscribe`。也就是说这个「模式」早就退化成
「一个用来推导布尔量的参数」，却让两块互不相干的代码（请求准入记账 / 抢占策略）看起来
绑在一起，还顺带允许了「承诺式的调度器 + 超卖的池子」这种自相矛盾、又不报错的组合。

**vLLM V1 没有这个开关**（0.28 全包 `grep preemption_mode` 零命中）。抢占是无条件的：

```python
while True:
    new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens, ...)
    if new_blocks is not None:
        break
    # The request cannot be scheduled. Preempt the lowest-priority request.
    if self.policy == SchedulingPolicy.PRIORITY:
        preempted_req = max(self.running, key=lambda r: (r.priority, r.arrival_time))
    else:
        preempted_req = self.running.pop()
    self._preempt_request(preempted_req, ...)
    if preempted_req == request:
        break
```

牺牲者的选法（`(priority, arrival_time)` / 尾部一条）、抢占后的处理
（`num_computed_tokens = 0` 全量重算、`waiting.prepend_request()` 插队首）、
连 `policy: "fcfs" | "priority"` 无条件可用——都和我们的实现同构。V0 时代有过
`preemption_mode: "swap" | "recompute"`，V1 把 swap 删掉，开关也一起没了。

**删掉的东西**：

| 位置 | 内容 |
|---|---|
| `Engine` | `preemption_mode` 参数（两个入口）、`_check_preemption_mode()`、`_check_scheduling_policy()` 里的 priority 闸门、`self.preemption_mode` |
| `Scheduler` | `preemption_mode` 参数与 `self.preemption_mode`（本来就是死状态） |
| `KVCachePool` | `over_subscribe` 参数、`_available_blocks()`、`promised_blocks` 的全部记账（`_plan_admission()` 的容量检查、`_commit_admission()` / `_commit_block_growth()` / `deallocate_block()` 里的增减） |
| `SequenceConfig` | `promised_blocks` 字段 |
| `_plan_block_growth()` | 承诺式那句「准入记账出错了」的 `raise`——现在不够就返回 `None` |

**保留 `InfeasibleRequest`**：它判的是「这条请求**单独跑**装不装得下」，与容量策略无关
——装不下就是装不下，谁让路都没用。删掉它，这种请求会永远排不动，最后撞上零进展守卫。

**准入现在只判一件事**：`max_request_blocks > num_kv_blocks` 就拒绝。所以池子小的时候
可以准入超过池子容量的请求数，不够时靠抢占腾——和 vLLM 一样。

**行为影响**（默认配置、fcfs、同 seed 同请求）：

最小的一例——池子 6 块、`block_size=4`、`max_num_seqs=2`，两条请求各要 4 块：

| | 旧默认（承诺式） | 新默认（无条件抢占） |
|---|---|---|
| 准入 | A 锁 4 块 → B 只剩 2 块不够 → B 一直等 | A、B 都进来，按需拿块 |
| 过程 | A 独占跑完，B 再跑 | B 被抢占 1 次、重算 |
| 目标 forward 次数 | 12 | **10** |
| 输出 | `A=[…] B=[…]` | **逐 token 相同** |

**输出永远相同**：抢占只改「什么时候算」，不改模型算什么。

| 扫描 | 结果 |
|---|---|
| 288 组 fcfs 配置（其中 102 组真的发生过抢占） | 输出、完成顺序、结束态资源**全部一致** |
| 160 组刻意压小的池子（1~4 块） | 完全一致 |
| 90 组 fcfs 配置的步数汇总 | 新默认更少 18 组、相同 72 组、**更多 0 组**（合计 −57 步） |

**这是一次行为变更，而且是默认配置上的**：承诺式是第三十六到四十三关的历史行为，
step44~51 的验收都以它为基线。旧行为现在只能从旧包里拿到（`step51/` 及更早原样保留），
step52 起没有开关可以还原——这正是我们想要的（一个已经不存在的模式不该留个开关假装还在）。
逐块对照脚本 `diff_step51_step52.py` 因此改成给 step51 显式传
`preemption_mode="recompute"`，再和 step52 比：11 个场景 × 2 seed **全部逐项一致**，
这同时也就是「删干净了、没有顺手改坏别的东西」的证据。

### 1.10 顺带：删掉一份已经不起作用的快照

`_reserve_blocks()` 里有一句 `running_snapshot = list(self.running)`，把它传给
`_make_room()` 当候选集合。它是**第四十四关留下的**，当时确实需要：

```python
# step44 的 _make_room
for victim in reversed(running):      # 直接遍历传进来的列表
    ...
    self._preempt(victim)             # 而 _preempt 会 self.running.remove(seq)
```

边遍历边删同一个列表，后面的元素会被跳过。传一份快照进去，遍历的就是不会变的那一份。

**第四十八关的重构让它的作用消失了**：候选的选择被抽成 `_victims_after()`，而它
**先把候选物化成新列表再返回**；`_make_room()` 遍历的是那个新列表，`_preempt()` 改
`self.running` 动不到它；本轮已被抢占的候选再由 `victim not in self.running` 滤掉。
加上 `_reserve_blocks()` 期间 `self.running` 只减不增，于是

```text
活列表 ⊆ 快照          差集 = 本轮已被抢占的那些 = 会被那句 continue 滤掉的
```

也就是**两者可证明等价**，快照纯粹是多余的。删法是连参数一起删：`_make_room(seq, num_tokens, running)`
→ `_make_room(seq, num_tokens)`、`_victims_after(seq, running)` → `_victims_after(seq)`。

**与 §1.8 删掉的那半句不同**：那个是**有害**的（会吞掉零进展守卫的报错），这个是**无害**的
（一步一次 `list()`，代价可忽略）。删它是因为「读到的集合」这个信息本来就不该由调用方钉成
快照——`_make_room()` 到底该按哪一刻的集合选人，快照反而让语义变含糊。真正要守住的那条
约束改用注释写在 `_victims_after()` 里：**它必须先把候选物化成列表**，将来谁把它内联回
`_make_room()` 就会重新踩上第四十四关的坑。

**等价性验证**（不是「测试都过了」，是差分）：800 组随机场景（fcfs/priority、池子 1~10 块、
`block_size` 1/2/4、动态到达、prefix 开关）的逐步轨迹指纹，删前删后**完全相同**：

```text
1df45fd65c007a572addf74c180acfc9
```

## 2. 不变量

```text
len(all_token_ids) == len(prompt_ids) + len(output_ids)
0 <= cache.length <= len(all_token_ids)          对仍在运行的请求
cache.length - start_cache_length <= kept_inputs  只保留被认可的输入
len(block_table) == ceil(cache.length / block_size)
```

§1.9 之后**没有「已承诺额度」这条不变量了**——准入不再锁未来容量，账本随之消失。

**投机每一步之后仍满足 `num_uncomputed_tokens == 1`**——这是它能连续投机的根据：
全部接受时 `cache.length` 与历史同步推进 `K+1`，首枚拒绝时只推进 1 而历史也只长 1。

被拒绝草稿的 KV 可能落在**不完整尾块**里，下一轮按位置覆盖写；落在被退掉的整块里
则整块回池子。两条路径都不会让拒绝的内容被当成命中。

## 3. 验证

### 3.1 纯函数与回滚（`benchmarks/check_step52_speculative.py`，53 项全通过）

- `propose_ngram`：找不到 / 找到 1 个 / 找到 2 个 / **多个匹配取最近** / `k` 截断 /
  `n=1` / 不自己匹配自己 / 直接吃只读视图；8192 长历史不复制整段；
- `verify_drafts`：三种验证结果 + 无草稿 + bonus 是 EOS + **草稿本身是 EOS** +
  第二枚草稿是 EOS + **输出上限不够时报错（不截断）** + 卡在 `K+1 == R` 边界仍正常
  + 行数不匹配报错；
- `truncate`：回滚到块边界 / 块内 / 0；非法目标报错；**不动已发布的完整块与其
  hash 双向索引**；被退掉的整块确实以「真正空闲」回队首；
- `can_grow`：只读（问完池子与请求状态一个字节没变）、判断与可分配链一致、不摘链；
- `_plan_drafts` 的四个上限逐条验：预算 / 输出上限 / 上下文 / 配置值，以及
  「还没生成过 token」「还差不止 1 个 token」两种不投机的情形；
- 配置校验：6 种不支持的组合与 3 种非法采样参数都**明确报错**，
  关掉投机后这些组合仍然合法。

### 3.2 端到端（`benchmarks/check_step52_engine.py`，40 项全通过）

**脚本模型**：包住真模型，KV、位置、块表照常由真实现推进，只把 logits 换成脚本
给定的 token。于是「目标模型会输出什么」完全可控，能造出确定性的用例：

| 用例 | 断言 |
|---|---|
| 全部接受 | 一次 forward 提交 `[2,3,7]`；`on_token` 序号连续；`on_finished` 只发一次；同样文本下投机 4 次目标 forward，普通贪心 6 次 |
| 首枚拒绝 | 草稿 `[2,3]` 提出来了，那一轮只提交 `[9]`；`num_scheduled_tokens=3`、`start_cache_length=8` |
| 部分接受 | 提交 `[2,9]`，第二枚之后的草稿不提交 |
| 草稿本身是 EOS | 提交 `[63]` 就停；提交时 `cache.length` 停在 `start+1`（只留 `x` 的 KV） |
| 找不到草稿 | 每一轮都是普通 1-token 路径 |
| 预算只够 1 个 | 草稿长度恒为 0 |
| `max_new_tokens=1` | 不投机（bonus 没地方放） |
| 上下文将满 | 没有任何一轮越过 `max_seq_len`，输出仍正确 |
| 回滚不变量 | 跨块边界反复回滚时，块表长度始终等于 `ceil(cache.length/4)` |
| 重算统计 | 首枚拒绝那一步 `recomputed_tokens == 0`、`high_water == start+1` |
| 结束状态 | 活动引用归零、链表成员等于真实可分配集合、无残留 hash、承诺归零 |
| 入口校验 | 空 prompt / 负 `max_new_tokens` / 浮点或越界 token / bool / 不可迭代 都**在 `add_request` 就报错**；`max_new_tokens=0` 仍合法 |
| 幽灵计划项 | 预算压到 1 个 token 时，计划里没有 0 token 的项（§1.8 删掉的那半句兜底的回归） |

**真模型等价性**：同一个随机小模型、同一个 prompt（重复片段多，n-gram 更容易命中），
`speculative_mode="ngram"` 与 step51 的普通贪心**逐 token 相同**：
`[54,6,54,6,…]`，且确实有一轮一次 forward 提交 3 枚（8 步产出 12 个 token，
普通贪心要 12 步）。这是本关最强的一条检查——接受也好拒绝也好，提交的永远是目标
模型自己的贪心 token，所以投机**不能**改变结果。

### 3.3 抢占：与旧的重算模式逐字节一致（`benchmarks/diff_step51_step52.py`，88 项全通过）

11 个场景 × 2 seed，逐步比较调度队列、本轮计划（含 `draft_ids`）、输入 token、
完成输出、每请求计数、抢占计数、**物理块编号**、**每请求 hash 链**、结束态。

§1.9 之后 step52 不再接受 `preemption_mode`，所以脚本给 **step51 显式传
`preemption_mode="recompute"`**，再和 step52 比——两者应当逐字节一致，**没有放宽
任何字段**。这一条同时是 §1.9「删干净了、没有顺手改坏别的东西」的证据：
删掉的那一堆记账在重算模式下本来就都是空操作。

（与 step51 的**默认**配置相比则不再一致：§1.9 有量化对比。）

### 3.4 设备与精度

| 配置 | 结果 |
|---|---|
| CPU / FP32 | 8 步产出 12 token，提出草稿 3 轮，引用归零 |
| CUDA / FP32（Torch attention） | 同上；投机 8 步 vs 普通 14 步，输出逐 token 相同 |
| CUDA / BF16（Torch attention） | 同上；投机 8 步 vs 普通 14 步，输出逐 token 相同 |

需求只要求「CPU 精确测试 + 一组 CUDA/Torch 冒烟」，这里是三档同一用例。

**本关不做吞吐矩阵**（普通关卡不做性能测试）：上面的步数只是「调用了几次目标模型」
的计数，不是端到端吞吐，也不构成加速承诺。

## 4. 接口变化与遗留

### 4.1 接口变化

**新增**：`Engine(...)` / `from_model_dir(...)` 的 keyword-only 参数
`speculative_mode=None|"ngram"`、`num_speculative_tokens=2`、`prompt_lookup_n=2`；
`step52.speculative.propose_ngram()` / `verify_drafts()` / `DraftVerification`；
`KVCachePool.truncate()` / `can_grow()`。

**计划项新增字段**：`scheduled_items` 的每一项多了 `draft_ids`、`start_cache_length`、
`num_sample_rows`。`num_scheduled_tokens` 的含义**变精确了**：它是本轮进模型的
token 数（投机时是 `K+1`），不是「验证后留下的 token 数」——后者要看
`cache.length - start_cache_length`。

**删除的参数**（传了会 `TypeError`，**不静默忽略**——一个已经不存在的开关不该假装还在）：

- `Engine(...)` / `from_model_dir(...)` 的 `preemption_mode`；
- `KVCachePool(...)` 的 `over_subscribe`；
- `Scheduler(...)` 的 `preemption_mode`（本来就是死状态）；
- `SequenceConfig.promised_blocks`、`KVCachePool.promised_blocks`、`KVCachePool._available_blocks()`。

**行为变化**：分成三类。

1. 只与投机有关的：只在 `speculative_mode="ngram"` 时发生，且该模式的合法组合被
   明确限定（见 §4.2）。`speculative_mode=None` 时投机相关的代码一行都不走。
2. **抢占变成无条件的**（§1.9）：默认配置下不再有「准入时按最坏情况锁未来块」的行为。
   step52 的行为等价于「step51 + `preemption_mode='recompute'`」，与 step51 的**默认**
   配置在池子小的时候不同（输出相同，步数与抢占次数可能不同）。要旧默认行为请用旧包。
3. **与两者都无关、所有配置都生效**：`add_request()` 现在拒绝 `prompt_ids=[]`
   （含与 `max_new_tokens=0` 的组合）、`prompt_ids` 里的非整数（bool 不算）或越界值、
   不可迭代的 `prompt_ids`、负的 `max_new_tokens`。这几种输入以前分别是
   「静默消失 / 静默截断 / IndexError / 守卫兜底报错」，见 §1.8 的对照表。
   `max_new_tokens=0` 仍然合法。

**未改**：调度顺序与优先级规则、KV 按需分配与淘汰、prefix hash 算法与命中规则、
采样结果、模型；`_shrink_draft()` 之外没有新的分配路径。

### 4.2 遗留

1. **只做单请求**：`max_num_seqs=1`，`_sample_plan()` 不做「多个请求各要不同行数」的
   混合行映射（结构上支持「每个请求若干行」，但只被单请求验证过）。
2. **只做贪心**：随机采样下「接受」的判定需要按概率比决定，本关明确拒绝非贪心请求。
3. **不做前缀缓存 / CUDA Graph 的组合**：那几块要么会与「被拒绝草稿的 KV」相互影响
   （prefix），要么与行映射冲突（graph），先把最小闭环站稳。（抢占那条限制随 §1.9
   一起取消：`max_num_seqs=1` 已经蕴含不会发生抢占——running 里只有一条请求，
   fcfs 下既没有更靠后的犠牲者，也不存在「顶掉名额」。）
4. **提议只有 n-gram**：没有 draft model、没有 EAGLE/Medusa 之类；
   `propose_ngram` 是线性扫描，长历史下应当换哈希索引（本关先不动）。
5. **不做异步提议、不做 `k` 的自适应**：K 固定由配置与四个上限决定，
   不按历史接受率调整。
6. **不承诺加速**：本关没有吞吐矩阵，步数只是目标 forward 次数。
7. **准入侧没有缓冲旋钮**：§1.9 之后池子很小时可以准入超过容量的请求数，靠抢占腾地方。
   vLLM 用 `scheduler_reserve_full_isl`（准入前检查整个 prompt 装不装得下）和
   `watermark`（留一部分块不参与准入）缓解这种 thrashing，我们两个都没有。
   本关的压测（1500 组刻意压小的池子）没有出现活锁或资源异常，但那是功能性结论，
   不是「不会有 thrashing」的性能结论。
