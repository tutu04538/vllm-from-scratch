# step53：批量投机验证与抢占恢复

- 对应代码：`step53/`（新增，从 `step52/` 复制，入口改名 `step53.py`）
- 包摘要 SHA256：`9bc0f9267309bcfb…`（16 个 .py / 3798 行，验收方 `source_digest()` 口径）
- 基线：`step52/`，指纹 `9bc0f9267309bcfb…`（16 个 .py / 3798 行），原样保留未改
- **改动 4 个文件**：

| 文件 | 改动 |
|---|---|
| `engine.py` | 放开 `max_num_seqs=1`；`_sample_plan()` 记录 `sample_offset`（筛选后偏移）；`_sample()` 拆成「投机整批 argmax」与「原采样后端」两条路；`_commit_drafts()` 改为接收调用方算好的 `greedy_ids` |
| `scheduler.py` | `_check_request_ids()` **返回**物化后的 `prompt_ids`（修迭代器被消费两次）；`_plan_tokens()` 里 fcfs 那句 `assert` 改成与 priority 同款的「真实 token 优先」截断 |
| `__init__.py`、`step53.py` | 包说明、入口改名 |

`cache.py`、`speculative.py`、`request.py`、`model.py`、`attention.py`、`norm.py`、`rope.py`、
`sampling.py`、`sampler.py`、`formats/` 未改。

## 0. 需求大概

第五十二关把投机做成了单请求闭环，但一个 batch 里只能有一条请求。本关把它接进
continuous batching：**同一步里同时有不同 K 的投机请求、无草稿的普通 decode 和
中间 prefill**，池子紧张时仍然缩草稿、抢占、恢复。

不新增 draft model，也不做随机接受算法——本关的进步是把投机接入批量运行时与资源压力。

## 1. 一句话版本 + 一个 4 步的例子

行映射是本关的核心：**原始输入行号**和**筛选后 logits 的偏移**是两个坐标系，
中间 prefill 占前者、不占后者。

下面这一次运行（脚本模型固定目标输出，KV 与位置仍由真实现推进）里，第 2 步就是
需求 §2 那张表的真实版本：

| 请求 | K | 实际输入数 | 筛选后偏移 | 接受/提交 | 回滚后 `cache.length` | 本轮抢占 |
|---|---:|---:|---|---|---:|---|
| A | 2 | 3（`[x,d0,d1]`） | `[0:3]` | 草稿全中，提交 3 枚 | 8 → **11** | 否 |
| C | 0 | 1（`[xC]`） | `[3:4]` | 无草稿，提交 1 枚 | 8 → **9** | 否 |
| D | 1 | 2（`[xD,dD0]`） | `[4:6]` | 草稿全中，提交 2 枚 | 2 → **4** | 否 |
| B | — | 14（中间 prefill） | 不采样 | — | 0 → **14** | 否 |

```text
sample_rows = [0, 1, 2, 3, 4, 5]      # A 占 0..2，C 占 3，D 占 4..5，B 的 14 行不取
一次 forward，一次 argmax，一次 .tolist()
```

第 4 步更能说明两个坐标系的区别：A、C 各 1 个 token 排在前两行，B 的 prefill 收尾
13 个 token 占了**原始行 2..14**，它要采样的那行是 **14**、而在筛选后的 logits 里
偏移只有 **2**：

```text
步 4: sample_rows = [0, 1, 14]        # 原始行号 14 -> 筛选后偏移 2
```

### 1.1 完整 4 步

```text
步 1: sample_rows=[7, 15, 17]        三条 prefill 同批跑完，各取片段末行
   A: K=0 输入=8 偏移=0 提交=[1]      cache.length 0 -> 8
   C: K=0 输入=8 偏移=1 提交=[10]     cache.length 0 -> 8
   D: K=0 输入=2 偏移=2 提交=[9]      cache.length 0 -> 2
步 2: sample_rows=[0, 1, 2, 3, 4, 5] 见上面那张表（B 同批做中间 prefill）
步 3: sample_rows=[0, 1, 2]          A/C/D 各 1 枚；B 的 17 行 prefill 不采样
   A: K=0 输入=1 提交=[0]  cache.length 11 -> 12
   C: K=0 输入=1 提交=[0]  cache.length  9 -> 10
   D: K=0 输入=1 提交=[0]  已到 max_new_tokens，本轮提交后释放
   B: K=0 输入=17 不采样   cache.length 0 -> 17
步 4: sample_rows=[0, 1, 14]         A/C 收尾各 1 枚；B 的 prefill 收尾 13 行
   A: 提交=[0] 已释放   C: 提交=[0] 已释放
   B: 输入=13 偏移=2 提交=[0]  cache.length 17 -> 30
```

（这个例子全程没有抢占——抢占那一条在 `check_step53_batch.py` 的容量压力段，
那里池子只有 4 块、两条请求各占 2 块。）

### 1.2 与需求 §2 那张表的对照

需求给的示意：

```text
请求 A：K=2  输入 [xA,dA0,dA1]  采样  raw rows 0,1,2
请求 B：中间 prefill [b0..b3]   不采样  （占 raw rows 3..6）
请求 C：普通 decode [xC]        采样  raw row  7
请求 D：K=1  输入 [xD,dD0]      采样  raw rows 8,9
sample_rows = [0,1,2,7,8,9]；筛选后 A=[0:3]、C=[3:4]、D=[4:6]
```

这份对照是 `check_step53_batch.py` 第 1 节**逐字**验的（手工造 `scheduled_items`）。
端到端跑不出「prefill 夹在两条 decode 中间」那种排列——FCFS 下没预算可用的请求总是
排在后面的那些，所以批尾才是中间 prefill 的常态；两种形状的映射逻辑是同一条。

## 2. 实现

### 2.1 行映射：两个坐标系

```python
def _sample_plan(scheduled_items):
    rows, picked, offset = [], [], 0
    for item in scheduled_items:
        offset += item["num_scheduled_tokens"]          # 中间 prefill 也占原始行
        if not item["can_sample"]:
            continue
        item["sample_offset"] = len(rows)               # 在**筛选后** logits 里的起点
        if item["draft_ids"]:
            item["num_sample_rows"] = item["num_scheduled_tokens"]
            rows.extend(range(offset - item["num_scheduled_tokens"], offset))
        else:
            item["num_sample_rows"] = 1
            rows.append(offset - 1)
        picked.append(item)
    return rows, picked
```

偏移记在**计划项**上（`sample_offset`），不再让调用方按 picked 顺序累加——批量下
「谁占几行」只有这里知道，让下游各推各的迟早对不上。

### 2.2 采样：整批一次 argmax，按区间切

```python
if self.speculative_mode is None:
    self._sample_with_sampler(logits, picked, notify)    # 原采样后端，随机/惩罚项都在
    return None
greedy = torch.argmax(logits, dim=-1).tolist()      # 整批一次 argmax
for item in picked:
    ids = greedy[item["sample_offset"]:item["sample_offset"] + item["num_sample_rows"]]
    if item["draft_ids"]:
        self._commit_drafts(item, ids, notify)
    else:
        self._commit_tokens(item["request"], ids, notify)
```

三件事：

1. **投机开着时不需要走采样后端**：`add_request()` 已经保证所有请求都是贪心且无惩罚项，
   所以整批一次 argmax 就够，而且只有**一次**设备回传。逐请求 `select()` + `.item()`
   会为每条请求付一次同步。
2. **不需要把 logits 转成 FP32**。模型跑 FP32 时它本来就是 FP32；跑 BF16 时它已经
   是 BF16——那是模型**算出来的**精度，事后加宽不恢复任何信息（BF16 → FP32 是精确加宽，
   值、顺序、并列关系都不变，argmax 结果必然相同）。实测三种组合都是 0 次不同：

   | 组合 | logits dtype | `argmax(原张量)` vs `argmax(转 FP32)` |
   |---|---|---|
   | CPU / FP32 | float32 | 0 次不同 |
   | CUDA / FP32 | float32 | 0 次不同 |
   | CUDA / BF16 | **bfloat16** | 0 次不同 |

   > 这里曾经写过一个**错误**的理由：说「BF16 只有 8 位尾数，两个 FP32 里分得开的
   > logits 会被舍入成同一个值，于是 argmax 选另一个 token」。那个数字（词表 64 时
   > 0.67%、151936 时 2.56%）测的是「**全精度算出来的** logits」与「**BF16 舍入后的**
   > logits」的差别，也就是两种**模型精度**的对比；而两条路径用的是同一个模型、同一个
   > dtype，张量里的数一模一样。widening 是精确的，argmax 不可能因此改变。

   **真正需要 FP32 的是普通路径**：`apply_penalties` 要在 FP32 上做算术、
   `torch.multinomial` 要浮点概率——那里转换是有用的，别照着投机这行把它删掉。

3. **提交顺序严格按 `picked`**。第五十二关是「先提交所有 plain、再提交 drafts」，
   单请求时看不出差别；批量下那会打乱本轮的事件顺序，所以改成一条循环走到底。

投机关闭时**原样保留 `TorchSampler` 路径**（`_sample_with_sampler`）：随机采样、
三种惩罚项、按请求隔离的随机数发生器都在那条路上，不能被裸 argmax 顶掉。

### 2.3 预算：真实 token 优先，草稿只吃余量

`_plan_tokens()` 的分法没变，本关把规则写清楚并补了一处：

1. 本轮每个 ready（只差最后一个 token）的请求**先各留 1 个真实 token**；
2. 剩下的额度才轮到草稿和 prefill，按遍历顺序先到先得；
3. 预算不够所有 ready 时，按 running 顺序保留靠前的，**其余的这轮不运行**。

第 3 条以前在 fcfs 下是一句 `assert len(ready_req) <= token_budget_remaining`
（「decode 预算不会超」），priority 下才是截断。但 `max_num_seqs > max_num_batched_tokens`
这个配置就能把它造出来——**把可配置的组合当成不可能发生的事**。现在两边走同一条规则：
被丢下的请求落到 `prefill_token_budget == 0` 那条分支，不会排出 0 token 的计划项。

预算 5、三条都 ready 时：先各留 1（共 3），余量 2 全给第一条的草稿，后面两条照样
各算 1 个真实 token——**草稿挤不掉别人的基础配额**。

### 2.4 容量：先缩草稿，再抢占

```text
草稿装不下 -> 先逐枚缩短 K（最坏退到只有真实 token）
真实 token 也装不下 -> 才走 _make_room() / 抢占
```

`_shrink_draft()` 只改这一轮的临时计划（`input_ids` / `draft_ids` /
`num_scheduled_tokens`），缩到 K=0 时 `draft_ids` 为空，`_sample_plan()` 自然按
普通 1-token 路径处理——**不需要另开一条「退化」分支**。

由此得到一条本关要守的性质：**可选草稿不能成为额外抢占别人的理由**。第 3 步的
例子（§5.3）里两条请求的草稿都装不下、都缩到 K=0，一次抢占都没发生。

被抢占的请求可能**已经排好草稿计划**（`_plan_tokens()` 排在 `_reserve_blocks()`
之前）。这种 item 必须整个作废：

```python
if seq not in self.running:
    continue        # 本轮被选成犠牲者，计划作废
```

它不会进模型、不会采样、不会发 token，也不会算进重算统计。

### 2.5 恢复：历史留着，KV 从重算进度追赶

抢占只丢物理 KV（`cache` 归零、`block_hashes` 清空），`all_token_ids` / `output_ids`
一字不动，所以恢复就是把历史重放一遍。中间的重算 chunk 不采样、不重放 `on_token`
（`can_sample = num_real == seq.num_uncomputed_tokens`）；**追到只差最后一个 token
就能继续投机**，不需要再经历一轮普通 decode——它那时已经满足 `is_ready_for_next_token`。

等待期间还有第五十二关之前就有的「阻塞者感知恢复」：队首的请求若它的阻塞者还没结束，
不急着重建 KV（建了大概率又被抢走，白算一遍）。

## 3. 不变量

```text
len(all_token_ids) == len(prompt_ids) + len(output_ids)
0 <= cache.length <= len(all_token_ids)          对仍在运行的请求
cache.length - start_cache_length <= kept_inputs  只保留被认可的输入
len(block_table) == ceil(cache.length / block_size)
block_usage[b] == 引用它的运行中请求数
set(可分配链) == {b | block_usage[b] == 0}，且无重复
```

**「`cache.length == len(all_token_ids) - 1`」不再是所有时刻的不变量**：抢占期间它
可以远小于历史长度（KV 归零、历史还在），只有「追上历史并完成采样的运行态」才满足。
测试里的每步检查因此分两类——块引用/链表/块表长度**每步都查**，
`cache.length` 与历史的关系只在 decode 之后查。

## 4. 验证

### 4.1 行映射与混批（`benchmarks/check_step53_batch.py`，35 项全通过）

- `_sample_plan()` 拿需求 §2 那张表**手工造** `scheduled_items` 直接验：
  `sample_rows == [0,1,2,7,8,9]`、偏移 `[0:3]/[3:4]/[4:6]`、中间 prefill 不进 picked；
- 端到端混批（§1 那张表）：一轮**只调用一次模型**、输入是全部计划项拼起来的一维、
  `sample_rows` 跳过批尾 prefill 的 14 行、三种 K 同轮出现、
  「提交顺序按 picked 而不是先 plain 后 drafts」、每条请求 `output_index` 连续；
- 预算压力：预算 2 / 4 条 ready 时不 assert、不排 0 token 项、不超预算、四条都跑完；
- 容量压力：**先缩草稿（0 次抢占）**、**真实 token 也不够才抢占**、
  被抢占的那一轮「计划里有草稿但整个 item 没执行」、B 没进模型也没发 token；
- 恢复：被抢占那一刻 `(KV 进度, 已提交输出, 被抢占次数, 阻塞者) == (0, 2, 1, "A")`、
  阻塞者没结束前不重建、重算一轮把历史一次追平、**追上之后马上又能投机**。

### 4.2 真模型等价性（`benchmarks/check_step53_engine.py`，51 项全通过）

同权重、同请求（4 条并发），`speculative_mode="ngram"` 与 step51 的普通贪心
**每个 request_id 的最终 token 序列完全相同**，且这次真的提了 5 轮草稿、
每条请求 `output_index` 连续、`on_finished` 恰好一次。CUDA FP32 / BF16 各跑一遍同样成立。

另外把第五十二关的用例整套搬了过来：脚本模型定点用例（全部接受 / 首枚拒绝 /
部分接受 / EOS）、入口校验、跨块回滚、重算统计。

### 4.3 旧路径不变（`benchmarks/diff_step52_step53.py`，88 项全通过）

`speculative_mode=None` 时 step53 与 step52 **逐步逐字节一致**：11 个场景 × 2 seed，
比较计划、队列、输入 token、完成输出、每请求计数、物理块编号、hash 链、结束态，
**没有放宽任何字段**。本关动的是投机路径和一条入口校验，不投机的路径一行不该变。

### 4.4 随机压测

| 压测 | 结果 |
|---|---|
| 600 组多请求投机（2~4 条、预算 1~12、池子 2~24 块、block_size 1/2/4、K 1~3、n 1~3、动态到达） | 0 崩溃、0 活锁、0 个 0-token 计划项；结束时引用归零、链表与真实空闲一致；每条请求 `output_index` 连续 |
| 400 组「投机 vs 普通贪心」逐 request_id 输出对照（同样的窗口） | **0 组不同**（其中 40 轮真的提出了草稿） |
| 第五十二关的 1500 组压小池子、600 组入口校验 | 重跑全过 |

### 4.5 验收方脚本

`benchmarks/review_step52.py`（验收方给的独立对照）重跑一致；它的「迭代器接口诊断」
依然会报 `采样行数 1 超过本轮 query 数 0`——那是 **step52** 的行为，本关只在
`step53/` 修，旧包按约定不动。

## 5. 接口变化与遗留

### 5.1 接口变化

**新增**：计划项多了 `sample_offset`（筛选后 logits 里的起点）、`num_sample_rows`
（`draft_ids` / `start_cache_length` 是上一关就有的）。

**签名变化**：`Engine._commit_drafts(item, greedy_ids, notify)` —— 第二参数从
「K+1 行的 logits 张量」变成「已经算好并切好的 Python token 列表」，因为整批 argmax
现在在 `_sample()` 里做。`Engine._sample_with_sampler()` 是新拆出来的原路径。

**放开**：`speculative_mode="ngram"` 不再要求 `max_num_seqs=1`。仍然明确拒绝的组合：
`priority`、开前缀缓存、非 Torch attention、CUDA Graph、非贪心或有惩罚项的请求。

**行为变化**：`add_request()` 的 `prompt_ids` 现在支持**任何可迭代对象**（含生成器），
校验函数返回物化后的 list 并交给 `SequenceConfig`——以前迭代器会被消费两次，
入队后 `prompt_ids=[]`，直到跑起来才炸。不原地改调用方的字典。

### 5.2 遗留

1. **不做 priority 投机的语义**：priority 模式下 `_budget_groups()` 仍按优先级分层，
   但投机本身仍被禁用（明确报错），没有验证过它与抢占恢复的交互。
2. **不做 prefix cache / Triton / CUDA Graph 下的投机**：前缀缓存的命中块与
   「被拒绝草稿的 KV」相互作用没验证过；Graph 与「每个请求行数不同」的图键冲突。
3. **不做随机投机**：接受判定需要按概率比决定，本关只做贪心的精确接受。
4. **不做接受率自适应**：K 仍由配置与四个上限决定，不按历史接受率调整。
5. **不承诺加速**：本关没有吞吐矩阵，步数只是目标 forward 次数。批量下每条请求
   每步输出数不同，跨请求完成顺序也会变——只保证「每条请求自己的 token 序列不变」。
