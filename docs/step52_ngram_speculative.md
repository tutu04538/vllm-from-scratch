# step52：单请求贪心 n-gram 投机解码

- 对应代码：`step52/`（新增，从 `step51/` 复制，入口改名 `step52.py`）
- 包摘要 SHA256：`dc9ff7ea4da56f58…`（16 个 .py / 3739 行，验收方 `source_digest()` 口径）
- 基线：`step51/`，指纹 `7115f26936cf9472…`（15 个 .py / 3425 行），原样保留未改
- **改动 5 个文件 + 新增 1 个**：

| 文件 | 改动 |
|---|---|
| `speculative.py`（新增，130 行） | 两个纯函数：`propose_ngram()` 提议、`verify_drafts()` 验证 + `DraftVerification` |
| `scheduler.py` | `_plan_drafts()` 按四个上限缩短 K；`_reserve_blocks()` 容量不够时逐枚缩草稿；计划项新增 `draft_ids` / `start_cache_length`；重算统计改按真实起点 |
| `cache.py` | 新增 `truncate()`（回滚 KV）与 `can_grow()`（只读容量查询） |
| `engine.py` | 投机配置与组合校验；`_sample_plan()` 改为「每个请求取几行」；`_commit_tokens()` / `_commit_drafts()` 拆出提交点 |
| `__init__.py`、`step52.py` | 包说明、入口改名 |

`request.py`、`model.py`、`attention.py`、`norm.py`、`rope.py`、`sampling.py`、`sampler.py`、
`formats/` 未改。

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
    committed_ids: list    # 真正要提交的输出（已按 EOS / 输出上限截断）
    kept_inputs: int       # 本轮输入的 KV 要保留几个
    stopped: bool          # 是否因 EOS 或输出上限提前停下
```

四条规则，逐条对应需求 §1 与 §3.D：

1. 逐枚比较 `greedy_ids[i] == draft_ids[i]`，第一个不同就停；
2. 候选 = `draft_ids[:accepted] + [greedy_ids[accepted]]`——全部接受时最后那枚就是 bonus；
3. 逐枚提交，**遇 EOS 或输出上限立即停**，后面的草稿与 bonus 都不提交；
4. 保留的输入 = `1 + 还需要当下一轮输入的草稿数`：
   - 被拒绝的草稿一律不留（就是需求表里的「保留哪些 KV」）；
   - bonus 永远不留（它还没进过模型）；
   - **被接受的草稿本身若是终止 token，它也不留**——它结束了这条请求，
     不会再作为下一轮输入，所以要留的是它**之前**那些草稿（`kept_drafts = index`）。

`kept_inputs` 是「回滚到哪儿」的唯一依据，所以把它算成一个显式的返回值，而不是
在提交循环里边走边推。

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

## 2. 不变量

```text
len(all_token_ids) == len(prompt_ids) + len(output_ids)
0 <= cache.length <= len(all_token_ids)          对仍在运行的请求
cache.length - start_cache_length <= kept_inputs  只保留被认可的输入
len(block_table) == ceil(cache.length / block_size)
```

**投机每一步之后仍满足 `num_uncomputed_tokens == 1`**——这是它能连续投机的根据：
全部接受时 `cache.length` 与历史同步推进 `K+1`，首枚拒绝时只推进 1 而历史也只长 1。

被拒绝草稿的 KV 可能落在**不完整尾块**里，下一轮按位置覆盖写；落在被退掉的整块里
则整块回池子。两条路径都不会让拒绝的内容被当成命中。

## 3. 验证

### 3.1 纯函数与回滚（`benchmarks/check_step52_speculative.py`，53 项全通过）

- `propose_ngram`：找不到 / 找到 1 个 / 找到 2 个 / **多个匹配取最近** / `k` 截断 /
  `n=1` / 不自己匹配自己 / 直接吃只读视图；8192 长历史不复制整段；
- `verify_drafts`：三种验证结果 + 无草稿 + bonus 是 EOS + **草稿本身是 EOS** +
  第二枚草稿是 EOS + 输出上限截断 + 行数不匹配报错；
- `truncate`：回滚到块边界 / 块内 / 0；非法目标报错；**不动已发布的完整块与其
  hash 双向索引**；被退掉的整块确实以「真正空闲」回队首；
- `can_grow`：只读（问完池子与请求状态一个字节没变）、判断与可分配链一致、不摘链；
- `_plan_drafts` 的四个上限逐条验：预算 / 输出上限 / 上下文 / 配置值，以及
  「还没生成过 token」「还差不止 1 个 token」两种不投机的情形；
- 配置校验：6 种不支持的组合与 3 种非法采样参数都**明确报错**，
  关掉投机后这些组合仍然合法。

### 3.2 端到端（`benchmarks/check_step52_engine.py`，28 项全通过）

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

**真模型等价性**：同一个随机小模型、同一个 prompt（重复片段多，n-gram 更容易命中），
`speculative_mode="ngram"` 与 step51 的普通贪心**逐 token 相同**：
`[54,6,54,6,…]`，且确实有一轮一次 forward 提交 3 枚（8 步产出 12 个 token，
普通贪心要 12 步）。这是本关最强的一条检查——接受也好拒绝也好，提交的永远是目标
模型自己的贪心 token，所以投机**不能**改变结果。

### 3.3 默认模式不改行为（`benchmarks/diff_step51_step52.py`，88 项全通过）

11 个场景 × 2 seed，逐步比较调度队列、本轮计划（含 `draft_ids`）、输入 token、
完成输出、每请求计数、承诺与引用、**物理块编号**、**每请求 hash 链**、结束态。
不开投机时 step52 与 step51 应当逐字节一致，所以这里**没有放宽字段**——比第五十
一关的对照更强。

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

**行为变化**：只在 `speculative_mode="ngram"` 时发生，且该模式的合法组合被明确
限定（见 §2 与 §4.2）。默认 `None` 时逐字节等同 step51（§3.3）。

**未改**：默认模式下的调度与优先级、KV 准入与淘汰、prefix hash 算法与命中规则、
采样结果、模型；`shrink_draft` 之外没有新的分配路径。

### 4.2 遗留

1. **只做单请求**：`max_num_seqs=1`，`_sample_plan()` 不做「多个请求各要不同行数」的
   混合行映射（结构上支持「每个请求若干行」，但只被单请求验证过）。
2. **只做贪心**：随机采样下「接受」的判定需要按概率比决定，本关明确拒绝非贪心请求。
3. **不做前缀缓存 / 抢占 / CUDA Graph 的组合**：那几块要么会与「被拒绝草稿的 KV」
   相互影响（prefix），要么与行映射冲突（graph），先把最小闭环站稳。
4. **提议只有 n-gram**：没有 draft model、没有 EAGLE/Medusa 之类；
   `propose_ngram` 是线性扫描，长历史下应当换哈希索引（本关先不动）。
5. **不做异步提议、不做 `k` 的自适应**：K 固定由配置与四个上限决定，
   不按历史接受率调整。
6. **不承诺加速**：本关没有吞吐矩阵，步数只是目标 forward 次数。
