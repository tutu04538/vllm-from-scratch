# step54：随机采样投机解码与拒绝修正

- 对应代码：`step54/`（新增，从 `step53/` 复制，入口改名 `step54.py`）
- 包摘要 SHA256：`e8f3e7200382d707…`（16 个 .py / 3983 行，本仓库 `source_digest()` 口径
  ——`name\0hash\n` 拼起来再 sha256；验收方 `review_step53.py` 用的是另一种拼法，
  同一份代码两个数字不同，比对时先确认口径）
- 基线：`step53/`（验收方记录 `6f4f362f04bbd3d0…`），原样保留未改
- **改动 5 个文件**：

| 文件 | 改动 |
|---|---|
| `speculative.py` | 新增 `verify_drafts_random()`（拒绝采样）与 `residual_probs()`；抽出 `_finish_candidates()` 给两条验证路径共用（EOS 截断 + 留多少 KV 只有一份实现） |
| `sampling.py` | 新增 `row_distribution()`：把「惩罚 → 温度 → top-k → softmax → top-p」这套**既有顺序**暴露出来给拒绝采样用，不复制第二份规则 |
| `engine.py` | `_sample()` 拆成「无草稿项走采样后端 / 贪心无惩罚走批量快路径 / 其余走随机验证」三条；新增 `_commit_drafts_random()`（逐行历史 + 临时计数）；`_sample_with_sampler()` 改名 `_sample_rows()` 并改为**只算 token、不提交**（提交统一按 picked 顺序） |
| `scheduler.py` | 删掉「投机只支持贪心且无惩罚项」的限制；顺带修掉 `request_id = request.get(...)` 写了两遍（验收方指出的） |
| `__init__.py`、`step54.py` | 包说明、入口改名 |

`cache.py`、`request.py`、`model.py`、`attention.py`、`norm.py`、`rope.py`、`sampler.py`、
`formats/` 未改。

## 0. 需求大概

第五十三关的投机只判断「草稿 == 目标模型 argmax」，等于把 argmax 当成了唯一正确答案。
用户给的 `temperature=0.8, top_k=20, top_p=0.9` 是要**采样**的，输出必须服从目标分布。

本关实现 n-gram 提议下的拒绝采样：接受概率 `min(1, p[d]/q[d]) = p[d]`（q 是确定性提议，
`q(d)=1`）、被拒后从「排除草稿的纠正分布」抽、逐行各自的惩罚历史、随机数归请求所有。
不新增 draft model、不写 GPU rejection kernel。

## 1. 一轮的具体数字

目标分布 `p = [0.6, 0.3, 0.1]`，草稿 `d = 1`：

| | 数字 |
|---|---|
| 提议分布 | `q = [0, 1, 0]`（n-gram 是确定性的） |
| 接受概率 | `p[d] = 0.3` |
| 拒绝时用的纠正分布 | `[6/7, 0, 1/7] ≈ [0.857, 0, 0.143]` |
| 最终 token0 的概率 | `0.7 × 6/7 = 0.6` ✓ |
| 最终 token1 的概率 | `0.3` ✓ |
| 最终 token2 的概率 | `0.7 × 1/7 = 0.1` ✓ |

**为什么不能拒绝后仍从原 p 抽**：那会让 token1 变成 `0.3 + 0.7×0.3 = 0.51`，而 token0 掉到
`0.7 × 0.6 = 0.42`——分布直接错了。本关的验收就是拿经验分布去对 p，这条会立刻被抓出来
（`check_step54_rejection.py` 用 20 万次采样断言经验分布落在 5σ 内、并且**明显偏离**那个错分布）。

KV 回滚同一轮里一起算：输入 `[x, d0]`（K=1）→ 两行 logits。若 `d0` 被拒、纠正 token 是 `c`：

```text
提交 [c]              本轮输入只留 [x]        cache.length = start + 1
若 d0 被接受、bonus 是 b：
提交 [d0, b]          本轮输入留 [x, d0]     cache.length = start + 2
```

## 2. 临时计数与真实计数的边界

已有真实输出 `[5]`、草稿 `[5, 7]`、目标输入 `[x, 5, 7]` 时，三行 logits 看到的生成历史是：

| logits 行 | 预测谁 | 该行看到的生成历史 |
|---|---|---|
| 0 | 第一枚草稿 / 纠正 token | `[5]` |
| 1 | 第二枚草稿 / 纠正 token | `[5, 5]` |
| 2 | bonus | `[5, 5, 7]` |

`_commit_drafts_random()` 为这些行造一份**临时计数**：

```python
temp = SimpleNamespace(prompt_token_ids=state.prompt_token_ids,
                       generated_counts=dict(state.generated_counts))   # 复制，不共享存储
```

- 只复制计数（两个属性），**不复制整段 token 历史**——逐行惩罚只需要计数与 prompt；
- 接受一枚草稿就往 `temp` 里加一枚；
- 真实 `sampling_state` 与 `all_token_ids` 在 `_commit_tokens()` 之前**一个字都不动**；
- 临时状态不重建、也不重置 RNG。

行的分布按「**全都接受**」构造：拒绝点之后的行根本不会被读到（`verify_drafts_random()`
首次拒绝就停），而拒绝点之前的行历史恰好就是「前 j 枚都被接受」，所以一次算完 K+1 行是对的。

**这条边界有判别性用例**（`check_step54_random.py`）：固定 logits（5 号 4.0、9 号 3.5）+
`frequency_penalty=0.7`，正确实现得到 `[5, 9, 5, 9, 5, 9]`，而「所有行都用真实计数」的
错误实现得到 `[5, 9, 5, 9, 5, 5]`——我先跑出差异再把它写成断言，不是先写断言再假设它有判别力。

## 3. RNG 消费约定

**随机数归请求，不归 batch 行**：uniform 与 categorical 都用该请求的
`SamplingState.generator`，不用 Python `random`，也不用全局 RNG。抢占不重置它。

消费顺序（测试会数调用次数）：

```text
逐位置（0..K-1）：
    p[d] == 0  -> 必拒，不抽 uniform
    p[d] == 1  -> 必接受，不抽 uniform
    否则        -> 抽 1 个 uniform，决定接受还是拒绝
首次拒绝        -> 抽 1 次纠正 token，本轮结束
被接受的草稿里出现终止 token -> 当场结束，**不抽 bonus**
全部接受        -> 抽 1 次 bonus
终止之后        -> 一个随机数都不再消耗
```

`p[d] ∈ {0, 1}` 不消耗 uniform 这一点是**约定**，不是实现细节里的巧合——文档和测试都
把它固定下来（`p[d]=0` 时抽了也是白抽；`p[d]=1` 时顺带避开「residual 全零」那条路）。

贪心请求没有 generator（第五十二关起只给随机采样建），也确实不需要：分布是 one-hot，
`satisfies p[d] ∈ {0,1}`，接受与否直接判定；纠正与 bonus 就是该行分布的 argmax。
代码里那个 `draw_uniform` 写成「被调用就抛 `AssertionError`」——真被调用说明上面这条推理错了。

## 4. 为什么「分布相同」不等于「同 seed 文本相同」

- **两条算法消耗的随机数不同**。普通随机解码每步抽 1 次；投机一轮可能抽 0～3 次（接受
  判定 + 纠正/bonus），而且 `p[d] ∈ {0,1}` 时还一次都不抽。同 seed 下两边的随机流很快就
  错位了。
- **K 会随 batch 变**。K 由配置、剩余预算、剩余输出额度、剩余上下文共同决定（§1.4 那四个
  上限），同一条请求在不同 batch 里可能拿到不同的 K，调用序列随之改变。
- 能保证的是**分布**：单步边缘分布、以及「接受则草稿 + 第二行的分布、被拒则只有纠正
  token」这个条件联合分布。这两条都在纯验证层用统计检验钉住了（20 万次 / 12 万次采样，
  5σ 容差，并且断言与错分布的距离远大于容差）。

所以本关**不要求**同 seed 与普通随机解码逐 token 相同——要求的是分布一致。
（贪心那条路仍然逐 token 相同，因为它是确定性的：`check_step54_random.py` 里
「贪心 + 三种惩罚：投机与普通路径的输出逐 token 相同」。）

## 5. 不变量

```text
len(all_token_ids) == len(prompt_ids) + len(output_ids)
0 <= cache.length <= len(all_token_ids)          对仍在运行的请求
cache.length - start_cache_length <= kept_inputs  只保留被认可的输入
generated_total == len(output_ids)                惩罚计数恰好等于已提交输出数
```

最后一条是**临时计数与真实状态边界**的可查形式：重算的历史不会重复计数、被拒的草稿
不会计数、被抢占恢复后也不会漏计。

## 6. 验证

### 6.1 验证层（`benchmarks/check_step54_rejection.py`，29 项全通过）

- **确定性分支**（注入 `draw_uniform` / `draw_token`）：全接受、首拒绝、部分接受、bonus、
  草稿 EOS、纠正 EOS、K=0；每一条都断言**提交的 token、保留的 KV 数、随机数调用次数**；
- **边界**：`p[d]=0` 必拒且不抽 uniform、`p[d]=1` 必接受、`K+1 > R` 报错、
  `residual` 无剩余质量时报错（不偷偷退回 argmax）；
- **分布正确性**：

  | 检查 | 结果 |
  |---|---|
  | `p=[0.6,0.3,0.1]`、草稿 1、20 万次 | 经验 `[0.5997, 0.3004, 0.0999]`，5σ 内 |
  | 同上，与错分布 `[0.42,0.51,0.07]` 的距离 | 远大于 10σ |
  | K=1 条件联合分布（接受 / 拒绝两条支路） | 实测 `{(0,0):0.1253, (0,1):0.3746, (1,):0.5001}` vs 理论 `{0.125, 0.375, 0.5}` |
  | 全部接受后 bonus 来自**第二行** | `[0.2514, 0.7486, 0]` vs 理论 `[0.25, 0.75, 0]` |

- **惩罚与过滤**：top-k / top-p 把草稿过滤掉时目标概率为 0；温度参与分布构造；
  重复草稿让第二行的 frequency 惩罚与第一行不同；逐行概率与「普通单步采样」的参考一致；
  贪心分布与 `TorchSampler` 的贪心取到同一个 token。

### 6.2 引擎状态（`benchmarks/check_step54_random.py`，21 项全通过）

- 被拒的草稿不进真实计数、不进已提交历史；逐枚重放提交序列得到的计数与真实计数一致；
- **抢占不消耗也不重置 generator**（抢占发生在 `schedule()` 里、forward 之前，那一步
  请求自己的随机流逐字节不变）；恢复后也没有回到初始状态；
- **重算不重复计入惩罚次数**：每次提交后 `generated_total == len(output_ids)`（含被抢占
  重算的那条）；
- 同批混跑 greedy / 随机 / 随机+惩罚 / 贪心+惩罚，四条请求的 `output_index` 都连续、都完成一次；
- 贪心 + 三种惩罚：投机与普通路径**逐 token 相同**；逐行历史有判别性用例（见 §2）；
- CUDA FP32 / BF16 冒烟。

### 6.3 组合验证（`benchmarks/check_step54_combinations.py`，17 项全通过）

| 检查 | 结果 |
|---|---|
| priority + 投机：容量抢占与**名额抢占**都发生，被抢占的那一步执行计划里不含它 | 抢占 2~3 次（名额 1） |
| priority + 投机：每步预算、池子自洽、无 0-token 项、四条请求序号连续 | ✓ |
| prefix + 投机：确实命中、确实投机、**开关前缀缓存不改变输出** | reused 8 / 草稿 2 轮 / 两组输出逐 token 相同 |
| prefix + 投机：**已发布块的 KV 在后续（含回滚）步骤里一字未改** | 快照 6 个块逐字节相同 |
| prefix + 投机：hash 双向索引一致、结束引用归零 | ✓ |
| priority + prefix + 投机（含随机与惩罚项）：抢占 1 / 命中 16 / 草稿 5 轮，三条请求都正常完成 | ✓ |

「已发布块的 KV 不被改」这条**做过反证**：故意往一个已发布块里写脏数据，这条断言立刻 FAIL
——不是「跑过了就算数」。

### 6.4 旧路径与回归

| 脚本 | 结果 |
|---|---|
| `check_step54_speculative.py`（第五十二关的纯函数 + 配置校验） | 53 项全过 |
| `check_step54_batch.py`（第五十三关的行映射 / 混批 / 预算 / 容量 / 恢复） | 35 项全过 |
| `check_step54_engine.py`（单请求与多请求等价性） | 51 项全过 |
| `diff_step53_step54.py`（投机关闭时与 step53 逐步对照，未放宽字段） | 88 项全过 |
| 随机压测 500 组（2~4 条请求、随机/惩罚/不同 K、预算 1~12、池子 2~24 块、动态到达） | 0 崩溃 0 活锁、0 个 0-token 计划项、计数与输出始终一致、引用归零 |
| 随机压测 500 组（priority / 前缀缓存随机组合 + 随机采样 + 惩罚项） | 同上，外加 hash 双向索引一致 |

## 7. 接口变化与遗留

### 7.1 接口变化

**新增**：`speculative.verify_drafts_random()`、`speculative.residual_probs()`、
`sampling.row_distribution()`；`Engine._commit_drafts_random()`。

**放开**：`speculative_mode="ngram"` 现在只拒绝两条**实现方式**决定的组合：

- `attention_backend != "torch"`：拒绝采样是逐请求的 Torch 参考循环，没有 Triton
  rejection kernel（需求明确不做）；
- `use_cuda_graph=True`：采样要在图**外**按请求逐行做设备同步，图里做不到。

除此之外都放开并验过：`max_num_seqs`（第五十三关）、采样参数（本关）、
`priority` 与 `enable_prefix_caching`（本关，见
`benchmarks/check_step54_combinations.py`）。

**放开抢占那条**不需要额外机制：抢占发生在 `schedule()` 里、forward **之前**，被抢占的
项整个作废、走不到采样与回调；**放开前缀缓存那条**靠的是「被回滚的整块从来没被发布过」
——回滚目标 ≥ 本轮起点 + 1，而发布的块严格在起点之前，两者不相交。

**内部签名变化**：`Engine._sample_with_sampler()` → `Engine._sample_rows()`，改为返回 token
列表而不提交（提交统一按 `picked` 顺序在 `_sample()` 里做）。

### 7.2 遗留

1. **一般 draft model 不做**：`q` 是确定性的，所以接受概率就是 `p[d]`，不需要整张 q 矩阵与
   `max(p-q,0)` 的归一化。换 draft model 时那两块要重写。
2. **不做 GPU rejection kernel**：现在是逐请求的 Torch 循环，允许必要的设备同步；
   批量无分支 kernel 留给后续关卡。
3. **不做异步调度下的采样**：`_sample_rows()` 里那句「不复制 logits」的前提是「只在本轮
   `step()` 内消费」——真要跨步持有，必须改回 `copy=True`（注释里写明了）。
4. **不承诺加速**：本关没有任何吞吐结论。
5. **`_filter_and_probs()` 的注释与实现对不上**：注释写「返回 (probs, keep_mask)」，
   实现只返回 `probs`——从第一次提交起就是这样，没有任何调用方解包过第二个值。
   本关顺手改正（只改 `step54/`，旧包按约定不动）。
6. **`p[d]` 的浮点边界**：`p[d]=0/1` 的判定用的是 `torch` 算出来的概率张量，理论上
   `softmax` 的结果可能落在 0/1 附近而不是精确的端点上；那时会走「抽 uniform」那条正常
   路径，结果仍然正确（只是多消耗一个随机数）。
