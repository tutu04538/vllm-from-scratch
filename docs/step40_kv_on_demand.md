# step40：KV 按需分配，容量不足时给一个明确结局

- 对应代码：`step40/`（新增，从 `step39/` 复制）
- 包摘要 SHA256：`306903707d0360624478a9269ab662caca553672abc59ab969d09a8c6e839ab4`（14 个 .py / 2694 行）
- 基线：`step39/`，指纹 `89fce66f…`（14 个 .py / 2585 行），原样保留未改
- 改动文件：`cache.py`、`scheduler.py`、`__init__.py`、入口改名 `step40.py`

## 0. 需求大概

这一关不是性能关，是**行为关**。第 36 关记录的缺陷：池子只有 2 块、队首 A 要预留 5 块、
后面的 B 只要 1 块时，`step()` 既不推进、也不报错、也不退出——调用方的
`while engine.has_unfinished_requests(): engine.step()` 是死循环。

根因两处叠加：`allocate_block()` 按 `ceil((prompt + max_new_tokens - 1) / block_size)`
**一次性预留整个输出上限**；`schedule()` 的接纳循环分配失败就 `break`，队首把后面全堵死。

三件事按顺序做（需求 §2），本关都做了。

## 1. 设计：把「准入」和「生长」分成两层

这是本关唯一需要想清楚的地方。

```text
准入 allocate_block(seq)   用**最坏情况**判断这条请求能不能跑完，并「承诺」相应块数
                           承诺是记账，不占物理块 —— 刚进来的请求占用块数是 0
生长 ensure_blocks(seq, n) 真正要写 KV 之前，按这一轮要写的 token 数补物理块
```

### 1.1 为什么准入那层不能省

**只做按需分配会引入一个新的死锁**，而且它比原来的那个更隐蔽：

```text
池子 4 块；A、B 各需 3 块（prompt 4 + 输出上限 8）

按需准入：A 拿 1 块、B 拿 1 块          （各自只算「这一轮要写的」）
step1    A、B 各写 4 个 token           用 2 块
step2    A、B 各 decode 1 个 token，都要第 2 块   用 4 块
step3    两者都要第 3 块，池子空了
         -> 零进展、无错误、无退出
```

现在的「一次预留整个输出上限」**恰好是防住这个的机制**——它让准入就把总量卡死。
所以按需分配不能简单地把预留删掉，得换成等价的记账。

**做法**：准入时按最坏情况算出一个额度记在 `seq.promised_blocks` 上，池子维护
`self.promised_blocks`（全局承诺之和）。准入条件从「有没有这么多空闲块」变成
**「空闲 + 可淘汰 − 他人已承诺 ≥ 本条需要的」**。这样：

- 已接纳的请求**一定**能长到它承诺的量（它的额度没被别人占）
- 账面总量不会超卖，所以也不会两条请求同时卡住

代价写清楚：**承诺占着的额度不能被别的请求用**。也就是说这一关**没有**把容量腾出来给
更多并发请求——那需要抢占（下面 §1.3 说为什么不选）。

### 1.2 容量真的不够时：明确拒绝（需求 §2.2 选 (a)）

如果一条请求**自己**要的块数就超过整个池子，它永远不可能完成。这时不再让它留在
`waiting` 里空转，而是在接纳那一刻抛 `InfeasibleRequest`，调度器接住并**明确结束这条请求**：

```python
try:
    admitted = self.kv_cache_pool.allocate_block(next_seq)
except InfeasibleRequest as exc:
    self.waiting.pop(0)
    self._fail(next_seq, str(exc))
    continue          # 队首被明确结束，后面的继续尝试
```

失败记录带着诊断信息进 `on_finished` 与 `step_done`（多一个 `"error"` 键）：

```text
请求 'A' 永远无法完成：它最多需要 7 块（prompt 18 + 输出上限 8 个 token，
block_size=4），而整个 KV 池只有 2 块。当前池子：空闲 2 块、闲置缓存 0 块、
他人已承诺 0 块；命中的前缀块 0 块（已抵扣）。请调大 num_kv_blocks，
或调小这条请求的 max_new_tokens
```

需求 §4.6 要「哪条请求、缺多少容量、当前池子什么状态」，这三样都在里面。

### 1.3 队首策略：保持 FCFS（需求 §2.3 选 FCFS）

明确选 **FCFS**，不跳过队首：

- 唯一能让请求永远卡住的情形是「它自己就不可能在池子里装下」，而这种请求现在**在准入时
  就被结束掉了**，不会留在队首。
- 剩下的「暂时不够」是会自解的：running 里的请求一定跑得完（§1.1），跑完就还块。
- 跳过队首会把「请求顺序语义」改掉，短请求可能长期插队饿死长的。既然 FCFS 已经不会死锁，
  就不付这个代价。

代码里表现为：`allocate_block` 返回 `False` 时仍旧 `break`（整队等），只有抛
`InfeasibleRequest` 才越过队首。

**不选抢占（§2.2b）**：需求 §5 明确说「如果按需分配 + 明确失败就够，那就停在那里」。
抢占要引入重算、优先级和防活锁，是另一个量级的东西，本关不做。

### 1.4 零进展守卫（需求 §2.4）

`schedule()` 末尾加了一条不变量检查：

```python
if (self.has_unfinished_requests() and not self.scheduled_items
        and not self._admitted_this_step and not self._failed_this_step):
    raise RuntimeError("调度没有任何进展：...")
```

结构上它不该触发（running 非空必然排出 token；running 为空时队首必然被接纳或被拒），
留着是为了**下次改这块时如果破坏了前提，会立刻响**，而不是回到静默空转。

## 2. 改动内容

| 位置 | 改动 |
|---|---|
| `cache.py` | `allocate_block` 改为「前缀匹配 + 最坏情况可行性 + 承诺额度」，不再一次占满；新增 `ensure_blocks`、`_evictable_block_indices`、`_available_blocks`、`InfeasibleRequest`；`deallocate_block` 归还未用额度 |
| `cache.py` | `KVCachePool.promised_blocks`、`SequenceConfig.promised_blocks` |
| `scheduler.py` | 接纳捕获 `InfeasibleRequest` 并走 `_fail`；计划后用 `ensure_blocks` 按需补块；零进展守卫 |
| `__init__.py` | 导出 `InfeasibleRequest` |
| `attention.py` / `model.py` / `norm.py` / `rope.py` / `engine.py` / `formats/` | **未改** |

`engine.py` 一行没动：失败出口走的是 `scheduler.on_finished` / `step_done`，已有通道。

## 3. 验证

### 3.1 第 36 关的复现用例（需求 §4.1）

```text
A: prompt 18、max_new_tokens 8（最坏 7 块）   B: prompt 4、max_new_tokens 4（最坏 2 块）
池子 2 块

step 1: running=0 waiting=0 用块=0 计划=1 明确结束=['A']
        -> 全部结束
```

**明确的结局 + 是哪一个**：A 被带诊断信息地结束，B 正常跑完。调用方的 while 循环会退出。

### 3.2 块占用随进度增长（需求 §4.2）

`prompt=18, gen=8, block_size=4, max_num_batched_tokens=8`，逐步打印实际占用块数：

```text
step39（旧）    [7, 7, 7, 7, 7, 7, 7, 7, 7, 0]     一进来就占满整个输出上限
step40（按需）  [2, 4, 5, 5, 5, 6, 6, 6, 6, 0]     随 chunked prefill 与 decode 增长
```

前三个数 2 → 4 → 5 就是 prompt 的 18 个 token 分三次 chunk prefill 写进去的；
后面 6 是 decode 期间跨块。**不是「预留得少一点」，是预算随进度释放。**

### 3.3 回归：与基线逐项一致

用验收方 15 个 `verify_step39_*.py` 的副本（把指向改成 `step40`）跑：

| 脚本 | step39 基线 | step40 |
|---|---|---|
| contract | 96/96 | **96/96** |
| external | 53/53 | **53/53** |
| features | 29/29 | **29/29** |
| io_contract | 46/46 | **46/46** |
| merge | 41/41 | **41/41** |
| norm | 56/56 | **56/56** |
| precision | 23/23 | **23/23** |
| qwen3 | 21/21 | **21/21** |
| real_attention | 56/56 | **56/56** |
| rope | 33/34 | **33/34**（同一项失败，见下） |
| sampling | 45/45 | **45/45** |
| selection | 32/32 | **32/32** |
| stride | 9/9 | **9/9** |
| structure | 通过 | **通过** |
| tiles | 14/14 | **14/14** |

**逐项相同，没有引入任何回归。**

`verify_step*_rope.py` 有一项「越界位置被拒绝」失败（33/34）。**这是既有失败，不是本关
引入的**——用同一脚本跑 step39 得到同样的 33/34。记在这里以免被误算到本关账上。

### 3.4 压力测试（需求 §4.3）

随机 prompt 长度（1–20）、随机 `max_new_tokens`（0–12）、故意配小池子，6 种配置 × 60 个种子：

```text
池子/块大小/并发/预算/前缀缓存
  2/4/2/8/关    3/4/3/8/关    5/3/4/6/关
  7/5/8/10/开   11/4/3/7/开   13/4/16/12/开

跑 360 组：{'完成': 206, '明确失败': 154, '卡住': 0}
三选一不变量成立：要么全部完成，要么带诊断信息明确失败，没有卡住
```

覆盖了需求点名的几类：池子是质数（3、5、7、11、13）、`block_size` 与请求长度互质、
`max_num_seqs` 大于池子能容纳的请求数、前缀缓存开着。

### 3.5 性能不回退（需求 §4.5）

step39 vs step40，同进程轮转配对，每点 6 轮：

| 负载 | step39 | step40 | 配对中位差 | 同向 |
|---|---:|---:|---:|---:|
| decode_c1 | 394.7 ms | 385.9 ms | −0.02% | 3/6 |
| decode_c8 | 663.9 ms | 653.7 ms | −0.8% | 4/6 |

方向对半、幅度 <1%，在噪声内 → **没有回退**。分配是 Python 侧记账（几十微秒），
摊在几百毫秒的 GPU 工作上不显。

## 4. 接口变化与遗留

### 4.1 接口变化

**新增**：

| 接口 | 说明 |
|---|---|
| `InfeasibleRequest`（从包导出） | 请求在任何容量配置下都不可能完成 |
| `KVCachePool.ensure_blocks(seq, num_new_tokens)` | 按需补块 |
| `KVCachePool.promised_blocks` / `SequenceConfig.promised_blocks` | 承诺额度记账 |

**语义变化**：

| 接口 | 之前 | 现在 |
|---|---|---|
| `KVCachePool.allocate_block(seq)` | 一次分配整个输出上限的块 | 只记承诺额度，不占物理块；不可能完成时抛 `InfeasibleRequest` |
| `on_finished` / `step_done` 的记录 | `{request_id, output_ids}` | 失败时多一个 `"error"` 键（带诊断信息），其余字段不变 |
| 刚接纳请求的 `block_usage` | 立刻等于最坏情况的块数 | 从 0 开始，随进度增长 |

**对既有调用方的影响**：`add_request` / `step` / `has_unfinished_requests` / `on_finished`
的用法都不用改；只有直接读 `block_usage` 或依赖「接纳即占满」的代码会看到差异。
`contract` / `io_contract` 等回归全过，说明既有断言没有依赖这一点。

### 4.2 遗留

1. **不选抢占**，所以「承诺额度占着不放」——一条长请求会让池子里的一部分容量在它运行
   期间不可用。这是需求 §5 允许的取舍，但它是与 vLLM 在并发吞吐上的主要差别。
2. **失败在准入时判定，运行中不再有失败路径**。这是承诺记账带来的性质（承诺过的额度必然
   拿得到）；如果以后改成允许超卖（比如加抢占），`ensure_blocks` 里那条 `RuntimeError`
   就会变成真的会被触发的分支，需要重新设计。
3. **没有测容量受限时的吞吐**。需求 §4.5 说「容量受限时的吞吐提升单独给」——按需分配本来
   应该让更多请求同时跑，但本关的承诺记账把它抵消了（§1.1 的代价），所以没有可报的提升。
4. **尾延迟没测**：`InfeasibleRequest` 只在接纳时判定，不影响已运行请求的延迟；但
   FCFS 下长请求在前时，后面短请求的等待时间没有量化。
5. **`verify_step*_rope.py` 的「越界位置被拒绝」仍是既有失败**（step39/step40 都 33/34）。
   不在本关范围内，但记在这里以免重复占用。
