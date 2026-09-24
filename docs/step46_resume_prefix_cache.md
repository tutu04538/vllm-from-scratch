# step46：抢占恢复与前缀缓存整合

- 对应代码：`step46/`（新增，从 `step45/` 复制，入口改名 `step46.py`）
- 包摘要 SHA256：`1cfbcb68c5de4869…`（14 个 .py / 2987 行，验收方 `source_digest()` 口径）
- 基线：`step45/`，指纹 `8e9bb0ccfb0cc798…`（14 个 .py / 2929 行），原样保留未改
- **改动文件只有 4 个 + 入口改名**：

| 文件 | 改动 |
|---|---|
| `cache.py`（+58 / −12） | ① `SequenceConfig` 新增 `reused_tokens`（真实复用 token 数，累加）；② `find_matched_prefix_blocks(token_ids, max_tokens=None)` 改为对**完整已知历史**查找，新增可选上限参数（默认不设上限，见 §2.C）；③ `allocate_block()` 用 `all_token_ids` 查命中、**可行性改按最坏逻辑块数**判定、成功借到后才累加 `reused_tokens`；④ `publish_completed_prompt_blocks` → **`publish_computed_blocks`**，按 `all_token_ids` 建 hash 链，覆盖范围由 `cache.length` 决定 |
| `scheduler.py`（+15 / −2） | ① `post_step()` 在 recompute + prefix 模式下**每步**为所有 running 请求发布完整块；② 完成路径改用新函数名；③ `_preempt()` 清掉 `seq.block_hashes`（物理块已还回去，hash 链跟着作废） |
| `engine.py`（+9 / −10） | 删掉「`recompute` 必须配 `enable_prefix_caching=False`」的构造期报错；非法 `preemption_mode` 仍报 `ValueError`；`_check_preemption_mode()` 不再需要 `enable_prefix_caching` 参数 |
| `__init__.py`（+1 / −1） | 包 docstring 改成第 46 关 |
| `step46.py` | 由 `step45/step45.py` 改名而来，只有包名引用变化 |

**未改的 10 个文件**：`attention.py`、`model.py`、`norm.py`、`rope.py`、`sampler.py`、`sampling.py`、
`formats/` 三个 —— 模型 forward、kernel、采样算法、外部格式适配一个字节没动。
`Engine` 也没有新参数：本关只是放开已有的配置组合。

## 0. 需求大概

第四十四、四十五关里 `preemption_mode="recompute"` 强制关闭 prefix cache，恢复时从 0 重算。
B 已经算过 12 个 token，因 A 抢占丢掉 KV；如果其中前 8 个 token 对应的两个完整块仍在缓存里，
就该先复用这 8 个，只重算剩下的历史。

**命中是机会，不是保证**——块被 LRU 淘汰了，B 仍要能从 0 正确重算。

## 1. 三个长度必须分清

这是本关最容易混的地方（需求 §7 第 1 步就要求先画清楚）：

| 长度 | 含义 | 谁在维护 |
|---|---|---|
| 已知 token | `all_token_ids = prompt_ids + output_ids` | `SequenceConfig.all_token_ids` |
| 已计算 token | **已经进入模型并写入 KV** 的数量 | `cache.length`（唯一真相） |
| 已缓存完整块 | 整块被 `cache.length` 覆盖、KV 确实写进了物理块 | `block_hashes` / `block_hash` |

`prompt=3, block_size=4` 的实测轨迹（`benchmarks/check_step46_resume_prefix.py` 前几项）：

```text
准入后（还没算）        已知 3  已计算 0  已发布 0
prompt 算完、刚采样 y0  已知 4  已计算 3  已发布 0   ← 刚采样的 y0 还没进模型，不发布
y0 真的进过模型之后      已知 5  已计算 4  已发布 1   ← 块 [p0,p1,p2,y0] 现在可命中
```

第三行正是需求 §4.3 要的：**刚采样、尚未计算的 token 绝不能伪装成命中**。

## 2. 四个闭环

### A. 放开配置组合

`preemption_mode="recompute"` 现在可以与 `enable_prefix_caching=True` 同时开。
`None` 模式、`recompute + prefix 关` 的行为都不变（§4.8 逐项验证过）。

### B. 只有**真正算完**的完整块才登记

`publish_completed_prompt_blocks` 改名并升级为 `publish_computed_blocks`，成为
prompt 与生成 token **共用的唯一发布入口**，不再各写一套 hash 逻辑：

```python
all_ids = seq.all_token_ids
full_blocks = min(len(all_ids), seq.cache.length) // self.block_size
```

覆盖范围完全由 `cache.length` 决定，三条约束自然满足：

- 不完整尾块不登记（整除）；
- 刚采样未计算的 token 不在 `cache.length` 里，进不了块；
- hash 链按 `all_token_ids` 的块顺序算，前块 hash 参与后块 hash，生成 token 落进的块也有稳定身份。

发布时机：`post_step()` 里、forward 之后、**采样 append 之后**——看似矛盾，其实正是要点：
`cache.length` 在本轮 forward 后就已是真实进度，采样只是把新 token 记进 `output_ids`，
`cache.length` 没跟着动，所以新 token 不会被算进任何完整块。

放在**每步**而不是只在抢占/完成前，是为了让被抢占的请求"进度早就在缓存里"，
恢复时（它自己或别的请求）才有东西可复用。

**为什么这个条件里要带 `preemption_mode == "recompute"`？** 只带 `enable_prefix_caching`
技术上也能跑——实测把条件放宽后 legacy 相关回归（contract 96/96、capacity 17/17、
selection 32/32、features 29/29、io_contract 46/46、numerics 12/12）全过，没有任何东西坏掉。
带上前提是为了让 `preemption_mode=None` 走的代码路径与 step45 **逐字节相同**：
需求 §4.8 要求旧配置行为一致，加条件后这条一致是**由构造保证**的，不加则变成
「我测了 135 组没发现差异」——证据强度不同。那 135 组（一次性到达 / 动态到达 / 3~6 块紧容量池子）
之所以测不出差异，是因为 legacy 下前缀复用是**数值透明**的：命中的块本来就是同样 token、
同样权重算出来的 KV，命中与否不改变 logits，只会改变资源账（谁和谁共享物理块、
哪些块可淘汰、准入成不成功）。**这是一个保守的边界选择，不是已被证明的必要性**——
它依赖「当前复用数值透明」这个性质，以后加 logits cache 或部分块复用就不成立了。

### C. 恢复时查最长可用前缀

`find_matched_prefix_blocks(token_ids, max_tokens=None)` 现在对 `all_token_ids` 查找。
**「留一个 token」是准入的策略，不是查询原语的语义**——第一次实现把默认值写成
`len(token_ids) - 1`，直接调用它的既有回归（`lru_cpu` / `lru_graph` 期望 8 个 token 的两个
整块都返回）当场挂掉。现在原语默认不设上限，准入显式传 `max_tokens=len(all_token_ids) - 1`：

```python
all_ids = seq.all_token_ids
matched_blocks, matched_hashes = self.find_matched_prefix_blocks(
    all_ids, max_tokens=len(all_ids) - 1)
```

本实现没有缓存 logits，不留一个 token 就采不出下一个 token 的分数。命中还必须是完整块，
所以实际上限再向下取整到块边界。

需求给的例子（历史长 9、块大小 4、前两块仍在）：`max_tokens = 8` → 命中 2 块 → `cache.length = 8`，
第 9 个 token 仍进 forward ✓。

首次准入时 `all_token_ids == prompt_ids`，与旧实现的 `find_matched_prefix_blocks(prompt_ids[:-1])`
完全等价（`(len(prompt)-1) // bs`）——所以新旧走的是同一条路径，不需要分支。

### D. 容量与引用计数

- **可行性按最坏逻辑块数判定**：`worst_case > num_kv_blocks` 就明确拒绝，**不能用命中块放宽**。
  命中的块同样占物理块——这条请求要同时持有 `worst_case` 块才能跑完（命中块被引用期间不可淘汰），
  池子装不下它就是要拒绝。旧代码用的是 `need = worst_case - len(matched)`，属于过度乐观，
  本关按需求 §D 改正。
- `ensure_blocks()` 失败仍无副作用；闲置 cached block 仍按 LRU 淘汰，被引用的共享块不可淘汰。
- 抢占只释放活动引用，缓存条目保留；`block_hash` / `block_to_hash` 双向索引保持一致，
  `_evict_block()` 同时删两边，所以物理块一旦复用，旧 hash 不会再命中它。

## 3. 可观测性

三个量不再混在一起：

```text
历史 token 数      = len(all_token_ids)
实际重算 token 数  = seq.recomputed_tokens   （进入模型的历史 token，与旧高水位求交）
从缓存复用 token 数 = seq.reused_tokens      （本关新增；命中的完整块数 × block_size）
```

`reused_tokens` 在 `allocate_block()` **成功借到之后**才累加——走失败路径
（返回 `False` / 抛 `InfeasibleRequest`）一块都不计。这一条有专项检查：
构造一条前 8 个 token 能命中缓存、但整条请求最坏需要 42 块（池子 32 块）的请求，
确认它抛 `InfeasibleRequest` 时 `reused_tokens` 仍为 0、块表与 `cache.length` 都没被部分改动。

## 4. 验证（只做功能正确性）

`benchmarks/check_step46_resume_prefix.py`，**47 项全部通过**：

| 需求 §4 条目 | 检查 | 结果 |
|---|---|---|
| A | `recompute` + prefix 可同时开，两个开关都没被偷偷改掉 | PASS |
| A | 非法 `preemption_mode` 仍在构造期报 `ValueError` | PASS |
| A | `from_model_dir` 同样支持 `recompute` + prefix，三种组合输出一致 | PASS |
| 3 | prompt=3、block_size=4：刚采样 y0 时已计算 3、已发布 0 | PASS |
| 3 | y0 真进过模型后发布含 y0 的块，**命中长度 4 > prompt 长度 3** | PASS |
| 2 | 正命中：1 次抢占、复用 12 token、实际重算 3（关闭 prefix 时重算 15） | PASS |
| 2 | 正命中：输出与独占运行一致、结束时块引用与承诺归零 | PASS |
| 4 | 边界：历史 8 / 7 / 9 的命中都 `< 历史长度` 且是整块倍数 | PASS |
| 4 | 边界：第二块内容不同时，第一块之后立刻停 | PASS |
| 5 | 共享：后到请求复用同一批物理块；两条完成后引用归零 | PASS |
| 5 | 淘汰：`block_hash`/`block_to_hash` 双向一致，无 hash 指向已被占用的物理块 | PASS |
| 6 | 容量不足：最坏逻辑块数超池仍明确失败（命中不能蒙混过关） | PASS |
| 6 | 多请求压力 48 组：有界完成、输出正确、无负引用、无承诺泄漏 | PASS |
| 7 | 固定 seed 随机采样 / 惩罚计数：抢占 + 复用后仍与独占运行一致 | PASS |
| 7 | `on_token` 拼接等于最终输出、索引连续；`on_finished` 每请求一次 | PASS |
| 7 | 分块 prefill 正常完成 | PASS |
| — | 阻塞者规则保留：prefix 开/关都是 4 次抢占、20 次阻塞跳过、完成序 `A,B,C,D,E` | PASS |
| 8 | 旧三种配置（`None` + prefix 开/关、`recompute` + prefix 关）输出与 step45 一致 | PASS |
| 4 | 查询原语默认不设上限（8 个 token 返回两整块），准入口径才留 1 个 token | PASS |
| 3 | 命中但随后不可行：`reused_tokens` 仍是 0，块表与 `cache.length` 没被部分改动 | PASS |

GPU：`recompute + enable_prefix_caching=True`、4 块池子在 CUDA/Torch、Triton eager、Triton Graph
三条路径上都 14 步有界完成、各 1 次抢占、无越界或块表破坏（验收方的 GPU 冒烟脚本改指向 step46 跑）。

前两关的不变量脚本指向 step46 后仍是 **3 / 3**（承诺账本全程为 0、全局 FCFS、准入不算进展）。

### 正命中实例（同一配置、同一权重）

```text
pool 8 块、block_size 4、两条 prompt 6 / 输出上限 12
prefix 开：抢占 1 次，复用 12 token，实际重算 3
prefix 关：抢占 1 次，实际重算 15
```

**本关不做性能测试**（按 `02_每次更新后的性能测量约定`）。重算 token 数减少是功能指标，
不是吞吐结论。

## 5. 接口变化与遗留

### 5.1 接口变化

**新增**：`SequenceConfig.reused_tokens`；`KVCachePool.publish_computed_blocks()`
（替换 `publish_completed_prompt_blocks()`，旧名不再存在）；
`find_matched_prefix_blocks()` 新增可选参数 `max_tokens`。

**行为变化**：`recompute` + `enable_prefix_caching=True` 从构造期报错变成合法组合。

**未改**：`Engine(...)` / `Engine.from_model_dir(...)` 的参数列表、`step()` 返回格式、
`on_finished` / `on_token` 语义、`preemption_mode` 的取值、全部模型/kernel 接口。

### 5.2 遗留

1. **不做部分块复用**：只复用完整 KV 块，因此不存在"两个请求同时写同一个不完整块"的写时复制问题。
2. **没有 logits cache**：每个恢复的请求至少有一个历史 token 要重新进模型。
3. **不做 swap/offload、priority preemption、异步在途批次**。
4. **`worst_case` 的可行性判定改了**（旧代码用 `need`，会把命中块当成"省下来的容量"）。
   旧三种配置的回归全过，但这确实是一处**行为变化**——池子刚好卡在边界、且命中前缀块时，
   现在会明确拒绝而不是接纳后跑不完。
5. **性能未测**。复用的收益与 LRU 淘汰带来的额外缓存占用没有量化，等专题收尾再统一测。
