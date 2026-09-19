# step18：跨请求前缀缓存与共享块管理

- 对应代码：`step18/step18.py`（未提交，工作区改动 +156 / -107）
- 源码 SHA256：`3b64f063e7a1b8daf05d692d6159fb4360d1f1e486803d1bead7a10358a47a59`

## 0. 需求大概

在第七关到第十七关的基础上，让同一个 Engine 里先后到达的请求复用相同的 prompt 前缀，不要反复计算同一段 KV。要求：

- 一个开关 `enable_prefix_caching`，关掉就回到第十七关的行为；
- 只缓存 prompt 中填满的块，回答部分和未填满的尾块不缓存，不做 copy-on-write；
- 一个请求结束，不能把别的请求正在读的共享块释放掉；没有活动请求也不等于这个块可以立刻被覆盖；
- 池子被缓存占满时要能淘汰"没人用但还留着"的条目继续跑，淘汰不能碰正在被使用的块。

验收方三次打回：一次是调度计划没按轮重置导致越界写入；一次是分配时"总块数"与"新增块数"混用，加上释放请求时把前缀缓存条目一起删了；最后一次是 `_stable_hash` 的编码把 token ID 卡在 0–255（19/20 里唯一未通过的那条）。

## 1. 改动内容

| 位置 | 改动 |
|---|---|
| `KVCachePool.__init__` | 新增 `block_to_hash`（物理块 → 缓存条目 hash）、`block_last_used`、`lru_seq` |
| `KVCachePool._free_block_indices` | 新增，区分"真正空闲"与"闲置缓存" |
| `KVCachePool._evict_block` / `_mark_used` | 新增，淘汰条目与刷新 LRU 序号 |
| `KVCachePool.allocate_block` | 重写：三个数量分开、容量预检、命中块保护、失败零副作用 |
| `KVCachePool.deallocate_block` | 只遍历块表减引用，不再用 `zip(block_table, block_hashes)`，不再删除缓存条目 |
| `KVCachePool.publish_completed_prompt_blocks` | 新增，请求结束时登记完整 prompt 块，接上 hash 链 |
| `KVCachePool.find_matched_prefix_blocks` | 返回值改为 `(matched_blocks, matched_hashes)` |
| `_stable_hash` | 改用 `json.dumps((previous_hash.hex(), block))` 编码，token ID 不再限制在 0–255 |
| `SequenceConfig` | 删掉 `all_ids` / `new_hashes` / `generate_new_block_hashes` / `update_block_hashes`；`block_hashes` 改为"已确定的 hash 链" |
| `Scheduler.schedule` | 删掉计划期的 `generate_new_block_hashes` 调用 |
| `Scheduler.post_step` | 完成分支里先 `publish_completed_prompt_blocks` 再 `deallocate_block` |

## 2. 设计要点

### 2.1 块有两个正交的事实，不是一个计数

| 字段 | 含义 |
|---|---|
| `block_usage[i]` | 有几个活动请求正在引用 |
| `block_to_hash[i]` | 这个块是否仍被缓存索引保留 |

由此导出三种状态，这是本关的地基：

- 有引用 → 不能覆盖；
- 无引用 + 被索引保留 → **闲置缓存**，先淘汰条目才能分配；
- 无引用 + 无索引 → **真正空闲**，可直接分配。

「请求结束」只让第一列归零，不改变第二列。所以 `deallocate_block` 里不能删条目，`allocate_block` 里也不能把 `block_usage == 0` 一律当成可覆盖空间。

### 2.2 索引里一条 key 只记一个块，不记整条块表

```python
self.block_hash[hash_value] = block_idx      # 单个物理块编号
```

命中时逐块累加，每一步只贡献"这条 key 指向的、当前活着的那个块"：

```python
matched_blocks.append(self.block_hash[current_hash])
```

这**不是省内存的问题，是正确性问题**。查找的每一步都当场重算链式 hash、当场查索引，是活的；而"这条 key 的前缀由哪些块组成"走一遍就能推出来。把它一起存进值里，就多出一份必须与 `block_to_hash`、淘汰、重新登记持续保持一致的冗余状态，而且它过期时没有任何机制会发现。

失效路径（A/B 实测复现，池 4 块、块大小 4）：

```
W：[0,1,2,3,0,1,2,3] 留下 H0 -> b0、H1 -> [b0, b1]
X：需要 3 块，按 LRU 淘汰 H0，b0 被回收并写入 X 的 KV
D：占掉 b0，又换了一次内容
Z：用 [0,1,2,3] 重算，H0 重新登记 -> b3
    此时 H0 的活条目指向 b3，而 H1 的快照里还记着 b0
E：用同样的 8-token 前缀来借用
    单块表示 -> [b3, b1]，第 1 块是 [0,1,2,3] 的 KV   ✅
    快照表示 -> [b0, b1]，第 1 块是 D 的数据          ❌
```

根因是 `matched_blocks = self.block_hash[current_hash]`——快照**整条覆盖**了前面正确走出来的结果。单块表示下没有快照，也就没有"过期快照盖掉新结果"这回事。

一句话：**别把"走一遍就能得到的东西"存进索引。**

推论：`block_hash` 与 `block_to_hash` 必须始终互为逆映射（一个块不被两条 key 指着，一条 key 也不指两个块）。登记时靠"新块必然来自真正空闲或刚被淘汰"保证，淘汰时靠 `_evict_block` 一次清掉两边。

### 2.3 分配：三个数量必须分开

```
total_blocks_needed = ceil((len(prompt) + max_new_tokens - 1) / block_size)   # 块表最终长度
matched_blocks                                                                # 已经借到的
new_blocks_needed   = total_blocks_needed - matched_blocks                    # 还需申请的私有块
```

左边的列表长度是「目前总共有多少块」，右边减完之后是「还需要新增多少块」，两者不能放进同一个比较里。

先做容量预检（`真正空闲 + 可淘汰闲置缓存 ≥ new_blocks_needed`），**确认够用才开始改状态**，因此接纳失败不留副作用：不淘汰条目、不刷新 LRU、不泄漏半份私有块。

### 2.4 命中块的保护是"先加引用"而不是"加白名单"

`matched_blocks` 在改状态阶段第一步就把 `block_usage` 加 1，于是它们自动落进"有引用 → 不能覆盖"那一类；可淘汰集合再显式排除它们，覆盖"命中块此刻引用还是 0（闲置缓存）"的窗口。

### 2.5 hash 链必须接上

块 key 是 `H(前一块 hash, 本块 token)`，所以命中 h 块之后 `seq.block_hashes` 不能是空的：

- 命中时把匹配到的 hash 一起取回来，写进 `seq.block_hashes`；
- 登记时从 `len(seq.block_hashes)` 接着往后算。

第二块的 key 必须是 `H(第一块 hash, prompt[4:8])`，而不是 `H(空, prompt[4:8])`——同一小段 token 出现在不同前文后面，KV 不一样。

交给 sha256 之前，`(前一个 hash, 本块 token)` 要先编码成 bytes。这里必须用**结构化**的编码：

```python
data = json.dumps((previous_hash.hex(), block)).encode("utf-8")
```

不能用 `bytes(block)`——那是"创建若干字节，值分别为这些整数"，一个字节只能放 0–255，`vocab_size=512` 时 token 256 直接 `ValueError: bytes must be in range(0, 256)`。也不能对 256 取模来绕开异常，那会把不同 token 混成一个 key。JSON 是结构化的，`(1, 23)` 与 `(12, 3)` 也不会撞。

### 2.6 发布时机回到"请求结束"，顺带删掉一整条机制

需求允许把发布时机提前到 forward 之后，真实 vLLM 甚至在调度阶段就登记映射；但它也把"请求完成时才登记"列为简化策略。按最小实现选后者之后，`all_ids` / `new_hashes` / `generate_new_block_hashes` / `update_block_hashes` / `update_hash` 全部失去用途——它们存在的唯一理由是在 schedule 里增量算 hash。

删掉它们同时消掉了验收反馈里的两个问题：`self.all_ids = prompt_ids` 的列表别名、以及"decode 已登记 hash 但 `all_ids` 里没有输出 token"。现在登记面只取 `min(len(prompt_ids), cache.length) // block_size`，输出 token 永远不参与块 key。

### 2.7 复用上限与登记上限是两条规则

```
当前请求最多读取：(len(prompt_ids) - 1) // block_size     # 至少留最后一个 prompt token 重算 logits
请求完成时最多登记：len(prompt_ids) // block_size
```

前者由 `allocate_block` 传 `seq.prompt_ids[:-1]` 实现，后者由 `publish` 的 `min(...)` 实现。长度 8 的 prompt 只用 4 个 token 的缓存，却登记两块，供未来长度 9 的请求命中。

### 2.8 LRU 只解决"空间不够时先舍弃哪一份"

`lru_seq` 递增，登记新条目和被接纳的请求命中时刷新。不做抢占：正在被请求使用的块永远不在候选里。

## 3. 验证

conda 环境 `vllm-omni-dev`。`TinyCausalLM` 按 `torch.cuda.is_available()` 落在 cuda:0，float32；验收方仍在 CPU float32 上跑。

- **需求给的三个例子**：B 第一次 prefill 输入 `[2,3]`、位置 4/5，前 4 个位置的 K/V 与关闭缓存时逐值相同；B/C 共享块 0 各自私有块、一条结束时另一条不受影响；缓存占满后按 LRU 腾空间继续运行。
- **复用上限表**：prompt 5/6/8/9/12/13 → 复用 4/4/4/8/8/12 个 token，与需求给的复用上限表一致。
- **开关对照**：零预算、EOS 提前结束、回调顺序、chunked prefill，开启/关闭输出一致。
- **随机对照 300 组**（block_size ∈ {1,2,4}、池 2–8 块、并发 1–3、预算 1–8）：输出与完成顺序一致，开启缓存的请求不晚于关闭时完成，结束后 `block_usage` 全零、无引用泄漏。
- **失败路径**：单请求需求超过池容量时 `allocate_block` 返回 False，且 `block_usage` / `block_hash` / `block_to_hash` / `lru_seq` 全部不变。
- **淘汰后**：旧 key 不再命中，重新提交同一前缀会重新计算并重建条目。
- **条目内容不变量**：120 组随机场景（池 2–6 块、强制淘汰）跑完后，用同权重独立模型逐条重算每个活条目对应前缀的 K/V，与物理块里的值对照，0 条不符。判据要用 `torch.allclose(atol=1e-5)`——批量 GEMM 与单序列 GEMM 的浮点末位有差异，`torch.equal` 会误报。
- **索引不变量**：300 组场景、60027 次 `step()` 逐步检查 `block_hash` 与 `block_to_hash` 互为逆映射，结束时 `block_usage` 全零。
- **表示 A/B**：同一场景下把 key 的值换成整条块表快照，E 借到的第 1 块变成已被复用的 b0；单块表示借到 b3（见 2.2）。
- **大 token ID**：`Engine(vocab_size=512, max_num_batched_tokens=8)`、prompt `[0, 255, 256, 511, 1]`，开启缓存不再报错，输出与关闭时一致；随机对照 300 组改用 token ∈ {0, 255, 256, 511, 3}、`vocab_size=512` 重跑，输出不一致数 0。

未跑：性能基准（冷/暖/未命中分别记录）由验收方测。

## 4. 接口变化与遗留

- `find_matched_prefix_blocks(prompt_ids)` 返回值由 `list[int]` 变为 `tuple[list[int], list[bytes]]`。
- `KVCachePool.block_hash` 的 value 语义已明确为**单个物理块编号**（当前值就是这一块的编号），不再是"到这一块为止的整条块表"。
- `KVCachePool.update_hash` / `SequenceConfig.generate_new_block_hashes` 已删除。
- `SequenceConfig.all_ids` 已删除。
- 遗留：`Scheduler.num_scheduled_tokens`、文件顶部 `from more_itertools import last`、`from torch import ceil` 均无人使用，属第十七关遗留，本次未清理。
- 未提交。
