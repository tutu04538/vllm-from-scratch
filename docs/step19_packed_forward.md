# step19：无 padding 打包与 prefill/decode 混合 forward

- 对应代码：`step19/step19.py`（新增，未提交）
- 源码 SHA256：`2b09d7350eb03f8ca9e190aafda45115b6a563f997944cbb3d2abb76feb4b4e5`
- 基线：`step18/step18.py` 原样保留、未修改

## 0. 需求大概

换一个浪费来源：请求长度不同，模型为了凑矩形 batch 要补零，补出来的假 token 也进了 embedding 和 Q/K/V 投影；而且 prefill 和 decode 分两次模型调用。本关要求：

- 本轮所有真实 token 拼成**一维** tensor，不加 padding，`N = sum(num_scheduled_tokens)`；
- prefill 和 decode **共用一次**模型调用；没有计算任务就不调用；
- 返回 `logits[N, vocab_size]`，行顺序与输入 token 相同；
- 每个请求只能看见自己的历史 KV 和自己的本轮 token，本轮内部仍满足因果；
- KV 仍原位追加，长度增量等于该请求的 count，共享前缀块不被覆盖；
- 采样只对就绪请求，取自己片段的最后一行；中间 prefill chunk 不采样；
- step18 的全部能力（prefix cache、共享、LRU、EOS、零预算、预算规则、回调顺序）不变。

## 1. 改动内容

| 位置 | 改动 |
|---|---|
| `TinyCausalLM._forward_append` | `input_ids` 由 `(B, T)` 改为 `(N,)` 一维；先按 offsets 切分逐请求追加 KV，再逐请求 gather 自己的历史做 attention；返回 `(N, vocab_size)` |
| `Scheduler.__init__` / `schedule` | `prefill_scheduled_items` 与 `decode_scheduled_items` 合成一份 `self.scheduled_items`；两个 prefill 分支合并为 `min(prefill_len, budget)` |
| `Engine.step` | 删掉 `_forward` 闭包与两次分组调用；一次打包 + 一次 forward + 一次采样 |
| `Engine._sample` | 从 `step()` 内的闭包变成方法；按片段末行取 logits 并写回输出 |

预算规则、FIFO 接纳、最大并发、prefix cache、LRU、`post_step` 一行未动。

## 2. 设计要点

### 2.1 打包下标不是模型位置

```text
A 输入：[2]        历史 KV 长度 3
B 输入：[1,0,3]    历史 KV 长度 0
C 输入：[2]        历史 KV 长度 0

input_ids    = [2, 1,0,3, 2]
counts       = [1, 3, 1]
offsets      = [0, 1, 4, 5]         # 片段边界，A=input_ids[0:1] 等
position_ids = [3, 0,1,2, 0]        # 每个请求从自己的 cache.length 开始
```

`offsets` 只是"这段属于谁"，`position_ids` 才是"模型眼里的第几个位置"。命中前缀的请求从命中后的 `cache.length` 开始，所以同一个扁平数组里可能同时出现位置 3 和位置 0。

### 2.2 一次投影，但 attention 必须逐请求做

embedding、Q/K/V 对 `[N]` / `[N, d_model]` 各调用一次，没有补零位置。

但 attention 不能在整条扁平数组上套一个普通下三角 mask——那样 B 会读到排在前面的 A。做法是投影完成后按 offsets 切片，每个请求 gather 自己的历史 KV，和自己的本轮 Q/K/V 做一次 attention，再按计划顺序拼回 `logits[N, vocab]`。

这是本关允许的简化实现：仍有 Python 循环和 gather 复制，**不等于 PagedAttention，也不保证端到端更快**。

### 2.3 先写 KV 再 gather

`append` 在前、`gather` 在后：本轮 token 的 KV 写进池子后，`gather(cache)` 直接返回"历史 + 本轮"，attention 不需要自己拼 `past_k` 和本轮 K/V 两段。长度增量严格等于该请求的 count，共享的前缀块只读，写入只落在私有块上。

mask 用**数组下标**而不是 clamp 后的位置：`key_pos > query_pos` 就屏蔽，query 的位置是 `[cache.length - n, cache.length)`。这样即使序列超过 `max_seq_len`、embedding 位置被 clamp，因果边界仍然正确。

### 2.4 采样行 = 片段末行

```text
A → logits[0]      （片段 [0:1]）
B → logits[3]      （片段 [1:4]）
C → logits[4]      （片段 [4:5]）
```

行号就是 `offset - 1`，和请求从**同一份计划**里对应出来，不需要另建映射。`can_sample=False` 的中间 prefill chunk 一律跳过，尽管它的 KV 已经更新；行和请求都由 `scheduled_items` 现算，不存在两份状态不同步的问题。

### 2.5 计划只有一份

`schedule()` 里仍然是一次遍历 `self.running`：`prefill_len == 0` 的请求安排 1 个 decode token，其余在 `prefill_token_budget` 里分块。decode 优先预留、队首不能接纳时不让后面绕过，都与 step18 相同——只是结果写进同一个列表，不再分两组。

## 3. 验证

conda `vllm-omni-dev`；`TinyCausalLM` 落在 cuda:0，float32。

- **需求 §1 的例子**（A decode 1、B prefill 3、C prefill 1，预算 5）：扁平输入 `shape=(5,)`、`token 数=5`、`logits shape=(5, 64)`、模型调用 **1** 次；采样行 A→0、B→3、C→4。同场景 step18 投影 7 个位置、调用 2 次。
- **step18 / step19 行为等价**：5 个定向场景（基础、prefix cache 命中、共享块、关闭缓存、零预算+EOS）+ 300 组随机场景（并发 1–3、预算 1/2/4/8、块大小 1/2/4、池 2–8 块、缓存在 1/3 场景关闭），**逐轮返回与回调顺序不一致 0 例**，结束后 `block_usage` 全零。
- **逐步 KV 与独立参考一致**：40 组随机场景、353 次逐步核对，把每个活动请求 `(prompt + output_ids)[:cache.length]` 用同权重独立模型重算 K/V，与 `gather` 结果逐值对照（`allclose(atol=1e-5)`），**不符 0**。
- **每个生成 token 都是参考的 argmax**：193 个生成 token，逐个用独立模型在 `prompt + out[:j]` 上重算，**不符 0**。这直接验证了采样行取的是正确位置的 logits。
- **从不 padding**：300 组场景里每次 `_forward_append` 都满足 `input_ids.numel() == sum(num_scheduled_tokens)`。
- **中间 prefill chunk 不采样**：预算 2、prompt 7 个 token，逐轮 `(cache长度, 已生成数, 剩余prefill) = (2,0,5) (4,0,3) (6,0,1)`，中途生成数始终为 0。
- **位置上限**：`max_seq_len=8`、prompt 10 个 token，step19 不崩溃，输出与"同样 clamp 的参考"逐 token 一致。
- **prefix cache 回归**：step18 的三套测试（条目内容不变量 120 组、索引双射 300 组 / 60027 次 step、开关输出对照 300 组）改用 step19 重跑，全部通过。
- **参考量级**：300 组随机场景的投影位置总数 step18=2118、step19=2073（减少 2.1%）。随机场景里大多数轮次各请求 count 相同，补零本来就少；收益只在长度不齐时明显（§1 的例子是 7→5）。**这个数字不能当性能结论**，性能由验收方按相同负载对比。

未跑：真实场景的性能基准。

## 4. 接口变化与遗留

- `_forward_append(input_ids, num_scheduled_tokens, past_kv, kv_cache_pool)`：`input_ids` 由 `(B, T)` 变为 `(N,)`，返回由 `(B, T, vocab)` 变为 `(N, vocab)`。参数名与顺序未变。
- `Scheduler.prefill_scheduled_items` / `Scheduler.decode_scheduled_items` 删除，合并为 `Scheduler.scheduled_items`。
- `Engine._sample` 由 `step()` 内的闭包变为私有方法。`Engine` 的公开方法与构造参数未变。
- `step18.py` 未修改；`TinyCausalLM.forward`（完整历史参考实现）保留未动。
- 遗留：逐请求 attention 仍是 Python 循环 + `gather` 逐位置 stack，不是变长 attention kernel；`_forward_append` 里每个请求各做一次 softmax，小模型下这批开销可能盖过省下的投影。
- 发现但未处理：`step18.py` 在序列超过 `max_seq_len` 时会崩（`kv_max_seq_len` 用的是 clamp 之后的位置，导致补零长度为负）。step19 用数组下标做 attention，不受影响。按需求"不修改旧 step18.py"未动它。
- 未提交。
