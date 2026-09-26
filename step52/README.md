# step52：单请求贪心 n-gram 投机解码

从 `step51/` 复制。`step51/` 原样保留，未改。

## 结论

第一个真正的投机解码闭环：**n-gram 从已提交历史猜 1~2 个 token → 目标模型一次 forward
验证 → 只提交被认可的 → 拒绝部分的 KV 撤回**。

```python
draft_ids = propose_ngram(seq.all_token_ids, n=2, k=2)   # 纯函数，只读历史
item["input_ids"]  = [x] + draft_ids                     # 草稿只进本轮计划
item["num_scheduled_tokens"] = 1 + len(draft_ids)        # 预算/KV 都按真实输入数走
...
greedy = argmax(logits[:K+1])                            # 一次 forward 的 K+1 行
result = verify_drafts(draft_ids, greedy, eos, remaining) # 纯函数：接受几枚、提交哪些、留多少 KV
kv_cache_pool.truncate(seq, start + result.kept_inputs)   # **先回滚**
self._commit_tokens(seq, result.committed_ids, notify)    # 再逐枚提交
```

三条不能破的线：

1. **草稿不是已提交历史**。`seq._all_token_ids` / `_output_ids` 在目标模型认可之前一个
   草稿都没有；`append_output_ids()` 仍是唯一写入点。
2. **K+1 才是本轮真实输入数**。`num_scheduled_tokens` 记的是进模型的 `K+1` 个 token，
   KV 补块、token 预算、`max_seq_len` 都按它算；不被接受的部分靠 `truncate()` 退回去，
   而 `truncate()` 必须发生在 `post_step()` **之前**。
3. **验证不改变结果**。提交的永远是目标模型的贪心 token，所以投机跑出来的文本和普通
   贪心**逐 token 相同**——这是本关最强的等价性检查。

## 明确限制（不支持的组合直接报错，不静默退化）

`speculative_mode="ngram"` 只支持：`max_num_seqs=1`、`scheduling_policy="fcfs"`、
`enable_prefix_caching=False`、`attention_backend="torch"`、`use_cuda_graph=False`，
请求必须是贪心且无惩罚项。CPU 与 CUDA/Torch 都可以。

（`max_num_seqs=1` 已经蕴含不会发生抢占：running 里只有一条请求，fcfs 下既没有
更靠后的犠牲者可选，也不存在「顶掉名额」。）

## 抢占不再有模式开关

删掉了 `preemption_mode` 与 `KVCachePool.over_subscribe`：**抢占是无条件的**，和 vLLM V1
一致（0.28 全包 `grep preemption_mode` 零命中）。准入只判「这条请求单独跑装不装得下」，
不再按最坏情况锁未来容量；不够时 `ensure_blocks()` 返回 False，由调度器从尾部选犧牲者
（priority 模式按 `(priority, arrival_order)`）。

顺带发现：`Scheduler.preemption_mode` 本来就是**死状态**——写了从来没人读，真正的开关
一直是池子的 `over_subscribe`。

代价与收益（默认配置、fcfs）：池子 6 块、两条各要 4 块的请求，旧默认让 B 一直等到 A
跑完（12 次目标 forward），新默许两条并行、B 被抢占一次（10 次），**输出逐 token 相同**。
288 组配置里输出/完成序/结束态全部一致；90 组步数汇总里新默认更少 18、相同 72、更多 0。

## 顺带：请求内容的入口校验

`add_request()` 以前只校验字段名、`priority` 类型、采样参数范围，`prompt_ids` 的内容
和 `max_new_tokens` 的符号没人管。补上之后，这几种坏输入在**入队、分配 KV 之前**就报错：

| 输入 | 以前 | 现在 |
|---|---|---|
| `max_new_tokens = -1` | 请求**静默消失**（被 waiting 的过滤器滤掉，不回调不报错） | `ValueError` |
| `prompt_ids` 里有浮点 | **静默截断**成整数（hash 按 2.5 算、模型看到 2） | `ValueError` |
| `prompt_ids` 越界 / 为负 | embedding 里抛 `IndexError` | `ValueError` |
| `prompt_ids = []` | 调度器排不出 token，靠零进展守卫兜底 | `ValueError` |
| `max_new_tokens = 0` | 合法（只算 prompt、不生成） | 不变 |

随之删掉 `_plan_tokens()` 里的 `num_uncomputed == 0` 半句兜底——那个状态只有空 prompt
能造出来，入口拒绝之后它是死代码，而且是会**吞掉零进展报错**的死代码。

## 怎么证明它是对的

| 检查 | 结果 |
|---|---|
| `benchmarks/check_step52_speculative.py`：纯函数 + 回滚 + 配置校验 | **53 项全通过** |
| `benchmarks/check_step52_engine.py`：脚本模型定点用例 + 真模型等价性 + 回滚不变量 + 入口校验 | **40 项全通过** |
| `benchmarks/diff_step51_step52.py`：与 step51 的重算模式逐步对照 | **88 项全通过**（11 场景 × 2 seed，未放宽任何字段） |
| CPU FP32 / CUDA FP32 / CUDA BF16 | 三者输出一致，投机 8 步 vs 普通 14 步 |

关键用例（脚本模型固定目标输出，KV 与位置仍由真实现推进）：

- **全部接受**：一轮提交 3 枚，`on_token` 序号连续，`on_finished` 只发一次；
  同样文本下投机 4 次目标 forward，普通贪心 6 次
- **首枚拒绝**：那一轮真正进模型 3 个 token，却只提交 1 枚
- **部分接受**：提交 `[d0, t1]`
- **草稿本身是 EOS**：只提交到它，且只保留它**之前**那些草稿的 KV
- **退回普通路径**：找不到草稿 / 预算只够 1 个 / 输出上限只剩 1 个 / 上下文将满
- **真模型等价性**：随机小模型 + 重复 prompt，投机与普通贪心输出逐 token 相同，
  且确实有一轮一次 forward 提交 3 枚

复现：

```bash
python benchmarks/check_step52_speculative.py
python benchmarks/check_step52_engine.py
python benchmarks/diff_step51_step52.py
```

完整说明见 [`docs/step52_ngram_speculative.md`](../docs/step52_ngram_speculative.md)。

**本关不承诺加速**：是否更快取决于接受率与 K+1 行验证的额外计算，这里只把闭环做对。
