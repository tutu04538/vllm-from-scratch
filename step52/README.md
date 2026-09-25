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
`preemption_mode=None`、`enable_prefix_caching=False`、`attention_backend="torch"`、
`use_cuda_graph=False`，请求必须是贪心且无惩罚项。CPU 与 CUDA/Torch 都可以。

## 怎么证明它是对的

| 检查 | 结果 |
|---|---|
| `benchmarks/check_step52_speculative.py`：纯函数 + 回滚 + 配置校验 | **53 项全通过** |
| `benchmarks/check_step52_engine.py`：脚本模型定点用例 + 真模型等价性 + 回滚不变量 | **28 项全通过** |
| `benchmarks/diff_step51_step52.py`：不开投机时与 step51 逐步对照 | **88 项全通过**（11 场景 × 2 seed） |
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
