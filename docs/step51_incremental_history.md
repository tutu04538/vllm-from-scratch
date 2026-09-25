# step51：像 vLLM 一样增量维护完整 token 历史

- 对应代码：`step51/`（新增，从 `step50/` 复制，入口改名 `step51.py`）
- 包摘要 SHA256：`1976084f4f51b33a…`（15 个 .py / 3423 行，验收方 `source_digest()` 口径）
- 基线：`step50/`，指纹 `a896e0f9e56620f5…`（15 个 .py / 3335 行），原样保留未改
- **改动 4 个文件**：

| 文件 | 改动 |
|---|---|
| `request.py`（+83 / −7） | 新增 `ReadOnlyTokenList`（只读视图）；`SequenceConfig` 增量维护 `_all_token_ids` / `_output_ids`，`append_output_ids()` 是唯一写入点 |
| `cache.py`（+4 / −0） | `publish_computed_blocks()` 没有新完整块时**早返回** |
| `engine.py`（+1 / −1） | 采样提交改走 `seq.append_output_ids(output_id)` |
| `scheduler.py`（+4 / −2） | 完成记录显式 `list(seq.output_ids)`，不把只读视图泄漏给外部 |
| `__init__.py`、`step51.py` | 包说明与入口改名 |

`attention.py`、`model.py`、`norm.py`、`rope.py`、`sampler.py`、`sampling.py`、`formats/` 未改。

## 0. 需求大概

剖析（第四十八关后那份）测到第二个热点：`_plan_tokens()` 里 `seq.all_token_ids[start:end]`
每次都要先算 `prompt_ids + output_ids`——**每步把整段历史复制一遍**。16 请求、各有 8192
输出 token 时，一次 `_plan_tokens()` 约 83.63 μs（只有 1 个输出 token 时约 6.06 μs）。

需求把「只在读取时拼」改成「像 vLLM 的 `Request` 那样增量维护」，并保留两份列表：
`_all_token_ids` 按逻辑位置切模型输入、算 prefix hash；`_output_ids` 直接回答
「生成了多少 / 最后是什么 / 返回给用户什么」。

这不是「两份状态互相矛盾」——它们由**同一个写入点**同步更新，且不变量可查。

## 1. 实现

### 1.1 只读视图

```python
class ReadOnlyTokenList(Sequence):
    __slots__ = ("_backing",)
    def __getitem__(self, index): return self._backing[index]
    def __len__(self):            return len(self._backing)
    ...
```

为什么不用「每次返回 `list(...)` / `tuple(...)`」来做只读：**那又会把整段历史复制一遍**，
正是本关要消掉的开销。这里只包一层引用；切片 `all_token_ids[3:8]` 直接落到原 list 上，
只复制那一段 ✓

对外接口尽量与普通 list 一致，因为既有脚本会这么写（我核对过验收回归）：

| 用法 | 从哪儿来 |
|---|---|
| `view[i]`、切片、`len`、`in`、迭代 | `collections.abc.Sequence` |
| `prompt_ids + output_ids` | 自定义 `__add__` / `__radd__`（验收回归里确实有两处） |
| `view == [...]` | 自定义 `__eq__` |

**没有** `append` / `extend` / `insert` / `pop` / `__setitem__` ✓

两个属性是**只读 property**：赋值会 `AttributeError`。这一点值得单独说——如果只把
`__init__` 里赋成视图、属性本身可写，`seq.output_ids = [...]` 能悄悄把视图换掉，
于是 `append_output_ids()` 改的私有列表就和外界看到的分家了。

### 1.2 唯一写入点

```python
def append_output_ids(self, token_ids):      # int 或 list[int]
    if isinstance(token_ids, int):
        token_ids = [token_ids]
    self._output_ids.extend(token_ids)
    self._all_token_ids.extend(token_ids)
```

`Engine._sample()` 里原来的 `seq.output_ids.append(output_id)` 改走这里。
追加一次的摊销代价是 O(1)，换掉了每步 O(len(history)) 的复制。

### 1.3 发布时早返回

```python
full_blocks = seq.cache.length // self.block_size
if full_blocks <= len(seq.block_hashes):
    return          # 本轮没有新算满的完整块
```

`full_blocks` 不需要 `min(len(all_ids), cache.length)`：`cache.length` 恒**严格小于**
`len(all_token_ids)`——刚采样的那个 token 已经在历史里、但还没进模型（不缓存 logits，
至少留一个历史 token 走模型），所以永远差至少 1 个。实测 54 组负载 747 次调用，
差值最小为 1、相等 0 次。

**要说清楚它买到了什么**：下面的 `for i in range(len(block_hashes), full_blocks)` 在这时
**本来就是空的**，所以这一句**不是正确性需要的**——hash 链不动、token 不切，是循环空转本身
保证的。它省掉的只是「为每个请求构造 `range` 并进入一次迭代器」，实测 16 请求约 0.3 μs
（三轮对照方向一致：有早返回 2.44~2.69 μs，去掉 2.76~2.90 μs）。

保留它是因为代价确实存在且需求也要求了这一步；但注释最初写成「不碰 hash 链、不切任何 token」
是**把循环本来就保证的事算在了自己头上**，已改成实话。

### 1.4 必须保留的边界

**追加进 `_all_token_ids` ≠ 增加 `cache.length`**，也 ≠ 允许发布它所在的 KV 块。
刚采样的 token 还没进模型，`publish_computed_blocks()` 用 `cache.length` 判断完整块，
所以它天然不会被发布 ✓ 这条边界原样保留（也是将来投机解码「未接受草稿不能提前进历史」
的落点）。

## 2. 不变量

```text
len(all_token_ids) == len(prompt_ids) + len(output_ids)
all_token_ids[:len(prompt_ids)] == prompt_ids
all_token_ids[len(prompt_ids):] == output_ids
0 <= cache.length <= len(all_token_ids)      # 对仍在运行的请求
```

`cache.length` 仍是「已计算并写入 KV 的长度」，**没有**和 `len(all_token_ids)` 合并。

## 3. 验证

### 3.1 定点性能（需求 §验收）

`benchmarks/profile_step50_long_history.py step50 step51`，16 请求、每个都有 N 个输出 token，
所有请求都处于 decode 就绪、发布时没有新完整块。原始数据：
`benchmarks/results/step50_long_history_profile.json`。

| 输出长度 | `_plan_tokens` step50 → step51 | `_publish_computed_blocks` step50 → step51 |
|---|---:|---:|
| 128 | 7.97 → 5.59 μs | 4.11 → 1.90 μs |
| 2048 | 23.65 → 6.04 μs | 20.35 → 2.18 μs |
| 8192 | 81.62 → **6.20 μs（13.2×）** | 76.37 → **2.42 μs（31.6×）** |
| 128→8192 增长 | **10.2× → 1.11×** | **18.6× → 1.27×** |

验收要求「8192 时至少快 5 倍、128→8192 不再近似线性」——两项都远超。
另外 `all_token_ids[-1:]` 在 8192 长历史下仍是 0.10 μs（step50 是 4.28 μs）。
**这是定点 CPU 基准，不是端到端吞吐。**

### 3.2 单请求与不变量（`benchmarks/check_step51_history.py`，36 项全通过）

- 初始化：`all == prompt`、`output` 为空；`prompt_ids` 是独立副本（外部改原列表不影响请求）；
- 连续追加 1 个 / 多个 / 空列表；每次后四条不变量都成立；
- prompt / output / 跨边界切片；切片返回普通 list；
- 视图没有 8 个可变方法；两个属性不能被赋值；支持 `len`/`in`/迭代/`== list`/`+ list`；
- 8192 长历史下切最后 1 个 < 5 μs（实测 ~0.1 μs）；
- 四个负载（普通 decode、chunked prefill、prefix 命中、容量压力抢占后重算）：
  输出逐 token 与 step50 相同，`on_token` 事件序列相同，记录里是普通 list。

### 3.3 逐步对照（`benchmarks/diff_step50_step51.py`，88 项全通过）

同 seed、同请求、同到达时刻，逐步比较调度队列、本轮计划、输入 token、完成输出、
每请求计数、承诺与引用、结束态。11 场景 × 2 seed。

本关**不改 KV 分配与调度**，所以比对**比上一关更强**——连这些也要求逐项一致：

- **物理块编号**（`block_table`）；
- **每个请求的 hash 链**（历史改成增量维护后必须完全一样）；
- **已提交历史的长度与末尾内容**（`len(all_token_ids)`、`len(output_ids)`、最后 3 个）。

### 3.4 其余

| 项 | 结果 |
|---|---|
| step47 / step46 / step45 的自查（新引擎换成 step51） | 42 / 44 / 38 项全过 |
| step44 的三个不变量（指向 step51） | 3 / 3 |
| CPU 随机压测 120 组 | PASS |
| GPU：CUDA/Torch、Triton eager、Triton Graph | 各 14 步有界完成，引用归零 |
| 既有 18 个回归脚本 | 全过（副本里同步了 `block_hash` → `hash_to_block` 改名） |

## 4. 接口变化与遗留

### 4.1 接口变化

**新增**：`request.ReadOnlyTokenList`；`SequenceConfig.append_output_ids(int | list[int])`、
`SequenceConfig.all_token_ids_len`。

**语义变化**：`SequenceConfig.all_token_ids` / `.output_ids` 从普通 list / 属性变成
**只读视图属性**——支持读取与切片，**不支持 append / extend / 赋值**。
`step()→step_done` 与 `on_finished` 给出的 `output_ids` 仍是**普通 list**（显式转换）✓

**未改**：`Engine` / `from_model_dir` 参数、调度与优先级、KV 分配与淘汰、prefix hash 算法与
命中规则、采样结果、模型。

### 4.2 遗留

1. **不做 token_slice()**（原第五十一关草案的接口）——需求已修订为增量维护完整历史。
2. **不做异步占位符、流式会话、多模态分支**，那些是 vLLM 的其它扩展。
3. **`_output_ids` 与 `_all_token_ids` 是两份列表**，靠单一写入点与不变量保证一致；
   本关没有为「投机解码未接受草稿」预留位置，但边界已经明确（见 §1.4）。
4. 性能是定点 CPU 结论，不做整套 GPU 吞吐矩阵。
