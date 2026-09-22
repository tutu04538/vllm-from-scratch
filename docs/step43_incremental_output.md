# step43：增量输出观察点（`on_token`）

- 对应代码：`step43/`（新增，从 `step41/` 复制 —— 需求 §2 明确说不要从没有实现的 `step42/` 复制）
- 包摘要 SHA256：`febd8c3663a061f6…`（14 个 .py / 2729 行，验收方 `source_digest()` 口径）
  —— **这是 §5 参数位置修复后的最终值**；首次验收快照是 `a60cb36de5fe3877…`（2726 行），
  之后 `engine.py` 的签名改动让行数与指纹都变了
- 基线：`step41/`，指纹 `e862960a…`（14 个 .py / 2714 行），原样保留未改
- **改动文件只有两个**：`engine.py`、`__init__.py`（+ 入口改名 `step43.py`）
  —— 模型 forward / kernel / 采样算法 / 调度策略 / KV 分配 / Graph key 都没动

## 0. 需求大概

现在 `step()` 只在请求**完成**时把结果交出来。目标：一条请求产生第 1 个 token 就通知一次，
第 2 个再通知一次……中间仍能在 step 边界提交新请求。

这不是重新实现 continuous batching（逐步调度与动态入队早就有），而是**补一个公开的
增量输出观察点**，并用持续到达场景验证已有能力。

## 1. 新增接口

```python
Engine(..., *, on_token=None)                 # keyword-only，在所有旧参数之后
Engine.from_model_dir(..., *, on_token=None)
```

**`on_token` 是 keyword-only 且排在最后。** 首次验收就是因为把它插在旧参数中间、
导致旧的位置调用错位（详见 §5）。传入函数后，每产生并写入请求状态的一个新 token 调用一次：

```python
on_token({"request_id": "A", "token_id": 123, "output_index": 0})
```

不传时保持原行为。回调在调用 `step()` 的同一线程内同步执行。

## 2. 改在哪里

需求 §4 指出要找的位置是「token 已经成为 Python 整数、已经提交到请求状态，但请求还没被
结束和清理」。就是 `Engine._sample()` 里那个循环：

```python
output_ids = torch.stack(tokens).tolist()          # 复用既有的整批回传，不额外 .item()/同步
notify = self.on_token
for item, output_id in zip(picked, output_ids):
    seq = item["request"]
    index = len(seq.output_ids)                    # 提交前的长度 == 这次输出的序号
    seq.output_ids.append(output_id)
    seq.sampling_state.note_output_token(output_id)
    if notify is not None:
        notify({"request_id": seq.request_id, "token_id": output_id,
                "output_index": index})
```

四个设计点：

1. **复用 `torch.stack(tokens).tolist()` 的结果**，没有为了通知再去 `.item()` 或
   `torch.cuda.synchronize()`（需求 §4 的硬要求）。
2. **序号取提交前的 `len(seq.output_ids)`** —— 它天然就是「第几个生成 token」，
   不用另开计数器；零预算请求一进来就结束，这个循环不会经过它。
3. **顺序沿用 `picked`**（本轮采样结果的顺序），不是 `running` 列表的顺序（需求 §3.5）。
4. **`notify = self.on_token` 提到循环外**，且每次新建一个只有 ID/整数的字典 ——
   需求 §4 要求未设置回调时「不创建事件字典、不取时间戳、不累计事件历史」，
   引擎也不永久保存任何通知。

`step()` 的返回值、`on_finished` 的参数与调用次数**一个字没改**（需求 §3.7）。

## 3. 验证

`benchmarks/check_step43_on_token.py`（本关新增），全部通过：

| 检查 | 结果 |
|---|---|
| 6 个生成 token → 恰好 6 次通知 | PASS |
| `output_index` 从 0 连续递增 | PASS `[0,1,2,3,4,5]` |
| `on_finished` 仍只发一次、带完整 `output_ids` | PASS |
| 多请求：事件按请求拼接后 **等于**最终 `output_ids` | PASS（A 5 个、B 7 个） |
| 零预算请求：不发 token 通知，仍产生完成记录 | PASS |
| 不可能完成的请求：不发 token 通知，仍走既有明确失败路径 | PASS |
| 回调里改字典不改变引擎状态 | PASS |
| 开关回调不改变生成结果 | PASS |
| chunked prefill（prompt 24、预算 8）：中间块不发通知，只发 3 次 | PASS `[0,1,2]` |

### 回归

用验收方 `verify_step41_*.py` 的副本（指向改成 step43）跑：

| 脚本 | 结果 |
|---|---|
| contract | 96 / 96 |
| io_contract | 46 / 46 |
| external | 53 / 53 |
| merge | 41 / 41 |
| selection | 32 / 32 |
| features | 29 / 29 |
| precision | 23 / 23 |
| qwen3 | 21 / 21 |
| capacity | 17 / 17 |
| tiles | 14 / 14 |
| numerics | 12 / 12 |
| slots | 10 / 10 |
| stride | 9 / 9 |
| real_attention | 56 / 56 |
| sampling | 45 / 45 |
| structure | 通过 |
| rope | 33 / 34（**既有失败**，step39/41 同样 33/34） |

## 4. 举一个收到 token 事件的例子

需求 §7 说可附一个小例子，不必跑正式性能实验：

```python
events = []
engine = Engine.from_model_dir(MODEL_DIR, on_token=lambda e: events.append(dict(e)))
engine.add_request({"request_id": "A", "prompt_ids": ids, "max_new_tokens": 5})

while engine.has_unfinished_requests():
    engine.step()
    # 每步结束就能看到这一刻已经产生的 token，不必等 A 结束
    print(events)
```

`events` 会随着 step 逐步变长（`output_index` 0,1,2,…），而不是最后一次性出现 5 条。
**这就是本关要说清的接口位置与语义**；TTFT / ITL / 完整延迟的统计由验收方负责。

## 5. 首次验收后的修正：参数位置

首次验收（`152_第四十三关首次验收`）发现一个 API 兼容性问题：**我把 `on_token` 插在了
旧参数 `enable_prefix_caching` 的中间**，于是旧的位置调用整体错位：

```python
Engine(1, 4, 4, 8, 5, 8, 32, None, False)   # 原意：on_finished=None, prefix cache 关
# 错位后：False 被当成 on_token，prefix cache 反而变回 True
# 下一步 _sample 里 notify({...}) -> TypeError: 'bool' object is not callable
```

**给新参数默认值，不代表旧调用兼容。** Python 按位置配对，不知道调用者心里的 `False`
是什么意思。

改法（验收 §4）：把 `on_token` 挪到**所有旧参数之后**，并做成 keyword-only：

```python
# Engine.__init__
..., norm_backend="torch", rope_backend="torch", model=None, *, on_token=None

# Engine.from_model_dir
..., norm_backend="torch", rope_backend="torch", *, on_token=None
```

`*` 放在旧参数**之后**，所以旧的位置参数一个都没挪位。

实测（`benchmarks/check_step43_on_token.py` 之外单独跑）：

| 检查 | 结果 |
|---|---|
| `Engine(1,4,4,8,5,8,32,None,False)` 的逐字段绑定与 step41 一致 | PASS（`enable_prefix_caching=False`、`on_token=None`） |
| 同一段调用实际跑一步 | PASS，不再抛 `TypeError` |
| `on_token=events.append` 的 keyword 调用仍正常 | PASS |
| `from_model_dir` 的位置参数同样不再错位 | PASS |

`step41/` 未改（需求 §4 第 3 条）。

## 6. 接口变化与遗留

### 6.1 接口变化

**新增**：`Engine(..., on_token=None)`、`Engine.from_model_dir(..., on_token=None)`、
`Engine.on_token` 属性。

**未改**：`step()` 的返回格式、`on_finished` 的参数与调用次数、`add_request`、
`has_unfinished_requests`、以及全部模型/kernel/调度/KV 接口。

### 6.2 遗留

1. **回调不重入、不提交新请求、不抛异常** —— 本关约定，与需求 §3 一致。重入、
   异常恢复、慢消费者队列都不在本关范围内。
2. **没有做任何延迟统计**。TTFT / ITL / 完整延迟的测量工具按需求 §6 由验收方写；
   本关只提供观察点。
3. **性能没测**。需求 §6 把性能分两层（静态六点、持续到达），都归验收方。
   本关只保证未设置回调时**没有事件构造、没有额外 GPU 回传**；
回调关闭时仍然增加了属性读取、分支和 `len()` 等 CPU 操作，
是否测得出开销由实验判断（验收 §5 的更正）。
4. **持续到达驱动（跨到达波次、引擎清空后再到达）没有实现** —— 需求 §5 给了单线程
   驱动的写法，但明确说工具由验收方负责，本关不另写 benchmark 系统。
