# step47：优先级调度与抢占

从 `step46/` 复制。`step46/` 原样保留，未改。

## 结论

高优先级请求晚到时不该一直排在低优先级长请求后面。本关把优先级做进**三种资源**，
语义对齐 vLLM 的 `priority` 模式：**数字越小越优先，同级按到达时间**。

```python
Engine(..., *, scheduling_policy="fcfs")     # 默认：忽略 priority，与 step46 完全一致
Engine(..., scheduling_policy="priority")    # 本关新路径
# 请求字典：{"priority": -1, ...}，默认 0；必须整数，bool 不算
```

`priority` 只支持 `preemption_mode="recompute"`，配 `None` 在构造期报错。

## 稳定排序键

```python
sort_key = (priority, arrival_order)
```

`arrival_order` 是调度器发的内部递增整数，抢占与恢复都不改它。不使用外部 `request_id`
——引擎没有建立「ID 唯一」的约束。

## 三种资源

| 资源 | 做法 |
|---|---|
| 运行名额 | 名额满时，队首可以顶掉一条**严格更低优先级**的 running 请求；**同级不得互相顶替** |
| KV 物理块 | 犠牲者从**排序键严格更靠后**且本轮未跑的请求里选——可以是优先级更低，也可以是**同级但到达更晚**（后者必需，否则同级占满池子谁也动不了） |
| token budget | 按优先级分层分配，高优先级层先拿额度；同级内部保持「ready 各留 1 个、其余给 prefill」 |

## 实测（功能正确性，45 项全通过）

| 场景 | 结果 |
|---|---|
| `max_num_seqs=1`，L 已生成 token、H 后到 | H 顶掉 L，完成序 `H, L`，`num_priority_preemptions=1` |
| `max_num_batched_tokens=1`，L 在 decode，H 需 prefill | priority 下 H **第 1 步**拿到额度；fcfs 下被饿住 **6 步** |
| 三档优先级 | 后到的高优先级先完成，同级按到达序 |
| 同级容量犠牲者 | 两条同级请求都能有界完成，先到的先完成 |
| 阻塞关系 | 低优先级不越过等待中的高优先级 |
| KV 压力（prefix 开/关） | 有界完成、输出正确、引用与承诺归零、hash 双向一致 |
| `fcfs` vs `step46` | 相同负载输出与完成顺序一致 |

复现：`benchmarks/check_step47_priority.py`。

**本关不做性能测试**（按 `02_每次更新后的性能测量约定`）。

## 遗留

不做 aging / 防抖（严格优先级可能让低优先级等很久）、不抢占已拿到本轮块的请求、
不做 swap/offload 与异步在途批次。

完整说明见 [`docs/step47_priority_scheduling.md`](../docs/step47_priority_scheduling.md)。
