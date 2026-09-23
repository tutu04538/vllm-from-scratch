# step45：阻塞者感知的恢复准入

从 `step44/` 复制。`step44/` 原样保留，未改。

## 结论

第四十四关会正确重算了，但恢复太急：刚被 A 抢占的 B 下一步又被准入，A 还没完成、内存再次紧张，
B 又被 A 抢一次。固定 5 请求、4 块池子全部完成却发生 **17 次抢占**——「输出正确」不等于「调度合理」。

本关只加一条策略：**记录是谁迫使你让路；在它结束之前，不要急着恢复你。**

```python
# _make_room 里把「当前是谁要块」传下去
self._preempt(victim, blocker=seq)
# _preempt 里记下
seq.resume_blocker = blocker
```

准入 waiting 队首时，如果它的 `resume_blocker` 还在 `running`/`waiting` 里，就跳过**并 `break`**
——被阻塞的队首同样不能让更晚到达的请求越过它，哪怕 `running` 还有空名额。
阻塞者一旦结束（或被明确拒绝），关系当场失效并清掉引用。

`Engine` 没有新参数：这套行为是 `preemption_mode="recompute"` 的一部分。legacy 模式与
prefix cache 路径一个字节没动。

## 实测（功能正确性）

| 场景 | step44 | step45 |
|---|---:|---:|
| A–E、4 块池子：抢占次数 | 17 | **4** |
| 完成顺序 | `A,B,C,D,E` | `A,B,C,D,E` |

- A/B：被 A 挡住之后、A 完成之前，B 不再回到 `running`；两条输出与独占运行逐 token 一致。
- 动态到达：B 被挡期间加入 F，F 不会越过 B；A 完成后按 FCFS 恢复为 `A,B,F`。
- 容量充足：不抢占、阻塞跳过次数为 0、输出与 step44 一致。
- legacy：输出与 step44 一致，承诺账本与 step43 一致。
- 72 组随机配置：有界完成、输出正确、无残留块引用与阻塞引用。

复现：`benchmarks/check_step45_blocker.py`。

**本关不做性能测试**（按 `02_每次更新后的性能测量约定`，从第四十五关起不再每关跑性能矩阵）。
抢占次数减少是功能指标，不是吞吐结论——这套策略可能减少并发，不保证总吞吐更高。

完整说明见 [`docs/step45_blocker_aware_resume.md`](../docs/step45_blocker_aware_resume.md)。
