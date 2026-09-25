# step51：像 vLLM 一样增量维护完整 token 历史

从 `step50/` 复制。`step50/` 原样保留，未改。

## 结论

剖析测到的第二个热点：`_plan_tokens()` 里 `seq.all_token_ids[start:end]` 每次都要先算
`prompt_ids + output_ids`——**每步把整段历史复制一遍**（8192 输出 token 时单次 83.63 μs）。

改成**增量维护**：初始化复制一次 prompt，之后每生成一个 token 只追加。

```python
self._all_token_ids = list(self.prompt_ids)   # 按逻辑位置切输入、算 prefix hash
self._output_ids    = []                      # 直接回答生成了多少 / 最后是什么
def append_output_ids(self, token_ids):       # 唯一写入点，两份同步追加
    ...
```

两份列表**不是冗余**：它们由同一个写入点更新，且四条不变量可查（长度和前缀/后缀关系）。
`cache.length` 仍是「已计算并写入 KV 的长度」，**没有**和 `len(all_token_ids)` 合并。

对外暴露成**只读视图**（`collections.abc.Sequence` 包装，不复制底层 list）：

- 不支持 `append` / `extend` / 赋值——改内容只有 `append_output_ids()` 一条路
- 支持索引、切片、`len`、`in`、迭代、`== list`、`+ list`（既有脚本会这么写）
- 两个属性是**只读 property**：能赋值的话，一次 `seq.output_ids = [...]` 就会让视图和
  私有列表分家

顺带一处：`publish_computed_blocks()` 没有新完整块时**早返回**（每步只有 1 个 token 进模型，
块很久才满一次，这是热路径）。

## 顺带解耦：更新 prefix cache 不再看 `preemption_mode`

`_publish_computed_blocks()` 原本只在 `recompute` 模式下每步登记，承诺式仍是「请求结束时
才登记」。**这两件事本来就不相干**——什么时候更新 prefix cache，和「准入时要不要按最坏情况
承诺未来块」是两套机制。本关把它们解耦。

**这是一次有意偏离第四十六关 §4.8（旧配置行为一致）的行为变化**，用户明确要求在本关完成
并会通知验收方。实测 768 组 legacy 负载：

| 模式 | 完成顺序不同 | 输出不同 | 资源异常 |
|---|---:|---:|---:|
| prefix 关 | 0 | 0 | 0 |
| prefix **开** | **108** | **0** | **0** |

只影响「谁先跑完」，不影响输出与资源守恒。**也不违反 FCFS**——FCFS 的三条排序规则
（waiting 按到达序、犠牲者从 running 尾部取、被抢占者回队首）都没动；变的是后到的请求
**够不够格**准入（它命中了正在运行的请求尚未释放的块，记账因此更准），不是「轮到谁」。

## 效果（需求指定的定点基准，原始数据在 `benchmarks/results/`）

16 请求、每个都有 N 个输出 token：

| 输出长度 | `_plan_tokens` | `_publish_computed_blocks` |
|---|---:|---:|
| 128 | 7.97 → 5.59 μs | 4.11 → 1.90 μs |
| 2048 | 23.65 → 6.04 μs | 20.35 → 2.18 μs |
| 8192 | 81.62 → **6.20 μs（13.2×）** | 76.37 → **2.42 μs（31.6×）** |
| 128→8192 增长 | **10.2× → 1.11×** | **18.6× → 1.27×** |

验收要求「8192 时至少快 5 倍、128→8192 不再近似线性」，两项都远超。

## 怎么证明行为没变

| 检查 | 结果 |
|---|---|
| `benchmarks/check_step51_history.py`：单请求 + 不变量 + 只读视图 | **36 项全通过** |
| `benchmarks/diff_step50_step51.py`：逐步对照 | **88 项全通过**（11 场景 × 2 seed） |
| 其中**物理块编号、hash 链、历史长度**都要求逐项一致（本关不改 KV 分配与调度） | 一致 |
| step47/46/45 自查、step44 三不变量、CPU 压测 120、GPU 三路径、18 个既有回归 | 全过 |

复现：`benchmarks/profile_step50_long_history.py step50 step51`（性能）、
`benchmarks/diff_step50_step51.py`（行为）、`benchmarks/check_step51_history.py`（不变量）。

完整说明见 [`docs/step51_incremental_history.md`](../docs/step51_incremental_history.md)。
