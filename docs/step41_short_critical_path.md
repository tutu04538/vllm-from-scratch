# step41：缩短 decode 一步的关键路径（slot mapping 不再每请求每步 H2D）

- 对应代码：`step41/`（新增，从 `step40/` 复制）
- 包摘要 SHA256：`870f6e3cbc8498999798a444541ef3d5c13e07db22d0de18ebbff72f4e53262f`（14 个 .py / 2714 行）
- 基线：`step40/`，指纹 `1abf01f1…`（14 个 .py / 2698 行），原样保留未改
- 改动文件：`cache.py`、`__init__.py`、入口改名 `step41.py`

## 0. 需求大概

依据第四十关的 vLLM 同条件基准：单请求已经打平或反超，**差距全在并发**（`short_c8` 1.55×、
`decode_c8` 1.35×）。而 `decode_c8` 拆开看是**严格串行**的：

```text
[CPU 合计 4.84 ms] → [GPU 3.34 ms，CPU 在里面等着]     步墙钟 4.87 ms
理想是 max(CPU, GPU)
```

目标是让步墙钟与 GPU 忙碌时间脱钩，向 `max(CPU, GPU)` 靠。

需求 §2 给两条允许的路，**只做一条**：
1. 去掉 `_sample` 里 `torch.stack(tokens).tolist()` 那个同步
2. 缩短同步之前的关键路径（`forward_cpu`）

§6 要求**先测量再决定**，不猜。

## 1. 测量改变了目标

用 `review_cpu_phase.py` 拿基线，再把 `forward_cpu` 单独插桩拆开：

| 阶段 | 每步 |
|---|---:|
| `schedule` + `build_inputs` + `post_step` | 0.037 ms |
| **`forward_cpu`** | **1.491 ms** |
| — 其中 `build_slot_mapping` | **1.028 ms（69%）** |
| — 其中 `graph.replay()` | ~0.1 ms |
| **`_sample`**（GPU 同步点） | **3.065 ms** |

需求里把 `forward_cpu` 那 1.7 ms 描述成「元数据准备 + 图重放 + logits 收集」，但**图重放只占
0.1 ms**，真正的大头是 `build_slot_mapping`。所以按 §6 的要求先测量是对的——照需求里那句
描述去改，会改错地方。

**为什么不直接做第 1 条（去掉同步）**：需求 §3 规定「EOS 与 `max_new_tokens` 的判定时点不变」，
而 `post_step` 每步都要看采样出来的 token 才能判 EOS。所以同步不能凭空拿掉，只能改成
**投机执行**（拿上一轮的结果先跑、发现该停就丢掉那一步），那是一个更大的改动。先把关键路径
上白花的 1 ms 拿掉，收益确定、风险低。

## 2. 根因：每个请求每步两次 H2D

```python
def _slots_of_range(self, block_table, start, count):
    positions = torch.arange(start, start + count, device=self.device)        # H2D（传起点）
    blocks = torch.tensor(block_table, device=self.device, dtype=torch.long)  # H2D（Python 列表 -> GPU）
    return blocks[positions // self.block_size] * self.block_size + positions % self.block_size

def build_slot_mapping(self, caches, counts):
    return torch.cat([self._slots_of_range(...) for ...])
```

**decode 时 `count=1`**：传的数据只有 1 个，成本全是发射与 pageable H2D 的开销。
8 个请求 × 2 次 H2D + 每个请求若干小 kernel + 一次 `torch.cat` ≈ 1.0 ms/步。

## 3. 改法：在 CPU 上按「块区间」展开

```python
def _slots_of_range(self, block_table, start, count):
    bs = self.block_size
    slots = []
    for blk in range(start // bs, (start + count - 1) // bs + 1):
        lo = max(start, blk * bs)
        hi = min(start + count, (blk + 1) * bs)
        base = block_table[blk] * bs
        slots.extend(range(base + lo - blk * bs, base + hi - blk * bs))
    return slots

def build_slot_mapping(self, caches, counts):
    slots = []
    for cache, count in zip(caches, counts):
        slots.extend(self._slots_of_range(cache.block_table, cache.length, count))
    return torch.tensor(slots, dtype=torch.long)      # CPU，接口不变
```

关键点：位置区间 `[start, start+count)` 按块切开后，**每一块对应的槽位是连续的**，所以用
`range` 展开就够了（C 层），既不用逐位置写 Python 循环，也不建任何临时张量。
循环次数是「跨了几个块」而不是「有多少 token」——decode 时是 1 次。

**返回类型保持不变**（CPU 上的 `torch.long` 张量）：验收的
`verify_step2X_slots_contract.py` 里断言了 `slots.dtype == torch.long and slots.device.type == 'cpu'`。
本来想直接返回 `list`，看到断言后改成张量。

调用方 `self.slot_buffer[:num_tokens].copy_(slot_mapping)` 不用改——只是从「16 次小 H2D」
变成「1 次整批 H2D」。

## 4. 验证

### 4.1 数值：逐位一致

200 组随机输入（`block_size` ∈ {1,3,4,7,16}、块表乱序、`length`/`count` 随机，
含 `count=0`）与 step40 的实现对照：

```text
200 组随机输入，与 step40 逐位一致: True
```

### 4.2 单函数耗时

```text
decode 形状（8 请求 × 1 token）    step40 0.077 ms  ->  step41 0.004 ms   （19×）
prefill 形状（8 请求 × 256 token） step40 0.093 ms  ->  step41 0.074 ms   （也略快）
```

### 4.3 归因对照（需求 §4.1 要的那张表）

`review_cpu_phase.py --case decode_c8`，两个版本各跑一次：

| | step40 | step41 | 变化 |
|---|---:|---:|---:|
| `forward_cpu` | 1.644 ms | **0.638 ms** | **−1.006 ms** |
| `sample`（等 GPU） | 3.093 ms | 3.246 ms | +0.153 ms |
| **CPU 合计** | **4.771 ms** | **3.913 ms** | **−0.858 ms** |
| **步墙钟** | **4.771 ms** | **3.913 ms** | **−0.858 ms（−18%）** |
| GPU kernel 时间 | 2.974 ms | 3.119 ms | 噪声 |

两点必须说清楚：

1. **墙钟省下的量与 CPU 减少量相同（0.858 ms）**，说明**串行链变短了，但还没有重叠**——
   CPU 合计仍然等于墙钟。需求 §2 的目标是「墙钟明显小于 CPU 合计」，**这一条还没达到**。
2. **GPU kernel 时间没有下降**（2.974 → 3.119 是两个进程之间的噪声）——收益不来自
   「少算了什么」，符合需求 §4.6 的红线。

`sample` 略微上升是因为 CPU 更早到达同步点、在那里等得更久；总量仍是下降的。

### 4.4 输出逐 token 不变（需求 §3 硬约束）

| 负载 | 结果 |
|---|---|
| `short_c1`（1 条请求 × 32 token） | 逐 token 相同 |
| `decode_c8`（8 条请求 × 1024 token） | 逐 token 相同 |

### 4.5 回归

用验收方脚本的副本（指向改成 step41）：

| 脚本 | 结果 |
|---|---|
| contract | 96 / 96 |
| external | 53 / 53 |
| io_contract | 46 / 46 |
| merge | 41 / 41 |
| selection | 32 / 32 |
| features | 29 / 29 |
| precision | 23 / 23 |
| qwen3 | 21 / 21 |
| capacity | 17 / 17 |
| numerics | 12 / 12 |
| stride | 9 / 9 |

**全过。**

`verify_step2X_slots_contract.py`（5 个）**跑不起来**，报
`TypeError: ... not 'NoneType'`（`Path(m.__file__)`，`m.__file__` 是 None）。
**在原始目录里对 step23 也是同样的错**，与本次改动无关；它们测的契约（返回 CPU long 张量、
槽位值正确）由 §4.1 的 200 组对照覆盖。

## 5. 接口变化与遗留

### 5.1 接口变化

**对外接口没有变化**：`build_slot_mapping(caches, counts)` 仍然返回 CPU 上的 `torch.long` 张量。

内部变化：

| | 之前 | 现在 |
|---|---|---|
| `_slots_of_range` 返回 | GPU 张量 | Python `list` |
| 每个请求的 H2D 次数 | 2 | 0 |
| 整批 H2D 次数 | 0（`torch.cat` 在 GPU 上） | 1 |
| `gather()`（参考路径） | 直接用 GPU 张量 | 多一次 `torch.tensor(..., device=...)` |

`_slots_of_range` 没有被任何验收脚本直接引用，所以可以改返回类型。

### 5.2 遗留

1. **本关没有达成「重叠」**。需求 §2 的目标是步墙钟 < CPU 合计；现在只是把串行链缩短了
   0.86 ms，CPU 合计仍等于墙钟。按需求 §5 的话，剩下的是「同步点的等待（`sample` 3.2 ms）」
   与「forward_cpu 剩下的 0.638 ms」，那是下一关。
2. **`sample` 仍是 3.2 ms 的同步等待**。要消掉它必须做投机执行（先用上一轮 token 跑下一步，
   判到 EOS 再丢弃），而需求 §3 对「被丢掉的那一步不能改动任何对外可见状态」有明确要求——
   这是一个需要单独设计的改动，不在本关。
3. **没有跑六点 A/B**（需求 §4.4）。单点的归因对照已经给出收益，但 `c1` 两点是否回退、
   `c8` 两点各自拿到多少，都还没测。
4. **`forward_cpu` 还剩 0.638 ms** 没有继续拆（positions 构建、sample_buffer、三处 H2D、
   attention 元数据 fill/upload）。需求 §6 的下一步本来就是这个方向的延伸。
