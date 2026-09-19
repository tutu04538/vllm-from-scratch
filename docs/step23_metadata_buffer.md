# step23：复用 attention 元数据缓冲区，一次上传

- 对应代码：`step23/step23.py`（新增，未提交）
- 源码 SHA256：`d609f8665972be594e88354d6eac6ba2c95fbf85a7cc27dc9857e7daaaf261ce`
- 基线：`step22/step22.py`（SHA256 `4f196fee…`）原样保留、未修改

## 0. 需求大概

step22 的 attention 已经只启动一个 kernel，但调用前的 `_triton_attention()` 每轮都在重建元数据：每轮新建 GPU `block_tables`（还要逐请求把一份小 GPU 张量复制到对应行）、`seq_lens`、`token_to_req`、`query_pos`。

观测到的实际开销是：`[1,3,1]` 三请求时，attention 前有 **6 次 CPU→GPU 复制、3 次 GPU→GPU 复制**，而 attention 本身只有一次 kernel。

本关把这套元数据改成**固定容量的持久缓冲区**：CPU 一份、GPU 一份，四个区域是同一段连续 int32 存储的视图；每轮只改内容、不换存储，用**一次 `copy_`** 整段上传。不改 attention 数学，不改 Triton kernel。

## 1. 改动内容

| 位置 | 改动 |
|---|---|
| `AttentionMetadata` | 新增：固定容量缓冲对象，含 CPU/GPU 主缓冲、四个区域的视图、`fill()`、`upload()` |
| `TinyCausalLM.__init__` | 新增 `attention_metadata=None` |
| `TinyCausalLM._triton_attention` | 从"每轮新建 4 个 GPU 张量"改为 `fill()` → `upload()` → 把视图交给 `paged_attention` |
| `Engine.__init__` | triton 后端时按配置创建 `AttentionMetadata` 并交给模型 |

kernel、`paged_attention()`、slot mapping、KV 写入、采样、调度全部未改。

## 2. 设计要点

### 2.1 一段存储，四个视图

容量来自引擎配置：

```text
Bmax = max_num_seqs
Nmax = max_num_batched_tokens
Kmax = ceil(max_seq_len / block_size)

seq_lens      [Bmax]            偏移 [0, Bmax)
block_tables  [Bmax, Kmax]      偏移 [Bmax, Bmax + Bmax*Kmax)
token_to_req  [Nmax]            偏移 [...]
query_pos     [Nmax]            最后

一共 Bmax*(1+Kmax) + 2*Nmax 个 int32
```

需求 §2 的例子（Bmax=3、Nmax=5、Kmax=8）正好 37 个整数。四个区域用切片 + `view` 解释成各自形状，`untyped_storage().data_ptr()` 与主缓冲一致——已核对。

在 `Engine.__init__` 里按配置分配，**不是**按第一次恰好出现的 batch 大小分配；同一 Engine 运行期间地址不变（切片与 `view` 都只是新的头部，不换存储）。

### 2.2 每轮：填 → 一次上传 → 交给 kernel

```python
def fill(self, past_kv, num_scheduled_tokens):   # 只用真实用量覆盖有效区域
    for req_idx, (_cache, count) in enumerate(...):
        self.cpu_seq_lens[req_idx] = _cache.length
        self.cpu_block_tables[req_idx, :len(bt)] = torch.tensor(bt, dtype=torch.int32)
        token_to_req.extend([req_idx] * count)
        query_pos.extend(range(_cache.length - count, _cache.length))
    self.cpu_token_to_req[:N] = ...
    self.cpu_query_pos[:N] = ...

def upload(self):
    self.gpu_buffer.copy_(self.cpu_buffer)       # 整段一次 H2D，non_blocking=False
```

然后把**整段容量视图**交给 kernel。这是安全的，因为 kernel 由 `grid=(N,)` 和 `seq_len` 驱动，只会读：

- `seq_lens[req]`，其中 `req = token_to_req[row]`，恒小于本轮真实请求数；
- `block_tables[req, blk]`，`blk < ceil(seq_len / block_size)`，不超过该请求的真实块表长度；
- `token_to_req[row]` / `query_pos[row]`，`row < N`。

所以尾部残留的旧数据不会被读到，既不需要每轮清零，也不需要按真实长度重建视图（重建视图虽然不复制数据，但会让"传给 kernel 的张量"每轮换对象）。

### 2.3 下一轮比上一轮小

需求 §4 的边界，实测：

```text
第一轮 counts=[1,3,1]
  seq_lens      = [4, 3, 1]
  token_to_req  = [0, 1, 1, 1, 2]
  query_pos     = [3, 0, 1, 2, 0]

第二轮 counts=[1]
  有效部分      = [0] / [1] / [2]
  尾部残留      = token_to_req[1:] = [1,1,1,2,0,0,0]   ← 仍在缓冲里，但不会被读到
```

把整个 CPU 缓冲先填成 `-12345` 再跑同一个负载，输出与干净缓冲完全一致——这是"尾部残留不参与计算"的直接证据。

### 2.4 容量超限要报错，不静默截断

三处都检查，错误信息指明是哪一项容量、由哪个配置决定：

```text
本轮请求数 2 超过元数据缓冲容量 1（由 max_num_seqs 决定）
本轮 query 数 3 超过元数据缓冲容量 2（由 max_num_batched_tokens 决定）
块表长度 4 超过元数据缓冲容量 2（由 ceil(max_seq_len / block_size) 决定）
```

注意最后一条相对 step22 是**行为收紧**：step22 的块表长度只受池容量限制，而这里还受 `Kmax = ceil(max_seq_len / block_size)` 限制。prompt + 生成长度超过 `max_seq_len` 的负载现在会明确报错而不是继续跑（位置本来就会被 clamp，属于本关划定的容量边界）。

## 3. 验证

conda `vllm-omni-dev`，RTX 5090 Laptop，torch 2.13.0+cu130，Triton 3.7.1，FP32。

### 3.1 缓冲区与生命周期

- **布局**：Bmax=3、Nmax=5、Kmax=8 → 缓冲 `numel() == 37`，四个视图形状 `(3,) (3,8) (5,) (5,)`。
- **共享存储**：四个 GPU 视图的 `untyped_storage().data_ptr()` 都等于主缓冲。
- **地址不变**：同一引擎跑完一个多轮负载后，CPU/GPU 主缓冲的 `data_ptr` 与初始一致。
- **脏数据**：整段填 `-12345` 后跑同一负载，输出与干净缓冲逐项一致。
- **§4 边界**：batch 大→小后，有效区域正确，尾部残留不被读取（见 2.3）。
- **容量报错**：三类超限各返回明确的 `ValueError`。

### 3.2 拷贝次数（PyTorch Profiler，预热后单次调用）

只统计 attention 准备 + kernel，`[1,3,1]` 三请求：

| | CPU→GPU | GPU→GPU | kernel |
|---|---:|---:|---:|
| step22 | **6** | **3** | 1 |
| step23 | **1** | **0** | 1 |

这与需求里记录的观测值（6 次 H2D、3 次 D2D）完全吻合，改完后降到 1 次 H2D、0 次 D2D。

整个 `step()` 层面（还包含 input_ids、采样同步等其他拷贝）：HtoD 9 → 5，DtoD 2 → 0。

### 3.3 正确性回归

- **step22 vs step23**：200 组随机场景（并发 1–3、预算 1/2/4/8、块大小 1/2/3/4、池 4–8 块、`max_seq_len` 16/32），**逐轮返回与回调顺序不一致 0 例**。
- **prefix cache 三套回归**：条目内容不变量（120 组，不符 0）、索引双射（300 组 / 60027 次 step）、开关输出对照（300 组，不一致 0）在 step23 上全部通过。

### 3.4 计时（粗测，同机同数据、预热后）

attention 准备 + kernel，`[1,3,1]` 三请求：

| | 耗时 |
|---|---:|
| step22 逐轮建 GPU 张量 | 167.1 μs |
| step23 固定缓冲一次上传 | **66.2 μs** |

完整 Engine（8 请求 × prompt 500 + 生成 24，块大小 128，预算 64，三轮取最快）：

| | 轮数 | 最快 |
|---|---:|---:|
| step22 | 88 | 98.9 ms |
| step23 | 88 | **62.9 ms** |

元数据路径本身约 2.5x；端到端约 1.57x。绝对数值会随机器状态波动（同一天不同次测量的 step22 完整引擎出现过 98.9 ms 与 141.6 ms 两个值），**倍数应在同一批进程内比较**；正式测量由验收方分层进行。

## 4. 接口变化与遗留

- 新增 `AttentionMetadata(max_num_seqs, max_num_query_tokens, max_blocks_per_request, device)`，含 `fill()`、`upload()` 与八个视图属性。
- 新增 `TinyCausalLM(..., attention_metadata=None)`。
- `Engine(device=..., attention_backend="triton")` 会自动创建并挂上元数据缓冲；torch 后端下为 `None`。
- kernel、`paged_attention()`、`CacheConfig`、调度器、KV 池、slot mapping 未改。
- 遗留：`fill()` 仍按请求用 Python 循环写入，每个请求建一个临时 CPU 张量（需求允许）；容量恒为配置上限，未按真实用量压缩上传字节数。
- 遗留：`upload()` 用 `non_blocking=False`，未用 pinned memory、未做异步复制——需求明确本关不引入。
- 遗留：块表容量 `Kmax = ceil(max_seq_len / block_size)` 收紧了对超长 prompt 的容忍度（见 2.4）。
- 未提交。
