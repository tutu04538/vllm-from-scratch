# step22：整批请求，一次 Triton kernel 调用

- 对应代码：`step22/step22.py`（新增，未提交）
- 源码 SHA256：`4f196fee5ad21e05cdec381c5fcb3a27de32fd8c0cb1c0fc41f08caf805768ac`
- 基线：`step21/step21.py`（SHA256 `c427cca1…`）原样保留、未修改

## 0. 需求大概

step21 证明了"不拼完整历史也能算对 attention"，但执行方式还是 Python 遍历请求、再遍历 KV 块、每块调一串 PyTorch 算子。本关把整批 attention 交给**一次** Triton kernel 调用，让不同 query 的工作由 GPU 调度。

要完成的是一个闭环：准备 GPU 元数据 → 整批 attention kernel → 接回 Engine。块循环进入 kernel，online softmax 的数学不变。

约束：保留两个可选后端（`attention_backend="torch"` 走 step21 的分块实现，`"triton"` 走 kernel）；旧调用仍能工作；CPU 选 triton 要**明确报错**而不是悄悄换后端；slot mapping 与批量 KV 写入仍用现有 PyTorch 实现；调度、hash、LRU、回调仍在 CPU；没有本轮 query 时不启动 kernel。

## 1. 改动内容

| 位置 | 改动 |
|---|---|
| `_paged_attention_kernel` | 新增：`@triton.jit` 的整批 attention，一个 program 负责一个 query |
| `paged_attention(...)` | 新增：Python 侧包装，填 grid、传 stride 与 constexpr |
| `TinyCausalLM.__init__` | 新增 `device`（不传时仍是"有 CUDA 就用 CUDA"）与 `attention_backend` |
| `TinyCausalLM._triton_attention` | 新增：把本轮的 packed 输入整理成四个元数据张量并调用 kernel |
| `TinyCausalLM._forward_append` | 按后端二选一；triton 路径一次 `lm_head(out)`，torch 路径保留 step21 的逐请求分块 |
| `Engine.__init__` | 新增 `device`、`attention_backend` 两个可选关键字；校验后端与设备组合 |

## 2. 设计要点

### 2.1 谁负责谁：一行 query 一个 program

```text
打包行号            0  1  2  3  4
token_to_req        0  1  1  1  2
query_pos           3  0  1  2  0
seq_lens = [4,3,1]

program 0 → A 的 query → out[0]
program 1 → B 的第一个 query → out[1]
...
```

`grid = (N,)`，`row = tl.program_id(0)`。program 不是线程，实际并发由硬件调度。这是最简单的工作划分：一个 program 只处理一个 query，块循环在 program 内部。

### 2.2 元数据：不让 kernel 猜请求信息

```python
q[N,D], k_pool[num_blocks,S,D], v_pool[num_blocks,S,D],
block_tables[B,max_blocks_per_request], seq_lens[B], token_to_req[N], query_pos[N] → out[N,D]
```

- `token_to_req[N]`：第 i 个打包行属于哪个请求 → 由它索引块表；
- `query_pos[N]`：该 query 的**绝对位置**，用于 causual mask；
- `seq_lens[B]`：写入本轮 KV **之后**的长度，决定遍历几个逻辑块；
- `block_tables[B, max_blocks]`：不同长度的块表补成矩形。kernel 的循环上界是 `tl.cdiv(seq_len, S)`，**未使用的列不会被读**。

Python 每轮重建这四个小张量并上传；`CacheConfig`、字典、请求对象都不进 kernel。

### 2.3 `tl.arange` 要求长度是 2 的幂 → 两个填充维度

这是本关最容易漏的地方。`tl.arange(0, S)` 在块大小是 3 时直接编译失败，`tl.arange(0, D)` 在 `d_model=12` 时也一样。所以两块都补到 2 的幂，各自带 mask：

```python
offs_s = tl.arange(0, S_POW2)
mask_kv = (offs_s < S)                     # 补宽出来的位置不属于本块
        & (offs_s < seq_len - blk * S)     # 尾块只读有效部分
        & (blk * S + offs_s <= qpos)       # query 不能看到未来的 key
```

**第一个条件不能省**：块大小 3、`seq_len=8` 时，逻辑块 0 的 `seq_len - 0 = 8`，`offs_s=3` 也能通过后两个条件——但它其实是块 1 的位置，读进来的是别的数据（或垃圾）。补宽出来的位置必须单独屏蔽，且**无效 key 的分数置 `-inf`**，不能只把 K 读成 0 就参与归一化。

`D` 那一侧同样：`offs_d < D` 用于读 q、读 K/V、写 out。

### 2.4 causal 用逻辑位置，不用物理块号

`key_positions = blk * S + offs_s`，与物理块编号无关，所以块表乱序（`[7,1]`）照样正确。遍历上界 `tl.cdiv(seq_len, S)` 保证预留但未写入的块不参与。

### 2.5 online softmax 与 step21 完全一致

```python
m_new = tl.maximum(m_i, tl.max(score, axis=0))
scale = tl.exp(m_i - m_new)
p = tl.where(mask_kv, tl.exp(score - m_new), 0.0)
z_i = scale * z_i + tl.sum(p, axis=0)
acc = scale * acc + tl.sum(p[:, None] * v, axis=0)
m_i = m_new
```

初始 `m_i = -inf, z_i = 0, acc = 0`，第一块自然完成初始化；全被 mask 的行 `m_new = m_i`、`scale = 1`、`p = 0`，贡献零且不出 NaN。最后 `acc / z_i` 写回。

### 2.6 后端选择与设备校验

```python
Engine(device="cpu",  attention_backend="torch")    # 可以
Engine(device="cuda", attention_backend="torch")    # 可以，同设备参考
Engine(device="cuda", attention_backend="triton")   # 可以
Engine(device="cpu",  attention_backend="triton")   # ValueError，明确报错
```

`device=None` 保持旧行为（有 CUDA 用 CUDA）；`attention_backend` 默认 `"torch"`，所以旧调用完全不变。模型、KV 池、元数据、q 全部落在同一个 device 上——`TinyCausalLM` 的 `device` 由 Engine 显式传入，不再各处自己判断。

## 3. 验证

conda `vllm-omni-dev`，RTX 5090 Laptop，torch 2.13.0+cu130，Triton 3.7.1，FP32。

### 3.1 kernel 层

- **需求 §3 的五行对应关系**：`token_to_req=[0,1,1,1,2]`、`query_pos=[3,0,1,2,0]`，与完整 attention 参考最大误差 `1.192e-07`，输出 `(5,8)`。
- **块大小 1/3/4、物理块乱序、尾块**：60 组随机夹具，最大误差 `3.576e-07`。
- **非 2 的幂的 d_model**：5 / 8 / 12，最大误差 `1.192e-07 ~ 2.384e-07`（验证两个填充维度的 mask）。
- **极端分数**：K/V 与 q 放大 50 倍，输出全部有限。
- **一次调用一次启动**：把 kernel 换成计数桩，一次 `paged_attention` 启动 **1** 次；q 与池的 K/V 逐值未被改写。

### 3.2 Engine 层

- **CPU + torch 后端可用**：模型参数与 KV 池都在 cpu，能正常跑完。
- **同设备、同一份 `state_dict`，torch vs triton**：123 组随机场景（并发 1–3、预算 1/2/4/8、块大小 1/2/3/4、池 4–8 块、缓存在 1/3 场景关闭），**逐轮返回与回调顺序不一致 0 例**；每次 `_forward_append` 恰好启动 1 次 kernel。
- **与 step21 的 CPU 分块参考对照**：45 组场景，输出不一致 0 例。
- **prefix cache 回归**：条目内容不变量（120 组，不符 0）、索引双射（300 组 / 60027 次 step）、开关输出对照（300 组，不一致 0）在 step22 上全部通过。

### 3.3 三层计时（粗测，同数据、预热、含 CUDA 同步）

`d_model=64`、块大小 128、并发 8、每请求 1 个 query：

| 历史长度 | ① 纯 kernel（元数据已在 GPU） | ② 含元数据构造与上传 | ③ step21 逐请求 torch |
|---:|---:|---:|---:|
| 512 | 31.3 μs | 141.9 μs | 7933.5 μs |
| 2048 | 42.8 μs | 155.4 μs | 25655.6 μs |
| 8192 | 151.6 μs | 260.2 μs | 104305.1 μs |

**同设备下 kernel 比逐请求 Python 路径快 250–690 倍**，这正是本关要解决的"请求串行"瓶颈。但**元数据构造与上传本身要约 100 μs**，与历史长度基本无关（它是按请求数的 Python 工作），在 512 这种小规模下占了②的七成。这是下一步该优化的地方，本关按需求"允许每轮上传这些小 Tensor"如实记录。

### 3.4 完整 Engine

8 请求 ×（prompt 500 + 生成 24），块大小 128，预算 64，三轮取最快：

| 后端 | 轮数 | 最快耗时 |
|---|---:|---:|
| torch | 88 | 566.2 ms |
| triton | 88 | **141.6 ms** |

端到端约 **4.0x**。注意这个倍数远小于纯 attention 的 250–690 倍——投影、KV 写入、lm_head、采样和 Python 调度都还在，attention 只是其中一段。**这是同机同数据的粗测，不替代验收方的三层正式测量。**

## 4. 接口变化与遗留

- 新增 `Engine(device=None, attention_backend="torch")`；旧调用不受影响。
- 新增 `TinyCausalLM(device=None, attention_backend="torch")`。
- 新增 `paged_attention(q, k_pool, v_pool, block_tables, seq_lens, token_to_req, query_pos) -> out[N,D]` 与 `_paged_attention_kernel`。
- `KVCachePool`、调度器、hash、LRU 未改；slot mapping 与 `append_batch` 仍是 step20 的 PyTorch 实现。
- 遗留：元数据每轮用 Python 循环重建并上传，小规模下开销占比高（见 3.3 的②）。
- 遗留：一个 program 只处理一个 query，没有做 query 分块或一个 program 处理多个 query；未使用 Tensor Core，分数是逐元素乘加再归约（`d_model` 很小时 `tl.dot` 也不划算）。
- 遗留：采样结果转成 Python token ID 仍有同步；未做 CUDA Graph。
- 未提交。
