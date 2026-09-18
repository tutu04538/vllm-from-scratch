# step25：多头 attention 与 GQA

- 对应代码：`step25.py`（新增，未提交）
- 源码 SHA256：`937ae3866d083b353dcd324c5f8d18ffa98a97b9f49dc56de64376555659d927`
- 基线：`step24.py`（SHA256 `f0129cb6…`）原样保留、未修改

## 0. 需求大概

前面几关把调度、分页 KV、prefix cache、Triton、CUDA Graph 都跑通了，但模型始终是单头：一个 token 只有一个 query 向量和一份 key/value。本关补上接入常见模型所需的结构——**多个 query head，并允许多个 query head 共享同一个 KV head**。

同一套参数表达三种情况：

| Q 头 / KV 头 | 名称 |
|---|---|
| 4 / 4 | MHA |
| 4 / 2 | GQA |
| 4 / 1 | MQA |

要求把 head 这一维从投影一路贯穿到 KV 缓存和 GPU kernel：投影输出维度、KV 池布局、Torch 参考、Triton kernel、CUDA Graph 都要一起改；仍保留单层、FP32、小词表和现有位置 embedding。

## 1. 改动内容

| 位置 | 改动 |
|---|---|
| `KVCachePool.__init__` | 参数由 `d_model` 改为 `num_kv_heads, head_dim`；池形状 `[num_blocks, block_size, num_kv_heads, head_dim]` |
| `KVCachePool.gather` | 空返回改为 `(0, num_kv_heads, head_dim)` |
| `TinyCausalLM.__init__` | 新增 `num_q_heads` / `num_kv_heads`；校验配置；Q 投影出 `num_q_heads*head_dim`，K/V 只投影出 `num_kv_heads*head_dim`；新增无 bias 的 `o_proj` |
| `TinyCausalLM._embeds_qkv` | 投影后 `view` 成 `[N, heads, head_dim]` |
| `TinyCausalLM._heads_to_logits` | 新增：各 head 按顺序拼回 `[N, d_model]` → `o_proj` → `lm_head` |
| `TinyCausalLM.block_attention` | 改成按 query head 循环，score 用 `1/sqrt(head_dim)`，只读对应 KV head |
| `_paged_attention_kernel` | grid 由 `(N,)` 改成 `(N, num_q_heads)`；按 `kv_head = q_head // GROUP_SIZE` 取 KV |
| `paged_attention` | 新增 `group_size` 参数；传 q/k/v/out 的 head 维 stride |
| `Engine.__init__` | 新增 `num_q_heads=1, num_kv_heads=1`；池用 `model.num_kv_heads` / `model.head_dim` 构造 |

## 2. 设计要点

### 2.1 形状链条

```text
输入 hidden        [N, 32]
q_proj → view      [N, 4, 8]        4 个 query head
k_proj → view      [N, 2, 8]        只有 2 个 KV head
v_proj → view      [N, 2, 8]
attention          [N, 4, 8]
reshape            [N, 32]          按 head 顺序拼接，不求平均
o_proj             [N, 32]
lm_head            [N, vocab]
```

三处容易漏：

1. **K/V 投影输出维度不再等于 `d_model`**，而是 `num_kv_heads * head_dim`。单头时两者恰好相等，所以默认配置下旧形状不变。
2. **score 除以 `sqrt(head_dim)`**，不是 `sqrt(d_model)`。单头时 `head_dim == d_model`，两者也恰好相等——这正是"默认配置下旧行为不变"的原因。
3. **`o_proj` 是新增的**，即使单头也有。所以拿旧权重跑单头对比时要把 `o_proj` 设成单位矩阵（`o_proj` 输入输出都是 `d_model`，形状对得上）。

### 2.2 分组是连续的，不是取余

```python
kv_head = q_head // group_size        # group_size = num_q_heads // num_kv_heads
```

4 个 Q 头、2 个 KV 头时：`[0, 0, 1, 1]`。这与 PyTorch 文档里按 head `repeat_interleave` 的对应关系一致，也要求 `num_q_heads % num_kv_heads == 0`。

**给同组的 Q 头复制 KV 权重做对照时，要按 head 块重复**（每 `head_dim` 行一组），不是按行重复——`repeat_interleave(weight, group, dim=0)` 是按行重复的，需要先 `view(num_kv_heads, head_dim, d_model)` 再重复。按行重复会让每个 KV head 只拿到半个头的权重，输出看起来"有结果但完全不对"。

### 2.3 KV 池一个 slot 装全部 KV head

```text
k_cache / v_cache:  [num_blocks, block_size, num_kv_heads, head_dim]
扁平视图:           [num_blocks * block_size, num_kv_heads, head_dim]
```

一个 slot 仍然是一个 token 的缓存位置，只是这一行现在装了该 token 的**全部 KV head**。所以：

- 块表、有效长度、引用计数、prefix cache 的 key 全都**按 token 算**，不用为每个 head 各来一套；
- `build_slot_mapping` 一行没改；
- 写入仍是每 token 一次 `index_copy_(0, slots, k)`，只是 `k` 的第 0 维还是 token 数，后面多了 head 维；
- GQA 不复制历史 KV——kernel 和 Torch 路径都是按 `kv_head` 索引到那**一份**，没有先扩成 4 份再算。

实测（块大小 4、8 块 = 32 个 slot、head_dim 8、FP32）：

| 配置 | 池形状 | K+V 字节 |
|---|---|---:|
| 4 / 4 | `(8, 4, 4, 8)` | 8192 |
| 4 / 2 | `(8, 4, 2, 8)` | **4096** |
| 4 / 1 | `(8, 4, 1, 8)` | **2048** |

与需求里的算例一致。

### 2.4 kernel 从"一个 query"变成"一个 query 的一个 head"

```python
grid = (N, num_q_heads)
row    = tl.program_id(0)
q_head = tl.program_id(1)
kv_head = q_head // GROUP_SIZE        # GROUP_SIZE 是 constexpr
```

每个 program 的 K/V 地址多了 `kv_head * stride_kh` 这一项，其余（块循环、三重 mask、online softmax）与上一关完全相同。**整批仍然只有一次 kernel 调用**——head 维是 grid 的第二维，不是 Python 循环。

`head_dim` 可能不是 2 的幂（比如 d_model=32、4 头时 head_dim=8 是；但 d_model=13 时不是），所以 `D_POW2` 的填充与 `offs_d < D` 屏蔽照旧保留。

### 2.5 Torch 参考路径按 head 循环

`block_attention` 外层遍历 `q_head`，取 `k_block[:, kv_head, :]`，用同一套 online softmax 分块累计，最后 `out[:, q_head, :] = u / z`。per-request、per-head 的直观循环，便于对照。

### 2.6 Graph 不受影响

同一个 Engine 内头数和维度固定，`gpu_forward` 的形状仍然只由 N 决定，所以还是按 N 缓存图。`o_proj` 也在图内（它紧接 attention 之后、`lm_head` 之前）。

## 3. 验证

conda `vllm-omni-dev`，RTX 5090 Laptop，torch 2.13.0+cu130，Triton 3.7.1，FP32。

### 3.1 配置校验

```text
d_model=8 不能被 num_q_heads=3 整除
num_q_heads=4 不能被 num_kv_heads=3 整除
头数必须为正，收到 num_q_heads=0、num_kv_heads=1
```

### 3.2 数值正确

- **与独立 CPU 参考逐 token 对照**：配置 1/1、4/4、4/2、4/1，各 3 组随机权重与提示，torch 与 triton 两个后端共 **48 条对照，不一致 0**。参考实现按 head 分别做完整历史的 causal attention，再拼接、`o_proj`、`lm_head`。
- **不同 head 的数值确实不同**：4 个 q head 在最后一个 token 上两两不同，2 个 kv head 也不同——避免"所有 head 一样"掩盖索引错误。
- **分组关系**：`q head -> kv head = [0,0,1,1]`。
- **单头旧行为**：把 `step24` 的权重载入 `step25` 并把 `o_proj` 设为单位矩阵，两个引擎的输出**完全一致**。
- **人为等价的 MHA 对照**：把 GQA(4/2) 的 K/V 投影按 head 块复制给 MHA(4/4) 的同组 Q 头，两者输出**完全一致**（此时 GQA 的 KV 只有 4096 字节，MHA 是 8192 字节）。
- **逐步 KV 与独立参考一致**：93 次核对（含 1/1、4/2、4/4），不符 0。

### 3.3 缓存与执行

- **42 组随机场景**：prefix cache 开/关的回调与结果一致；torch / triton / triton+graph **三个后端逐轮返回完全一致**。
- **写入边界**：20 组场景里把池填成已知值后跑一轮，变化的 slot 集合**恰好等于**本轮该写的 slot 集合，没有越界、没有覆盖本轮之外的历史。
- **索引不变量**：120 组场景逐步检查 `block_hash` 与 `block_to_hash` 互为双射，结束时引用计数全零。
- **Graph**：4/2、4/4、4/1 三种配置下 eager 与 graph 输出一致，每 N 一张图。

### 3.4 计时（粗测，同机同数据，三轮取最快）

完整 Engine：8 请求 × prompt 500 + 生成 24，`d_model=32`，块大小 128，预算 64：

| 配置 | torch | triton | triton + graph |
|---|---:|---:|---:|
| 1/1 | 316.6 ms | 111.3 ms | 75.6 ms |
| 4/4 | 1276.2 ms | 100.7 ms | 91.0 ms |
| 4/2 | 1316.6 ms | 105.9 ms | 107.8 ms |
| 4/1 | 1168.6 ms | 95.4 ms | 95.1 ms |

**不能从这张表直接读出"GQA 更快"**：不同 head 配置是不同的模型参数结构，不是同一个模型的两种跑法。这里只能看出 torch 参考路径按 head 循环后明显变慢（head 越多循环越多），而 triton 与 graph 的量级基本不变。

## 4. 接口变化与遗留

- `Engine(...)` 新增 `num_q_heads=1, num_kv_heads=1`；`TinyCausalLM(...)` 同样。
- `KVCachePool(block_size, num_kv_blocks, num_kv_heads, head_dim, device, enable_prefix_caching)`：第 3、4 个参数由原来的 `d_model` 拆成两个。
- `paged_attention(q, k_pool, v_pool, block_tables, seq_lens, token_to_req, query_pos, group_size)`：多了 `group_size`。
- `TinyCausalLM` 新增 `o_proj`，`state_dict` 因此多了 `o_proj.weight`；加载旧 checkpoint 需要补这一项。
- `gather()` 返回形状变为 `[length, num_kv_heads, head_dim]`。
- 遗留：仍是单层、无 RoPE / RMSNorm / FFN；`DummyModel` 未随 head 结构更新（它本来就没被 Engine 使用）。
- 遗留：`block_attention` 是 per-request × per-head 的双层 Python 循环，只作参考路径。
- 未提交。
