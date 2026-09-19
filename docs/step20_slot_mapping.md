# step20：slot mapping 与批量 KV 读写

- 对应代码：`step20/step20.py`（新增，未提交）
- 源码 SHA256：`9c6d204ba2e90efa2f55c77dddda121709f84a0ddc8270e116e7518a6075a248`
- 基线：`step19/step19.py`（SHA256 `2b09d735…`）原样保留、未修改

## 0. 需求大概

step19 已经把输入打包成 `[N]`，但 KV 存取还是逐 token 的 Python 循环：写入时每个新 token 各算一次块号和偏移、分别赋值 K/V；读取历史时每个位置取一行、塞进列表再 `stack`。请求越长，这些细碎索引越多。

本关一次做完三件关联的事：

- 算出 slot mapping——本轮打包输入第 i 个 token 的 K/V 该写到哪个物理槽位；
- K/V 各用**一次**批量索引写入，不再逐 token 赋值；
- 读取用**一次**索引选出整段历史，不再逐行 stack。

只改 KV 的地址与数据流，不重写分配器：Engine 接口、调度规则、token 预算、采样、回调顺序、prefix cache、共享引用、LRU、最大预算预留全部不变；模型仍收一维真实 token、一次 Q/K/V 投影、返回 `[N, vocab_size]`。

## 1. 改动内容

| 位置 | 改动 |
|---|---|
| `KVCachePool.__init__` | 新增 `k_flat` / `v_flat`：`[num_kv_blocks * block_size, d_model]`，由 `k_cache.view(-1, d_model)` 得到，与底层存储共享 |
| `KVCachePool._slots_of_range` | 新增：请求内逻辑位置 `[start, start+count)` → 物理槽位，用 Tensor 运算算一整段 |
| `KVCachePool.build_slot_mapping` | 新增：把各请求的槽位段按打包顺序拼成 `LongTensor[N]` |
| `KVCachePool.append_batch` | 新增：整批 `index_copy_` 写 K、写 V，写完再各请求 `length += count` |
| `KVCachePool.gather` | 改为 `index_select` 一次选出 K 和 V，按逻辑顺序返回 |
| `KVCachePool.append` | 删除（逐 token 版本，已无调用方） |
| `TinyCausalLM._forward_append` | 逐请求的写入循环换成一次 `append_batch(past_kv, num_scheduled_tokens, k, v)` |

## 2. 设计要点

### 2.1 slot 是第三种编号，和前两种都不是一回事

同一个 token 在这套代码里有三个编号，别混：

| 编号 | 含义 | 范围 |
|---|---|---|
| token ID | 词表里的第几个 | `0..vocab_size-1` |
| position ID | 模型眼里的第几个位置，决定位置 embedding | `0..max_seq_len-1`（超了 clamp） |
| **slot** | 池子里的第几行存储 | `0..num_kv_blocks*block_size-1` |

```text
slot = 物理块编号 * block_size + 块内偏移
```

所以 `position 5` 落在一块 `block_size=4` 的块里，是"这个请求的第 5 个位置"；它到底存在哪一行，要查块表才知道——同一位置在不同请求里可以是不同 slot，同一个 slot 在不同时刻也可以属于不同请求。

`[num_kv_blocks, block_size, d_model]` 的前两维本来就是一排连续的 token 槽位，`view(-1, d_model)` 只是换个形状看同一块存储，`data_ptr` 不变，不做任何复制。

### 2.2 地址必须用写入前的 length

```python
positions = torch.arange(start, start + count)              # start 是本轮写入前的 length
blocks = torch.tensor(block_table)
slots = blocks[positions // block_size] * block_size + positions % block_size
```

本轮写入的是 `[length, length+count)`，块号和偏移都由**旧** length 推。写完再统一 `cache.length += count`。顺序反了就会整体偏移一格。

需求 §3 的夹具正好卡在块边界上：`A.block_table=[5,2], length=5, count=3` 从块 2 的偏移 1 开始写，`B.block_table=[1,7], length=1, count=2` 从块 1 的偏移 1 开始，得到 `[9,10,11,5,6]`。

### 2.3 一次写、一次读，长度更新留在批量接口里

```python
slot_mapping = self.build_slot_mapping(caches, counts)
self.k_flat.index_copy_(0, slot_mapping, new_k)
self.v_flat.index_copy_(0, slot_mapping, new_v)
for cache, count in zip(caches, counts):
    cache.length += count
```

- `index_copy_` 是原位写入，只有目标 slot 变，旧历史、其他请求、共享前缀块、预留未用区域都不动。
- 有效请求的写入 slot 互不重复（每个请求写自己的私有区域），所以不依赖重复索引的覆盖顺序。
- 长度更新放在这里而不是调用方，是为了让"写入"和"长度前进"绑在一起——分开写容易出现写完忘了加长度、或者加了长度又拿新长度去算地址。

### 2.4 读取的顺序由块表决定，不由物理位置决定

```python
slots = self._slots_of_range(cache.block_table, 0, cache.length)
return self.k_flat.index_select(0, slots), self.v_flat.index_select(0, slots)   # 新 Tensor
```

`slots` 是按请求逻辑位置 `0..length-1` 生成的，所以块表乱序（比如 `[7,1]`）也能按逻辑顺序返回。`index_select` 返回新 Tensor，**仍然存在 gather 复制**——这一关不是"零复制 attention"。

length 为 0 时返回 `[0, d_model]` 空 Tensor，dtype/device 从池上取（`k_cache.new_empty`），不手写 `torch.empty` 以免默认 dtype 不一致。

### 2.5 允许保留的循环

按需求 §5：遍历请求、给每个请求构造位置范围和块表 Tensor、拼接索引、更新 length，都允许用 Python 循环。被消掉的是**请求内部逐 token** 的那一层——算地址、收集 slot、逐行读写，现在都用 Tensor 运算表达。

attention 仍逐请求计算，各请求自己的 `gather` 也仍逐请求调用。

### 2.6 地址构造本身有开销，这是本关新引入的成本

对"每请求 1 个 token"的 decode 场景，`build_slot_mapping` 要为每个请求建一个块表 Tensor、算几个下标、再 `torch.cat`，这些固定开销摊不开。实测（见 3.5）写入路径比逐 token 版本**慢约 2.5 倍**，而读取路径快 15 倍以上。净效果取决于"历史长度 vs 并发数"，需求也明确提醒过"少写 Python 循环不保证小模型一定更快"。

## 3. 验证

conda `vllm-omni-dev`；`TinyCausalLM` 落在 cuda:0，float32。

### 3.1 需求 §3 的地址例子

```
A.block_table=[5,2] length=5 count=3
B.block_table=[1,7] length=1 count=2
slot_mapping = [9, 10, 11, 5, 6]                  与需求一致
写完后 length: A=8  B=3                            与需求一致
每个 slot 的前后内容逐项核对                          5/5 正确
```

### 3.2 存储与边界

- **共享存储**：`k_flat.data_ptr() == k_cache.data_ptr()`、`v_flat` 同理，`k_cache.shape=(16,4,8)` → `k_flat.shape=(64,8)`。
- **原位写入**：先把整个池填成已知序列，写入 2 个 slot 后逐行比对，K/V 各只有目标那 2 行变了。
- **读取顺序**：`block_table=[7,1]`、length=6，先往物理块 7、1 写已知值，`gather` 返回的 6 行与逻辑顺序逐值一致。
- **空读取**：length=0 返回 `(0, 8)`，dtype/device 与池一致。
- **块大小 1/3/4**：`length` 落在块中间、一次追加跨过块边界，生成的 slot 与手算一致（如 `block_size=4, length=3, count=6 → [23, 8, 9, 10, 11, 28]`）。

### 3.3 回归

- **step19 vs step20 等价**：300 组随机场景（并发 1–3、预算 1/2/4/8、块大小 1/2/4、池 2–8 块、缓存在 1/3 场景关闭），逐轮返回与回调顺序**不一致 0 例**；投影位置数与模型调用次数两者完全相同（2073 / 1127）。
- **KV 与独立参考一致**：353 次逐步核对，**不符 0**。
- **每个生成 token 都是参考的 argmax**：193 个，**不符 0**。
- **prefix cache 回归**：条目内容不变量（120 组，不符 0）、索引双射（300 组 / 60027 次 step）、开关输出对照（300 组，不一致 0）在 step20 上全部通过——共享前缀没有被覆盖。

### 3.4 计时口径

块大小 4、`d_model=64`、池 256 块、历史长度 128、CPU 单线程、取多次平均，**只测 KV 读写本身，不含模型**。

### 3.5 粗略计时：写入变慢，读取变快

每轮一次写入、各请求 1 个 token（decode 场景）：

| 并发 | step19 逐请求逐 token | step20 一次整批 |
|---:|---:|---:|
| 1 | 5.91 μs | 17.06 μs |
| 2 | 12.82 μs | 30.79 μs |
| 4 | 22.66 μs | 57.37 μs |
| 8 | 45.89 μs | 114.99 μs |
| 16 | 89.79 μs | 220.56 μs |

读取历史（每请求 128 个位置）：

| 并发 | step19 逐 token | step20 index_select |
|---:|---:|---:|
| 1 | 280.93 μs | 18.43 μs |
| 4 | 1149.42 μs | 80.17 μs |
| 16 | 4565.69 μs | 312.43 μs |

写入路径的差距来自 `build_slot_mapping` 的固定开销（每个请求建块表 Tensor、`arange`、`cat`）摊不开；读取路径省掉的是每位置一次 Python 级索引，所以历史越长、并发越高差距越大。**这不是性能结论**，完整 Engine 的对比由验收方按相同负载测。

## 4. 接口变化与遗留

- 新增 `KVCachePool.build_slot_mapping(caches, counts) -> LongTensor[N]`、`KVCachePool.append_batch(caches, counts, new_k, new_v)`、`KVCachePool.k_flat` / `v_flat`。
- `KVCachePool.gather(cache)` 签名不变，返回仍是新 Tensor。
- `KVCachePool.append(cache, new_k, new_v)` **已删除**（逐 token 版本，唯一调用方已改用 `append_batch`）。旧接口名不同、语义不同，不能混用。
- `_forward_append` 的签名与返回值未变。
- 遗留：`build_slot_mapping` 每个请求都要 `torch.tensor(block_table)` 转换一次整张块表；块表长、写入少时这部分开销占比很高。若以后要优化，可以考虑把块表本身存成 Tensor 或按需要的块段转换，但本关按需求"先用 PyTorch 把地址和数据流做对"未做。
- 遗留：`gather` 仍返回新 Tensor，attention 仍是逐请求 Python 循环，不是 PagedAttention kernel。
- 未提交。
