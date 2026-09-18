# step24：把 GPU forward 接入 CUDA Graph

- 对应代码：`step24.py`（新增，未提交）
- 源码 SHA256：`f0129cb66b2b4ff067d60ccc1fc22b6b1da64084bd66652f43f36c48f490d74c`
- 基线：`step23.py`（SHA256 `d609f866…`）原样保留、未修改

## 0. 需求大概

step23 合并了 attention 元数据上传，但一次 `[1,3,1]` 的混合 step 仍有 **37 个 GPU kernel 事件**：attention 只有一个，其余来自位置/slot 准备、embedding、投影、KV 写入、采样等。不是要把它们融成一个 kernel，而是说：attention 融合后，每轮仍要由主机逐个提交不少 GPU 操作。

本关把 **embedding → QKV → KV 写入 → Triton attention → lm_head** 这一段接进 CUDA Graph：capture 一次，以后更新输入内容再 replay。调度、数据准备、采样仍在图外。

## 1. 改动内容

| 位置 | 改动 |
|---|---|
| `TinyCausalLM._prepare_inputs` | 新增：图外准备——算 position/slot（用旧 length）、推进 Python 状态、填固定输入缓冲、填并上传 attention 元数据 |
| `TinyCausalLM._embeds_qkv` / `_write_kv` | 新增：从固定缓冲投影、按 slot 原位写 KV（纯 GPU，不动 Python 状态） |
| `TinyCausalLM.gpu_forward` | 新增：**纯 GPU 段**，图内执行的那一段 |
| `TinyCausalLM._torch_forward` | 新增：参考路径，与 `gpu_forward` 共用准备和 KV 写入，只换 attention 算法 |
| `TinyCausalLM._capture_graph` | 新增：side stream 预热 → `torch.cuda.graph` 捕获，按 N 缓存 |
| `TinyCausalLM._forward_append` | 拆成 准备 →（eager | graph.replay），不再自己更新 length |
| `TinyCausalLM.__init__` | 新增 `max_num_query_tokens`、`use_cuda_graph`；分配三个固定输入缓冲 |
| `Engine.__init__` | 新增 `use_cuda_graph=False`；校验只支持 CUDA + Triton |
| `Engine.step` | `input_ids` 改为在 CPU 组装，由图外准备一次写进固定 GPU 缓冲 |

## 2. 设计要点

### 2.1 三段职责切开

```text
图外 准备：position_ids / slot_mapping（用旧 length）
          每条请求 cache.length += count     ← Python 状态只在这里前进
          填 input/position/slot 固定缓冲
          attention 元数据 fill + 一次上传

图内 计算：embedding + 位置 → Q/K/V 投影
          按 slot_mapping 原位写 K/V
          Triton paged_attention
          lm_head

图外 消费：采样、更新 output_ids、回调、释放
```

关键是把 **`append_batch()` 拆开**：它原来同时写 KV、更新 Python length，整个包进 capture 会让 warmup/capture/replay 各加一次长度。现在：

- 地址计算留在图外（`build_slot_mapping`，用旧 length）；
- 写入换成纯 GPU 的 `k_flat.index_copy_(0, slot_buffer[:N], k)`；
- **长度更新搬到准备阶段，一轮只加一次**。

`Engine.step` 里的采样、`.item()`、回调、调度全在图外 ✓。

### 2.2 一 N 一图，内容变化由图外缓冲表达

第一版按本轮真实 query 数 `N` 缓存图，不做 padding bucket：

```python
graph = self.graphs.get(num_tokens)
if graph is None:
    graph = self._capture_graph(num_tokens, kv_cache_pool)
graph.replay()
return self.graph_outputs[num_tokens]
```

请求数、历史长度、块表的变化，全部通过**图外缓冲的内容**表达，不在捕获时固化进 Python 分支：

- `input_buffer` / `position_buffer` / `slot_buffer`：固定容量 `max_num_batched_tokens`，每轮 `copy_` 前 N 行；
- attention 元数据：step23 就已经是固定容量的整段视图，天然满足。

所以同一个 `graph[5]` 可以服务"三个请求 counts=[1,3,1]"和"两个请求 counts=[2,3]"两种分组。

### 2.3 预热写的是本轮真实 KV，不是假 KV

```python
warmup_stream = torch.cuda.Stream()
warmup_stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(warmup_stream):
    for _ in range(3):
        self.gpu_forward(num_tokens, kv_cache_pool)
torch.cuda.current_stream().wait_stream(warmup_stream)

graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    output = self.gpu_forward(num_tokens, kv_cache_pool)
```

`gpu_forward` 只读固定缓冲、只往**本轮该写的 slot** 写本轮的真实 KV，所以反复执行是幂等的：不产生假 KV，不覆盖别处，也不碰共享前缀。`cache.length` 已经在准备阶段更新完，预热/捕获/重放都不会再动它。

侧流的等待关系按官方示例写：预热前 `wait_stream(current)`，预热后 `current.wait_stream(side)`。

### 2.4 图输出的存储会被复用

`graph_outputs[N]` 的内存由图的内存池持有，每次 replay 覆盖同一块。所以采样必须**消费完本轮结果再进下一轮**——`Engine.step` 里采样紧跟 forward ✓。外部若要长期保存旧 logits，得自己 clone。

### 2.5 保留但没有调用方的 `append_batch`

`KVCachePool.append_batch()` 现在没有调用方了（triton 和 torch 两条路径都走 `build_slot_mapping` + `_write_kv`）。**保留它是为了不打断既有回归**——验收方会重跑前几关的缓存测试，那些测试会用到它。它已不在热路径上。

## 3. 验证

conda `vllm-omni-dev`，RTX 5090 Laptop，torch 2.13.0+cu130，Triton 3.7.1，FP32。

### 3.1 开关与错误

- `Engine(...)` 默认 `use_cuda_graph=False`，旧调用不受影响。
- `use_cuda_graph=True` + CPU → `ValueError`；+ `attention_backend="torch"` → `ValueError`（信息里带当前 device 与 backend）。

### 3.2 数值一致（同权重 `state_dict`，graph vs eager）

- **逐轮返回与回调顺序**：95 组随机场景（并发 1–3、预算 2/4/8、块大小 2/4、池 4–8 块、max_seq_len 16/32、缓存在 2/3 场景开启），**不一致 0 例**。
- **池内容**：跑完同一批请求后，两个引擎的 `k_flat` / `v_flat` **逐值相等**（`torch.equal`）。
- **logits**：同一份缓冲与元数据下，eager 的纯 GPU 段与 graph replay 的输出最大差 **0.000e+00**。

### 3.3 捕获与重放

- **每个 N 只捕获一次**：一次多轮负载里出现过 N=2/3/6/8，捕获序列正好是这四个值各一次，没有重复。
- **同一 N 不同分组复用同一张图**：第一轮 N=5 分组 `[('R1',2),('R2',3)]`，第二轮 N=5 分组 `[('R1',1),('R2',1),('R3',3)]`——N=5 只捕获一次，两轮结果与 eager 完全一致。
- **确实在 replay**：给 `CUDAGraph.replay` 装计数桩，多轮负载里计数随轮次增长。
- **空轮次不建图、不 replay**：没有请求时、只有零预算请求时，`len(model.graphs) == 0`。

### 3.4 状态只前进一次

第一次 step 同时包含 3 次预热 + 1 次捕获，之后各请求 `cache.length` 为 `{A: 6, B: 2, C: 0}`——正好等于本轮各自被调度的 token 数（6 / 2 / 0），没有被预热放大成 4 倍。长度翻倍会让后续所有输出错乱，随机对照 0 例不一致是更强的佐证。

### 3.5 固定地址

`input_buffer` / `position_buffer` / `slot_buffer` / attention 元数据主缓冲的 `data_ptr` 在多轮负载前后完全一致。

### 3.6 图中是多个 kernel，不是融合

Profiler 捕获一次 replay：

```text
37 个 GPU kernel 事件，16 种
  6 x elementwise_kernel_with_index
  4 x gemmSN_TN_kernel
  3 x vectorized_elementwise_kernel（多种）
  3 x index_elementwise_kernel
  2 x CatArrayBatchedCopy
  2 x indexSelectSmallIndex
  1 x _paged_attention_kernel        ← attention 仍然只有一个
  1 x softmax_warp_forward
  1 x reduce_kernel
  ...
```

37 这个数字与需求里记录的观测吻合。Graph 减少的是**主机提交开销**，不是把 kernel 融合成一个。

### 3.7 既有回归

prefix cache 三套（条目内容不变量 120 组、索引双射 300 组 / 60027 次 step、开关输出对照 300 组）在 step24 上全部通过。

### 3.8 计时（粗测，同机同数据，三轮取最快）

完整 Engine：8 请求 × prompt 500 + 生成 24，块大小 128，预算 64：

| | 轮数 | 最快 |
|---|---:|---:|
| step23 eager | 88 | 110.0 ms |
| step24 eager | 88 | 98.9 ms |
| step24 **graph** | 88 | **71.1 ms** |

graph 相对同文件 eager 约 **1.39x**。

首次捕获成本：

| | 耗时 |
|---|---:|
| 第一次 step（含 3 次预热 + 捕获 N=64） | 4.94 ms |
| 第二次 step（已缓存，直接 replay） | 0.35 ms |

捕获开销约 **4.59 ms**，需要摊到后续所有轮次上。**这是同机粗测，不替代验收方的分层测量**；绝对数值会随机器状态波动，倍数应在同一批进程内比较。

## 4. 接口变化与遗留

- 新增 `Engine(use_cuda_graph=False)`；只支持 CUDA + Triton，其他组合明确报错。
- 新增 `TinyCausalLM(max_num_query_tokens=None, use_cuda_graph=False)`、`gpu_forward(num_tokens, pool)`、`_prepare_inputs(...)`、`_capture_graph(...)`，以及 `input_buffer` / `position_buffer` / `slot_buffer` / `graphs` / `graph_outputs`。
- `_forward_append` 签名与返回值未变；`Engine.step` 的对外行为未变（`input_ids` 改为在 CPU 组装）。
- `KVCachePool.append_batch` 保留但已无调用方（见 2.5）。
- 遗留：一 N 一图，没有 padding bucket；N 的种类多时会累积多张图（每张都要单独预热+捕获）。
- 遗留：图只覆盖 GPU forward，`input_ids` 组装、元数据填充、采样、`.item()` 同步仍在图外。
- 遗留：未用 pinned memory、未做异步 H2D；未加多流、未加 `torch.compile`。
- 未提交。

---

# 附：`_capture_graph` 逐行解释与 CUDA Graph 原理

> 这一节是补充说明，不属于每篇固定的五节结构。里面的每条行为都在本机实测过
> （RTX 5090 Laptop / WSL2 / torch 2.13.0+cu130 / Triton 3.7.1）。

## 5. `_capture_graph` 里每个 API 是什么

```python
def _capture_graph(self, num_tokens, kv_cache_pool):
    warmup_stream = torch.cuda.Stream()                                  # ①
    warmup_stream.wait_stream(torch.cuda.current_stream())               # ②
    with torch.cuda.stream(warmup_stream):                               # ③
        for _ in range(3):
            self.gpu_forward(num_tokens, kv_cache_pool)                  # ④
    torch.cuda.current_stream().wait_stream(warmup_stream)               # ⑤

    graph = torch.cuda.CUDAGraph()                                       # ⑥
    with torch.cuda.graph(graph):                                        # ⑦
        output = self.gpu_forward(num_tokens, kv_cache_pool)             # ⑧

    self.graphs[num_tokens] = graph                                      # ⑨
    self.graph_outputs[num_tokens] = output
    return graph
```

### 先建立两个概念

- **stream（流）**：一条 GPU 工作队列。提交到同一条流的操作按顺序执行；不同流之间默认互不等待。PyTorch 有"当前流"的概念（默认流或线程自己设的流），绝大多数算子都提交到当前流。
- **提交（launch）**：每个算子都要主机调用一次驱动接口把工作放进队列。**这一步是主机开销**。小 kernel 时，提交开销可能比 kernel 本身还大——这正是 CUDA Graph 要省的东西。

### ① `torch.cuda.Stream()`

新建一条 CUDA 流。它只是一个队列句柄，本身不执行任何东西，也不花 GPU 时间。

### ② `warmup_stream.wait_stream(torch.cuda.current_stream())`

让 `warmup_stream` **等**当前流上"此刻已经排队的全部工作"做完。

注意这是**设备侧依赖**（内部是一个 event 记录 + 等待），不是主机阻塞——主机不会卡在这里，只是排了一个"等待"进队列。

为什么要这一句：预热不是凭空开始的，它要接在当前流已有的工作之后，不能越过它们先跑。

### ③ `with torch.cuda.stream(warmup_stream):`

上下文管理器：块内把**当前流**临时切成 `warmup_stream`，出块后自动还原。所以块里的 `gpu_forward` 都提交到预热流上。

### ④ `for _ in range(3): self.gpu_forward(...)`

预热。为什么必须做，见第 7 节——**实测冷捕获完整 forward 会直接失败**。

预热跑的是同一个 `gpu_forward`，只读固定缓冲、只往本轮该写的 slot 写本轮的真实 KV，所以重复执行是幂等的。

### ⑤ `torch.cuda.current_stream().wait_stream(warmup_stream)`

反向等待：让当前流等预热流做完。这样后面的捕获能看到预热的结果（内存分配、cuBLAS 句柄、Triton 编译结果都已就绪）。

②和⑤是一对：**先用②让预热接在已有工作后面，再用⑤让后续工作接在预热后面。**

### ⑥ `torch.cuda.CUDAGraph()`

创建一个图对象（主机侧句柄）。此时还没有任何 GPU 工作。它承担两件事：`capture_begin/end` 记录图，`replay()` 重放图。

### ⑦ `with torch.cuda.graph(graph):`

**捕获上下文**。查过本机 `torch/cuda/graphs.py` 的源码，它进入时做四件事：

```python
torch.cuda.synchronize()        # 等 GPU 全部空闲
torch.cuda.empty_cache()        # 把缓存分配器里的空闲块还给驱动，好让图内存池能用
self.stream_ctx.__enter__()     # 切到它自己的内部侧流（类级共享，不是当前流）
self.cuda_graph.capture_begin(...)   # 开始记录
```

退出时 `capture_end()` 结束记录并实例化图。

两个要点：

1. **捕获在它自己的侧流上进行**，不是当前流；
2. 块里的 CUDA 操作**被记录，但不执行**（"捕获成功"≠"算了一遍"）。
   所以第一次 replay 才是这一轮真正第一次算。

### ⑧ `output = self.gpu_forward(...)`

被记录的调用。块内的算子不会真的执行，但 Python 照常跑——`output` 是一个**占位张量**，它的内存在图的私有内存池里，每次 replay 覆盖同一块。

### ⑨ 保存图的引用

`graph` 和 `output` 都必须长期持有：图要留着 replay，`output` 的存储由图管理，被回收会出问题。

## 6. CUDA Graph 的原理

### 6.1 eager 执行的问题

普通执行时，主机逐个提交算子：

```text
主机: 提交 kernel A -> 提交 kernel B -> 提交 kernel C -> ...
GPU :       执行 A           执行 B           执行 C
```

GPU 很快，主机提交慢的时候 GPU 会饿着。我们这个小模型一次 forward 有 37 个 kernel，每个都不大，**提交开销占比很高**。

### 6.2 Graph 的三个阶段

```text
① 捕获  cudaStreamBeginCapture / cudaStreamEndCapture
        -> cudaGraph_t：有向无环图，节点是 kernel / memcpy / memset，边是依赖关系

② 实例化 cudaGraphInstantiate
        -> cudaGraphExec_t：驱动把图编译成一份可直接调度的方案

③ 重放  cudaGraphLaunch
        -> 一次调用把整张图交给 GPU
```

`graph.replay()` 就是第③步。

### 6.3 关键：记录的是"操作 + 地址"，不是数据

捕获记录的是**指针**（kernel 参数里的地址）和**依赖**。数据内容完全不进图。

所以：

- **重放时读到的是那个地址上的当前内容**——这就是"改内容不改地址，replay 就得到新结果"的原因；
- **地址变了就出事**——如果把变量重新赋值成新张量（新地址），图里记的还是旧地址，会读到旧数据，甚至已被释放的内存。

一句话：**图固化的是"怎么算"和"在哪算"，不固化"算什么"。**

### 6.4 它省什么、不省什么

| | |
|---|---|
| 省 | 每轮逐个提交算子的主机开销 |
| 不省 | kernel 的执行时间；**也不自动融合 kernel** |
| 不自动获得 | 并行性——依赖关系还是捕获时那一套 |

需求 §7 也点了这一点：图中依然是 37 个 kernel，不是 1 个。profiler 看到的 kernel 事件一个不少，少的是提交次数。

## 7. 编写要点

### 7.1 逐条核对你的三点理解

**"图内计算要全部是 GPU 内的计算"** —— 基本对，但要说准：图里的节点只能是 **CUDA 能记录的操作**：kernel、memcpy、memset、event 记录/等待。不是"必须都是自定义 kernel"，只要有对应的 CUDA 调用就行。

**"不能涉及 CPU 到 GPU 的数据搬运"** —— 对，而且本机实测比文档更严：

| 捕获块里的操作 | 结果 |
|---|---|
| GPU 上的逐元素运算 / kernel | 成功 |
| `torch.empty(...)`（图内分配） | 成功（走图内存池） |
| `x.item()` | 失败 `AcceleratorError`（同步操作） |
| `torch.cuda.synchronize()` | 失败 `AcceleratorError` |
| 未 pinned 的 CPU→GPU `copy_` | 失败：`Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the CPU tensor is pinned` |
| GPU→CPU `copy_` | 同上 |
| `torch.tensor(..., device="cuda")` | 同上（内部就是 H2D） |
| **pinned 的 CPU→GPU `copy_`** | **本机仍然失败** `CUDA error: operation failed due to a previous error during capture` |
| 纯 CPU 张量操作 / `print` | "成功"，但**不会进图**（见 7.4） |

PyTorch 的报错信息说 pinned 内存的 H2D 是允许的，但在这台机器（WSL2 + CUDA 13）上实测仍然失败。所以本项目的做法是**图内零 CPU↔GPU 拷贝**：输入在准备阶段就写进固定 GPU 缓冲，图只读固定地址。

**"从 GPU 中获取数据时需要使用固定的地址"** —— 方向对，范围要扩大：**图内读写的所有张量都必须是固定地址**，不只是"取数据"的时候。包括：

- 输入（`input_buffer` / `position_buffer` / `slot_buffer`）
- attention 元数据缓冲
- 模型权重（`nn.Parameter` 本身的地址，所以**不能中途给 model 换一份 state_dict**）
- KV 池（`k_cache` / `v_flat`）
- 输出（`graph_outputs[N]`）

而 `.item()` / `.cpu()` 这类**回读**属于同步操作，捕获时就报错，只能在图外做——这也就是本关把采样、回调、调度都放在图外的原因。

### 7.2 捕获前必须预热

实测：拿完整 `gpu_forward` 直接捕获

```text
预热=False: AcceleratorError: CUDA error: operation failed due to a previous error during capture
预热=True : 捕获并 replay 成功，logits 形状 (4, 64)
```

原因是首次执行会做一堆**懒初始化**：cuBLAS 句柄创建与算法选择、内存分配器扩容、Triton 首次编译并加载模块。这些主机侧动作混进捕获就会失败。先跑几次，让这些一次性开销在图外发生掉。

（单独捕获 `paged_attention` 时冷捕获也能过，因为那一段不含 cuBLAS；但完整 forward 不行——**不要靠"某次侥幸能过"来省掉预热**。）

### 7.3 形状固定：一张图只服务一种形状

图记录的是 kernel 的 grid 和参数，形状变了图就不适用。本关的做法是**按真实 N 缓存不同的图**（`graph[5]`、`graph[1]`…）。请求数、历史长度、块表的变化由图外缓冲的**内容**表达，不改形状。

### 7.4 图内分配的存储会被复用，且 CPU 代码不会被重放

实测两件事：

```text
纯 CPU 操作：捕获成功，Python 计数在 capture 后是 4；5 次 replay 后仍然是 4
             -> 主机侧代码只在捕获时执行了一遍，replay 完全不碰主机

图内分配的输出：第一次 replay 后快照 [2,4,6,8]
                换数据再 replay 后，static_out 变成 [18,18,18,18]
             -> 同一块存储被覆盖，想留旧结果必须自己 clone
```

"捕获成功"不等于"进了图"。**判断某段代码是否真的在图里，不能只看有没有报错，要看 replay 时它有没有重新生效。**

### 7.5 其余几条

- **保存长期引用**：`graph`、图内用到的输入/输出张量都要持有；图不会阻止 Python 回收张量。
- **捕获是进程级串行的**：同一时刻只能有一处捕获；`capture_error_mode="global"`（默认）下，其他线程做 CUDA 调用也可能让捕获失败。多线程服务要额外小心，本关不涉及。
- **`replay()` 提交到当前流**：重放只是把图排进当前流，不会同步。要读结果（比如 `.item()`）时照常需要同步。
- **不要为了省事把整个 `Engine.step()` 塞进图**：调度、`.item()`、回调、CPU↔GPU 拷贝都在里面，捕获必然失败。本关只圈住纯 GPU 的那一段。
- **多张图可以共享内存池**：`torch.cuda.graph(g, pool=...)`，本关未用。

### 7.6 为什么预热要放在另一条流上

先看实测（同一份 `gpu_forward`，四种预热方式，各自独立进程）：

| 预热方式 | 用 `torch.cuda.graph` 捕获 |
|---|---|
| 侧流 + 两条 `wait_stream`（本关代码的写法） | 成功 |
| **直接在当前流上预热** | **也成功** |
| 侧流，但不加 `current.wait_stream(side)` | 也成功 |
| 侧流，但不加 `side.wait_stream(current)` | 也成功 |
| 完全不预热 | **失败** |

所以"预热必须放侧流"这句话，在 `torch.cuda.graph` 这个高层 API 下**并不是捕获成功的必要条件**。原因是这个上下文管理器进入时会先做一次全设备同步（源码见 5.⑦）：

```python
def __enter__(self):
    torch.cuda.synchronize()        # ← 全设备同步，把流顺序问题兜掉了
    torch.cuda.empty_cache()
    self.stream_ctx.__enter__()     # 切到它自己的内部侧流
    self.cuda_graph.capture_begin(...)
```

预热次数也实测过：**0 次失败，1 次就够**，代码里写 3 次是余量（官方示例也用 3）。

**那为什么还要写成侧流？四条理由：**

**1. 捕获本身就必须在非默认流上进行。** 手动捕获时一试就知道：

```python
g.capture_begin()          # 此时当前流是默认流
# RuntimeError: CUDA graphs must be captured on a non-default stream.
```

`torch.cuda.graph` 内部有一条**类级共享的侧流**（源码里的 `default_capture_stream`），替你满足了这条要求。所以"用侧流"是捕获的硬性要求，只不过位置从"你的代码"挪到了"库内部"。

**2. 你没法把预热放进捕获流。** 捕获流在库内部，你拿不到它。既然预热和捕获必然在两条不同的流上，"预热先于捕获"就必须显式声明——官方那两条 `wait_stream` 就是这个声明。不写它们，正确性就全押在第 1 行那次 `synchronize()` 上。

**3. 不要依赖那次同步。** 它在源码里看得见，但属于实现细节，不是契约。一旦你改成手动 `capture_begin/capture_end`，或者换一个库版本，兜底就没有了；显式侧流写法在这几种情况下都成立。

**4. 不占当前流。** 真实推理里当前流上还排着别的 kernel（准备、采样、其他层）。预热放侧流，它的顺序是"接在当前流已有工作之后"（`side.wait_stream(current)`），不会插进当前流已有工作的中间。另外传统默认流与其他流之间有隐式同步，行为跟普通流不一样；预热走普通侧流，与捕获流（也是普通流）语义一致。

**一句话**：侧流不是"让捕获能成功"的开关（那次全设备同步才是），而是把"预热先于捕获"这条依赖**显式写出来**，不去依赖库内部的实现细节。真正不能省的是**预热本身**——不预热，完整 `gpu_forward` 直接捕获必然失败（7.2 的实测）。

### 7.7 `torch.cuda.synchronize()` 是什么

一句话：**阻塞主机，直到当前设备上所有流的工作全部做完。**

**前提是 CUDA 调用是异步的。** 主机调用 kernel 只是把工作放进队列，然后立刻返回，GPU 稍后才执行。实测（4096³ 的 fp32 矩阵乘，预热后）：

```text
单次:   入队用时     0.62 ms
        实际执行时间 5.53 ms      ← 主机早就往下走了

连续 20 次: 20 次入队总共  0.50 ms（每次 0.025 ms）
            全部算完       123.98 ms
```

主机用 0.5 毫秒把 20 个矩阵乘排进队列就走人了，GPU 要 124 毫秒才算完。`synchronize()` 就是"停下来等 GPU 追上"的那个动作。

**三个层级的等待**：

| API | 等什么 |
|---|---|
| `torch.cuda.synchronize()` | **当前设备上所有流**的工作；可传 `device` 指定 |
| `torch.cuda.current_stream().synchronize()` | 只等**当前流** |
| `stream.synchronize()` / `event.synchronize()` | 只等某条流 / 某个事件 |

实测三者的差别（往侧流 `s` 提交一个大矩阵乘之后）：

```text
current_stream().synchronize():  0.582 ms   ← 当前流没活，立刻返回
s.synchronize():                 5.224 ms   ← 等到侧流算完
torch.cuda.synchronize():        0.017 ms   ← 此刻全设备已空闲
```

注意第一行：**"等当前流"根本不会等侧流上的工作**。所以"同步"这个词要看清等的是谁——这也正是 7.6 里那两条 `wait_stream` 存在的意义。

**什么时候必须用**：

1. 要在主机上读 GPU 的值。`.item()` / `.cpu()` / `print(tensor)` 都会**隐式同步**，因为要把数据拷回主机。
2. 计时。不同步测到的是"入队时间"而不是"执行时间"——上面 20 次矩阵乘的例子，不同步会测出 0.5 ms，真实是 124 ms。
3. 释放或复用显存之前。
4. **CUDA Graph 捕获之前**：`torch.cuda.graph.__enter__` 第一行就是它，为的是让设备在开始记录前彻底空闲（这就是 7.6 里"预热放哪条流都能过"的原因）。

**为什么捕获期间绝对不能调用**：捕获只记录操作、不执行，同步等于去等一件根本不会发生的完成事件。

```text
torch.cuda.synchronize() 在捕获块内
  -> AcceleratorError: CUDA error: operation failed due to a previous error during capture
```

同理 `.item()` 和 CPU↔GPU 拷贝在捕获块里也都会失败——它们本质上都要求"等工作真的做完"。

**代价**：每次同步都把异步流水线拍平一次（主机停下、GPU 排空、再重新灌满），频繁调用会打掉吞吐。

所以这个项目里同步只出现在**必须把结果拿回主机**的地方——采样后把 token ID 变成 Python 整数（`output_ids.append(output_id.item())`），这也是需求里"采样结果转成 Python token ID 的同步暂时允许"的原因。其余能不同步的（整个 GPU forward）都留在 GPU 上跑完为止。
