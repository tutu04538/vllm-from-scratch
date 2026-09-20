# step33：把 RMSNorm 融合成一个 Triton kernel

- 对应代码：`step33/`（新增，未提交）
- 包摘要 SHA256：`5b9f90e53990068443cda38bebfa96461b19510faf290474837f6903691495c7`
  （`sha256("\n".join(sorted("路径 sha256(文件)")) + "\n")`，只算 `step33/**/*.py`）
- 基线：`step32/` 原样保留、未修改

## 0. 需求大概

上一关用了 BF16，但 `RMSNorm.forward()` 还是这样：

```text
转 FP32 → 平方 → 求均值 → 加 eps → rsqrt → 乘输入 → 乘 weight → 转回 BF16
```

数学上是一件事，执行时拆成了多次 GPU 运算，每一步还生成一个中间 Tensor。实测单次 eager RMSNorm：**FP32 是 6 个 kernel，BF16 是 9 个**；真实 Qwen3 一个 forward 有 `28 × 4 + 1 = 113` 次 norm 调用。

**需求：不改变归一化语义，把这串操作合进一个 Triton kernel，并接进真实模型。**

## 1. 改动内容

| 文件 | 内容 |
|---|---|
| `step33/norm.py` | **新增**：`_rms_norm_kernel`（一个 program 一行）+ 包装函数 `rms_norm(x, weight, eps)` |
| `step33/model.py` | `RMSNorm(size, eps, backend="torch")`，forward 按后端分派；`DecoderLayer`/`TinyCausalLM` 透传 `norm_backend` |
| `step33/engine.py` | `Engine` / `from_model_dir` / `build_model_from_config` 新增 `norm_backend`；`_check_runtime` 校验设备组合 |
| `step33/step33.py` | 新增 `--norm-backend {torch,triton}`，并打印实际生效的后端 |

## 2. 设计要点

### 2.1 「一行」是什么：最后一维，不是整个张量

统一把**最后一维**当归一化维度，于是任何连续张量都能看成一整片 `[行数, 行宽]`：

| 输入 | 行数 | 行宽 |
|---|---:|---:|
| hidden `[3, 1024]` | 3 | 1024 |
| Q `[3, 16, 128]` | 48 | 128 |
| K `[3, 8, 128]` | 24 | 128 |

**Q 的 16 个 head 各算自己的均方值**，`[3,16,128]` 是 48 行而不是 3 行——如果把 head 混在一起归一化，那就不是 RMSNorm 了。

kernel 就按这个来：`grid = (行数,)`，一个 program 负责一行，`stride_row` 由 `x.stride(-2)` 给出。

### 2.2 kernel 里的三个要点

```python
row  = tl.program_id(0)
offs = tl.arange(0, BLOCK)          # BLOCK 是补齐到 2 的幂后的宽度
mask = offs < N                     # N 是实际宽度

x = tl.load(x_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)
mean_sq = tl.sum(x * x, axis=0) / N          # ← 除以 N，不是 BLOCK
rstd = tl.rsqrt(mean_sq + eps)
w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
tl.store(out_ptr + row * stride_row + offs, (x * rstd * w).to(out_ptr.dtype.element_ty), mask=mask)
```

1. **`tl.arange` 只能取 2 的幂**，所以宽度 6 / 14 / 30 要补到 8 / 16 / 32。补出来的列靠 `mask` 挡住，`other=0.0` 让它们不污染平方和。
2. **分母是实际宽度 `N`**，不是补齐后的 `BLOCK`。这是这一步最容易错的地方：补齐的列虽然值是 0，但除错了就整体缩小。
3. **累计和乘权重都在 FP32，只在 `tl.store` 时做一次舍入**——`tl.store` 的第三个参数写成 `.to(out_ptr.dtype.element_ty)`，所以 FP32 模式是空操作，BF16 模式下就是唯一一次舍入。和上一关 Torch 路径的精度约定完全一致：`x * rstd * w` 的乘法顺序也一样。

   `element_ty` 是**指针类型里的元素类型**：`out_ptr` 是 Triton 的指针值，它的 `.dtype` 是 `!tt.ptr<bf16>` 这样的**指针类型**，不是标量类型；`.element_ty` 才取出被指向的 `bf16`。写成 `.to(out_ptr.dtype)` 会直接编译失败（拿指针类型当转换目标）。

   实测这一句**不是正确性必需**：去掉 `.to()` 之后 Triton 会自己插入转换，PTX 与逐位结果都完全一样（bf16 下都是 `mul.f32x2` + `cvt.rn.bf16x2.f32`，65536 个数 0 个不同）。保留它是因为这一关的主题就是「把精度边界写出来」——**乘法在 FP32、只在写回时收窄一次**，这件事应该出现在源码里，而不是留给后端去决定。

补齐只影响执行形状，不影响结果：**分母是实际宽度 `N`，补齐列由 mask 挡住**，所以宽度 1 / 2 / 3 / 4 / 6 / 14 / 30 全部与参考一致（最坏误差 1.19e-07）。

`num_warps` 按宽度分档（`min(8, max(1, BLOCK // 256))`）：宽度 1024 给 4 个，宽度 128 给 1 个。**宽度 128 只给 1 个是有依据的**——PTX 里能直接看到机制：

| num_warps（BLOCK=128） | shared 字节 | `bar.sync` | `shfl.sync` | 寄存器 |
|---:|---:|---:|---:|---:|
| **1** | **0** | **0** | 5 | 22 |
| 2 | 8 | 2 | 6 | 17 |
| 4 | 16 | 2 | 7 | 16 |

整行都在同一个 warp 里时，`tl.sum(x*x)` 走 warp 内的 `shfl` 蝶形规约——**不用共享内存，也不用 barrier**。一开到 2 个 warp 以上，各部分和必须经共享内存汇合，于是多出 16 字节共享内存和两次 `bar.sync`；而 128 个元素本来每线程才 4 个，换成每线程 1 个换不来多少并行度。这一档是 latency-bound，不是 throughput-bound。

**行数变化时谁更快？用纯 GPU 时间量（下面那张表是修正过的，见文末说明）：**

| 行数（宽度 128） | nw=1 | nw=2 | nw=4 | nw=8 |
|---:|---:|---:|---:|---:|
| 304（模型实际规模：19 token × 16 head） | **2.90** | 3.05 | 3.36 | 3.15 |
| 1216 | **3.21** | 3.55 | 4.30 | 5.26 |
| 4864 | **7.62** | 7.95 | 9.24 | 14.02 |
| 19456 | **19.46** | 19.14 | 23.66 | 42.41 |
| 38912 | **34.04** | 34.58 | 45.46 | 82.42 |

宽度 128 时 **1 个 warp 在每一档都是最快的**，而且行数越多优势越大。原因是每行那个固定开销：多开 warp 就得付共享内存往返 + 两次 `bar.sync`，这个代价**不随行数摊薄**——行越多，总共要付的 barrier 越多。38912 行时 nw=1 跑到 1170 GB/s，nw=4 只有 877 GB/s。

宽度 1024（模型的 hidden norm）则基本无所谓，四档在任何规模下都只差百分之几：

| 行数（宽度 1024） | nw=1 | nw=2 | nw=4 | nw=8 |
|---:|---:|---:|---:|---:|
| 19（模型实际规模） | 3.40 | 3.40 | **3.20** | 3.65 |
| 304 | **4.15** | 4.36 | 4.30 | 4.35 |
| 1216 | 8.62 | 9.13 | 8.93 | **8.46** |

所以按宽度分档的公式给出的是「宽度 ≤256 给 1、1024 给 4」，落在这两档的实测最优附近（1024 那一档本来就是平的），是个够用的护栏，不是精密调参。**在真实模型的两种规模上（19 行 × 1024、304 行 × 128），各种 `num_warps` 的差距都在 15% 以内。**

> **这张表修正过一次。** 更早的版本量的是「Python 循环里发射 + 同步」的端到端时间，而在这个规模上它几乎全是 Triton **Python 侧的发射开销**，不是内核时间：

| 测法（304 行 × 128，nw=1） | 时间 |
|---|---:|
| A 每次调用端到端（Python 循环 + 同步） | 16.88 us |
| B 只把命令塞进队列（不同步） | 16.55 us |
| C CUDA Graph replay（纯 GPU） | **3.64 us** |

> A ≈ B，内核本身只占 22%。配置之间的真实差异（~1 us）被那 17 us 的发射开销完全淹没，于是得出了相反的结论。上表改用 CUDA Graph 把 20 次发射打包再测 replay，量的才是 GPU 时间。

宽度 4 时曾经写过 `max(width, 16)` 的下限，实测**既非正确性要求（BLOCK=4/8/16/32 结果都对）也没有可复现的收益（8.62 vs 8.85 us）**，已经去掉，现在是 `triton.next_power_of_2(width)`。

### 2.3 输入布局：明确拒绝而不是悄悄算错

kernel 用 `row * stride_row + column` 寻址，等于认定**整个张量就是行挨着行的一整片**。所以包装函数要求整块连续：

```python
if not x.is_contiguous():
    raise ValueError(...)
if not weight.is_contiguous():
    raise ValueError(...)
```

**只查最后两维是不够的。** `base[::2]` 这类切片的最后两维各自连续（`stride(-1)==1`、`stride(-2)==width`），但组与组之间有空洞：

| 想读的位置 | 正确偏移（元素） | `row * stride_row` 算出的偏移 |
|---|---:|---:|
| `x[0,0,0]` | 0 | 0 |
| `x[1,0,0]` | 48 | `4 × 6 = 24` |

第二组会读到本应跳过的数据。`weight` 同理：`arange(1,13)[::2]` 的 stride 是 2，而 kernel 按相邻地址读成 `[1,2,3,4,5,6]`。两处都是「读到了错的数据但不报错」，比直接崩掉更危险。

CPU 张量、`float16`、weight 形状/dtype 不符也都明确报错。**不修改输入和 weight**（`out = torch.empty_like(x)`），**不经 CPU 拷贝**，**不用 `.item()`**。

真实模型里 113 个调用点全部满足这个约束（否则会直接报错，不是静默退化成 Torch）。

### 2.4 norm 后端与 attention 后端相互独立

需求明确要求能只换 norm 做 A/B，所以 `norm_backend` 是**独立参数**，不和 `attention_backend` 绑：

```python
Engine.from_model_dir(dir, attention_backend="triton", norm_backend="torch")   # 对照
Engine.from_model_dir(dir, attention_backend="triton", norm_backend="triton")  # 融合
```

四种组合都实测过。后端在**创建 Engine 时确定**，运行中不切换；默认 `"torch"`，旧行为一字不变。

`RMSNorm` 自己持有后端（`self.backend`），forward 里分派——这样 `norm1 / norm2 / q_norm / k_norm / 最终 norm` 五处自动覆盖，不需要在模型里逐处判断。没有 Q/K norm 的模型（`use_qk_norm=False`）不受影响，因为那两份参数根本不存在。

CPU 保留 Torch；显式选 `CPU + triton` 直接报错。

### 2.5 捕获 CUDA Graph 之前先回收垃圾

修完布局检查后，验收脚本里出现了一个只在**整个脚本连跑**时才复现的失败：某个 graph 用例报

```
CUDA error: operation failed due to a previous error during capture
RuntimeError: info.status != cudaStreamCaptureStatusInvalidated INTERNAL ASSERT FAILED
```

打开 `CUDA_LOG_FILE=stderr` 看到真正的元凶：

```
Returning 900 (CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED) from cuGraphExecDestroy   ← 4 次
Returning 901 (CUDA_ERROR_STREAM_CAPTURE_INVALIDATED) from cuLaunchKernelEx
Capture was invalidated by a prior API call
```

**Python 的垃圾回收在捕获期间回收了旧的 `CUDAGraph` 对象，而它的析构函数会调用 `cuGraphExecDestroy`——那是捕获期禁止的 API，一次调用就把整段捕获作废。**

为什么只在连跑时出现：脚本每个用例都会新建一个带图的 Engine，旧 Engine 掉出作用域后其 `CUDAGraph` 变成垃圾；当捕获过程中的分配刚好触发 GC，析构就落在捕获区间里。单独跑那个用例时进程里没有旧图，所以从不复现。

修法是在进入捕获前先 `gc.collect()`：

```python
def _capture_graph(self, num_tokens, kv_cache_pool):
    # 捕获期间不能有 Python 终结器跑：旧 CUDAGraph 被回收时会调用 cuGraphExecDestroy，
    # 那是捕获期禁止的 API，会直接把这次捕获作废。先把待回收的对象清掉再进捕获。
    gc.collect()
    ...
```

`gc.collect()` 放在这里就够了，因为**在自己的捕获区间内不会再产生新的垃圾 CUDAGraph**——`self.graphs` 全程被引用着。待回收的只可能是之前累积下来的。

### 2.6 融合的是「执行」，不是「语义」

Torch 路径原样保留在 `RMSNorm.forward()` 里，作为参考实现和 CPU 路径。两条路径的公式、精度约定、乘法顺序完全一致，所以：

- **FP32 下两者输出逐位相同**（实测 5 种配置 × 3 条请求全一致）；
- **BF16 下两者差在 0.25 量级**（与上一关 attention 两个后端之间的 0.31 同源，是 BF16 的舍入敏感性，见 `docs/step32` §3.9）。

## 3. 验证

conda `vllm-omni-dev`，RTX 5090 Laptop，torch 2.13.0+cu130，Triton 3.7.1，Transformers 5.14.1，TF32 关闭。

### 3.1 算子本身

对照 Torch 参考公式，12 组 shape / 宽度 / dtype / eps 组合：

| shape | 宽度 | dtype | eps | 最大绝对误差 | 输入与 weight 未被修改 |
|---|---:|---|---:|---:|---|
| `(3,14)` | 14 | FP32 / BF16 | 1e-6 | 2.38e-07 / **0.00e+00** | ✓ |
| `(3,4,6)` | 6 | FP32 / BF16 | 1e-6 | 2.38e-07 / **0.00e+00** | ✓ |
| `(48,30)` | 30 | FP32 / BF16 | 1e-6 / 0.003 | 4.77e-07 / **0.00e+00** | ✓ |
| `(3,128)` | 128 | FP32 / BF16 | 1e-5 / 1e-6 | **0.00e+00** / **0.00e+00** | ✓ |
| `(3,1024)` | 1024 | FP32 / BF16 | 1e-6 | 9.54e-07 / **0.00e+00** | ✓ |
| `(1,16)` `(3,17)` | 非 2 的幂 | FP32 / BF16 | 1e-6 | **0.00e+00** | ✓ |

weight 全部是非单位的随机值（`randn * 0.5 + 1.0`），不是全 1。**BF16 全部逐位相同**——因为两条路径都是「FP32 算完舍入一次」，中间没有第二次舍入。

**布局检查**（修掉验收方指出的缺口后）：

| 布局 | 结果 |
|---|---|
| `x = randn(6,4,6)[::2]`（最后两维连续、整块不连续） | `ValueError: 要求 x 整块连续，shape=(3,4,6)、stride=(48,6,1)` |
| `weight = arange(1,13)[::2]`（stride=2） | `ValueError: 要求 weight 整块连续，shape=(6,)、stride=(2,)` |
| 连续输入（12 组 shape/宽度/dtype） | 全部照常工作 |

其他拒绝路径：CPU 张量 / `float16` / weight 形状不符 / weight dtype 不符，也都给出明确的 `ValueError`。

### 3.2 kernel 数

单次 `RMSNorm.forward()`（hidden `[3,1024]`），10 次调用取平均：

| dtype | torch 后端 | triton 后端 |
|---|---:|---:|
| FP32 | **6 个 kernel** | **1 个**（`_rms_norm_kernel`） |
| BF16 | **9 个 kernel** | **1 个** |

与需求给的数字完全吻合。真实模型单次 forward（19 个 prompt token）：

```
norm=torch : 总 kernel 1786 个，其中融合 kernel 0 个
norm=triton: 总 kernel 1221 个，其中融合 kernel 113 个   ← 28×4+1
```

**kernel 总数 1786 → 1221（−31.6%）**，norm 那部分从 113×6=678 个变成 113 个。

### 3.3 真实模型确实走融合路径

按 `_rms_norm_kernel` 计数：单次 forward 恰好 **113** 个，等于 `28 层 × 4 + 最终 norm 1`。不是只改了配置名。

所有 113 个调用点都满足「行连续」的约束（否则会直接抛错）。

### 3.4 数值与功能

| 检查 | 结果 |
|---|---|
| FP32，norm=torch vs norm=triton 的 logits | 最大差 2.003e-05，平均 3.044e-06，argmax 一致 |
| 31 个生成 token | 与 norm=torch **逐位一致**，也与官方参考一致 |
| 小模型 `tiny_gqa` / `tiny_mqa`，norm×attention 六种组合 | 全部与 `norm=torch + attention=torch` 一致 |
| FP32 真实模型：整批 / 分块 20 / 分块 8 / prefix cache / Graph | 五种配置**全部 3/3 请求逐位相同** |
| BF16 真实模型：Graph / 分块 prefill | 都工作，引用计数归零 |
| 保存 → 以融合 norm 重载 | 输出一致、参数逐位相同；`config.json` 里**没有** norm 后端字段 |
| 真实模型 Graph + 融合 norm（FP32 / BF16） | 都工作，Graph 与 eager 逐位一致 |
| 第二十九关三个脚本 | 96/96、45/45、21/21 |
| **第三十三关验收脚本 `verify_step33_norm.py`** | **56/56**（连跑 4 次稳定） |
| FP32 + norm=torch 与 step32 | 输出逐位一致（老路径没变） |

### 3.5 性能

真实 Qwen3-0.6B，同一 prompt、同样 **31 个输出 token**（EOS 处停止），预热后 9 组取中位数，attention 固定为 Triton。

**先看 eager（不捕获 Graph）——这是使用者直接看到的端到端数字，但它包含「发射开销」的节省：**

| 精度 | norm 后端 | TTFT | ITL | 完整生成 | 吞吐 |
|---|---|---:|---:|---:|---:|
| FP32 | torch | 30.8 ms | 31.15 ms | 0.965 s | 32.1 tok/s |
| FP32 | **triton** | **28.5 ms** | **26.98 ms** | **0.838 s** | **37.0 tok/s** |
| BF16 | torch | 37.2 ms | 39.79 ms | 1.231 s | 25.2 tok/s |
| BF16 | **triton** | **19.7 ms** | **27.34 ms** | **0.840 s** | **36.9 tok/s** |

**再看 Graph 模式——Python 侧的发射开销被整段捕获掉，量到的才是 GPU 计算时间：**

| 精度 | norm=torch | norm=triton | 提升 |
|---|---:|---:|---:|
| FP32 | 0.182 s（169.9 tok/s） | 0.174 s（178.0 tok/s） | **+4.7%** |
| BF16 | 0.162 s（191.8 tok/s） | 0.139 s（223.1 tok/s） | **+16.3%** |

**两张表差很多，差的就是发射开销。** 一次 forward 的 kernel 数从 1786 降到 1221，省掉 565 次发射；eager 模式下每次发射都要过一遍 Triton/PyTorch 的 Python 启动器（上一节实测 **约 16 us/次**），光这一项就接近 9 ms——比内核本身省下的时间大得多。所以：

- **+15% / +46%（eager）** 是真实用户能拿到的收益，但主要来自「少发 565 次命令」；
- **+4.7% / +16.3%（Graph）** 才是这一关真正省下的 GPU 计算时间。

两个都要报，不能只报大的那个。

峰值显存**没有变化**（FP32 3.546 GB、BF16 1.800 GB，两种 norm 后端相同）：融合省掉的是转瞬即逝的中间 Tensor，本来就不构成峰值；它也**不增加**任何常驻缓冲。

**BF16 受益更大**：上一关测到 BF16 的 eager 反而比 FP32 慢（多了转换开销），而融合正好把那些转换算子一起吃掉了——Graph 模式下 BF16 + 融合 norm 是四种组合里最快的。

## 4. 接口变化与遗留

- 新增模块 `step33/norm.py`：`rms_norm(x, weight, eps)`、`_rms_norm_kernel`。
- `RMSNorm(size, eps, backend="torch")` 新增 `backend` 与属性 `self.backend`。
- `DecoderLayer(..., norm_backend="torch")`；`TinyCausalLM(..., norm_backend="torch")` 并有属性 `model.norm_backend`。
- `Engine(...)` / `Engine.from_model_dir(...)` / `build_model_from_config(...)` 新增 `norm_backend="torch"`。
- `step33/step33.py` 新增 `--norm-backend {torch,triton}`。
- **没有**改动权重格式：norm 后端是运行选项，`config.json` 与参数名都不变。
- 遗留：只融合了 RMSNorm。attention、`o_proj`、残差相加、SwiGLU 仍是多个小算子（真实模型单次 forward 仍有 1221 个 kernel，其中 680 个是 torch 原生）。没做 `tl.dot`、没融合整个 Transformer、没实现 backward。
- 遗留：融合 kernel 要求输入**整块连续**（不只是最后两维），非连续输入直接拒绝而不是自动 `contiguous()`。
- 遗留：`_capture_graph()` 开头新增一次 `gc.collect()`，代价只在捕获时付一次。
- 未提交。
