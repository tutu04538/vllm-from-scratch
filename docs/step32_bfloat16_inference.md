# step32：真正使用 BF16 推理

- 对应代码：`step32/`（新增，未提交）
- 包摘要 SHA256：`62caddc9e2015c6a226dac810b85cd32ff88cc536089cdcc757d49e002bb95f9`
  （`sha256("\n".join(sorted("路径 sha256(文件)")) + "\n")`，只算 `step32/**/*.py`）
- 基线：`step31/` 原样保留、未修改

## 0. 需求大概

上一关虽然读 BF16 文件，进 Engine 后全变成了 FP32。本关给 Engine 加一个**运行精度**选项：同一个模型可选 FP32 或 BF16，并用真实测量回答省了多少显存、快不快、误差可不可接受。

关键前提：**低精度推理不等于把所有数字都换成 BF16。** 三个位置的精度要求不同：

| 位置 | BF16 模式下 |
|---|---|
| 磁盘权重 | 仍允许 FP32 / BF16 文件 |
| 模型权重、主要 hidden/QKV、KV cache | BF16 |
| 归一化统计量、attention 的关键累计量 | FP32 |

`Engine.from_model_dir(..., dtype=torch.bfloat16)`；默认仍是 FP32，旧例子不变。

## 1. 改动内容

| 文件 | 内容 |
|---|---|
| `step32/model.py` | `TinyCausalLM(..., dtype=...)`；RMSNorm、RoPE、`block_attention` 三处显式精度边界 |
| `step32/attention.py` | Triton 算子：load 后转 FP32，store 时转回池的精度 |
| `step32/cache.py` | `KVCachePool(..., dtype=...)`，K/V 池按运行精度分配 |
| `step32/engine.py` | `Engine` / `from_model_dir` / `build_model_from_config` 新增 `dtype`；`_check_runtime` 校验设备与精度的组合；装权重时转成运行精度；采样前把 logits 转 FP32 |
| `step32/formats/__init__.py` | `read_raw_weights()` 不再强制转 FP32，改成原样返回（转不转由调用方按运行精度决定） |
| `step32/step32.py` | 新增 `--dtype {float32,bfloat16}`，加载后打印**实际**精度与显存占用 |

## 2. 设计要点

### 2.1 三处精度边界，各自都是显式写的

**RMSNorm**：平方、均值、rsqrt、权重缩放全在 FP32，最后舍入一次回输入精度。

```python
x_fp32 = x.float()
variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
return ((x_fp32 * torch.rsqrt(variance + self.eps)) * self.weight.float()).to(x.dtype)
```

`x.float()` 在 FP32 输入下是空操作，所以 **FP32 路径的数值逐位不变**（见 §3.2 的实测）。

**RoPE**：角度表**恒为 FP32**。这个模块只 `.to(device)`、不 `.to(dtype)`——如果把表也转成 BF16，就违反了「频率、角度计算保持 FP32」。但随之而来有个坑：

```python
rotated = x.float() * cos + _rotate_half(x).float() * sin
return rotated.to(x.dtype)    # ← 这一行不能省
```

FP32 的 `cos/sin` 与 BF16 的 `x` 做乘法，**类型提升会把结果变成 FP32**。不显式转回去的话，从这一行开始整个网络都会悄悄变回 FP32——这正是需求里警告的「不能因为 cos/sin 是 FP32，就让后续整条网络意外回到 FP32」。

**分页 attention**：Q/K/V 从缓存读出来是 BF16，**先转 FP32 再参与运算**，点积、max、exp、归一化分母、加权和全程 FP32，最后写回时舍入一次。

```python
q_fp32 = q.float()
k_head = k_block[:, kv_head, :].float()
v_head = v_block[:, kv_head, :].float()
...
out[:, q_head, :] = (u / z).to(out.dtype)    # 恢复运行精度
```

Triton 算子同理，在 load 之后 `.to(tl.float32)`，store 时 `.to(out_ptr.dtype.element_ty)`：

```python
q = tl.load(q_ptr + ..., mask=mask_d, other=0.0).to(tl.float32)
...
tl.store(out_ptr + ..., (acc / z_i).to(out_ptr.dtype.element_ty), mask=mask_d)
```

**为什么必须显式写**：先用 BF16 乘、得到舍入后的结果再转 FP32，**不等于**先转 FP32 再乘。实测这两者差 `2.6e-02`（小模型 6 个 token、16 个元素上），不是可以忽略的量级。只把 `acc` 声明成 FP32 而让输入留在 BF16，是这一关最容易犯的错。

### 2.2 整数索引绝不跟着变

token IDs、position IDs、slot mapping、block table、attention 元数据缓冲全是整数。它们在构造时就写明 `dtype=torch.long` / `torch.int32`，且**不经过任何 dtype 转换路径**。实测 BF16 模式下这些仍是 `int64` / `int32`。

### 2.3 文件精度与运行精度解耦

```
读 safetensors → 检查实际 dtype（白名单由适配器声明）
              → 原样返回，不转
              → 改名 → tied 核对
              → t.to(model.dtype)  ← 唯一一次精度转换，在装进模型的那一刻
              → load_state_dict(strict=True)
```

这样：
- BF16 文件 → BF16 模型：**零转换**，直接装（比上一关少一次 FP32 中转，加载也更快）；
- BF16 文件 → FP32 模型：精确扩宽；
- FP32 文件 → BF16 模型：在这里舍入一次。

转换点只有一个，且就在「装进模型」这一步，不会散落在各处。

### 2.4 dtype 在创建时确定，两个 Engine 互不影响

`dtype` 是构造参数，存成 `model.dtype`，KV 池、输入缓冲、Graph 捕获全按它走。全程**不碰 `torch.set_default_dtype`**，所以 FP32 与 BF16 的两个 Engine 可以同时存在、分别跑，实测互不影响（`torch.get_default_dtype()` 也仍是 float32）。

CPU 只保留 FP32，`dtype=torch.bfloat16` 且设备不是 CUDA 时明确拒绝。`dtype=torch.float16` 同样拒绝（本关不做 FP16）。

Graph 在目标精度的模型和缓存就位**之后**才捕获，捕获后不替换任何存储。

### 2.5 采样前转 FP32

logits 本身保持运行精度（BF16 模式下 `lm_head` 输出就是 BF16），只在做「决定」的那一刻转 FP32：

```python
output_ids = self.sampler.sample(logits[rows, :].float())
```

FP32 模式下 `.float()` 是空操作。

## 3. 验证

conda `vllm-omni-dev`，RTX 5090 Laptop，torch 2.13.0+cu130，Triton 3.7.1，Transformers 5.14.1，TF32 关闭。

### 3.1 精度确实贯通（不是「配置写 BF16、实际跑 FP32」）

| | FP32 模式 | BF16 模式 |
|---|---|---|
| 参数 / embedding / lm_head | `torch.float32` | `torch.bfloat16` |
| KV 池 K/V | `torch.float32` | `torch.bfloat16` |
| RoPE 角度表 | `torch.float32` | **`torch.float32`（按设计恒定）** |
| logits | `torch.float32` | `torch.bfloat16` |
| 输入 / 位置 / slot 缓冲 | `torch.int64` | `torch.int64` |
| attention 元数据缓冲 | `torch.int32` | `torch.int32` |

### 3.2 FP32 路径没有回归

- `step32` 与 `step31` 在 Torch / Triton / Triton+Graph 三条路径上输出完全一致；同种子随机模型的权重**逐位相同**。
- 把验收方第二十九关的三个脚本指到 `step32` 上跑：**96/96**、**45/45**、**21/21**（与 `step31` 相同）。

### 3.3 精度边界不是摆设

```
先转 FP32 再乘 vs 先 BF16 乘再转：最大差 2.648e-02
（6×6 的分数矩阵里 24 个元素不同，不是可以忽略的量级）
```

### 3.4 BF16 下的三条路径都工作（小模型）

- `tiny_gqa`：FP32 三条路径与 BF16 三条路径**六条全部一致**。
- `tiny_mqa`：BF16 三条路径之间一致，但与 FP32 在第 4 个 token 分歧（该样例 `head_dim=6`，BF16 的舍入影响相对更大）。这正是 §3.9 讲的现象在小模型上的体现。

### 3.5 文件精度 → 运行精度

| 文件 | 运行精度 | 结果 |
|---|---|---|
| BF16 | FP32 | 参数 `float32`，可用 |
| BF16 | BF16 | 参数 `bfloat16`，零转换直装 |
| FP32 | BF16 | 参数 `bfloat16`（在这里舍入） |

同一份权重：BF16 文件直装的结果与 FP32 文件舍入到 BF16 的结果**逐位相同**。

### 3.6 两个 Engine 共存 / CPU

- FP32 Engine 的输出在 BF16 Engine 跑前跑后完全相同；全局默认 dtype 未被改动。
- `CPU + bfloat16` → `ValueError: dtype=torch.bfloat16 需要 CUDA 设备`；`dtype=torch.float16` → `ValueError: 不支持的 dtype`；**CPU + FP32 仍可用**。

### 3.7 保存 / 重载

- `save_model()` 仍写 FP32 文件（`config` 里 `dtype=float32`），保存的是**当前参数值**。
- 以 BF16 重载后参数**逐位相同**、`eos_token_ids` 一致、输出一致；同一目录以 FP32 加载也可以（旧文件继续可用）。没有新增文件格式版本。

### 3.8 真实 Qwen3-0.6B

| | FP32 | BF16 |
|---|---:|---:|
| 参数字节 | 3.007 GB | **1.503 GB**（−50.0%） |
| K+V 缓存（128 块） | 470 MB | **235 MB**（−50.0%） |
| 进程峰值显存（eager） | 3.498 GB | **1.759 GB**（−49.7%） |
| 加载耗时 | 5.34 s | 2.92 s |

**与参考对照**：

- FP32：31 个 token 与官方参考逐位一致（沿用上一关结论）。
- BF16：同样是 31 个 token，**与 FP32 逐位一致**；停止于 EOS。
- BF16 与**官方 BF16 实现**比：argmax 一致、top5 id 一致、logits 最大绝对误差 4.062e-01。
- BF16 与**本实现 FP32**比：logits 最大绝对误差 2.682e-01，平均 3.342e-02。
- **Graph 与 eager 逐位相同**（最大差 `0.000e+00`，token 序列一致）；同一路径重复跑也逐位相同。

**时序**（同一 prompt、同样 31 个 token，预热 3 次后测 1 次）：

| | eager | +Graph |
|---|---:|---:|
| FP32 | 27.2 tok/s | 141.3 tok/s |
| BF16 | 22.5 tok/s | **188.9 tok/s** |

**BF16 在 eager 下反而更慢，加上 Graph 才快 33.7%。** 原因是本关没做 `tl.dot`：attention 里的点积是 FP32 的逐元素乘加，BF16 在那里得不到张量核的好处，反而多了转换开销；Graph 摊掉的是每步的调度与启动开销，所以两条线都被抬高，BF16 因为数据量更小而受益更明显。**不预先假定「低精度一定更快」，以测量为准。**

### 3.9 BF16 的误差来自哪里：massive activations

多层之后 BF16 与 FP32 的 logits 差异到 0.27 量级。追到逐层 hidden state：

| 层 | hidden 幅值 max | FP32 平均幅值 | BF16 平均幅值 | BF16 在该量级的 1 个 ULP | 实测最大差 | 差 / ULP |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 7.7 | 0.2264 | 0.2265 | 0.06 | 0.04 | 1.32 |
| 3 | 8140.3 | 1.0493 | 1.0500 | 32.0 | 19.72 | 0.62 |
| 7 | 8136.2 | 1.2104 | 1.2116 | 32.0 | 23.84 | 0.75 |
| 13 | 8117.3 | 1.5578 | 1.5599 | 32.0 | 42.66 | 1.33 |
| 27 | 2539.8 | 10.2491 | 10.2396 | 16.0 | 84.21 | 5.26 |

**Qwen3-0.6B 有 massive activation**：从第 3 层起，少数维度的幅值到 8000 上下，而平均幅值只有 1。BF16 只有 8 位尾数，在 8000 这个量级上 1 个 ULP 就是 32——**实测的 FP32/BF16 差异大多不到 1 个 ULP，已经到 BF16 能表示的极限**。也就是说这个差异是量化精度决定的，不是算法错误。

同样的机理解释了路径间差异：任何一点运算顺序的不同都会翻转某些元素的舍入方向，而一次翻转就是整整一个 ULP，再逐层放大。所以：

- **同一条路径是确定的**（重复跑逐位相同），**Graph 与 eager 也是逐位相同**——这是本关最硬的内部一致性问题，通过了；
- 但**不同实现之间不再能要求逐位一致**。三条短文本请求（24 token）在 BF16 下与 FP32 比：

  | 对比 | q0 | q1 | q2（8 token） |
  |---|---|---|---|
  | FP32/torch → BF16/torch | 0/24 不同 | 6/24 不同 | 0/8 不同 |
  | FP32/torch → BF16/triton | 5/24 不同 | 6/24 不同 | 0/8 不同 |
  | BF16/torch → BF16/triton | 5/24 不同 | 0/24 不同 | 0/8 不同 |

  分歧之后文本仍然通顺，只是走了另一条路（`，其核心目的是…` vs `。其核心思想是…`）。参考 prompt 上全程最小 top1-top2 间距是 **0.0327**（FP32）/ **0.125**（BF16），远小于 0.27 的误差量级，所以在这个模型上出现翻转是可预期的。

**结论：BF16 下「数值接近」的判据不能是逐位相等，只能是 argmax/候选间距。** 本关没有出现「文字看着正常但数值错」的情况：与官方 BF16 参考的 argmax 一致，且 Graph 与 eager 逐位相同。

### 3.10 其他功能

- BF16 下分块 prefill 预算 20 / 64、prefix cache（命中 1 个前缀块）、Graph（捕获 N = 1 / 19）都继续工作。
- 从未被分配过的块，K/V **仍然全 0**；跑完后引用计数全 0、缓存索引为空。

## 4. 接口变化与遗留

- `TinyCausalLM(...)` 新增 `dtype=torch.float32` 与属性 `model.dtype`。
- `Engine(...)` / `Engine.from_model_dir(...)` / `build_model_from_config(...)` 新增 `dtype=torch.float32`。
- `KVCachePool(..., dtype=torch.float32)` 新增参数。
- `formats.read_raw_weights(model_dir, allowed_dtypes)` 现在**原样返回**读到的权重，不再统一转 FP32；精度转换移到 `engine._load_weights_into()`。
- `step32/step32.py` 新增 `--dtype {float32,bfloat16}`，并打印实际参数/KV/缓冲的精度与占用。
- 未新增文件格式版本：`save_model()` 仍写 FP32。
- 遗留：没做 `tl.dot`，attention 的点积不是张量核运算，所以 BF16 在 eager 下没有提速；没做 INT8/INT4、算子融合、自动精度选择或 FP16；CPU 不支持 BF16。
- 遗留：tied 权重仍是两份独立参数，BF16 下也没有做物理存储去重。
- 未提交。
