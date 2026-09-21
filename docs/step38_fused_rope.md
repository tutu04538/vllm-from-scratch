# step38：融合 RoPE

- 对应代码：`step38/`（新增，从 `step37/` 复制）
- 包摘要 SHA256：`45c0e5c61783695f36894446f05232520e5af0f8e60ddac7400108564357319a`（14 个 .py / 2511 行）
- 基线：`step37/`，指纹 `ca06af2a…`（13 个 .py / 2350 行），原样保留未改
- 改动文件：**新增 `rope.py`**；`model.py`、`engine.py`、`__init__.py`、入口改名 `step38.py`

## 0. 需求大概

依据第三十七关的归因：真实模型 eager decode 一步里，`RotaryEmbedding.forward()`
每次执行 **12 个 GPU kernel**，Qwen3 每层对 Q/K 各调一次，28 层合计 **672 个 kernel**。
计算本身只是一组二维旋转，中间步骤却被拆得很碎。

这一关只做一件事：**为 `RotaryEmbedding` 增加 Triton 后端，一次 forward 用一个 kernel。**

约束：保留 Torch 路径作为 CPU 路径与参考；支持 CUDA FP32/BF16；返回值 shape/dtype/设备与输入相同；
Q 与 K 仍各调用一次（不合并 Q/K、不融合 QK norm、不把 KV 写入塞进来）；不改 tiled attention、
不改权重、不改 RoPE 数学定义；后端选择是运行配置，不进模型权重。

## 1. 改动内容

| 文件 | 改动 |
|---|---|
| `rope.py` | **新增**：`_rope_kernel` 与 `rope()` 入口 |
| `model.py` | `RotaryEmbedding` 增加 `backend` 并按后端分派；`TinyCausalLM` 增加 `rope_backend` 并构造时传入 |
| `engine.py` | `_check_runtime` / `build_model_from_config` / `Engine.__init__` / `from_model_dir` 传递 `rope_backend` |
| `step38.py` | 新增 `--rope-backend`，并在启动信息里打印实际后端 |
| `attention.py` / `cache.py` / `scheduler.py` / `norm.py` / `sampling.py` / `formats/` | **未改** |

## 2. 设计要点

### 2.1 配对是前后半段，不是相邻两个

```text
x = [a, b, c, d]     配对是 (a, c)、(b, d)
y[i]     = x[i]     * cos[i] - x[i + D/2] * sin[i]
y[i+D/2] = x[i+D/2] * cos[i] + x[i]     * sin[i]
```

角度表存的是**半宽** D/2，每个角度只存一份。这带来一个附带好处：Torch 路径里
`torch.cat((cos, cos), dim=-1)` 那一步（把 [N,D/2] 拼成 [N,D]）在 kernel 里根本不需要——
直接按 `pos * D_HALF + offs` 读半宽表，输出两个位置时复用同一个 `c`/`s`。
`rotate_half(x)` 那个中间张量同样不需要。

### 2.2 「一个 program 管一行的一个 head」

grid = `(N, num_heads)`。每个 program 只做 2×BLOCK 个元素的搬运与一次旋转，
不需要 block 间通信，也不需要共享内存。

| 维度 | 处理 |
|---|---|
| `D/2` 不是 2 的幂（如 head_dim=14 → 7） | `BLOCK = next_power_of_2(D/2)`，补出来的列由 mask 挡住 |
| `D < 16` | 不需要特判——这里没有 `tl.dot`，不涉及矩阵乘法单元的维度下限 |
| Q 与 K 的 head 数不同（16 / 8） | 各是一次独立 launch，grid 的第二个维度自然不同 |

### 2.3 位置必须查表，不能用行号

打包的两行可能属于两个请求，真实位置可能是 `[100, 7]`，而行号是 `[0, 1]`。
kernel 里 `pos = tl.load(pos_ptr + row)`，取的是**传入的 positions**。这一点单独做了
带区分度的检查（§4.1 第 3 项）——用行号查表的误差是 `6.02e+00`，用真实位置是 `2.33e-07`，
差了七个数量级，说明这个检查确实能抓住错误。

### 2.4 精度边界

- **角度表恒为 FP32**，不从输入精度走。kernel 里 `c`/`s` 直接按 fp32 读入，不做 dtype 转换。
- **BF16 输入先扩 FP32 再旋转**：`tl.load(...).to(tl.float32)`，算完在 store 时转回。
- **只在写回时舍入一次**。这与 Torch 路径的舍入位置一致。

### 2.5 `fp_fusion`：默认关闭，并说明为什么

`x0*c - x1*s` 若允许编译器收缩成一次 FMA，只舍入一次而不是两次。这改变了舍入次数，
需求要求「明确处理或报告，不能直接扩大误差容限」。处理方式是：

- 默认 `fp_fusion=False`，让每一步乘法/加法各自舍入一次。这样 **FP32 下与 Torch 路径逐位相同**
  （实测 `torch.equal == True`），是一条可以直接检验的性质。
- 保留开关，并把差异量化（§4.1 第 6 项）：**BF16 下最多影响 2～5 个元素 / 131072，且差异
  不超过一个满量程 BF16 ulp**；FP32 下开/关之差不超过一个 FP32 ULP。
- 顺带记下一个方向性事实：**开融合反而离 FP64 真值更近**（FP32 下 `2.680e-07` vs `3.487e-07`），
  因为 FMA 只舍入一次；而关融合是与 Torch 逐位一致。两者取舍不同，不是谁对谁错。

### 2.6 不做的事（需求划定）

不合并 Q/K 成一次调用；不融合 QK norm；不把 KV 写入塞进来；不原地改输入；
不分配 KV 块、不推进请求长度、不改采样状态。角度表仍是 `persistent=False` 的 buffer，
不进 `state_dict`。

### 2.7 寻址：入和出各按自己的 stride

沿用分页 attention kernel 的约定——**多维度张量各传自己的 stride**：

```python
q.stride(0), q.stride(1), q.stride(2),        # stride_q*
out.stride(0), out.stride(1), out.stride(2),  # stride_o*     ← 各一套
```

第一版这里只传了**一套** stride（`x.stride(0), x.stride(1)`），读写共用。当时靠一条
`x.is_contiguous()` 检查把它焊死才没出事。问题在于 `out = torch.empty_like(x)`
给的不一定是同一套 stride：

```text
x.transpose(0,1)   x.stride=(256, 4096, 1)  →  out.stride=(256, 4096, 1)   ← 一样（换序但稠密）
x[:, ::2, :]       x.stride=(4096, 512, 1)  →  out.stride=(2048, 256, 1)   ← 补成连续，不一样
```

`empty_like` 对**换序但稠密**的布局保留 stride，对「切片切出来的空洞」则补成连续。
两类都会遇到，所以不能假设它和 x 一致。

拿 x 的 stride 去写 out，最大触到偏移 991，而 out 只有 512 个元素——**越界 480 个**。
这不是算错，是内存越界。

现在 x 与 out 各传三个 stride，列 stride 也走参数而不是写死 `+1`，于是：

| 布局 | 改前 | 改后 |
|---|---|---|
| `x[:, ::2, :]`（head 间有空洞） | 拒绝（否则越界写） | 算对，误差 2.19e-07 |
| `x[..., ::2]`（列 stride=2） | 拒绝 | 算对，误差 2.10e-07 |
| `x.transpose(0,1)` | 拒绝（虽然本来能算对） | 算对，误差 2.17e-07 |
| `x[:, 1:6:2, :]` | 拒绝 | 算对，误差 2.26e-07 |

`x` 的连续性检查随之**去掉**——不再是能力限制，而是根本不需要。
`cos_table` / `sin_table` / `positions` 保留连续要求：它们是模块自己的 buffer 与视图，
本来就是连续的，kernel 也按单位列 stride 读；留着只是把隐含前提写出来。

代价：kernel 多 4 个 int 参数，实测无性能差异（prefill_c8 84.80 ms、decode_c1 443.20 ms，
与改前一致）。

位置的下标边界**不在 kernel 入口检查**——那要在热路径上对 GPU 张量取 `.item()`，强制一次同步。
边界由调用方保证：Engine 已经在推进长度之前校验过位置区间。

## 3. 验证

### 3.1 算子数值（`benchmarks/check_step38_fused_rope.py`，21/21 通过）

三份参考各自回答不同问题：独立参考（CPU/FP64，按配对旋转定义直接算）、Torch 路径、被测 kernel。

形状与精度覆盖：

| 用例 | 形状 | dtype | 最大绝对差 | 容限 |
|---|---|---|---:|---:|
| decode 形状 Q | (1,16,128) | fp32 | 0.000e+00 | 3.95e-07 |
| decode 形状 K | (1,8,128) | fp32 | 0.000e+00 | 3.61e-07 |
| prefill 形状 Q | (64,16,128) | bf16 | 1.528e-02 | 3.33e-02 |
| prefill 形状 K | (64,8,128) | bf16 | 1.489e-02 | 3.55e-02 |
| 小 head_dim=8 | (7,3,8) | fp32 | 1.387e-07 | 3.32e-07 |
| 偶数非 2 的幂 head_dim=14 | (7,3,14) | fp32 | 1.493e-07 | 3.59e-07 |
| MQA + head_dim=14 | (5,1,14) | bf16 | 3.683e-03 | 1.72e-02 |
| 行数少于 head 数 | (3,2,128) | bf16 | 4.571e-03 | 2.95e-02 |

容限来自**输出的存储精度**：参考在 FP64 下算，结果最后要舍入回输入 dtype，
所以地板是 `4 × max|ref| × eps(dtype)`。输入本身已经是目标 dtype、与参考用的是同一份 x，
不构成差异来源。（第一版把地板算成了 FP32 级，导致 BF16 用例全部误报，已更正。）

其余检查：

| 检查 | 结果 |
|---|---|
| FP32 下与 Torch 路径**逐位相同** | PASS（`最大差=0.000e+00`） |
| 位置按 `positions` 查表而非行号 | PASS（用位置 2.33e-07，用行号 6.02e+00） |
| 输入 x / positions / 角度表都未被改动 | PASS |
| 返回新张量，不是输入的别名 | PASS |
| 非连续布局算对：`x[:, ::2, :]` / `x[..., ::2]` / `x.transpose(0,1)` / `x[:, 1:6:2, :]` | PASS ×4（误差 ~2e-07） |
| 奇数 head_dim 被拒绝 | PASS |
| 非 FP32 角度表被拒绝 | PASS |
| BF16 下开/关融合差异 ≤ 1 个 BF16 ulp 且只影响极少数元素 | PASS（1/131072，2.441e-04 ≤ 1.749e-02） |
| 关融合时 FP32 与 FP64 参考差在几个 ULP 内 | PASS（0.61 ULP） |
| 开/关融合的 FP32 差 ≤ 1 ULP | PASS（0.83 ULP） |

### 3.2 引擎语义（`benchmarks/check_step38_engine.py`，13/13 通过）

| 检查 | 结果 |
|---|---|
| `rope_backend` 传到 `RotaryEmbedding`（torch / triton） | PASS |
| FP32 也支持 triton RoPE | PASS |
| 未知 `rope_backend` 被拒绝 | PASS |
| BF16 / FP32 下两条后端的 logits **逐位相同** | PASS（都是 `0.000e+00`） |
| 三个不同长度请求：Graph 与 eager 输出一致 | PASS |
| **相同 shape、不同 positions 的 Graph 重放正确** | PASS |
| **decode 图被复用，未按位置重新捕获** | PASS（只新增 B 的 prefill 图 `(37,1,True,True)`） |
| 真实 Qwen3-0.6B 两条后端 greedy 输出一致 | PASS |

「相同 shape 不同 positions」这一条是需求点名的重点。decode 步的 RoPE 输入 shape 恒为
`[1, H, D]`，但位置随请求变化；若图把首次捕获时的 positions 固化，第二个请求就会算错。
验证方式是让同一引擎先跑 prompt=5、再跑 prompt=37 的请求，两者都与 eager 一致，
且图的数量只增加了 1（B 的 prefill），说明 decode 图确实被复用。

### 3.3 算子测量（`benchmarks/bench_step38_rope.py --op`）

一次 RoPE 调用真正花在 GPU kernel 上的时间（profiler 的 `self_device_time_total`）：

| 形状 | Torch | Triton | 提升 |
|---|---:|---:|---:|
| prefill Q `[2048,16,128]` | 173.6 µs | 15.7 µs | 11.05× |
| prefill K `[2048,8,128]` | 61.6 µs | 8.7 µs | 7.08× |
| decode Q `[1,16,128]` | 14.2 µs | 1.0 µs | 14.66× |
| decode K `[1,8,128]` | 14.2 µs | 1.0 µs | 14.83× |

数字自洽：Q 的 head 数是 K 的两倍，prefill 下 15.7 / 8.7 ≈ 1.8；decode 两个形状都触到 1 µs 下限。

**两次测量方法上的修正**（都记下来，因为两次都得出过错误的数）：

1. 先用 CUDA Event 计时，得到 torch 215 µs / triton 32 µs。查下来那是 **CPU 发射开销**——
   Torch 路径一次 12 个 kernel，每 kernel 约 18 µs 的发射时间盖过了 GPU 时间。
2. 改用 CUDA Graph 消掉发射开销，结果自相矛盾：prefill K 只有 prefill Q 的 1/18，
   decode K 的 triton 反而更慢。捕获改变了分配行为，这个口径不可用。
3. 最终用 profiler 读 kernel 执行时长，得到上表。

### 3.4 kernel 数量（需求的核心指标）

真实 Qwen3-0.6B，eager 一步：

| | Torch RoPE | Triton RoPE | 差 |
|---|---:|---:|---:|
| 一步 decode 的全部 GPU kernel | 1200 | **584** | −616 |
| 一步 prefill（2048 token）的全部 GPU kernel | 1233 | **617** | −616 |

`_rope_kernel` 恰好出现 **56 次** = 28 层 × Q/K 各一次 ✓。
616 ÷ 56 = **11**，即每次 RoPE 从 **12 个 kernel 收到 1 个**（56×12 = 672 → 56×1 = 56），
与第三十七关验收方的归因数字完全吻合。

### 3.5 完整引擎 A/B（`benchmarks/bench_step38_rope.py`）

同版 step38、同一份输入、同样开 Triton norm / tiled attention / Graph、同样 KV 容量，
唯一变量是 `rope_backend`。每点预热 2 次、测量 5 次取中位。

| 测点 | Torch RoPE | Triton RoPE | 变化 |
|---|---:|---:|---:|
| short_c1 | 134.70 ms | **105.00 ms** | −22.0% |
| short_c8 | 216.00 ms | **177.90 ms** | −17.6% |
| prefill_c1 | 13.40 ms | **10.90 ms** | −18.7% |
| prefill_c8 | 120.30 ms | **84.80 ms** | −29.5% |
| decode_c1 | 542.30 ms | **442.80 ms** | −18.3% |
| decode_c8 | 832.20 ms | **757.60 ms** | −9.0% |

六个点全部变快。单步 prefill（2048 token）的墙钟对照：

| | Torch RoPE | Triton RoPE | 提升 |
|---|---:|---:|---:|
| Graph 开 | 51.31 ms | 40.83 ms | 20.4% |
| Graph 关 | 51.97 ms | 40.93 ms | 21.2% |

开不开 Graph 结果几乎相同——说明 CPU 发射开销本来就已经被图盖住了，
这一关省下的是**真实的 GPU 执行时间**，不只是提交次数。

### 3.6 一处不能用 profiler 数字解释的矛盾

在引擎上下文里对**一次 Graph 重放的 prefill 步**取 profiler 的 kernel 时间合计：

```text
rope=torch   1233 个 kernel，合计 36.29 ms
rope=triton   617 个 kernel，合计 39.40 ms      ← triton 反而更高
```

kernel 数确实少了 616 个（符合预期），但 GPU 时间合计反而略增。这与**所有**墙钟测量矛盾
（§3.5 的六点表、单步墙钟、纯算子微基准都显示 triton 更快）。

第三十七关验收方也遇到过同类现象（kernel 区间并集 1.444 s 而墙钟 0.611 s，口径不自洽）。
**本关不使用这个 profiler 数字下任何结论**，正式性能结论一律来自不带 profiler 的独立计时。

## 4. 接口变化与遗留

### 4.1 接口变化

**新增**：

| 接口 | 说明 |
|---|---|
| `rope.rope(x, positions, cos_table, sin_table, fp_fusion=False)` | 融合入口，返回同 shape/dtype 的新张量；`x` 可为任意稠密布局（含非连续） |
| `rope._rope_kernel` | 融合 Triton kernel；x 与 out 各三个 stride，与分页 attention 同一约定 |
| `RotaryEmbedding(head_dim, max_seq_len, theta, backend="torch")` | 新增 `backend` 参数 |
| `RotaryEmbedding.backend` | 实际后端，可查询 |
| `TinyCausalLM(..., rope_backend="torch")` | 新增参数 |
| `Engine(..., rope_backend=...)` / `Engine.from_model_dir(..., rope_backend=...)` | 新增参数 |
| `Engine.model.rope_backend` | 实际后端 |
| CLI `--rope-backend {torch,triton}` | 默认 `torch` |

**未改**：`paged_attention` / `tiled_paged_attention` 及第 37 关全部接口、
`AttentionMetadata`、`rms_norm`、调度与 KV 相关接口、`model_config()` 与权重格式。

默认仍是 `torch`：需求要求「默认保留 Torch，测试时明确选择 Triton；不要悄悄回退后
仍打印使用 Triton」。启动信息里打印的是 `model.rotary.backend`（对象上的实际值），
不是命令行入参。

### 4.2 遗留

1. **第 37 关 decode_c1 约 3.6% 的回退仍单独记录，原因未查明。** 本关 rope=triton 下
   decode_c1 是 442.80 ms，比 step37 的 549.72 ms 快得多，但**不能因此声称此前回退的
   原因已经找到**——需求 §6.5 明确要求保留这一条。它是两个独立事实。
2. **`fp_fusion` 默认关闭是权衡，不是最优。** 开融合离 FP64 真值更近且更快，但会与
   Torch 路径产生极少数元素的最后一位差异。若后续要默认开启，需要先补一次更宽的
   随机与真实模型验证，而不是只靠本次的 20 个种子。
3. **非连续输入现在支持了，但没有引擎级的用例**：只有算子级检查覆盖了四种非连续布局，
   引擎内实际传的仍是 `.view()` 出来的连续张量。
4. **没有做 Q/K 合并**（需求明确不做），所以每层仍是两次 launch。这是下一处可能的
   收益点，但需求把范围划在这里。
5. **`enable_fp_fusion` 依赖 Triton 的编译选项**。若 Triton 版本变化导致该选项失效，
   会静默变成开融合——目前没有断言保护。第 2 节的逐位对照检查可以抓住这一点。
