# 第八十五关：Norm、激活融合与 Gemma4 packed QKV

> 需求：`单卡执行优化需求/085_归一化激活融合与QKV打包.md`
> 日期：2026-10-30。前置 84（RoPE 共享紧凑表），后接 86（Gemma4 原生分页 attention）。
> 包摘要：`07ee10305da0fc6b2e68af52f97cf84a581e02b60f08e2b68324f773cec927d5`（177 个 `.py`）
> 结论先行：本关完成 A/B/C 三块——RMSNorm/GemmaRMSNorm 融合、`GeluAndMul`/`SiluAndMul`
> 共享融合、Gemma4 target 统一 `qkv_proj` 与 `k_eq_v`/共享层权重路由。Norm 和 QKV 的固定
> shape 微测明确变快；激活在 decode 小 shape 仍可能是 launch-bound，未宣称全面提速。
> 83 的 BF16 输出分叉在本次固定复放中不再出现，但 position 70 的 top1 变成精确并列
> （margin=0），**不是“BF16 数值问题已修”**。

## 1. 这关解决什么痛点

B=1 decode 时每一步只有一行 token，GPU 的大部分时间不是算矩阵，而是等小 kernel。
原 `RMSNorm.forward` 的无残差分支约等于：

```text
x.to(fp32) → pow(2) → mean → +eps → rsqrt → x32 * ... → to(dtype) → * weight
```

一次 norm 八步左右，每步读写一遍中间结果；真实 Gemma4 target 35 层，每层 4 个 block
norm，再加 q_norm、非共享层的 k/v norm、final norm 和 PLE norm，单 token 前向有 **242 次
RMSNorm 调用**。QKV 则是 q/k/v 三次小 GEMM；打包成一次更宽的 GEMM 可以减少 launch、
提高张量核利用率，但 Q-only 共享层会多算用不到的 K/V 行，这个取舍必须记录。

本关**不写矩阵乘库**：继续用现有 `F.linear` / `QKVParallelLinear`，只改算子实现、布局和
模型接线。

## 2. 路径映射

| 本项目 | 参考 | 状态 |
|---|---|---|
| `minivllm/layers/ops/layernorm.py` | `vllm/model_executor/layers/layernorm.py` 的 `RMSNorm` / `GemmaRMSNorm` 语义 | 本地 Triton + Torch native |
| `minivllm/layers/ops/activation.py` | `vllm/model_executor/layers/activation.py` 的 `GeluAndMul` / `SiluAndMul` | 本地 Triton + Torch native |
| `minivllm/layers/activation.py` | 同名 `GeluAndMul` / `SiluAndMul` / `get_act_and_mul_fn` | 共享实现；删除 Gemma4 私有 `_GeluAndMul` |
| `minivllm/layers/linear.py::QKVParallelLinear` | `vllm/model_executor/layers/linear.py::QKVParallelLinear` | 复用现有 shard loader；新增加载记账 |
| `minivllm/models/gemma4.py` | `vllm/model_executor/models/gemma4.py` | target 统一 `qkv_proj`、共享层/k_eq_v 路由 |
| `minivllm/models/gemma4_mtp.py` | `vllm/model_executor/models/gemma4_mtp.py` | assistant 保持 Q-only `q_proj`，只换共享激活 |

## 3. A：RMSNorm 的语义先于融合

### 3.1 接口与语义

保留上游签名：

```python
forward(x, residual=None) -> out | (out, residual_new)

无 residual：
    out = x / sqrt(mean(x^2) + eps) * w

有 residual：
    residual_new = x + residual
    out = residual_new / sqrt(mean(residual_new^2) + eps) * w
    return out, residual_new
```

`GemmaRMSNorm` 的权重解释是 `(1 + w)`，与普通 `w` 分开；`has_weight=False` 走真正的
无权重分支，不分配全 1 向量。

### 3.2 实现

`layers/ops/layernorm.py`：

- CPU：`rms_norm_native`，显式 Torch 参考；方差 FP32 归约。
- CUDA：本地 Triton `_rms_norm_kernel`，一行一个 program；读 x/residual、加法、平方均值、
  rsqrt、weight 乘法、写 out/residual，中间不落地整份 FP32 hidden。
- 支持 1D/2D/3D，最后一维连续；2D 支持任意正 row stride（packed QKV 切片），3D 支持任意
  正 batch/head stride（Q/K norm 的 `[tokens, heads, head_dim]`）。
- 输出连续；`residual_new` 是新连续张量，不原地写 caller。
- 若最后一维不连续、3D 之外的 rank、shape/dtype/device 不匹配、weight 长度不符，入口直接
  报错；不做静默复制或回退。

### 3.3 cast 次序

上游 native 的普通 RMSNorm 是：

```python
x32 = x.float(); var = x32.pow(2).mean(-1)
normed = x32 * rsqrt(var + eps)
if weight is not None:
    normed = normed.to(weight.dtype) * weight
return normed.to(x.dtype)
```

本实现逐项保持这条次序。BF16 权重时先 cast 到 BF16 再乘，乘法结果也落回 BF16，最后写
output dtype；这样 BF16 只经历一次权重乘舍入。`GemmaRMSNorm` 先 `weight.float()+1`，在
FP32 里乘，最后回原 dtype。实测 CUDA kernel 与 native 在 BF16 下逐位相同，FP32 只剩
`~1e-6` 的归约差。

### 3.4 与上游的明确差异

1. 上游 CUDA 的 `fused_add_rms_norm` 自定义 op 会**原地**写 input/residual；本实现的
   `residual` 分支返回新 residual。上游 `forward_native` 也返回新张量，本仓库选这条更安全
   的语义；调用方不需要半更新状态。差异已在测试里用“输入不被改写”固定。
2. 本实现没有上游的 `var_hidden_size` / batch-invariant 分支；本仓库当前模型不使用。
3. `RMSNorm` 的 CUDA 是本地 Triton，不是上游 C++ `ops.rms_norm`；输入/输出契约相同，
   数值有 FP32 末位差。

## 4. B：GeluAndMul / SiluAndMul 共享实现

### 4.1 公式

```text
GeluAndMul: out = gelu(gate) * up
SiluAndMul: out = silu(gate) * up
输入最后一维 = [gate | up]，输出宽度减半
```

Gemma4 默认 `hidden_activation="gelu_pytorch_tanh"`，对应
`GeluAndMul(approximate="tanh")`；exact erf 与 tanh 近似不能互换。

### 4.2 实现

`layers/activation.py` 现在是唯一实现：

```python
GeluAndMul(approximate="none" | "tanh")
SiluAndMul()
get_act_and_mul_fn(name)
```

`gemma4_mtp.py` 的私有 `_GeluAndMul` / `_get_act_and_mul_fn` 已删除，`Gemma4MLP` 改用共享
工厂。`SwigluStepAndMul` 的 clamp 语义保持原样，没有顺便塞进这两个类。

CUDA kernel 在 `layers/ops/activation.py`：

- 输入支持 1D/2D/3D、最后一维连续、外层正 stride；
- 输出连续；
- decode（行数 ≤ 8 且 half > 4096）按列切 4096 的块，让单行也能占多个 CTA；
- prefill（行多）一行一个 program，减少调度层数；
- `gelu` 的 tanh 分支用 `2*sigmoid(2z)-1` 表示 tanh，避免依赖不同 Triton 版本的
  libdevice 绑定；exact 分支用 `tl.erf`。

### 4.3 数值

- CPU native 与上游 `GeluAndMul.forward_native` / `SiluAndMul.forward_native` 在 FP32 下
  逐位相同。
- CUDA FP32 与 Torch 参考有 `≤5e-7` 的末位差；BF16 写回与“FP32 参考再 cast BF16”逐位
  相同。
- 没有把 BF16 native 的 `F.gelu` 输出当成 CUDA kernel 的逐位参考；两者都保留，差异在
  `max_abs_error_vs_fp32_reference_bf16` 中记录。

## 5. C：Gemma4 target packed QKV

### 5.1 结构

`Gemma4Attention` 现在所有层都走一个 `QKVParallelLinear`：

```python
qkv, _ = self.qkv_proj(hidden_states)
q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
```

输出维按 `[num_heads*head_dim, num_kv_heads*head_dim, num_kv_heads*head_dim]` 切；GQA 下
不能三等分。`q_size/kv_size` 从 packed 层读取，不再由模型手算两套。

### 5.2 权重路由

`Gemma4ForCausalLM.load_weights` 新增：

```text
.q_proj → .qkv_proj 的 q shard
.k_proj → .qkv_proj 的 k shard
.v_proj → .qkv_proj 的 v shard
```

`QKVParallelLinear.weight_loader` 复用原有的 shard 写入；新增 `_shard_load_counts`：

- 非共享层加载完后必须看到 `{q,k,v}`；
- 共享层 forward 只消费 Q，允许 checkpoint 只提供 `q_proj`（HF tiny 文本栈就是这样）；
  真实 target checkpoint 里存在的 K/V 仍会装进 qkv 的 K/V 行，不丢权重；
- 同一 shard 被覆盖两次 → 明确报错；缺 shard → 明确报错，不留下未初始化行。

### 5.3 共享层

上游 Gemma4 target 的共享层也保留 `qkv_proj`；本实现对齐这一点。forward 中：

```text
共享层：q_norm → RoPE(Q) → attention(q, k, v) 的 Q-only 路径
        K/V 不归一化、不旋转、不写入
非共享层：q_norm → k_norm → RoPE(Q,K) → v_norm → attention
```

与上游的一处明确差异：上游还会为共享层构造 `k_norm` / `v_norm` 参数；本实现省略这两个
forward 不消费的参数。真实 checkpoint 里若有对应权重，加载器**显式跳过**，不把“没有参数
可装”伪装成已加载；这是为了同时兼容本机 tiny HF 文本栈（共享层没有 k_norm/v_norm 权重）
和真实 checkpoint（有这些权重）。

### 5.4 `attention_k_eq_v`

真实 E2B 的 `attention_k_eq_v=False`；本关仍按上游实现合成路径：

```text
full 层 checkpoint 只有 k_proj，没有 v_proj
  → loader 把 K 权重复制到 V 槽
  → forward 对原始 V 走无权重 v_norm
```

不是把已经过 `k_norm` 和 RoPE 的 `k` 直接赋给 `v`。测试专门比较了：

```text
expected_v = RMS(raw_k)                # 无权重
capture.v  = 正确
capture.k  = k_norm(raw_k) + RoPE      # 与 v 明显不同
```

### 5.5 assistant 不变

`Gemma4MTPAttention` 仍是 Q-only `q_proj`，没有为了统一类名制造 K/V；测试断言 assistant
每层没有 `qkv_proj` 且 `attn.q_only=True`。真实 assistant 77.7M 的加载/短生成由 80E/81
回归覆盖。

## 6. 测试与实跑

### 6.1 新增测试

`tests/step85/` 三个文件，实跑 **31 passed**：

| 文件 | 覆盖 |
|---|---|
| `test_layernorm_ops.py` | CPU 与上游 `ir.ops.rms_norm` 逐值；CUDA vs native；2D/3D、FP32/BF16、residual/no-weight/Gemma；packed slice stride；CUDA Graph；错误布局 |
| `test_activation_ops.py` | CPU 与上游 native；CUDA vs FP32 参考；exact/tanh/silu；2D/3D 非连续；错误维度 |
| `test_qkv_packing.py` | packed loader q/k/v；共享层 qkv + 跳过 K/V；k_eq_v K→K/V；缺 shard/重复覆盖；v=v_norm(raw_k) 而非 rotated k；assistant Q-only |

CPU-only 实跑：

```text
CUDA_VISIBLE_DEVICES='' python -m pytest tests/step85 -q
11 passed, 20 skipped
```

（19 个 CUDA kernel/图用例按设备跳过；CPU native 路径实跑，不是全 skip。）

### 6.2 命令

```bash
PY=/home/user/anaconda3/envs/vllm-omni-dev/bin/python
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY -m pytest tests/step85 -q
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY -m pytest \
  tests/step58 … tests/step85 -q
$PY benchmarks/check_step85_ops.py --all
```

全量回归（含真实 Gemma4 target+assistant、K=0/K=2、V1/V2、Graph）最终 **1231 passed,
0 failed, 0 skipped, 148 warnings**。真实 target 的 35 层、2 个 KV group、20 个 Q-only
共享层，assistant 4 层源映射测试都实际执行通过。

### 6.3 真实 target 加载 smoke

真实 bf16 target 加载后检查：

| layer | type | shared | qkv shape | q/k/v finite |
|---:|---|---:|---|---:|
| 0 | sliding | 否 | `(2560, 1536)` | 是 |
| 14 | full | 否 | `(5120, 1536)` | 是 |
| 15 | sliding | 是 | `(2560, 1536)` | 是 |
| 20 | sliding | 是 | `(2560, 1536)` | 是 |
| 34 | full | 是 | `(5120, 1536)` | 是 |

真实 checkpoint 的共享层 K/V 权重仍然存在并装进 qkv；forward 不消费。Q-only 层数量与
80D/81 的 20 层一致。

## 7. 性能微测（固定 shape，非端到端）

`benchmarks/check_step85_ops.py --micro`：3 次独立运行，每次 5 个 CUDA event loop 取
中位数，再取 3 次中位数；仍包含 Python launch 间隙。**单机噪声明显，下面是量级证据，
不是性能排名。**

单位 μs/call；ratio = fused/native（越小越好）：

### Norm

| case | fused | native | ratio |
|---|---:|---:|---:|
| decode hidden 1×1536 | 34.40 | 92.12 | 0.373 |
| prefill hidden 128×1536 | 50.87 | 85.32 | 0.596 |
| decode q_norm 1×8×256 | 36.45 | 92.44 | 0.394 |
| decode q_norm 1×8×512 | 39.16 | 84.63 | 0.463 |
| decode v_norm 1×512 无权重 | 30.07 | 83.32 | 0.361 |
| decode Gemma 1×1536 | 25.13 | 101.33 | 0.248 |
| decode hidden+residual 1×1536 | 33.75 | 120.56 | 0.280 |

所有固定 shape 的中位数都低于 Torch 分解路径；与 kernel 计数一致（见 §8）。

### 激活

| case | fused | native | ratio |
|---|---:|---:|---:|
| decode gelu 1×12288 | 31.41 | 28.12 | 1.117 |
| decode gelu 1×24576 | 29.54 | 25.85 | 1.143 |
| prefill gelu 128×24576 | 34.13 | 32.28 | 1.058 |
| decode silu 1×17920 | 27.89 | 18.79 | 1.484 |

**不掩盖：decode 激活的融合 kernel 中位数没有超过 Torch 分解路径。** 原因不是“融合没
生效”，而是 1 行、24–48 KB 的输入下，Triton Python launcher/单次 launch 开销大于两个
高度优化的小 CUDA kernel；prefill 的收益也不稳定。要证明端到端收益，必须看 88 的公平矩阵，
不能拿这组数外推。

### QKV GEMM

| case | packed | 3×GEMM | ratio |
|---|---:|---:|---:|
| decode sliding 1× (q=2048, kv=256) | 13.67 | 31.80 | 0.430 |
| decode full 1× (q=4096, kv=512) | 17.41 | 38.71 | 0.450 |
| prefill sliding 128× | 36.78 | 73.17 | 0.503 |

这是纯 GEMM 名义对比；真实 target 有 20/35 个 Q-only 共享层，packed QKV 会多算这些层
用不到的 K/V 行，净收益要按整层/端到端看，不能只看这三行。

## 8. 真实 K=0 eager 8-step trace

`benchmarks/check_step85_ops.py --trace`：真实 target、K=0、eager、单 prompt、预填后
正好 8 个 decode step，profiler 只用于数 kernel/launch。

```text
prompt_len = 18
tokens = [1018, 57137, 236743, 82056, 237221, 57137, 47910, 237214, 59013]
build_s = 5.30
profiled_decode_steps = 8
gpu_kernel_events = 15728
cpu_kernel_launch_events = 11504
distinct_gpu_kernels = 54
_rms_norm_kernel events = 1936 = 8 × 242
_act_and_mul_kernel events = 280 = 8 × 35
per_step gpu_kernel_events = 1966
per_step cpu_kernel_launch_events = 1438
memory.allocated = 9,665,132,544 B
memory.reserved = 10,175,381,504 B
memory.peak_allocated = 10,111,753,216 B
```

242 = 35×4 block norm + 35 q_norm + 15×2 k/v norm + final norm + PLE projection norm +
35 post-per-layer norm。两个 kernel 的计数与理论值逐 token 对齐，说明它们确实在热路径，
不是“代码里写了但没调用”。

CPU launch 事件（11504）少于 GPU kernel 事件（15728），是 PyTorch profiler 对 CPU
launch 采集的合并/丢失；**以 GPU kernel 事件为准**，不把它解释成“有 4224 个 kernel 没
launch”。

## 9. 83 固定历史复放与分歧状态

命令：

```bash
$PY benchmarks/check_step83_numerics.py --max-new-tokens 64 \
  --output-dir benchmarks/results/step85/replay83/bf16 \
  --dtype bfloat16 --blocks 512 --fp32-final-control
$PY benchmarks/check_step83_numerics.py --max-new-tokens 64 \
  --output-dir benchmarks/results/step85/replay83/fp32 \
  --dtype float32 --blocks 64
```

### BF16

```text
first_diff = None（K=0/K=2 64 token 输出完全相同）
target top1 在 64 个共同 position 上 64/64 相同
相对 step83 旧 trace：K=0 改 2 个 token（第 45/46 个）
  57208,25352 → 25352,33988
相对 step83 旧 trace：K=2 改 0 个 token
max final hidden relative L2 = 2.42e-2
max layer hidden relative L2 = 2.12e-2（model.layers.11）
position 70：K=0 top1=57208，margin=0.0，top values 15.6875 / 15.6875
```

**状态：83 原始“输出第 45 个 token 分叉”在 84/85 后的这次固定复放中不再出现；但不是
BF16 已修。** position 70 的 top1 已经精确并列，说明改变的是 near-tie 落在哪一侧；只要
换 prompt/长度/批布局，仍可能翻。相对 step84 的 64 token 输出，本次 0 变化。

### FP32

```text
first_diff = None
target top1 64/64 相同
相对 step83：K=0/K=2 都 0 个 token 变化
max final hidden relative L2 = 2.12e-6
max layer hidden relative L2 = 1.79e-6（model.layers.11）
position 70 margin：0.06539631 → 0.06539345
```

FP32 仍有 `~2e-6` 的归约差（FMA/归约顺序），但 token 与 top1 一致。**83 的结论保留：
不是 assistant/拒绝采样状态错误；BF16 near-tie 是数值边界。**

## 10. 已知边界/未跑项

1. **只做单卡、TP=1、纯文本**；没有量化、并行、多模态、PD、LoRA、Mamba。
2. **没有做 torch.compile**；本关只换算子/布局。
3. **没有接 87 的完整 Graph 系统**；只验证了单算子 CUDA Graph capture/replay，以及全量
   回归里已有的 Qwen3/Gemma4 图路径。
4. **没有跑完整 B×K 性能矩阵**；公平对照属于 88。本关 micro 含 Python launch 间隙，
   trace 的 profiler 有开销，不能当 tok/s。
5. **激活 decode 小 shape 可能更慢**；已如实记录，不承诺所有 shape 加速。
6. **共享 target 层省略 k_norm/v_norm 参数**；上游会建。真实 checkpoint 的对应权重被
   显式跳过；真实 target 的 K/V 权重仍装入 qkv，forward 不消费。
7. **`attention_k_eq_v` 真实 E2B=False**；K→K/V 路由是合成配置覆盖，不是当前真实热路径。
8. **RMSNorm residual 不原地写**，与上游 CUDA 自定义 op 不同；测试固定“输入不变”。
9. 真实模型结果只覆盖固定 revision、固定 prompt/8–64 token；不能外推到其他 checkpoint。
10. BF16 仍不是逐位等价；FP32 也不是 bitwise，只是 token/top1 一致。

## 11. 一句话

**85 把 decode 热路径上的 242 次 RMSNorm 和 35 次 GeGLU 收敛成两个本地融合 kernel，
把 Gemma4 target 的 q/k/v 三次 GEMM 收敛成一个 packed QKV，并把共享层、k_eq_v 和权重
shard 对账接回上游结构；Norm/QKV 的固定 shape 微测明显变快，激活在小 decode shape 仍是
launch-bound，83 的 BF16 分叉只是被舍入推到另一侧而不是修好。**
