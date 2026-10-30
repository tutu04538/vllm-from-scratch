# step84：RoPE 共享缓存与融合 Q/K 旋转

> 需求：`单卡执行优化需求/084_RoPE共享缓存与融合执行.md`
> 上游基线：vLLM 0.28.0（本机安装快照）
> 实测日期：2026-10-30
> 包摘要：`6e7481e59bd45326f997cfb56f6e28dff0f9defd420f2e3b1052906884fbb325`（175 个 `.py`）

## 1. 这一关解决什么

旧实现有两个叠加浪费：

1. **每层各建一张表**：`Gemma4Attention` 每层直接 `RotaryEmbedding(...)`，target 35 层 +
   assistant 4 层共 39 份；相同 sliding/full 配置的表内容完全一样。
2. **表里把 cos/sin 各复制了一份**：旧 `cos_sin_cache` 宽 `2 * apply_dim`，因为先
   `cat(freqs, freqs)` 再 `cat(cos, sin)`。上游紧凑布局只需要宽 `apply_dim`。

本关把两件事一起做：

```text
按完整配置共享一张表 + 紧凑半宽布局 + 融合 Q/K 原地旋转
```

不把 `max_position` 从 131072 改小来省显存，也不改 attention/scheduler/模型数学。

## 2. 路径与改动

| 本项目 | 上游参考 | 改动 |
|---|---|---|
| `minivllm/layers/rotary_embedding.py` | `model_executor/layers/rotary_embedding/{__init__,base,gemma4_rope}.py` | `get_rope`、`RotaryEmbedding`、`Gemma4RotaryEmbedding`；共享弱缓存；紧凑表；dtype/device 转换保共享 |
| `minivllm/layers/ops/rotary_embedding.py` | `RotaryEmbedding.forward_cuda` / `ops.rotary_embedding` 的输入输出约定 | 本地 Triton 原地 Q/K 旋转 + 显式 Torch 参考；支持 Q-only、partial、proportional、任意 2D stride |
| `minivllm/layers/ops/__init__.py` | 上游 kernel 目录边界 | 新子包 |
| `minivllm/layers/__init__.py` | 上游 re-export | 导出 `get_rope`、`Gemma4RotaryEmbedding`、缓存统计 |
| `minivllm/models/gemma4_mtp.py::_make_rotary_embedding` | `models/gemma4.py` / `models/gemma4_mtp.py` 的 `get_rope(...)` | 只转发配置，不再自行建表 |
| `minivllm/models/qwen3.py` | Qwen3 `get_rope` | 迁到统一工厂 |
| `minivllm/models/qwen3_eagle3.py` | EAGLE3 `get_rope` | 迁到统一工厂 |
| `minivllm/models/qwen3_dflash.py` | DFlash `get_rope` | 迁到统一工厂 |
| `minivllm/models/step3p5.py` | Step3p5 partial RoPE | 迁到统一工厂（`rope_dim=head_dim*partial`） |
| `minivllm/testing/torch_dflash.py` | 测试参考 | 上下文 K 改用 `apply_to_key`；不再把同一张量同时当 q/k |

所有模型调用点统一走 `get_rope`；旧 `RotaryEmbedding(..., proportional=...)` 的直接构造路径
已迁移，没有保留第二套实现。

## 3. 实施 A：共享与紧凑表

### 3.1 Cache key

`get_rope(head_size, max_position, is_neox_style=True, rope_parameters=None, dtype=None,
device=None)` 的 key 由以下部分组成：

```text
head_size
declared_rotary_dim            # rope_dim 或 int(head_size * partial_rotary_factor)
max_position
base                           # rope_theta
is_neox_style
rope_type                      # default / proportional
frozen(rope_parameters)        # 嵌套 dict/list 递归冻结、key 排序
dtype
device                         # cuda 统一成 cuda:<index>
```

`rope_parameters` 只做一层 `dict()` 复制 + 递归冻结，**不修改调用方字典**。同一 key 返回同一份
`cos_sin_cache` storage；base、rope type、partial/rope_dim、max_position、旋转风格、dtype、
device 任一不同都不共享，即使 shape 相同。

只实现 `default` 与 `proportional`。`llama3`/`yarn`/`mrope` 等在工厂处显式报错，不静默按
default 算。

### 3.2 紧凑布局

表形状：

```text
[max_position, apply_dim]
前半 [0 : apply_dim/2] = cos
后半 [apply_dim/2 : apply_dim] = sin
```

- 普通 RoPE：`apply_dim = declared_rotary_dim`，未旋转维不进表。
- Gemma4 proportional：`apply_dim = head_size`；前 `rope_angles=rotary_dim/2` 对用
  `head_size` 做频率分母，剩余 `nope_angles` 对补零，于是 `cos=1/sin=0`，是恒等旋转。
  不能把 cos/sin 直接写成 0。

表先在 CPU FP32 上按上游 `_compute_cos_sin_cache` 的公式生成，再一次性 `.to(dtype, device)`。
这与上游构造时的 CPU 表逐值一致；测试在 position 0 / 中间 / `max-2` / `max-1` 与整表
逐位对照。

### 3.3 生命周期

上游 `_ROPE_DICT` 是进程级**强引用**字典，靠进程生命周期兜底。本项目的差异：

```text
_ROPE_TABLE_CACHE: weakref.WeakValueDictionary[key, _RopeTable]
```

- 两个活 module 真正共享同一块 storage；不是每次返回副本。
- 缓存只弱引用表对象，**不持有 module/engine**；最后一个使用者释放后条目自动消失。
- 不同引擎同一 device/dtype/数学配置可以共享同一张不可变表；这是安全的，不会把旧引擎
  保活。
- 若外部测试/调用方仍持有 module，表当然还在；这是引用计数语义，不是泄漏。

### 3.4 dtype/device 转换

`RotaryEmbedding._apply(fn)` 先暂时摘掉 `cos_sin_cache`，让 `nn.Module._apply` 处理参数/其他
buffer，再用一个空 probe 推断 `fn` 的目标 dtype/device，重新向工厂取 canonical 表。这样：

- 不会对每层复制整张表；
- `model.to(bfloat16)` / `model.to("cuda")` 后，同 key 的层仍然共享；
- 转换回 FP32 也仍共享；
- 前向时 cache 与 query 的 dtype/device 不一致会明确报错，而不是每层偷偷 `.to()`。

## 4. 实施 B：融合 Q/K 旋转

### 4.1 入口与语义

生产入口：

```python
apply_rotary_embedding(positions, query, key, cos_sin_cache,
                       head_size, rotary_dim, is_neox_style=True)
```

- **CUDA**：一个 Triton kernel 同时处理 Q 和 K；原地写回并返回传入张量（对齐上游
  `ops.rotary_embedding` 的原地语义）。`key=None` 只旋转 Q（Gemma4 Q-only）。
- **CPU**：显式 Torch 参考，返回新张量（对齐上游 `forward_native`）。
- `rotary_dim` 是“实际应用宽度”：普通 partial 是声明的 rotary_dim，proportional 是
  `head_size`（表宽也是 `head_size`）。

kernel grid = `(num_tokens, max(q_heads, k_heads))`。每个 program 只读自己的 position、
cos/sin 两半，并写自己 head 的 Q/K；Q/K 头数不同时，多出来的 program 只做 Q 或只做 K。

### 4.2 stride 与 packed QKV

2D 输入支持任意正的 token/列 stride：

```text
qkv = ...                                  # [N, (Hq + 2*Hk) * D]
q = qkv[:, :Hq*D]                          # token stride = 整条 qkv 宽
k = qkv[:, Hq*D:(Hq+Hk)*D]
v = qkv[:, (Hq+Hk)*D:]
```

kernel 只写 Q/K 的列，V 区域逐字节不变（测试覆盖）。列 stride 为 2 的视图也直接支持，
不需要在每层 `.contiguous()`。零/负 token stride、零列 stride、query/key 行数不一致、
query/key 同起始地址都会在入口拒绝，避免重叠写。

### 4.3 精度

kernel 内把 Q/K、cos、sin 都扩到 FP32 计算，写回时按输入 dtype 只舍入一次；cos/sin 表
本身是 BF16/FP32。BF16 下与旧 Torch 路径按 BF16 逐步舍入会有中间 ulp 差异；FP32 下与
Torch 参考的 max abs 差约 `2e-7`，不是逐位相等（FMA/规约顺序不同）。测试按容差比较，
没有把“接近”写成“逐位相同”。

### 4.4 越界与 CUDA Graph

- 位置越界用 Triton `device_assert` 在设备端报错，不需要 host `.item()`，不新增 clamp。
  代价是越界会触发 device-side assert，**会污染该进程的 CUDA context**；所以越界测试在
  独立子进程跑。
- 位置读取、查表、写回都在 kernel 内；没有每层完整 cos/sin gather 中间张量，没有
  `.tolist()`/`.item()`。
- 单次 CUDA Graph capture + 两次 replay（改变 positions/输入）验证读的是新位置，不是
  capture 时的旧表/旧位置。

## 5. 测试

```bash
python -m pytest tests/step84 -q
# 46 passed
```

| 文件 | 数量 | 覆盖 |
|---|---:|---|
| `test_rope_cache_key.py` | 17 | 同 key 共享、不同数学配置不共享、嵌套 key 稳定、输入不改、紧凑布局、dtype/device 转换保共享、弱缓存释放、引擎 del 后不保活、非法配置 |
| `test_compact_cache_upstream.py` | 14 | 与上游 `RotaryEmbedding`/`Gemma4RotaryEmbedding` 整表逐位、position 0/中间/大位置/最后合法位置、native Q/K 逐位、neox/gptj、partial/proportional、越界报错 |
| `test_fused_rope_ops.py` | 12 | Q/K、Q-only、partial/proportional、BF16/FP32、packed QKV 不写 V、列 stride、CUDA Graph 两次 replay、非法布局、CUDA 越界子进程 |
| `test_gemma4_shared_rope.py` | 2 | 真实 config 的 35+4 层 → 2 张表；BF16 转换后仍 2 张；按布局算唯一字节 |
| `test_rope_engine_lifecycle.py` | 1 | tiny V2 引擎 shutdown/del 后 RoPE module 与缓存条目释放 |

另：`tests/step76/test_dflash_kv.py` 的参考实现从 `rotary_emb(pos, k, k)` 改为
`apply_to_key(pos, k)`。原因不是放宽断言，而是融合路径原地写回，同一个张量不能同时当
Q/K；改后该文件 **11 passed**。

## 6. 实测

命令：

```bash
PY=/home/user/anaconda3/envs/vllm-omni-dev/bin/python
PATH=/home/user/anaconda3/envs/vllm-omni-dev/bin:$PATH
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

$PY benchmarks/check_step84_rope.py --all \
  --output-dir benchmarks/results/step84
```

结果：6 个隔离 case 全部 `ok`。

### 6.1 表审计（真实 config，不加载权重）

target 35 层（28 sliding + 7 full）+ assistant 4 层（3 sliding + 1 full），
`max_position=131072`，BF16：

| 项目 | 数值 |
|---|---:|
| RoPE module 数 | 39 |
| 唯一 key / storage | **2** |
| sliding 表 | `[131072, 256]` BF16，64 MiB |
| full/proportional 表 | `[131072, 512]` BF16，128 MiB |
| 唯一字节 | **192 MiB** |
| 新紧凑布局“每层一份”名义字节 | 3008 MiB |
| 旧布局“每层一份”名义字节 | 6016 MiB |

说明：

- 旧布局宽 `2 * apply_dim`（sliding 512、full 1024），所以 39 份名义上是 6016 MiB；
- 新紧凑布局宽 `apply_dim`（sliding 256、full 512），39 份名义上是 3008 MiB；
- 共享后只剩 2 份，**192 MiB**；
- target 单独的旧名义值是 5.25 GiB；本关只把它当布局计算结果，不在测试里写死这个数。

### 6.2 真实 target + assistant 短生成

固定 revision 的真实权重，K=2、单 prompt、8 token。eager 与 FULL_DECODE_ONLY graph
各在独立子进程里跑：

```text
eager tokens = [1018, 57137, 47910, 236743, 55118, 237184, 1018, 108]
graph tokens = [1018, 57137, 47910, 236743, 55118, 237184, 1018, 108]
```

模型内审到 **39 个 RoPE module、2 张表、192 MiB 唯一 storage**。显存：

| 阶段 | eager allocated / reserved | graph allocated / reserved |
|---|---:|---:|
| build 后 | 9,743,730,688 / 10,422,845,440 | 9,810,851,840 / 10,582,228,992 |
| 峰值 allocated | 10,284,393,472 | 10,284,393,472 |
| shutdown + del + empty_cache 后 | 33,554,432 / 268,435,456 | 33,554,432 / 268,435,456 |

`allocated` 是整模型（权重/KV/workspace）总量，不是 RoPE 单独占用；RoPE 的 192 MiB 只是
其中一个已由唯一 storage 审计确认的子集。shutdown 后进程内的 RoPE 弱缓存条目为
`entries=0, bytes=0`；残留的 32 MiB allocated / 256 MiB reserved 是 CUDA/allocator
缓存，不是 RoPE 表。

### 6.3 RoPE 小型微测

单进程 CUDA event，BF16，每项 5 次计时取中位数；包含 Python launch 间隙，**不是**
kernel-only，也不是端到端吞吐：

| case | tokens | Q/K heads | fused μs/call | native μs/call | 比值（含 launch） | max abs err |
|---|---:|---:|---:|---:|---:|---:|
| decode sliding | 1 | 16 / 8 | 34.88 | 193.42 | 0.18 | 0.015625 |
| decode sliding Q-only | 1 | 16 / 0 | 28.77 | 97.28 | 0.30 | 0.015625 |
| decode full Q-only | 1 | 8 / 0 | 24.20 | 130.55 | 0.19 | 0.015625 |
| prefill sliding | 128 | 16 / 8 | 17.67 | 207.86 | 0.085 | 0.03125 |

这些数字只能说明“本次单进程里 fused 入口的 per-call 时间更低”，不能外推成端到端加速；
需求也明确不做完整性能矩阵。

### 6.4 复放 83 的固定历史

用同一 prompt、64 token、同一 K=0/K=2 配置重跑 `check_step83_numerics.py`，与
`benchmarks/results/step83/{bf16,fp32}` 的旧 trace 比较。

#### BF16

| 指标 | 旧 83 | 新 84 |
|---|---:|---:|
| K=0/K=2 第一分歧 | 45 | **None（64 token 相同）** |
| K=0 token 变化数（vs 旧） | — | 2（位置 45、46） |
| K=2 token 变化数（vs 旧） | — | 0 |
| K=0 共同位置 top1 相同 | — | 62 / 64 |
| K=2 共同位置 top1 相同 | — | 65 / 65 |
| position 70 K=0 top1 / margin | 57208 / 0.0625 | 25352 / 0.0625 |
| position 70 K=2 top1 / margin | 25352 / 0.25 | 25352 / 0.3125 |

新 K=0 与 K=2 的 64 个 token 完全相同；变化集中在旧的分歧点：

```text
旧 K=0: ... 237075, 57208, 25352, ...
新 K=0: ... 237075, 25352, 33988, ...
```

`K=2` 的 token 完全不变，但逐层 hidden 仍有变化：`max final hidden relative L2 =
0.0320`，`max layer hidden relative L2 = 0.0454`（`model.layers.13`），
第一层 >1e-3 出现在 `model.layers.0`。这来自 BF16 下融合 kernel 的 FP32 计算/一次舍入
与旧 Torch 路径逐步舍入的差异，不是缓存/位置错误。

`K=0` 的 `max final hidden relative L2=0.6118`、layer 1.0244 不能与 `K=2` 的 0.0320
直接比较：它在第 45 个 token 已经走了不同历史，后面是两条不同轨迹。

**这不是通用精度修复**。本次固定历史里 K=0 跨过了 near-tie，与 K=2 一致；换 prompt/长度
仍可能有不同翻转。

#### FP32

| 指标 | 旧 83 | 新 84 |
|---|---:|---:|
| K=0/K=2 第一分歧 | None | None |
| K=0 token 变化数 | — | 0 |
| K=2 token 变化数 | — | 0 |
| K=0 共同位置 top1 相同 | — | 64 / 64 |
| K=2 共同位置 top1 相同 | — | 65 / 65 |
| K=0 max final hidden relative L2 | — | 2.82e-6 |
| K=2 max final hidden relative L2 | — | 2.78e-6 |
| K=0/K=2 position 70 margin | 0.0653963 / 0.0653954 | 0.0653849 / 0.0653849 |

FP32 下 64 个 token 与旧 83 完全一致，top1 全同；hidden 的 `~2.8e-6` 差异与融合 kernel
的 FMA/规约顺序一致，没有出现 token 级变化。

## 7. 与上游的差异账本

| 编号 | 差异 | 说明 |
|---|---|---|
| D84-1 | 缓存生命周期用弱引用，不是进程级强引用 | 上游 `_ROPE_DICT` 强引用；本仓库要求 shutdown 不保活旧引擎。两个活 module 仍真正共享 storage |
| D84-2 | dtype/device 转换由 `_apply` 重新取 canonical 表 | 上游 `_match_cos_sin_cache_dtype` 可能在实例上替换 buffer；本仓库优先保持跨层共享，前向不再每层 `.to()` |
| D84-3 | CUDA kernel 是本地 Triton，不是上游 C++ `ops.rotary_embedding` | 输入/输出/原地/stride 语义对齐；不 import 上游生产 kernel。FP32 与上游 native 表/输出逐位一致，fused 与 native 约 2e-7（FMA） |
| D84-4 | 越界用 `device_assert` | 上游 C++ 路径不保证 host 可读的越界错误；本仓库加设备端边界检查，越界会污染 CUDA context。只在非法输入触发 |
| D84-5 | 只支持 default/proportional | 其他 rope_type 显式报错；不实现 dual_chunk/mrope/yarn 等 |
| D84-6 | key 包含完整 `rope_parameters` | 与上游一致；语义等价但写法不同的冗余配置可能拆表，不能只看 shape 就共享 |
| D84-7 | CPU 路径返回新张量，CUDA 路径原地 | 与上游 `forward_native` vs `forward_cuda` 的差异同型；模型调用方两种都接 |

## 8. 已知边界/未跑项

1. **没有跑 B×K 完整性能矩阵**；微测包含 Python launch 间隙，不据此给端到端加速比。
2. **BF16 不是逐 token 等价**；84 改变了部分舍入，本次 K=0 的历史与 K=2 合并，但不能外推。
3. **FP32 也不是逐位**；hidden 相对差 `~2.8e-6`，token/top1 全同。
4. **越界会污染 CUDA context**；生产输入协议必须保证 `0 <= position < max_position`。
5. **弱缓存是进程级**：同一 device/dtype/数学配置的不同活引擎共享同一张不可变表；若外部
   仍持有旧 module，表不会提前释放。这是引用计数语义。
6. **`_apply` 只保证 dtype/device 转换**：任意 `_apply(fn)` 的值变换不会重算表。
7. **只测文本路径**：Gemma4 视觉/音频塔未加载；TP/量化未接。
8. **未实现的 RoPE 变体**：`llama3`/`yarn`/`mrope` 等仍按显式不支持处理。
9. **没有独立测量表构建的 CPU 峰值**；审计按唯一 storage 和真实模型 `allocated` 记录。
10. **真实 target 的 35 层是当前 config**；其他 Gemma4/Step3p5 checkpoint 需要重新按 key 审计。

## 9. 一句话

**84 把 39 层各自一份的 6016 MiB 名义表，按完整配置压成 2 份、192 MiB 唯一 storage，
并把旧 Torch 的多步 Q/K 旋转换成一个原地 Triton 入口；真实 target+assistant 的 eager
与 CUDA Graph 短生成通过，FP32 复放 83 历史 64 token 完全相同，BF16 只改变 near-tie 附近
的舍入，不承诺逐 token 等价。**
