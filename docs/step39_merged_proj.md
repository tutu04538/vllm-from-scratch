# step39：把每层 7 次投影合并成 4 次

- 对应代码：`step39/`（新增，从 `step38/` 复制）
- 包摘要 SHA256：`250a22f423ddad3c33e4d2f25ec6059977d198cf4457fb2761b41f797eabae7d`（14 个 .py / 2602 行）
- 基线：`step38/`，指纹 `45c0e5c6…`（14 个 .py / 2511 行），原样保留未改
- 改动文件：`model.py`、`norm.py`、`__init__.py`、入口改名 `step39.py`

## 0. 需求大概

第 36 关的定位文档点名过这一处：Q/K/V 与 gate/up 各自执行投影，而 vLLM 用
`QKVParallelLinear` 与 `gate_up_proj`。第 38 关之后，decode 一步的预算里投影 GEMM 占
**76% 的 GPU 时间、196 次 launch**——同一件事被拆成了 196 次。

```text
现在： q_proj  k_proj  v_proj  o_proj  gate_proj  up_proj  down_proj
新：   qkv（一次）      o_proj   gate_up（一次）        down_proj
```

约束：不改 attention、KV、调度、采样、RoPE；`o_proj`/`down_proj` 不动；
不改磁盘格式与 `formats/qwen3.py` 认的外部键名；`model_config()` 字段不变；
`use_qk_norm` 的顺序不变（先拆 q/k，再各自 norm，再 RoPE）；
合并权重**不能在每次 forward 里重新 `torch.cat`**；显存不能悄悄翻倍。

## 1. 核心设计：一块存储，多个视图

关键约束是这样推出来的：

1. **不能缓存一份合并副本**。看起来那是「构造时 cat 一次、forward 用缓存」的简单做法，
   但验收工具里有大量用例**在构造之后原地改权重**：

   ```python
   e.model.layers[0].q_proj.weight.add_(0.123)     # 10+ 个 process_roundtrip 脚本
   e.model.v_proj.weight[0, 0] = 1
   ```

   缓存的合并副本在那一刻就**陈旧**了，forward 会拿旧权重算出错误结果，而且不报错。
   这比多占显存严重得多。

2. **不能改内部键名**。`q_proj` / `k_proj` / `gate_proj` 这些名字被约 110 个验收文件
   按名引用（`w['layers.0.q_proj.weight']`、`layer.q_proj.weight`），改名会打掉一大片
   既有套件。

于是只剩一条路：**合并权重与三个 Parameter 共用同一块存储**。

```python
storage = torch.cat([q, k, v], dim=0).contiguous()   # 构造时做一次
self.register_buffer("_qkv_storage", storage, persistent=False)
self.q_proj.weight = nn.Parameter(storage[:nq])       # 变成视图
self.k_proj.weight = nn.Parameter(storage[nq:nq + nk])
self.v_proj.weight = nn.Parameter(storage[nq + nk:])  # 原存储到这里释放
```

于是下面这些行为**全部不变**，因为它们本来就是同一块内存：

| 行为 | 为什么不变 |
|---|---|
| `layer.q_proj.weight.add_(x)` | 原地操作写穿到存储 |
| `load_state_dict({"...q_proj.weight": t})` | 逐参数 `copy_`，同样写穿 |
| `state_dict()` | 键名不变，取到的是视图，值正确 |
| `save_model()` | 对每个张量 `.contiguous()` 另存，值正确 |
| 显存（稳态） | **存储元素数 == 三个视图之和，零额外开销** |

### 1.0 显存要分「稳态」和「构建期瞬时」两笔账

三个独立分配要变成一整块连续存储，**中间必然有一次拷贝**——搬的过程中源和目标同时存在。
这一点原文只写了「零额外显存」，不完整，实测补上（Qwen3-0.6B 形状的一层）：

| | 稳态 | 构建**峰值** |
|---|---:|---:|
| step38 | 60.01 MB | 60.01 MB |
| step39 | 60.01 MB | **100.01 MB** |

差的 40.0 MB = 10.49M 元素 × 4 字节（构造时还是 FP32），就是那次 `torch.cat`。

**但它不累积**：`_fuse_projections` 是逐层调用的，方法返回时旧的三个 Parameter 引用消失、
旧存储立即释放，所以任意时刻只有一层的副本存在——峰值是 **+40 MB，不是 28 层累加的 +1.1 GB**。

**端到端实测连峰值都没有差异**：

```text
版本        构建后稳态      构建峰值      跑完峰值
step38     2349.8 MB     2349.8 MB    2447.0 MB
step39     2349.8 MB     2349.8 MB    2447.0 MB
```

构建期那 40 MB 之所以看不出来，是因为 `_load_weights_into` 会先把整份权重转成 BF16 放进一个
字典（`{name: t.to(model.dtype) ...}`），那个瞬时拷贝大得多，把高水位盖住了。

**一个容易看错的地方**：step39 下 `sum(p.numel() for p in parameters())` 是 15.73M，
再加上 buffer 的 10.49M = 26.2M，看起来翻倍了——那是**重复计数**，因为三个 Parameter
就是 buffer 的视图，实际分配仍是 15.73M（60.01 MB，与 step38 相同）。

### 1.1 `.to()` 会打断共享，用 `_apply` 补回来

`.to(device/dtype)` 会把每个 Parameter **各自**搬走，视图与存储的共享关系随之断掉。
存储本身带着同一份数据一起被搬走，所以搬完把视图重新指过去就恢复了：

```python
def _apply(self, fn, *args, **kwargs):
    super()._apply(fn, *args, **kwargs)
    self._rebind_split_views()
    return self
```

实测：`.to("cuda")`、`.to(torch.float64)` 之后仍然共享，且之后原地改权重照样写穿。

## 2. 改动内容

| 位置 | 改动 |
|---|---|
| `model.py` `DecoderLayer` | 新增 `_fuse_projections()` / `_rebind_split_views()` / `_apply`；`forward` 与 `_layer_qkv` 改为一条 GEMM + 切片 |
| `model.py` `_layer_qkv` | 一次 `F.linear(hidden, layer._qkv_storage)`，再按 head 数切开 |
| `model.py` `DecoderLayer.forward` | 一次 `F.linear(z, _gate_up_storage)`，再 `chunk(2, -1)` |
| `norm.py` | kernel 增加外层 stride 参数（见 §3） |
| `attention.py` / `cache.py` / `scheduler.py` / `rope.py` / `sampling.py` / `formats/` | **未改** |
| `engine.py` | **未改** |

`o_proj` / `down_proj` 各只有一次，不动。SwiGLU **没有融合**——需求说这一项在噪声边缘、
可做可不做，做了要单独报数字；本关只做「投影组织」，不混进来。

## 3. 合并之后发现的一处连续性假设（需求 §5 让我查的）

需求说 `rope.py` 与 attention 都按 stride 寻址，所以不需要 `.contiguous()`，
但让我自己确认链上还有没有别处隐含假设紧凑布局。**找到一处**：

```text
ValueError: 融合 RMSNorm 要求 x 整块连续，当前 shape=(8, 4, 16)、stride=(128, 16, 1)
```

`q_norm(q)` 拿到的是合并 QKV 切出来的视图：N=1 时它恰好紧凑，**N>1 时不紧凑**
（最后一维 stride 是 `(HQ+2*HKV)*head_dim` 而不是 `HQ*head_dim`）。所以 prefill 与
多请求打包会撞上，decode 不会。

**修法沿用本仓库的既有约定：传 stride，而不是 `.contiguous()`。** 需求明确说
`.contiguous()` 是「白白多一次拷贝」，而且这一轮我刚给 `rope.py` 做过同样的修改。

```python
def _rms_norm_kernel(x_ptr, w_ptr, out_ptr, eps,
                     stride_row, stride_outer, rows_per_outer,
                     out_stride_row, out_stride_outer, ...):
    row = tl.program_id(0)
    outer = row // rows_per_outer          # 行是「外层下标 + 内层下标」拼出来的
    inner = row % rows_per_outer
    x_base = outer * stride_outer + inner * stride_row
```

- 输入是 `[R, W]` 时让 `rows_per_outer = R`，外层下标恒为 0，
  **退化成 `row * stride_row`，与加参数之前逐位相同**（下面的数值检查证实了这一点）。
- 输入是 `[N, H, W]` 时行 (n, h) 的地址是 `n*stride(0) + h*stride(1)`，一个 stride 表达不了。
- 出和入各传自己的 stride——和 `rope.py` 一样，因为 `empty_like` 给的 stride 不一定和输入一致。

## 4. 验证

### 4.1 结构：launch 数 7 → 4，由 profiler 数出来

真实 Qwen3-0.6B（28 层），eager 一步 decode，按 kernel 名分类：

| | 一步 decode 的 kernel 总数 | 其中投影 GEMM/GEMV |
|---|---:|---:|
| step38 | 584 | **197**（28 层 × 7 + lm_head 1） |
| step39 | 500 | **113**（28 层 × 4 + lm_head 1） |

少掉的 84 = 28 × 3，正好是每层省下的 3 次。**不是「代码看起来是 4 次」，是数出来的。**

### 4.2 数值：逐位相同，而不是 ULP 级

需求允许「ULP 级的差别」，但实测结果是**逐位相同**。原因是合并只在输出维度上拼行，
**每个输出元素沿 d_model 的归约长度与顺序都没变**，所以 GEMM 的结果一模一样。

`benchmarks/check_step39_merged_proj.py`，step38 与 step39 装同一份权重、喂同一份输入：

| 场景 | logits 最大差 | 逐层 KV 最大差 |
|---|---:|---:|
| tiny 模型，BF16，3 请求打包（N>1） | `0.000e+00` | `0.000e+00` |
| tiny 模型，FP32，3 请求打包（N>1） | `0.000e+00` | `0.000e+00` |
| 真实 Qwen3-0.6B，rope=torch，prefill + 5 步 decode | `0.000e+00` | `0.000e+00` |
| 真实 Qwen3-0.6B，rope=triton，prefill + 5 步 decode | `0.000e+00` | `0.000e+00` |

**注意这只是本组输入上的实测，不是数学保证**：cuBLAS 对不同形状可能选不同的 kernel，
换一组形状未必仍然逐位相同。所以没有把容限放宽，也没有声称「一定逐位相同」。

### 4.3 既有语义（`check_step39_merged_proj.py` 第 4、5 节，全部 PASS）

| 检查 | 结果 |
|---|---|
| 合并后显存不变（存储元素数 == 三个视图之和） | PASS（4096 vs 4096） |
| 稳态显存与 step38 相同（整模型实测） | PASS（2349.8 MB vs 2349.8 MB） |
| 构建期瞬时峰值只多一层（+40 MB），不累积 | PASS（见 §1.0） |
| `q/k/v` 的 Parameter 仍然是存储的视图 | PASS |
| 原地改 `q_proj.weight` 写穿到合并存储 | PASS |
| `load_state_dict` 往返后存储仍然正确 | PASS |
| `state_dict()` 键名未变 | PASS |
| 再次 `.to("cuda")` 后视图仍指向存储 | PASS |
| 图的数量与键都和 step38 相同 | PASS（2 vs 2） |

### 4.4 性能：decode 稳定变快，其余在噪声内

需求 §6.4 要求「3 个独立进程、按顺序轮转、先给配对差」。**这一步我先做错过**：
第一版按顺序单跑两个版本，得出「`prefill_c8` 慢 8.2%」——那正是需求 §3 警告的
顺序漂移（同一负载两次运行能差几十个百分点）。改成同进程轮转配对后重测。

**稳态 decode 步（每轮 400 步，取 CUDA Event 计时，6 轮轮转配对）**：

| 负载 | step38 | step39 | 配对同向 |
|---|---:|---:|---|
| decode_c1 | 3.679 ms/步（最小） | 3.558 ms/步（最小） | **6/6**（−3.31%） |
| decode_c8 | 6.637 ms/步（中位） | 6.226 ms/步（中位） | **6/6**（−5.93%） |

「6/6 同向」是这里最硬的信号——幅度随口径在 −3% 到 −6% 之间浮动，但六轮没有一轮反向。
需求预估「decode 一步省 6–8%」，实测落在同一量级。

**六点整轮生成（同进程轮转配对，每点 10 轮）**：

| 测点 | step38 | step39 | 配对中位差 | 同向轮数 |
|---|---:|---:|---:|---:|
| **decode_c1** | 438.0 ms | **412.6 ms** | **−7.41%** | **9/10** |
| **decode_c8** | 707.2 ms | **655.9 ms** | **−6.12%** | **10/10** |
| short_c1 | 115.7 ms | 111.9 ms | −3.89% | 7/10 |
| short_c8 | 177.8 ms | 174.1 ms | −2.34% | 6/10 |
| prefill_c8 | 80.3 ms | 79.0 ms | +0.11% | 5/10 |
| prefill_c1 | 11.8 ms | 11.9 ms | +0.37% | 2/10 |

读法：**收益集中在 decode**，与投影在 decode 预算里占 76% 一致；`short_*` 含首轮 prefill
与后续 decode，收益小一些但方向一致；**两个 prefill 点在噪声内**（幅度 <0.4%、方向对半）。

prefill 上没有收益是可以解释的，也没有当成意外：prefill 的投影本来就是大 GEMM
（M=512+），cuBLAS 已经跑得比较满，合并只减少 launch 次数与中间读写，而 launch 早被
Graph 盖住了；decode 的投影是 GEMV，小且受带宽/发射支配，合并才有实质收益。

需求预估「decode 一步 6–8%」，实测 −6.1% 与 −7.4%，落在同一量级。

## 5. 接口变化与遗留

### 5.1 接口变化

**新增（都是内部的）**：

| 接口 | 说明 |
|---|---|
| `DecoderLayer._qkv_storage` / `_gate_up_storage` | 非持久 buffer，合并后的权重存储 |
| `DecoderLayer._fuse_projections()` / `_rebind_split_views()` | 构造时合并、搬移后重绑 |
| `DecoderLayer._apply()` | 重写以在 `.to()` 后恢复视图共享 |

**未改**：`q_proj` / `k_proj` / `v_proj` / `o_proj` / `gate_proj` / `up_proj` / `down_proj`
全部保留为 `nn.Linear`，名字、`state_dict()` 键、`model_config()`、外部权重格式、
`Engine` 公开接口、`benchmarks` 用的属性都没有变化。

`rms_norm` 的 kernel 签名多了 4 个 stride 参数——这是内部函数，调用点只有 `model.py`。

### 5.2 遗留

1. **SwiGLU 没有融合**（需求允许不做）。验收方测出它还能再省一点，但那是另一个变量。
2. **逐位相同只在实测的这组形状上成立**，没有做形状扫描去界定它的边界。
3. **`_apply` 重写依赖 `super()._apply()` 的行为**：如果将来 PyTorch 改了 `_apply` 的语义
   （例如不再逐个搬 Parameter），重绑的时机可能要跟着调。目前 torch 2.13 下实测正常。
4. **直接给 `layer.q_proj.weight` 赋一个新的 `nn.Parameter` 会打断共享**（视图被替换掉，
   存储不再被 forward 读到）。现有用例都是原地操作，没有这种写法；但这是一个已知的边界，
   没有加检测。
5. **两个 prefill 点没有收益，也没有单独优化**。原因（大 GEMM 本来就跑得满）是推断，不是
   实测到的归因；要确认得单独 profile prefill 步的投影时间。
6. **六点配对是「同一进程里轮流跑两个版本」**，两个引擎同时驻留会互相影响内存与缓存。
   需求 §6.4 要的是 3 个独立进程 + 顺序轮转；decode 两点 9/10 与 10/10 的同向性足以支持
   方向，但绝对幅度可能受这个安排影响。
