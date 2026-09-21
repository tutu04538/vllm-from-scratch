# step39：把每层 7 次投影合并成 4 次

- 对应代码：`step39/`（新增，从 `step38/` 复制）
- 包摘要 SHA256：`89fce66fe677e5071e0ab2d3c52dc1978f2750bba31a7ec9e5d5021beb99ec2c`（14 个 .py / 2586 行）
- 基线：`step38/`，指纹 `45c0e5c6…`（14 个 .py / 2511 行），原样保留未改
- 改动文件：`model.py`、`norm.py`、`formats/native.py`、`__init__.py`、入口改名 `step39.py`

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

## 1. 核心设计：把三/五个 Parameter 换成两个

```python
class DecoderLayer(nn.Module):
    def __init__(self, ...):
        self.qkv_proj = nn.Linear(d_model, (HQ + 2*HKV) * head_dim, bias=False)
        self.gate_up_proj = nn.Linear(d_model, 2 * intermediate, bias=False)
        self.o_proj = nn.Linear(HQ * head_dim, d_model, bias=False)
        self.down_proj = nn.Linear(intermediate, d_model, bias=False)
```

**就这么多。** 没有缓冲区、没有别名、没有 `_apply` 重写。融合 Parameter 本身就是存储，
前向一次 `F.linear` 拿 [N, (HQ+2*HKV)·D]，再沿输出维切开。

代价是**内部参数名不再与 HF 一一对应**：`state_dict()` 里是 `qkv_proj.weight` 与
`gate_up_proj.weight`。转换在装载权重时做，见 §1.1。

### 1.1 装载时合成：一个 hook，覆盖所有路径

`DecoderLayer._load_from_state_dict` 在装载前把旧的三键/两键沿输出维拼成融合键：

```python
for fused, parts in (("qkv_proj.weight", ("q_proj.weight", "k_proj.weight", "v_proj.weight")),
                     ("gate_up_proj.weight", ("gate_proj.weight", "up_proj.weight"))):
    if (prefix + fused) in state_dict:
        continue                       # 已经是融合键（新版目录），放行
    keys = [prefix + p for p in parts]
    if not all(k in state_dict for k in keys):
        continue
    state_dict[prefix + fused] = torch.cat([state_dict.pop(k) for k in keys], dim=0)
```

放在模块 hook 而不是适配器里，是因为它是**唯一的必经之路**：HF 目录、旧版 `save_model`
目录、以及直接调 `model.load_state_dict(三键)` 的夹具，全都从这里过。`formats/qwen3.py`
因此**一个字都不用改**。

**加和删两个动作都是必需的**，少一个都会报错：

| 动作 | 不做会怎样 |
|---|---|
| **加上** `qkv_proj.weight` | 子模块 `qkv_proj` 取不到键 → `Missing key: qkv_proj.weight` |
| **`pop` 掉** `q_proj.weight` 等 | 基类末尾会把它们报成 `Unexpected key` → 装载抛错 |

后者容易被当成顺手清理，其实不是：基类末尾会遍历本前缀下的键，把不属于任何子模块的
报成多余，而 `q_proj` 重构后已经不是子模块了。实测不 pop 时，装一次 HF 目录会报出
全部 10 个旧键。**根模块和 `layers` 这一层都不会报**（它们看到的 `input_name[0]` 是
`layers` / `0`，都合法），只有 `DecoderLayer` 这一层看到的是 `q_proj`。

**这里踩过一个坑，记下来**：第一版我建了一份 `dict(state_dict)` 副本、只把它传给
`super()._load_from_state_dict(...)`，结果一直报 `Missing key: qkv_proj.weight`。
原因是 torch 的 `load()` 是按前缀从**传进来的那个对象**给每个子模块过滤子字典的：

```python
def load(module, local_state_dict, prefix=""):
    module._load_from_state_dict(local_state_dict, prefix, ...)
    for name, child in module._modules.items():
        child_state_dict = {k: v for k, v in local_state_dict.items() if k.startswith(child_prefix)}
        load(child, child_state_dict, child_prefix)     # ← 用的是原对象
```

`qkv_proj.weight` 最后由 `qkv_proj` 这个 `nn.Linear` 子模块取，改副本它看不到。
**必须原地改**——torch 在 `load_state_dict` 入口已经复制过一次，所以不会改到调用方的对象
（源码里那句注释 "copy state_dict so `_load_from_state_dict` can modify it" 就是这个意思）。

### 1.2 格式版本

`save_model()` 写出的目录现在是 **v4**（参数名变了），`COMPATIBLE_FORMAT_VERSIONS`
扩到 `(1, 2, 3, 4)`：

- step39 读 v1–v3（三键）✓ 走 hook 合成
- step39 读 v4（融合键）✓ 直接装
- **旧版本读 v4 会明确拒绝**（版本号不认识），不会把融合键当成缺参数

### 1.3 与「一块存储 + 参数视图」方案的对比

第一版走的是另一条路：保留 `q_proj`/`k_proj`/… 五个 Parameter，让它们成为同一块
buffer 的视图，再重写 `_apply` 在 `.to()` 后重绑。那样能不改任何接口，但代价是引入一个
**隐蔽且可被破坏的不变量**（"三个 Parameter 指着同一块内存"），破坏了会静默算错。

换成融合 Parameter 之后，不变量变成"两个 Linear 各管各的"——显然、不可破坏。
接口的兼容性改由 §1.1 的装载 hook 承担，那里错了会直接报 `Missing key`，不会静默。

## 2. 改动内容

| 位置 | 改动 |
|---|---|
| `model.py` `DecoderLayer` | 五个投影 Parameter 换成 `qkv_proj` / `gate_up_proj`；新增 `_load_from_state_dict` 做装载时合成 |
| `model.py` `_layer_qkv` | 一次 `layer.qkv_proj(hidden)`，再按 head 数切开 |
| `model.py` `DecoderLayer.forward` | 一次 `layer.gate_up_proj(z)`，再 `chunk(2, -1)` |
| `norm.py` | kernel 增加外层 stride 参数（见 §3） |
| `formats/native.py` | `FORMAT_VERSION` 3 → 4，兼容列表加 4 |
| `formats/qwen3.py` | **未改**（HF 的三键由 §1.1 的 hook 合成） |
| `attention.py` / `cache.py` / `scheduler.py` / `rope.py` / `sampling.py` / `engine.py` | **未改** |

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

### 4.3 参数名、兼容装载与往返（`check_step39_merged_proj.py` 第 4、5 节）

| 检查 | 结果 |
|---|---|
| `state_dict()` 用融合键 | PASS（`layers.0.qkv_proj.weight` / `gate_up_proj.weight`） |
| 旧的三键/两键不再出现 | PASS |
| 融合权重元素数 == Q/K/V 三者之和 | PASS（4096 vs 4096） |
| **旧三键字典仍能装进来** | PASS |
| **两条装载路径的权重逐位相同** | PASS |
| **两条装载路径的前向输出一致** | PASS（最大差 `0.000e+00`） |
| 原地改融合权重生效 | PASS |
| 图的数量与键都和 step38 相同 | PASS（2 vs 2） |

四条装载路径的往返（独立脚本验证，全部**逐位相同**）：

```text
① Qwen3 目录（HF 三键）        -> step39            逐位相同
② save_model 写出 v4 融合键目录                    format_version=4
③ v4 融合键目录                -> step39            逐位相同
④ v3 三键目录（模拟旧版）      -> step39            逐位相同
```

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

### 5.1 接口变化（这是本关最需要知道的一件事）

**内部参数名变了**：

| 旧 | 新 |
|---|---|
| `layers.N.q_proj.weight` / `k_proj` / `v_proj` | `layers.N.qkv_proj.weight` |
| `layers.N.gate_proj.weight` / `up_proj` | `layers.N.gate_up_proj.weight` |
| `layer.q_proj` / `layer.k_proj` / ... （模块属性） | 不再存在 |

`o_proj` / `down_proj` / 各种 norm 的名字不变；`model_config()`、`Engine` 公开接口、
外部（HF）权重格式、CLI 都不变。参数**总量**也不变——只是存法从五个张量变成两个。

**装载**：新的融合键与旧的三键/两键**都能装**（§1.1），所以 HF 目录与旧版目录都不受影响。
**`save_model()` 写出的目录是 v4**，旧版本读它会明确拒绝而不是误装。

**已知会让哪些验收断言失败**（用户已确认会通知验收方调整）：

```text
step39_external_process.py:31   assert set(map(mapped, state_dict())) == set(hf.state_dict())
    mapped 是一对一改名，而 qkv_proj.weight 在 HF 里没有对应键
    -> 实测：改为融合键后该断言失败，报出 model.layers.0.self_attn.qkv_proj.weight

step39_process_roundtrip.py:19  e.model.layers[0].q_proj.weight.add_(0.123)
step39_test_helpers.py:21,28    w['layers.0.q_proj.weight']
step39_precision_reference.py:11 w[root + 'q_proj.weight']
    -> 属性访问与字典键访问都要改成 qkv_proj 并自己切片
```

其余断言（`set(payload) == set(model.state_dict())` 这类自洽比较）不受影响。

`rms_norm` 的 kernel 签名多了 4 个 stride 参数——内部函数，调用点只有 `model.py`。

### 5.2 遗留

1. **SwiGLU 没有融合**（需求允许不做）。验收方测出它还能再省一点，但那是另一个变量。
2. **旧版代码读不了 v4 目录**：老版本会因为版本号不认识而明确拒绝（这是正确行为），但要把
   v4 目录喂给 step38 及更早的版本，得先转回三键。没有写这个转换工具。
3. **逐位相同只在实测的这组形状上成立**，没有做形状扫描去界定它的边界。
4. **两个 prefill 点没有收益，也没有单独优化**。原因（大 GEMM 本来就跑得满）是推断，不是
   实测到的归因；要确认得单独 profile prefill 步的投影时间。
5. **六点配对是「同一进程里轮流跑两个版本」**，两个引擎同时驻留会互相影响内存与缓存。
   需求 §6.4 要的是 3 个独立进程 + 顺序轮转；decode 两点 9/10 与 10/10 的同向性足以支持
   方向，但绝对幅度可能受这个安排影响。
6. **装载 hook 依赖 torch 的 `load()` 按前缀过滤子字典这一行为**（§1.1 那个坑）。
   如果将来 PyTorch 改成把合并后的字典传给子模块，原地改仍然安全；但如果改成对每个子模块
   独立复制，这条路径需要重测。
