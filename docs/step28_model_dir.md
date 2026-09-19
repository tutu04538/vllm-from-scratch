# step28：模型目录与权重加载

- 对应代码：`step28/step28.py`（新增，未提交）
- 源码 SHA256：`5fd0691af7c41f33b3ea414d43567b04d5529fbe2f8f5854e71b884496e2f98f`
- 基线：`step27/step27.py`（SHA256 `5330e6ae…`）原样保留、未修改

> 代码统一放在各自子目录下（`stepNN/stepNN.py`）。本关的模型目录也写在 `step28/` 里。

## 0. 需求大概

模型公式已经能跑，但创建模型仍然依赖代码里的构造参数和随机初始化；测试时还得手动复制 `state_dict` 才能保证用同一套权重。

本关要求：**把现有小模型保存下来，退出进程后，只凭一个目录就能重新构建它，并由 Engine 继续生成。**

不下载真实权重、不接 tokenizer、不新增 Q/K norm 或低精度、不改变现有模型数学。

```text
my_tiny_model/
├── config.json          这个网络长什么样
└── model.safetensors    每个参数的数值是什么
```

**只保存随机种子不算保存模型**——改一层权重后再保存，也必须能准确恢复。

## 1. 改动内容

| 位置 | 改动 |
|---|---|
| `save_model(model, model_dir)` | 新增：写 `config.json` + `model.safetensors` |
| `load_model_config(model_dir)` | 新增：读并校验配置 |
| `build_model_from_config(...)` | 新增：按配置构造模型 |
| `load_model_weights(model_dir, model)` | 新增：读并校验权重，加载进已建在目标设备上的模型 |
| `_resolve_device` / `_check_runtime` | 新增：设备解析与后端/Graph 组合校验，两条构造路径共用 |
| `TinyCausalLM.__init__` | 新增 `vocab_size` / `max_seq_len` 必须为正的校验（随机初始化与目录加载共用） |
| `TinyCausalLM.model_config()` | 新增：导出这个模型**实际**的结构配置 |
| `Engine.__init__(..., model=None)` | 新增 `model` 关键字：给定时不再按维度参数随机初始化 |
| `Engine._init_runtime(...)` | 从 `__init__` 抽出的运行时装配：元数据、KV 池、调度器 |
| `Engine.from_model_dir(...)` | 新增类方法：只给目录和运行选项，返回可用的 Engine |

## 2. 设计要点

### 2.1 配置从模型导出，不写死

```python
config = {"format_version": 1, "model_type": "tiny_rope_decoder", "dtype": "float32"}
config.update(model.model_config())     # 取实际模型配置
```

`model_config()` 读的是模型自己的属性（`num_layers`、`rope_theta` 等），所以改了构造参数、加载了别的权重，写出来的都是**这个**模型的真实结构。

**运行选项不写进目录**：`device`、attention 后端、Graph 开关、并发数、token 预算、物理块数、prefix cache 开关都由加载方选择。`config.json` 只回答"网络长什么样"。

### 2.2 保存不改动原模型

```python
weights = {name: tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
           for name, tensor in model.state_dict().items()}
```

`detach().to(...)` 产生副本，所以保存之后原模型的权重、device、dtype 和已捕获的 Graph 都还能用。实测：保存前后逐位相同，`device` 仍是 `cuda`。

统一存成 CPU float32，与模型当时在 CPU 还是 GPU 无关——这也是跨设备往返能成立的原因。

RoPE 的 `cos/sin` 表**不写进 safetensors**：它们是 `register_buffer(persistent=False)`，由配置重算即可，不是可学习参数。

### 2.3 加载顺序：先把模型建好、权重装好，再装运行时

```python
config = load_model_config(model_dir)        # ① 读并检查配置
device = _resolve_device(device); _check_runtime(...)
model = build_model_from_config(...)         # ② 按配置构造模型（已在目标设备上）
load_model_weights(model_dir, model)         # ③ 读并检查权重，直接加载到目标设备
return cls(model=model, ...)                 # ④ 装运行时：元数据、KV 池、调度器
```

第 ④ 步里 `_init_runtime` 用 `model.num_kv_heads` / `model.head_dim` / `model.num_layers` 建 KV 池，所以池的尺寸来自**刚加载的模型**，不会被 Engine 的默认 `d_model=8`、两层之类覆盖。

顺序上把权重放在建池之前，是为了**加载失败时不留下半个 Engine**：`from_model_dir` 要么返回一个完整可用的 Engine，要么抛异常，调用方拿不到混着随机参数的对象。

Graph 是首次 `step()` 时才捕获的，所以"权重加载结束之后才捕获图"自然成立。

### 2.4 dtype 必须自己查

```python
for name, tensor in weights.items():
    if tensor.dtype != torch.float32:
        raise ValueError(f"权重 {name} 的 dtype 是 {tensor.dtype}，本实现只支持 float32")
model.load_state_dict(weights, strict=True)
```

`load_state_dict` **会静默把 dtype 转成目标参数的类型**——实测把 float64 权重装进 float32 模型，装完就是 float32，不报错也不提示。所以"用 strict=True 就够了"不成立：strict 管的是键与形状，不管 dtype。

键与形状交给 `strict=True`：缺参数、多参数、shape 不符都会抛。

### 2.4.1 strict 权重加载拦不住配置错误

`load_state_dict(strict=True)` 只管键与形状，**配置本身合不合法它管不着**。最直接的例子是 `max_seq_len`：

- RoPE 的 `cos/sin` 表不是可学习参数，**不写进权重文件**（见 2.2），加载时按配置重算；
- 所以把 `max_seq_len` 改成 0，不会造成任何 shape 不匹配，`strict=True` 一声不响地放行；
- 结果是一个"看起来有效"的 Engine：`from_model_dir` 正常返回，直到提交请求、`step()` 走到位置检查才报 `位置区间 [0, 1) 超出 max_seq_len=0`。

这类字段必须自己校验。现在 `TinyCausalLM.__init__` 的结构校验里包含：

```python
if vocab_size <= 0:   raise ValueError(f"vocab_size 必须为正，收到 {vocab_size}")
if max_seq_len <= 0:  raise ValueError(f"max_seq_len 必须为正，收到 {max_seq_len}")
```

放在模型构造里，**随机初始化和从目录加载两条路共用同一处校验**，不会出现"一条路拦住、另一条漏掉"。

`vocab_size` 是同一类洞：`0` 能构造成功、要到运行时查表才炸，`-1` 抛的是 torch 的 `Trying to create tensor with negative dimension`。验收方只点名了 `max_seq_len`，这一处是我按同一规则一并补上的（110 里已写明"模型维度非法要明确报错"）。

### 2.5 Engine 的接缝

`__init__` 原来自己建模型，没法注入一个已加载的模型。这次把"建模型"和"装运行时"分开：

```python
def __init__(self, ..., model=None):
    if model is None:
        ...按维度参数随机初始化...
    self._init_runtime(model, ...)      # 元数据、KV 池、调度器

@classmethod
def from_model_dir(cls, model_dir, ...):
    ...读配置、建模型、装权重...
    return cls(model=model, ...)        # 走同一条装配路径
```

后端与 Graph 开关以**模型上记的**为准（`model.attention_backend`、`model.use_cuda_graph`），避免"Engine 说是 triton、模型却不是"这种不一致。校验逻辑抽成 `_check_runtime(device, backend, use_cuda_graph)`，两条路径各调一次。

保留了原来的随机初始化入口，小配置测试仍然方便。

## 3. 验证

conda `vllm-omni-dev`，RTX 5090 Laptop，torch 2.13.0+cu130，Triton 3.7.1，FP32。

### 3.1 两个独立进程

子进程 A：`manual_seed(1234)` 建模型，**改掉一层的一个权重**（确保不是靠种子复现），保存。
子进程 B：`manual_seed(999)`，只凭目录加载。

```
子进程 save: saved
子进程 load: LOADED same=True n=21
```

21 个参数逐个逐位相同。

### 3.2 不同结构都能恢复

| 配置 | 配置 | 权重 | RoPE 表 | 加载后的 KV 池 |
|---|---|---|---|---|
| 1 层 1/1 theta=10000 | ✅ | ✅ | ✅ | `(1, 8, 4, 1, 32)` |
| 3 层 4/2 theta=500 | ✅ | ✅ | ✅ | `(3, 8, 4, 2, 8)` |
| 2 层 8/1 theta=100000 | ✅ | ✅ | ✅ | `(2, 8, 4, 1, 4)` |

`cos/sin` 表逐位相同——说明是**按配置重算**出来的，不是从文件读的。

### 3.3 保存前后与加载后的数值一致

- 保存不改动原模型：权重逐位相同、device 仍是 cuda、dtype 仍是 float32。
- 生成结果：原模型与加载的都是 `[('A', (3, 39, 2)), ('B', (33, 3, 39))]`。
- 加载后**逐层 KV** 与独立参考一致（不是只比最终输出）。

### 3.4 跨设备

```
CPU 保存 -> GPU 加载: 权重在 cuda:0，与 CPU 版逐位相同 = True
GPU 保存 -> CPU 加载: 权重在 cpu，与 GPU 版逐位相同 = True
```

### 3.5 加载后的 Engine 能继续用

- 分块 prefill 预算 1 / 4 / 16：输出一致。
- Graph 关：图数量 0；Graph 开：图数量 3，输出与 eager 一致。
- prefix cache：Q 命中 2 个块，`cache.length = 9`。

### 3.6 失败用例（都明确报错，不返回部分加载的 Engine）

| 情况 | 结果 |
|---|---|
| 缺配置文件 | `FileNotFoundError: 缺少配置文件 .../config.json` |
| 缺权重文件 | `FileNotFoundError: 缺少权重文件 .../model.safetensors` |
| `format_version=99` | `ValueError: 不支持的 format_version=99，本实现只支持 1` |
| `model_type='llama'` | `ValueError: 不支持的 model_type='llama'...` |
| `dtype='bfloat16'` | `ValueError: 不支持的 dtype='bfloat16'...` |
| 缺 `num_layers` | `ValueError: 配置缺少必需字段: ['num_layers']` |
| `num_q_heads=3` | `ValueError: d_model=32 不能被 num_q_heads=3 整除` |
| 参数缺失 | `RuntimeError: Missing key(s) in state_dict` |
| 额外参数 | `RuntimeError: Unexpected key(s) in state_dict` |
| shape 不符 | `RuntimeError: size mismatch for norm.weight` |
| 权重是 float64 | `ValueError: 权重 norm.weight 的 dtype 是 torch.float64...` |
| `max_seq_len=0` / `-5` | `ValueError: max_seq_len 必须为正，收到 0` |
| `vocab_size=0` / `-1` | `ValueError: vocab_size 必须为正，收到 0` |

后两类在**加载路径和随机初始化路径**下都会被拦（同一处校验），且合法模型不受影响——改成 0 的配置连 Engine 都返回不出来，不会拖到 `step()` 才失败。

非法运行选项仍在建模型之前被拒（`from_model_dir(device="cpu", attention_backend="triton")` 等）。

### 3.6.1 验收方脚本

`verify_step28_contract.py` **96/96**、`verify_step28_io_contract.py` **43/43**（补上 `max_seq_len` 校验前是 42/43）。

### 3.7 既有回归

prefix cache 开关输出对照 300 组：不一致 0；索引双射 + 无引用泄漏：300 组场景 / 60027 次 step 逐步检查通过。

## 4. 接口变化与遗留

- 新增模块级函数 `save_model` / `load_model_config` / `build_model_from_config` / `load_model_weights`。
- 新增 `Engine.from_model_dir(model_dir, device=None, attention_backend="torch", use_cuda_graph=False, max_num_seqs=1, max_num_batched_tokens=4, block_size=4, num_kv_blocks=8, on_finished=None, enable_prefix_caching=True)`。
- `Engine.__init__` 新增 `model=None`：**给定时忽略所有维度参数**，并以后端/Graph 也从模型上取。
- 新增常量 `MODEL_CONFIG_NAME` / `MODEL_WEIGHTS_NAME` / `FORMAT_VERSION` / `MODEL_TYPE` / `MODEL_DTYPE`。
- 新增依赖 `safetensors`（本机 0.8.0）；除此之外没有新依赖。
- 目录变化：本关代码在 `step28/step28.py`，`save_model` 产出的示例目录在 `step28/my_tiny_model/`。
- 遗留：只支持单个 safetensors 文件、FP32、本地目录；没有分片索引、共享权重恢复、量化、下载器。
- 遗留：不做热更新与失败回滚——加载不完整就不把对象交给调用方，仅此而已。
- 遗留：`step28/my_tiny_model/` 是内置 demo 生成的示例目录，可随时删除重建。
- 未提交。
