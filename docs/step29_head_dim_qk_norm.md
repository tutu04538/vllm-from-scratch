# step29：独立 head_dim 与 Q/K 归一化

- 对应代码：`step29/step29.py`（新增，未提交）
- 源码 SHA256：`c8b8d28051d14999674dbff586163f689a632cfd75ce58068e4b07c5fc5eafc1`
- 基线：`step28/step28.py` 原样保留、未修改

## 0. 需求大概

能加载自己的模型了，但**权重名字和 shape 对上不等于算得对**——另一种模型可能采用不同的计算步骤。本关向 Qwen3 的核心结构靠近，解决两个差异：

1. **每个 head 的宽度不一定等于 `d_model / num_q_heads`**。Qwen3-0.6B 是 `hidden_size=1024`、16 个 Q heads、`head_dim=128`，`16 × 128 = 2048 ≠ 1024`。
2. **Q、K 投影之后各还有一次归一化**（Q/K Norm），位置在 RoPE 之前。

只做小配置的计算对齐，不下载预训练权重、不接 tokenizer。

## 1. 改动内容

| 位置 | 改动 |
|---|---|
| `RMSNorm.__init__` | 参数 `d_model` 改名为 `size`——它现在也用于 `head_dim` |
| `DecoderLayer.__init__` | 新增 `head_dim`、`use_qk_norm`；`o_proj` 输入维度改为 `num_q_heads * head_dim`；开启时新增 `q_norm` / `k_norm` |
| `TinyCausalLM.__init__` | 新增 `head_dim=None`、`use_qk_norm=False`；整除检查只在推导时要求；新增 `attn_width`；补回 `d_model > 0` 校验 |
| `TinyCausalLM._layer_qkv` | 拆 heads 后、RoPE 前对 Q/K 做归一化（V 不做） |
| `TinyCausalLM.model_config()` | 记录 `head_dim` 与 `use_qk_norm` |
| `_attention_out` / `_torch_forward` | 两处 `reshape(num_tokens, self.d_model)` → `reshape(num_tokens, self.attn_width)` |
| `FORMAT_VERSION` | 1 → 2；新增 `COMPATIBLE_FORMAT_VERSIONS = (1, 2)` |
| `load_model_config` | v2 必须写明 `head_dim` / `use_qk_norm`；v1 按旧公式补齐 |
| `Engine.__init__` / `build_model_from_config` | 透传两个新参数 |

## 2. 设计要点

### 2.1 `attn_width`：拼接后的宽度不是 `d_model`

需求 §2 给了这组数：

```text
d_model=32，Q heads=4，KV heads=2，head_dim=16

Q 投影   32 → 64        拆成 4 × 16
K/V 投影 32 → 32        拆成 2 × 16
attention 输出  4 × 16  拼接后 64 个数
o_proj   64 → 32        才能与原 hidden 做残差相加
```

**要注意的不是 hidden 变宽了**——embedding、残差、MLP 输入、最终 norm 用的仍是 `d_model`（32）。变宽的只有 attention 那一段：`o_proj` 的输入是 `num_q_heads * head_dim`。

代码里原来的两处

```python
out.reshape(num_tokens, self.d_model)
```

在 `head_dim != d_model / num_q_heads` 时是错的。现在统一用一个属性：

```python
self.attn_width = num_q_heads * head_dim      # o_proj 的输入维度
```

三处（`o_proj` 定义、两条计算路径的 reshape）都引用它，不会各推各的。

### 2.2 head_dim：显式给定时不再要求整除

```python
if head_dim is None:
    if d_model % num_q_heads != 0:
        raise ValueError("d_model=... 不能被 num_q_heads=... 整除；要么让它整除，要么显式传 head_dim")
    head_dim = d_model // num_q_heads
if head_dim <= 0 or head_dim % 2 != 0:
    raise ValueError(f"RoPE 要求 head_dim 为正偶数，当前 head_dim={head_dim}")
```

- 整除只在**推导**那条路上要求；
- `head_dim` 本身仍是正偶数（RoPE 的前后半段配对需要）；
- GQA 的 `num_q_heads % num_kv_heads == 0` 不变。

实测 `d_model=30, Q=4, head_dim=16` 现在合法（`30 % 4 != 0`，旧代码会拒绝）。

### 2.2.1 解耦会弄丢「顺带发生」的检查

解耦之前，`d_model=0` 是**被间接拦住**的：

```text
d_model=0 → 推导 head_dim=0 → head_dim 正偶数检查拦住
```

解耦之后：

```text
d_model=0，显式 head_dim=16 → 不再推导 → head_dim 合法 → d_model 没人管
```

**解除了两个维度之间的绑定，也顺带解除了那条检查。** 现在 `d_model > 0` 是独立的一条，放在 `vocab_size` / `max_seq_len` 旁边。

注意没有把整除检查放回公共路径——那会误杀 `d_model=30, head_dim=16` 这种合法配置：

| d_model | Q heads | 显式 head_dim | 结果 |
|---:|---:|---:|---|
| 32 | 4 | 16 | 合法 |
| 30 | 4 | 16 | 合法 |
| 0 | 4 | 16 | 拒绝（`d_model 必须为正`） |
| 30 | 4 | 未提供 | 拒绝（无法整除推导） |

另外，`d_model=0` 时**权重 shape 自己看不出问题**：零宽度的 `Linear` / `Embedding` 是允许的，配置和权重可以完全自洽，所以不能靠 `load_state_dict(strict=True)` 替代配置校验。

### 2.3 Q/K Norm：在 RoPE 之前，只沿最后一维

```text
hidden → norm1 → Q/K/V 投影 → 拆 heads
                              ├─ Q → q_norm → RoPE
                              ├─ K → k_norm → RoPE → 写 K cache
                              └─ V ─────────────────→ 写 V cache
```

```python
if self.use_qk_norm:
    q = layer.q_norm(q)      # q: [N, Q_heads, head_dim]
    k = layer.k_norm(k)      # k: [N, KV_heads, head_dim]
return self.rotary(q, positions), self.rotary(k, positions), v
```

复用已有的 `RMSNorm`，`eps` 仍取 `rms_norm_eps`。归一化沿**最后一维**做，所以：

- 每个 token 的每个 head 单独算统计量，不跨 head、不跨请求；
- 权重形状是 `[head_dim]`，**不是** `[num_heads, head_dim]`——同一层的各 Q head 共用一份；
- Q 和 K 各有一份，不共用；
- V 不做，`norm1` 也不删。

**缓存里存的是「做过 qk_norm、也做过 RoPE」的 K**——顺序不能颠倒，否则与参考对不上（实测与 Qwen3 的 KV 缓存逐层一致）。

### 2.4 保存/加载：不许静默忽略新配置

`FORMAT_VERSION` 从 1 提到 2，新配置写进 `config.json`：

```json
"head_dim": 16,
"use_qk_norm": true
```

版本处理：

- **v2**：必须写明这两个字段，缺任何一项直接报错；
- **v1**（旧目录）：没有这两项，按旧公式处理（推导 `head_dim`、不做 Q/K norm）——显式的兼容，不是忽略。

"打开了 Q/K norm 但权重里没有 q_norm"会被 `load_state_dict(strict=True)` 拦下（Missing keys）——因为模型因为配置而**多出了**这两个参数，文件里必须有对应项。所以加载后不会悄悄退回旧公式。

## 3. 验证

conda `vllm-omni-dev`，RTX 5090 Laptop，torch 2.13.0+cu130，Triton 3.7.1，Transformers 5.14.1，FP32。

起步配置即需求 §5 那组：`vocab_size=11, d_model=32, num_layers=2, intermediate_size=48, Q/KV heads=4/2, head_dim=16, use_qk_norm=True, max_seq_len=64`。

### 3.1 与本机 Transformers 的 Qwen3 对照

把本实现的权重逐项拷进一个同配置的 `Qwen3ForCausalLM`（`attn_implementation="eager"`），再比同一段 token 的输出：

```
逐位置 logits 最大绝对误差 = 2.384e-07
逐位置 argmax 一致        = True
```

映射关系是干净的 1:1：`input_layernorm`→`norm1`、`post_attention_layernorm`→`norm2`、`self_attn.{q,k,v,o}_proj`、`self_attn.q_norm/k_norm`、`mlp.{gate,up,down}_proj`、`embed_tokens`/`norm`/`lm_head`。

### 3.1.1 验收方脚本

`verify_step29_contract.py` **96/96**、`verify_step29_io_contract.py` **45/45**、`verify_step29_qwen3.py` **21/21**（补上 `d_model` 校验前是 19/21）。

### 3.2 Q/K Norm 确实在用

把每层的 `q_norm.weight` 填成 1.7、`k_norm.weight` 填成 0.4：

```
与 Qwen3 的最大误差        = 1.863e-07     仍一致
与「权重全 1 时」的输出差异 = 8.292e-02     确实生效
```

### 3.3 每层 KV 也与 Qwen3 一致

比对 `past_key_values` 与本实现每层写进缓存的 K/V（K 是做过 qk_norm + RoPE 的）：

```
第 0 层 K/V 一致 = True
第 1 层 K/V 一致 = True
```

### 3.4 Engine 出口与公式一致

torch / triton / triton+graph 三条路径，两条请求，输出都与"按公式逐 token 生成"完全相同。

### 3.5 尺寸随 head_dim 走

| head_dim | KV 池 | K+V | RoPE 表 |
|---:|---|---:|---|
| 8 | `(2, 8, 4, 2, 8)` | 8192 字节 | `(64, 4)` |
| 16 | `(2, 8, 4, 2, 16)` | 16384 字节 | `(64, 8)` |

KV 池、RoPE 表和 attention 缩放因子都用实际 `head_dim`，没有第二处从 `d_model` 推导。

### 3.6 保存 / 加载与兼容

- 保存出的 `config.json`：`format_version=2`、`head_dim=16`、`use_qk_norm=true`；权重里有每层的 `q_norm` / `k_norm`。
- 换随机种子加载后 `head_dim` / `use_qk_norm` / `attn_width` 都正确，权重逐位相同。
- **v1 旧目录**仍能加载：`head_dim` 按推导值、`use_qk_norm=False`，权重一致。

### 3.7 缓存与执行

- 分块 prefill 预算 1 / 3 / 8：输出一致。
- prefix cache：命中 2 块、`cache.length=9`。
- 两层模型一轮调度 4 个 token → `cache.length=4`（不是 8），长度只推进一次。
- prefix cache 开关对照 300 组：不一致 0；索引双射 + 无引用泄漏：300 组 / 60027 次 step 通过。

### 3.8 失败用例

| 情况 | 结果 |
|---|---|
| `head_dim=0` / `15` / `-8` | `ValueError: RoPE 要求 head_dim 为正偶数...` |
| `d_model=30` 且未传 `head_dim` | `ValueError: d_model=30 不能被 num_q_heads=4 整除；要么让它整除，要么显式传 head_dim` |
| `num_q_heads=4, num_kv_heads=3` | `ValueError: num_q_heads=4 不能被 num_kv_heads=3 整除` |
| v2 配置缺 `head_dim` | `ValueError: format_version=2 的配置缺少字段: ['head_dim']` |
| v2 配置缺 `use_qk_norm` | `ValueError: format_version=2 的配置缺少字段: ['use_qk_norm']` |
| 配置开了 Q/K norm 但权重里没有 | `RuntimeError: Missing key(s) in state_dict` |
| `d_model=0`（显式 `head_dim=16`） | `ValueError: d_model 必须为正，收到 0` |

`d_model` 的检查在**随机创建与目录加载两条路**都生效；构造零宽度配置 + 零宽度权重的临时目录，加载器同样在建模阶段就拒绝，不会返回 Engine。八个维度字段（`vocab_size` / `d_model` / `max_seq_len` / `num_q_heads` / `num_kv_heads` / `num_layers` / `intermediate_size` / `head_dim`）各测 `0` 与 `-1`，**没有漏检**。

## 4. 接口变化与遗留

- `TinyCausalLM(...)` / `Engine(...)` 新增 `head_dim=None`、`use_qk_norm=False`。
- `RMSNorm(size, eps)`：第一个参数由 `d_model` 改名 `size`。
- `DecoderLayer(d_model, num_q_heads, num_kv_heads, head_dim, intermediate_size, eps, use_qk_norm=False)`：新增 `head_dim` 位置参数与 `use_qk_norm`。
- 新增属性 `TinyCausalLM.attn_width` 与 `use_qk_norm`。
- `config.json` 的 `format_version` 升到 2，新增 `head_dim` / `use_qk_norm`；v1 目录仍可加载。
- `state_dict` 在 `use_qk_norm=True` 时每层多出 `q_norm.weight` / `k_norm.weight`。
- 遗留：仍是普通 RoPE，没有 sliding window、没有 QKV bias、没有共享 embedding/lm_head 权重。
- 遗留：`from torch import ceil` 与 `from more_itertools import last` 仍未使用，是前几关带下来的。
- 未提交。
