# step30：直接读取外部 Qwen3 模型目录

- 对应代码：`step30/`（新增，未提交）
- 包摘要 SHA256：`8d0443e790c658b676a2e701633e6c9d95fd6edd8ab28c5a28d3ee090d4d0d71`
  （`sha256("\n".join(sorted("路径 sha256(文件)")) + "\n")`，只算 `step30/**/*.py`）
- 基线：`step29/step29.py` 原样保留、未修改

## 0. 需求大概

上一关证明了「拿到相同参数能算出正确结果」，但权重对应是验收脚本帮忙做的。本关假设同事给一个 Transformers 导出的 Qwen3 目录，**不能要求他先把配置和参数名改成我们的格式**——只给目录，Engine 就要能加载并生成 token。

同时从本关起，代码不再是一个大脚本，按模块拆分。

样例目录：`fixtures/step30_qwen3/tiny_gqa`（2 层、hidden=32、Q/KV heads=4/2、head_dim=16）和 `tiny_mqa`（3 层、hidden=30、Q/KV heads=4/1、head_dim=6）。**`tiny_mqa` 的 `30 // 4 = 7 ≠ 6`**，所以 head_dim 必须取目录里的值。

`head_dim=16` 的这次是：外部目录 → 转内部配置与权重名 → 建自己的模型 → 严格加载 → 装运行时 → 跑自己的 forward。

## 1. 改动内容

| 文件 | 内容 |
|---|---|
| `step30/attention.py` | 从 step29 拆出：Triton 算子、`paged_attention`、`AttentionMetadata` |
| `step30/cache.py` | 拆出：`_stable_hash`、`CacheConfig`、`SequenceConfig`、`KVCachePool` |
| `step30/model.py` | 拆出：`RotaryEmbedding`、`RMSNorm`、`DecoderLayer`、`TinyCausalLM`、`DummyModel` |
| `step30/sampler.py`、`step30/scheduler.py` | 拆出：`Sampler`、`Scheduler` |
| `step30/formats/native.py` | 自定义目录格式：校验配置、`save_model`、权重原样透传 |
| `step30/formats/qwen3.py` | **新增**：外部 Qwen3 的配置字段翻译与参数名映射 |
| `step30/formats/__init__.py` | **新增**：读 `config.json`，按 `model_type` 选适配器；读权重并统一检查 dtype |
| `step30/engine.py` | `Engine` / `build_model_from_config` / 加载入口，改走适配层 |
| `step30/__init__.py` | 汇总公开 API |
| `step30/step30.py` | 入口：`python step30/step30.py` 跑演示 |
| `.gitignore` | `step*/my_tiny_model/` → `step*/my_*_model/`（演示会多写一个目录） |

## 2. 设计要点

### 2.1 模块边界：依赖单向，各自只认自己那层

```text
formats/   只认 json 和 safetensors      -> 内部配置 + 内部参数名
cache      只认物理块与引用计数          -> 谁在用、写到哪个 slot
attention  只认打包好的一维 query         -> 每个 query 每个 head 的输出
model      只认「一批 token + 缓存」      -> logits
sampler    只认 logits
scheduler  只认请求与预算                -> 本轮计划
engine     把上面装起来，提供目录加载入口
```

`model` 不认请求，`scheduler` 不认模型（只通过 `kv_cache_pool` 要块）。这样拆的收益是**以后加格式不用碰模型，加模型结构不用碰调度**。

副作用：原来一个文件里的私有名字（`_paged_attention_kernel`、`_rotate_half`）现在住在各自的模块里。包顶层仍然 re-export 了它们，但 **monkeypatch 的目标变成了 `step30.attention._paged_attention_kernel`**——用 `patch.object(step30, ...)` 不会再影响 `attention.paged_attention` 里的全局查找。

### 2.2 适配层的形状：一个分支，不是注册中心

要求说「少量函数和一个格式分支就足够」。所以适配层只有三样东西：

```python
_ADAPTERS = {native.MODEL_TYPE: native, qwen3.MODEL_TYPE: qwen3}   # model_type -> 模块

adapter, raw = read_raw_config(model_dir)        # 读一次 config.json 并选适配器
config  = adapter.to_internal_config(raw)        # 外部字段 -> 内部字段
weights = adapter.to_internal_weights(...)       # 外部参数名 -> 内部参数名
```

适配器就是两个**纯函数**：吃外部数据，吐内部数据，不碰文件、不建模型、不认识 Engine。文件读取和 dtype 检查放在共用的 `read_raw_weights()` 里，两种格式只有一份。

**配置只读一次。** `Engine.from_model_dir` 拿到 `(adapter, raw)` 之后一路用下去；`load_model_config()` / `load_model_weights()` 保留成单独入口，供只想读配置的场景使用。

### 2.3 字段翻译：Qwen3 的字段名不同，模型不用重新设计

| 外部 | 内部 |
|---|---|
| `hidden_size` | `d_model` |
| `num_hidden_layers` | `num_layers` |
| `num_attention_heads` | `num_q_heads` |
| `num_key_value_heads` | `num_kv_heads` |
| `max_position_embeddings` | `max_seq_len` |
| `intermediate_size` / `rms_norm_eps` / `vocab_size` | 同名 |
| `head_dim` | `head_dim`（**必须取目录的值，不重新推导**） |
| `rope_parameters.rope_theta` | `rope_theta` |
| 没有这个字段 | `use_qk_norm = True` |

最后一行是关键：**Qwen3 配置里没有 `use_qk_norm`，不代表要关掉它**。它的结构里 Q/K 投影之后一定有归一化，缺字段就默认关闭等于静默换成另一套公式。所以这一项由适配器**明确写成 True**。

`head_dim` 单列一行也是同一个道理：内部格式允许「不传就按 `d_model // num_q_heads` 推导」，外部目录不允许——`tiny_mqa` 的 `30 // 4 = 7 ≠ 6`，一推导就错。

### 2.4 权重映射：改名，不转置，不丢源

```text
model.embed_tokens.weight               -> token_embedding.weight
model.norm.weight                       -> norm.weight
lm_head.weight                          -> lm_head.weight（同名，也在表里）
model.layers.<i>.input_layernorm.weight -> layers.<i>.norm1.weight
model.layers.<i>.post_attention_layernorm.weight -> layers.<i>.norm2.weight
model.layers.<i>.self_attn.{q,k,v,o}_proj.weight -> layers.<i>.{q,k,v,o}_proj.weight
model.layers.<i>.self_attn.{q,k}_norm.weight     -> layers.<i>.{q,k}_norm.weight
model.layers.<i>.mlp.{gate,up,down}_proj.weight  -> layers.<i>.{gate,up,down}_proj.weight
```

三件事分开管，谁出问题报谁的错：

| 问题 | 谁拦下 | 报错形式 |
|---|---|---|
| 参数名不认识（例如 `self_attn.rotary_emb.inv_freq`） | 映射表 | `ValueError: 权重里有本实现不认识的参数名: [...]` |
| 目标参数漏装 / 多装 / shape 不符 | `load_state_dict(strict=True)` | `RuntimeError: Missing key(s) / size mismatch` |
| 不是 FP32 | `read_raw_weights()`（改名**之前**） | `ValueError: 权重 X 的 dtype 是 float16` |

「不认识的参数名」必须在改名之后单独查一次：只靠 `strict=True` 的话，源文件里多出来的张量被静默丢掉，报的是「目标少了一个参数」，指向的是错的地方。

**这些线性层都是 PyTorch 的 `[out, in]` 布局，不转置。** 看到「权重 × 输入」就顺手 `.T` 是这一步最容易犯的错。

### 2.5 明确拒绝，不假装支持

范围外的外部配置一律在 `to_internal_config()` 里报错：

| 拒绝项 | 触发条件 |
|---|---|
| MoE | `num_experts` / `num_local_experts` 非空 |
| 非 FP32 | `dtype` 既不是 `float32` 也不是缺省 |
| attention bias | `attention_bias=True` |
| sliding window | `use_sliding_window=True` 或 `sliding_window` 不为 `null` |
| 逐层类型 | `layer_types` 里出现非 `full_attention`（**光看滑动开关不够**） |
| 非 silu MLP | `hidden_act` 不是 `silu` |
| 共享权重 | `tie_word_embeddings=True` |
| 别的 EOS | `eos_token_id != 4`（Engine 的约定是固定的） |
| 非普通 RoPE | `rope_type != "default"` |
| 分片权重 | 目录里有 `model.safetensors.index.json` |

`architectures`、`transformers_version` 之类不影响计算的字段忽略，不因为多一个无关字段就拒绝目录。

### 2.6 适配在加载时发生，不在 step 里

```text
read_raw_config -> to_internal_config -> build_model_from_config -> 严格装权重 -> _init_runtime
```

四步都在 `Engine.from_model_dir()` 里一次走完。权重装完才建 Engine，所以**加载失败不会交出半个 Engine**（比如权重少一层，`load_state_dict` 抛异常，调用方拿不到对象）。之后每次 `step()` 走的完全是内部那套，跟目录来源无关。

### 2.7 拆分没有改变计算

拆分是纯搬运。实测：同种子下 `step29.TinyCausalLM` 与 `step30.TinyCausalLM` 的 `state_dict` 键集合、逐位数值、buffer 名字全部相同；同样的请求在 CPU/Torch、GPU/Torch、Triton、Triton+Graph 四条路径上，输出与轮数都与 step29 完全一致。

## 3. 验证

conda `vllm-omni-dev`，RTX 5090 Laptop，torch 2.13.0+cu130，Triton 3.7.1，Transformers 5.14.1，FP32。

### 3.1 与官方 Qwen3 对照

用 `AutoModelForCausalLM.from_pretrained(目录, attn_implementation="eager")` 当独立参考（**只在验收脚本里用，没有进实现**），比同一段 token 的逐位置 logits 和每层 KV：

| 目录 | logits 最大绝对误差 | argmax 一致 | 每层 K/V 一致 |
|---|---:|---|---|
| `tiny_gqa` | 5.960e-08 | True | 2/2 |
| `tiny_mqa` | 7.451e-09 | True | 3/3 |

`tiny_mqa` 的 3/3 说明 MQA（Q/KV heads = 4/1）加上非 2 的幂的 `head_dim=6` 也对上了。

### 3.2 Engine 出口

三条路径（Torch / Triton / Triton+Graph）对三个请求的输出完全一致。分块 prefill 预算 1 / 2 / 3 / 4 / 8，**每个请求的输出都一致**（预算不同时请求完成的先后次序会变，所以按请求 id 比，不按回调顺序比）。prefix cache 命中 2 块、`cache.length=9`。

### 3.3 CPU 与 GPU

两个目录上，`device="cpu"` + Torch 与 `device="cuda"` + Torch 生成结果一致。

### 3.4 外部加载 → 自己的格式 → 再加载

`save_model()` 写出 `format_version=2`、`head_dim=16`、`use_qk_norm=true` 的目录；换随机种子只凭目录重建，输出一致、权重逐位相同。源目录文件未被修改。

### 3.5 新进程、不同 seed

三个独立进程分别在 `torch.manual_seed(0/12345/999)` 下只凭目录加载 `tiny_mqa`：权重摘要与生成结果完全相同。

### 3.6 失败用例

| 情况 | 结果 |
|---|---|
| `model_type=qwen3_moe` | `ValueError: 不支持的 model_type='qwen3_moe'` |
| `attention_bias=True` | `ValueError: 不支持 attention_bias=True` |
| `sliding_window=512` / `use_sliding_window=True` | `ValueError: 不支持 sliding window` |
| `layer_types` 含 `sliding_attention` | `ValueError: 不支持这些层类型 ['sliding_attention']` |
| `hidden_act=gelu` | `ValueError: 不支持的 hidden_act='gelu'` |
| `tie_word_embeddings=True` | `ValueError: 不支持 tie_word_embeddings=True` |
| `eos_token_id=2` | `ValueError: 不支持 eos_token_id=2` |
| `dtype=float16` | `ValueError: 不支持的 dtype='float16'` |
| `num_experts=8` | `ValueError: 不支持 MoE 配置` |
| `rope_type=linear` | `ValueError: 不支持的 rope_type='linear'` |
| 缺 `head_dim` / 缺 `rope_parameters` | `ValueError: 外部配置缺少字段 ...` |
| `hidden_size` 是字符串 | `ValueError: 外部配置字段 d_model 必须是整数` |
| 权重多一个不认识的名字 | `ValueError: 权重里有本实现不认识的参数名: [...]` |
| 权重少一层 | `RuntimeError: Missing key(s) in state_dict` |
| 权重 shape 不符 | `RuntimeError: size mismatch for lm_head.weight` |
| 权重不是 FP32 | `ValueError: 权重 lm_head.weight 的 dtype 是 torch.float16` |
| 分片目录（有 `index.json`） | `ValueError: ... 是分片权重目录` |

### 3.7 旧路径没有回归

- 旧自定义目录（`format_version=1` / `2`）仍能加载，权重逐位一致。
- 把验收方第二十九关的三个脚本原样指到拆分后的包上跑：

| 脚本 | 结果 |
|---|---|
| `verify_step29_contract.py` | **96/96**（其中两处 monkeypatch 的目标改成 `m.attention._paged_attention_kernel`，见 2.1） |
| `verify_step29_io_contract.py` | **45/45**（无需改动） |
| `verify_step29_qwen3.py` | 通过 |

回归脚本里的 96 个用例在改掉 patch 目标后全部通过，说明拆分没有改变行为，只是私有名字换了住处。

## 4. 接口变化与遗留

- **代码布局变了**：`step30/` 是一个包，实现分散在 `attention/cache/model/sampler/scheduler/engine/formats/` 里，`step30/step30.py` 只是入口和演示。导入方式：项目根加入 `sys.path` 后 `import step30`；`python step30/step30.py` 也能直接跑。
- `Engine.from_model_dir()` 签名不变，现在接受自定义目录**和**外部 Qwen3 目录，按 `model_type` 自动分派。
- 包顶层仍导出 `Engine` / `save_model` / `load_model_config` / `load_model_weights` / `build_model_from_config` / `FORMAT_VERSION` 等，并额外导出 `native_format`、`qwen3_format` 两个适配器模块。
- 新增 `formats/` 适配层；私有名字 `_paged_attention_kernel` / `_rotate_half` / `_stable_hash` 的所在地变成各自的模块（包顶层仍 re-export）。
- 遗留：仍是普通 RoPE，没有 sliding window、没有 QKV bias、没有共享 embedding/lm_head、不支持低精度与分片权重、不做 tokenizer 与文本接口、不反向导出 Transformers 格式。
- 遗留：外部加载只支持样例采用的那一种配置写法（`rope_parameters` 形式、必须写明 `head_dim`），历史版本配置不做兼容层。
- 未提交。
