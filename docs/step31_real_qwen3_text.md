# step31：用自己的 Engine 生成真实文本

- 对应代码：`step31/`（新增，未提交）
- 包摘要 SHA256：`a6c6f6842b4af9831d7236eb5b086d38e370788540cc096c75d1b5a629c90b9e`
  （`sha256("\n".join(sorted("路径 sha256(文件)")) + "\n")`，只算 `step31/**/*.py`）
- 基线：`step30/` 原样保留、未修改

## 0. 需求大概

不再增加 Transformer 公式，把已有的各部分接成一条能用的链路：

```text
一句话 → tokenizer → token IDs → 自己的 Engine → token IDs → tokenizer.decode → 一句话
```

真实模型在 `/home/user/proj/KuiperLLama/Qwen/Qwen3-0.6B`（只读）。它与上一关的随机样例有四处差异，每一处都要处理：

| 上关的限制 | 真实目录 |
|---|---|
| 配置 `dtype=float32` | `torch_dtype=bfloat16`，311 个 Tensor 全是 BF16 |
| `rope_parameters` 里的 theta | 顶层 `rope_theta=1000000`，`rope_scaling=null` |
| embedding 与 lm_head 独立 | `tie_word_embeddings=true`（两个名字都在，数值相同） |
| 固定 EOS=4 | config 里 151645，generation_config 里 `[151645, 151643]` |

本关约定：**文件允许 FP32/BF16，加载后统一 FP32；模型、KV cache 和计算全程 FP32**，所以不动 Triton 的累加精度，也不把权重改成 BF16。

## 1. 改动内容

| 文件 | 内容 |
|---|---|
| `step31/formats/__init__.py` | 改读 `config.json` + 可选 `generation_config.json`；`read_raw_weights()` 按适配器声明的 dtype 白名单检查并统一转 FP32 |
| `step31/formats/qwen3.py` | 认 `dtype`/`torch_dtype`；两种普通 RoPE 写法；tied 权重的补齐与核对；EOS 解析 |
| `step31/formats/native.py` | `FORMAT_VERSION` 2 → 3，新增 `eos_token_ids`；v1/v2 默认 `{4}` |
| `step31/model.py` | 新增 `DEFAULT_EOS_TOKEN_IDS`、`normalize_eos_ids()`、`TinyCausalLM.eos_token_ids` |
| `step31/scheduler.py` | `== 4` → `in self.eos_token_ids`；新增 `eos_token_ids` 参数 |
| `step31/engine.py` | 适配器接口多了 generation_config 与 `raw_config`；EOS 透传给调度器 |
| `step31/step31.py` | 文本入口：chat template 编码、提交、只 decode 新生成的部分 |

## 2. 设计要点

### 2.1 四个差异分别落在哪一层

接真实模型最容易出的错是「把加载期的差异漏到运行期」。这四处差异的属性并不一样：

| 差异 | 性质 | 处理位置 |
|---|---|---|
| BF16 文件 | **精度**，只影响加载 | `read_raw_weights()`：验完 dtype 就地转 FP32，之后所有代码都不知道还有 BF16 这回事 |
| `rope_theta` 位置 | **字段名**，只影响配置翻译 | `_rope_theta()`：两种写法读同一个数 |
| tied 权重 | **权重布局**，只影响加载 | `_reconcile_tied_weights()`：改名之后、`load_state_dict` 之前 |
| EOS | **生成控制**，要从配置一路走到结束判断 | 翻译进配置 → 装到模型 → 交给调度器 |

四条都在加载时一次性做完。之后 `step()` 走的完全是内部那套，与目录来源无关。

### 2.2 文件精度不等于运行精度

```text
读取 safetensors → 逐个 Tensor 查实际 dtype（不信配置里的字符串）
                 → 允许就转 FP32，不允许就报错
                 → 改名 → tied 核对 → load_state_dict(strict=True)
```

- 配置里的声明和实际 Tensor 的 dtype **分别检查**：声明是 `dtype` 或 `torch_dtype` 都认，但只接受 `float32` / `bfloat16`；实际 Tensor 也必须在适配器声明的白名单里。
- 白名单由适配器给出（`WEIGHT_DTYPES`）：外部 Qwen3 是 `(float32, bfloat16)`，自己写的目录只有 `(float32,)`。**允许 BF16 是外部格式的事，没有顺手放宽自己的格式。**
- 转换放在**改名之前**，所以后面 tied 权重的比较比的是最终会装进去的 FP32 数值。

`load_state_dict` 本来就会把 dtype 静默转成目标参数的类型——如果只靠它，一个 FP16 目录会被悄悄接受。所以检查必须在它之前自己做。

### 2.3 tied 权重：补齐、核对、不擅自覆盖

`tie_word_embeddings=true` 表示 embedding 与 lm_head 是同一套数值，但**目录里存几个名字是导出方决定的**。本地这份 1.5 GB 权重两个名字都写了，而且数值逐位相同；别的导出可能只写一个。

| 目录里有什么 | 处理 |
|---|---|
| 两个名字都有且相同 | 直接用 |
| 两个名字都有但**不一致** | 报错，不覆盖其中一个 |
| 只有 embedding | 用 embedding 补齐 lm_head |
| 只有 lm_head | 用 lm_head 补齐 embedding |
| 一个都没有 | 报错 |

`tie_word_embeddings=false` 时**完全不碰**：两个独立参数缺哪个由 `strict=True` 报错，不能因为名字像就擅自复制。

本关只要求推理等价，所以装进去的是两份数值相同的独立 FP32 参数，没做物理存储去重。**代价是显存里 embedding 存了两份**（0.6B 模型里这是 2 × 622 MB）。保存时也不去重——两份都写，共享底层存储不会让 CPU 保存出问题。

### 2.4 EOS：从写死的 4 变成随模型走的数据

优先级和归一化规则：

```text
generation_config.json 声明了 eos_token_id  →  用它
否则 config.json 声明了                      →  用它
两个都没声明                                 →  报错（不猜：真实模型里 4 是普通 token）
```

外部格式要求必须声明；自己的格式在 v1/v2 目录（那时行为就是写死的 4）上回落到 `{4}`，所以旧例子不受影响。

归一化 `normalize_eos_ids()`：接受一个整数或一个非空整数列表，统一成升序去重的 tuple，并检查每个 id 落在 `[0, vocab_size)`。空列表、非整数、越界都明确报错。

**存放位置**：`eos_token_ids` 是模型上的一个属性，因为它必须能随模型一起保存。但它**不写进 `model_config()`**——`model_config()` 描述的是模型结构，EOS 是生成控制信息，和 `format_version` / `dtype` 一样由 `save_model()` 另外写进 config。`FORMAT_VERSION` 因此提到 3：v3 必须写明 `eos_token_ids`，v1/v2 默认 `[4]`。这样「保存再加载」不会丢停止规则，也不会让 `model_config()` 的含义变浑。

`Engine` 建调度器时把 `model.eos_token_ids` 传下去，结束判断从 `== 4` 变成 `in self.eos_token_ids`。

### 2.5 文本入口

```python
text = tokenizer.apply_chat_template(messages, tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
prompt_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
```

- 角色标签和特殊 token 由官方模板加，不手工拼——手工拼很容易和官方输入差几个 token，那就不是同一个输入了。
- `enable_thinking=False`：本关先做简短回答（实测 prompt 末尾是空的 thinking 段 `151667, 271, 151668, 271`）。
- 提交 `prompt_ids` 给 Engine，**只 decode 新生成的 `output_ids`**，`skip_special_tokens=True`。
- 多条问题各自编码后一起提交，仍是一条消息、一批并发。

## 3. 验证

conda `vllm-omni-dev`，RTX 5090 Laptop，torch 2.13.0+cu130，Triton 3.7.1，Transformers 5.14.1，FP32，TF32 关闭。

### 3.1 与官方参考逐 token 对照

验收方给了一份官方 FP32 贪心生成的记录（同一个 prompt、`max_new_tokens=32`）。本实现：

```
output_ids 与参考逐位一致 = True   （31 个 token，最后一个 151645 在停止集合里）
decode 出的文本与参考逐字一致 = True
```

> KV cache 是一种用于缓存键值对（Key-Value Pair）的内存结构，用于减少重复访问和提高数据访问效率。

不是「最终文本差不多」——31 个 token 一个不差。

### 3.2 数值口径

| 检查 | 结果 |
|---|---|
| 与官方 FP32（`attn_implementation="eager"`）逐位置 logits 最大绝对误差 | 9.155e-05（logits 量级到 23） |
| 首 token top5 的 id | 与参考完全一致 |
| 首 token top5 的分数最大绝对误差 | 2.10e-05 |
| top1 与 top2 的分差 | 3.805（比误差大四个数量级） |
| 权重 dtype / tied 两份 | 全部 FP32 / 数值相同 |

逐层 KV 对照（19 个 prompt token）：

| 层 | K 绝对误差 | V 绝对误差 |
|---:|---:|---:|
| 0 | 3.05e-05 | **0.00e+00** |
| 13 | 1.69e-05 | 9.66e-06 |
| 27 | 3.58e-05 | 1.57e-04 |

**第 0 层的 V 是逐位相同**（0.00e+00），说明 embedding → norm1 → v_proj 这条路径完全没有偏差，分歧是从 attention 之后、随层数累积的 FP32 运算顺序差异（我的分块 online softmax 与显式 RoPE 查表 vs 官方实现）。到第 27 层 K/V 的相对误差约 1%，绝对误差仍在 1e-4 量级；最终 logits 误差 9e-05，argmax 与 31 个 token 全部一致。**本关要求「数值接近、不要求逐位相等」，这里没有出现 token 分歧。**

### 3.3 各路径与调度

三条短文本请求（19 / 16 / 13 个 prompt token）一起跑：

| 路径 | 步数 | 生成 token 数 | 输出 |
|---|---:|---|---|
| GPU / Torch | 24 | 24 / 24 / 8 | 一致 |
| Triton eager | 24 | 24 / 24 / 8 | 一致 |
| Triton + Graph | 24 | 24 / 24 / 8 | 一致（按 N 捕获，本次 N ∈ {2, 3, 48}） |

- 分块 prefill 预算 8 / 20 / 64：三步数不同但每个请求输出都与整批一致。
- prefix cache：两个相同请求输出相同；第三个相同请求命中 1 个前缀块（19 token / block 16，至少留最后一个 token 重算）。

### 3.4 加载边界（用随机小样例覆盖）

| 检查 | 结果 |
|---|---|
| BF16 目录（`torch_dtype` 写法 + 实际 BF16 张量） | 装入后 `torch.float32`，输出与 FP32 目录一致 |
| 声明 `dtype=float16` | `ValueError: 不支持的 dtype='float16'` |
| 实际张量是 float16 | `ValueError: 权重 ... 的 dtype 是 torch.float16` |
| 顶层 `rope_theta` + `rope_scaling=null` | 读到 `theta=10000.0` |
| `rope_scaling` 非空 | `ValueError: 不支持的 rope_scaling=...` |
| tie=true，两个都在且相同 | 通过 |
| tie=true，两个都在但冲突 | `ValueError: ... 数值不一致；不擅自覆盖其中一个` |
| tie=true，只给 embedding | lm_head 补齐且数值相同，输出与「两个都给的 tied 目录」一致 |
| tie=false 且缺 lm_head | `RuntimeError: Missing key(s) in state_dict` |
| EOS 来自 generation_config（列表 `[7,9]`） | `eos=[7, 9]`，**优先于** config |
| 只有 config 声明 | `eos=[4]` |
| 两个文件都没声明 | `ValueError: ... 本实现不猜停止规则` |
| EOS 越界 / 空列表 | 分别报「超出词表范围」「停止规则不能为空」 |

### 3.5 停止规则真的生效

用一个小模型先取它自然的贪心序列，再用这个序列反过来设停止规则：

```
自然贪心序列             [5, 3, 3, 3, 3, 3, 3, 3]
单 EOS = 序列第 1 个     输出 [5]        停在 5 ✓
多 EOS = {序列[1], 序列[3]} 输出 [5, 3]     停在集合内 ✓
EOS=10（≠4），prompt [4,7]  输出 [4, 6, 10]  穿过了普通 token 4 ✓
遇不到停止符             输出 4 个 = max_new_tokens ✓
```

第三行验证的是「4 不再是特殊数字」：贪心序列里出现 4 时不会提前结束；最后一行验证达到 `max_new_tokens` 也会正常收尾。

### 3.6 保存 / 重新加载

- 真实模型 → `save_model()` → 3.01 GB，`format_version=3`、`eos_token_ids=[151643, 151645]`。
- 重新加载后 31 个 token 逐位一致，权重逐位相同。
- v2 旧目录（手工把 `format_version` 改回 2 并删掉 `eos_token_ids`）→ `eos=[4]`，仍可加载；v3 缺 `eos_token_ids` 明确报错。
- tied 权重保存后文件里两个名字都在、数值相同，**共享底层存储没有让 CPU 保存报错**。

### 3.7 源目录未被改动

跑完整套之后重新计算 `config.json`、`generation_config.json`、`tokenizer_config.json`、`vocab.json`、`tokenizer.json`、`merges.txt` 以及 `model.safetensors` 文件头的 SHA256：**全部与跑之前相同**。全程只读。

### 3.8 上一关没有回归

- `fixtures/step30_qwen3` 两个样例仍能加载，EOS 取到 `{4}`（它们的 `generation_config.json` 就写着 4），输出与第 30 关一致。
- 把验收方第二十九关的三个脚本原样指到 `step31` 上跑：

| 脚本 | 结果 |
|---|---|
| `verify_step29_contract.py` | **96/96**（monkeypatch 目标为 `m.attention._paged_attention_kernel`） |
| `verify_step29_io_contract.py` | **45/45**（无需改动） |
| `verify_step29_qwen3.py` | **21/21** |

`verify_step29_io_contract.py` 里有一处 `assert e.model.model_config() == config(variant)`，正是它让我把 `eos_token_ids` 从 `model_config()` 里拿出来、改由 `save_model()` 单独写——`model_config()` 的契约是「模型结构」，不该被生成控制信息污染。

## 4. 接口变化与遗留

- `TinyCausalLM(...)` 新增 `eos_token_ids=None`（默认 `(4,)`）与属性 `model.eos_token_ids`（升序 tuple）。
- `Scheduler(...)` 新增 `eos_token_ids=None`；判断从 `== 4` 改为 `in self.eos_token_ids`。
- `Engine(...)` 新增 `eos_token_ids=None`；`Engine.from_model_dir()` 签名不变。
- 适配器接口变了：`to_internal_config(raw, generation)`、`to_internal_weights(weights, raw_config)`，并新增类属性 `WEIGHT_DTYPES`。
- `formats.read_raw_config()` 现在返回 `(adapter, raw, generation)` 三元组；`read_raw_weights(model_dir, allowed_dtypes)` 多了第二个参数。
- 自定义格式 `FORMAT_VERSION` 升到 3（新增 `eos_token_ids`）；v1/v2 仍可加载，EOS 回落到 `{4}`。
- `model_config()` 的返回**没有变**：仍然只描述模型结构。
- 新增 `step31/step31.py` 文本入口：`python step31/step31.py [问题 ...] [--model-dir ...] [--max-new-tokens N] [--backend torch|triton] [--graph]`。
- 遗留：tied 权重没有做物理存储去重，显存里是两份；没有做低精度计算、量化、并行或 HTTP 服务；仍不支持 sliding window、QKV bias、分片权重、非 FP32/BF16 文件与反向导出官方格式。
- 遗留：CPU 路径在真实模型上没有跑（0.6B 逐层 forward 太慢），CPU 正确性由随机小样例覆盖。
- 未提交。
