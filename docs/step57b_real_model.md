# step57B：把真实模型接进协议

- 对应代码：`step57/`（在 57A 的骨架上继续长；新增 `layers/`、`attention/`、`models/`、
  `model_loader/`、`sample/`，以及 `worker/` 下的三个文件）
- 包摘要 SHA256：`5aafc312b8fe066c…`（49 个 .py / 4436 行；口径 = 包内 `*.py` 按相对路径排序，
  每个文件取自身 sha256，拼成 `名字\0哈希\n` 再取 sha256）
- 验收脚本：`benchmarks/check_step57_{runner_inputs,weight_loading,model_logits}.py`
  （对应需求里点名的 `test_runner_inputs.py` / `test_weight_loading.py` / `test_model_logits.py`）
- 演示入口：`python step57/step57.py`（本机 `models/Qwen3-1.7B`，短生成；`--trace` 打印第一轮
  喂给模型的数字）
- 参考实现：本机 `vllm 0.28.0`（`v1/worker/gpu_model_runner.py`、`v1/worker/gpu_input_batch.py`、
  `v1/worker/block_table.py`、`model_executor/model_loader/*`、`model_executor/models/utils.py`、
  `model_executor/models/{qwen2,qwen3}.py`、`model_executor/layers/*`）
- 外部参照（只在测试里当 oracle，不参与实现）：`transformers 5.14.1` 的 `Qwen3ForCausalLM`

## 0. 需求大概

198 定的这一段：把真实模型放到 57A 那套协议后面——Qwen3 小配置、权重加载、Attention 边界、
Runner 输入打包、简单物理 KV，单卡 eager。判定点是"GQA/qk norm/RoPE 不遗漏、打包加载正确、
full/chunk/decode logits 对照、输入打包数字与 198 一致、Scheduler 未访问 GPU tensor"。

200 §57B 给的三件交付物是 `test_model_logits.py`、`test_weight_loading.py`、
`test_runner_inputs.py`，外加一个本地模型的短生成示例。这一关**不做**性能优化（逐请求的
Torch attention 是刻意的参考实现），也不碰前缀缓存/抢占（57C）。

## 1. 改动内容（新增的层与它们各自回答什么）

| 文件 | 对应 vLLM | 它回答的问题 |
|---|---|---|
| `models/qwen3.py` | `models/{qwen2,qwen3}.py` | 模型长什么样：GQA + q/k norm + RoPE + 残差双链；`forward/embed_input_ids/compute_logits/load_weights` 四个接口 |
| `layers/linear.py` | `layers/linear.py` | 打包参数怎么建、**每个参数自己怎么把分片写进去**（`weight_loader`）；**TP=1，没有通信** |
| `layers/layernorm.py` / `activation.py` / `rotary_embedding.py` | 同名文件 | 融合残差的 RMSNorm（FP32 方差）、`SiLU(gate)*up`、预计算 cos/sin 表的 RoPE（扁平入出） |
| `attention/layer.py` | `attention/layer.py` | 模型里的"注意力边界"：按层名从 forward 上下文取 metadata、把 KV 交给后端 |
| `attention/forward_context.py` | `forward_context.py` | 一次 forward 的"层名 → AttentionMetadata"，`finally` 恢复，异常也恢复 |
| `attention/metadata.py` | `v1/attention/backends/*` | 分页 KV + 变长 batch 的四个张量（本关子集，不冒充统一签名） |
| `attention/backends/torch_sdpa.py` | `v1/attention/backends/torch_sdpa.py` | 写 KV（`slot_mapping` 散射）+ 按**绝对位置** causal 的逐请求 attention |
| `model_loader/weight_utils.py` | `model_loader/weight_utils.py` | 目录 → `(名字, 张量)` 的**流式**读取 + 检查点自洽性校验 |
| `model_loader/auto_weights_loader.py` | `models/utils.py` | 名字路由（打包映射）、递归委派给子模块、缺失/未知/重复检查、**覆盖检查** |
| `model_loader/base_loader.py` | `model_loader/base_loader.py` | "选类 → 构造 → 灌权重"里与来源无关的那部分 |
| `model_loader/default_loader.py` / `loader.py` | 同名文件 | 本地 safetensors 目录；按目录布局选加载器 |
| `sample/sampler.py` | `v1/sample/sampler.py` | 最小采样：贪心 / 温度随机；其余参数**明确报错**（属 57D） |
| `worker/block_table.py` | `v1/worker/block_table.py` | 块表镜像（CPU 权威副本 + 一次 H2D）与 `slot_mapping` 计算 |
| `worker/gpu_input_batch.py` | `v1/worker/gpu_input_batch.py` | 请求 ↔ batch 行的账本：定长缓冲、原地写、`condense` 重排 |
| `worker/gpu_model_runner.py` | `v1/worker/gpu_model_runner.py` | 一轮：`_update_states` → `_prepare_inputs` → 模型 → `sample_tokens` → 记账 |
| `step57/step57.py` | （无对应，演示入口） | 真实模型短生成 |

57A 的四个文件因为 57B 的接线做了小改（都在文档 §3 的账本里）：

- `worker/worker.py`：没注入 Runner 时**建真实 Runner**（原来直接报错）；`initialize_from_config`
  真的去分配并绑定 KV；
- `engine/core.py`：拿到容量后立刻 `initialize_kv_cache(cache_config)`（195 §8 的第三步），
  并在 `preprocess_add_request` 做输入边界检查；
- `attention/layer.py`：`Attention` 改成 `nn.Module`（不然它不出现在 `named_modules()` 里，
  Runner 找不到层、KV 无处可绑）；
- `testing/fake_runner.py`：补一个空的 `initialize_kv_cache`（协议里有这一格）；
- `engine/core.py` 与 `worker/worker.py` 的改动都能被 57A 的用例观察到，所以那三个脚本重跑过。

## 2. 设计要点

### 2.1 三种状态、三种角色，不能混

    控制端权威（Scheduler / KVCacheManager）
      → 执行端镜像（CachedRequestState + InputBatch + BlockTable 的 CPU 副本）
        → GPU 输入副本（每轮上传的 input_ids / positions / slot_mapping / 块表）

镜像的进度（`num_computed_tokens`）**每轮由协议校正**，执行侧不自增。这一条是踩出来的：
第一版只把协议值写进 `CachedRequestState`、忘了写 `InputBatch` 的镜像，于是 `positions`
按旧值算、`seq_lens` 永远追不平已知历史，**任何请求都判不出 ready**，整条链路原地空转
（表现是调度器的"排不出 token"保护报错，跟真正的原因隔了两层）。同理，块表镜像也要在
续跑时**追加**新块、在 resumed 时**整表替换**——只改 `CachedRequestState` 是不够的，
模型读的是镜像。

### 2.2 输入打包（198 §4 的逐值对照）

`_prepare_inputs` 全在 CPU 上用 Torch 算术算，不上 kernel（198 §4 明确要求先分开职责）：

```text
req_indices = repeat(arange(num_reqs), num_scheduled)      # [0,0,1,1,1]
query_pos   = concat(arange(n_i))                          # [0,1,0,1,2]
positions   = num_computed_tokens[req_indices] + query_pos  # 绝对位置，不是行内序号
input_ids   = token_ids_cpu.flatten()[positions + req_indices * max_model_len]
seq_lens    = num_computed_tokens + num_scheduled
slot        = block_table[req, pos // bs] * bs + pos % bs
```

需求给的那组数字（A 续跑 2 个 token、B 新来 3 个，`block_size=4`、块表 A=[7,2] B=[9]）在
用例里是**逐值断言**的：`input_ids=[4,5,6,7,8]`、`positions=[3,4,0,1,2]`、
`query_start_loc=[0,2,5]`、`seq_lens=[5,3]`、`slot_mapping=[31,8,36,37,38]`、
`logits_indices=[1,4]`。

两条容易忽略的：

- `positions` 是**绝对位置**。写成"本轮第几个"的话，chunked prefill 的第二块和 decode 都会
  错，而错误只体现在 attention 的 mask 和 RoPE 上——数值上"看起来还在跑"。
- `slot_mapping` 越界要**当场报错**：`pos // bs` 超出这一行已有的块数，说明控制面给少了块
  （或 positions 算错），此时按 0 号块写进去会覆盖别人的 KV。这条检查在 `BlockTable` 里。

### 2.3 只对 ready 行采样，并且**映射要显式重建**

"算完之后追平已知历史"的请求才能产出 token（`seq_lens == num_tokens_no_spec`）。中间 prefill
块即使有 hidden 也不该产出。本关直接从采样阶段就不算它们（198 §4 允许的显式教学差异：
本机是"先算采样行、再在 `_bookkeeping_sync` 里丢掉不该提交的结果"）。

筛行之后**不能拿原列表 zip**：`sample_row → InputBatch 行 → req_id` 这条链要显式走一遍
（`InputBatch.req_id_at(row)`）。行号不是身份——`condense()` 会把后面的行搬到前面，
token 缓冲、块表、采样参数、**generator** 都要跟着请求走；漏搬一个就会出现"token 是 A 的、
随机流是 B 的"。

### 2.4 权重加载：三层 + 覆盖检查

```text
loader.get_model           选加载器（本地 safetensors）
  └─ base_loader.load_model   选模型类（registry）→ 构造 → 灌权重
       └─ DefaultModelLoader.get_all_weights   目录 → 流式 (名字, 张量)
            └─ Qwen3ForCausalLM.load_weights    顶层：跳过 lm_head.
                 └─ Qwen3Model.load_weights     这一层挂 hf_to_vllm_mapper（打包映射）
                      └─ Parameter.weight_loader   某个 shard 写进目标参数的哪一段
```

"第几段"这个信息 vLLM 是**挂在张量对象的属性上**（`tensor.shard_id = "q"`，加载器再
`getattr(weight, "shard_id", None)` 取回来），所以改名与分段在一次遍历里同时完成。本关照抄
这条通道；非打包参数收到 `shard_id` 会直接报错（说明映射指错了目标）。

**覆盖检查**是本关对 `strict=False` 的替代：加载完把"已填写的参数名"与
`named_parameters()` 对账，缺一个就报错。漏掉 `q_norm` 这类权重是最典型的静默错误——模型
能跑、输出只是"有点不对"。豁免只有两种：`skip_prefixes` 声明过的（tied 的 `lm_head.`）
与共享参数（`named_parameters()` 本来只报第一个名字）。

流式读取的预检值得单说：**只看名字**先把检查点跑一遍（index 点名的 shard 是否存在、张量有没有
放错 shard、名字有没有缺、有没有重名、前缀组是否连续），全过了才开始读张量。"检查点坏了"
不该变成"加载到一半才发现"。

### 2.5 tied embedding：共享对象 + 跳过检查点里那一份

Qwen3-1.7B 的检查点里**同时**有 `model.embed_tokens.weight` 与 `lm_head.weight`（内容相同），
而模型里这两者是同一个 `Parameter`（`ParallelLMHead.tie_weights()` 直接共享对象，省 1.2 GB）。
于是检查点里的 `lm_head.weight` 必须**跳过**：不跳过的话第二份会原地再覆盖一遍，数值上无害，
但"哪份是真相"就说不清了。

### 2.6 Attention 边界的两个坑

- **`Attention` 必须是 `nn.Module`**。第一版是普通 Python 类，结果 `named_modules()` 里根本
  没有它——Runner 靠遍历模型找 Attention 层来绑 KV、挂 metadata，找不到就"KV 无处可绑"。
- **层名必须与模块路径逐字相同**。`Attention.layer_name` 是 forward 上下文的键，
  Runner 用它建 map。前缀拼接写成 `f"{prefix}.attn"` 时顶层前缀是空串，层名会多一个开头的点，
  对应关系当场断掉（`maybe_prefix()` 就是为了这个点）。Runner 在建 map 时会断言两者相等。

### 2.7 两步协议与失败停摆

`execute_model()` 跑完模型**返回 `None`** 并把 logits 存进 `execute_model_state`；
`sample_tokens()` 才消费它。两条边界都要报错：上一轮没消费就再来一轮、没有 execute 就 sample。
模型前向抛异常时**清掉状态并让 Runner 停摆**（`failure` 字段），不把没有依据的半成品留给下一次
采样——GPU 已经写进去的 KV 不回滚，也不假装能回滚。

## 3. 与 vLLM 的差异账本

| 差异 | 原因 / 后续 |
|---|---|
| **只支持 TP=1**，`layers/linear.py` 里的 `*ParallelLinear` 没有通信 | 保留名字是为了源码一一对应；文档与 README 都写明，改 tp_size 不会跑 |
| 加载器只有 `WeightsMapper.orig_to_new_stacked` 一种映射 | vLLM 还有正则/前缀/后缀/重命名与 `__or__` 合并；本关只需要打包映射 |
| **有覆盖检查**（vLLM 没有） | "漏加载"在 vLLM 里是静默的（只是结果不对）；本关把它挡在加载阶段。副作用：不能靠外层 `skip_prefixes` 跳掉子模块的整层参数（子模块自己的加载器会拦），这是有意的 |
| `.bin` / GGUF / 量化检查点**明确报错** | 本关只读 safetensors；静默退回空模型比直接失败难查 |
| `InputBatch.condense()` 用**保序压实** | vLLM 用尾部行交换进空洞（少搬几行）；保序让"行序 == 加入顺序"，`_prepare_inputs` 的数字才好逐值对照 |
| 采样器只有贪心 / 温度随机，逐请求采样 | top_k/top_p/惩罚项与批采样属 57D；配置里出现就报错，不静默忽略 |
| `_bookkeeping_sync` 里不做 `_update_states_after_model_execute` | 198 §5 明确说那是 hybrid 模型的投机状态修正，本关 Full Attention 不需要 |
| 采样行之外的请求不产出（`[]`） | 198 §4 允许的显式教学差异，不照搬 `generator.get_offset() - 4` 那套 RNG 记账 |
| 空轮（0 token）不碰模型，直接回空结果 | 与 57A 一致：这条约束落在执行侧 |
| `num_gpu_blocks` 手动配置 | vLLM 靠显存 profiling 自动定容；容量由执行侧回报这条设计已保留在 `get_cache_config` 的位置上 |
| `preprocess_add_request` 拦 prompt 过长、截 `max_tokens` | 对齐 vLLM `v1/engine/processor.py` 的位置；不拦的话只会看到调度器空转报错（57A 的账本里记过这个缺口，这里补上） |

## 4. 验证

| 脚本 | 项数 | 覆盖 |
|---|---:|---|
| `check_step57_engine_protocol.py` | 23 | 57A：三轮轨迹逐字段、空轮、按 ID 映射、快照隔离、结束清理两段式、协议包无活对象 |
| `check_step57_request_progress.py` | 29 | 57A：prompt 复制、唯一写入点、状态映射、排序与队列、`check_stop` 六条分支 |
| `check_step57_scheduler_basic.py` | 25 | 57A：统一预算、两个上限、分配失败原子性、结束两段式、abort、空转保护 |
| `check_step57_runner_inputs.py` | 29 | 198 §4 逐值对照、只有 ready 行采样、`_update_states` 七步、行号不是身份（含 generator）、两步协议与失败停摆、块表越界、Scheduler 无 GPU 张量（端到端真模型）、入口边界 |
| `check_step57_weight_loading.py` | 25 | 文件层五种坏检查点各报各的错、前缀分组检查、packed qkv/gate_up 区间逐值、覆盖检查抓漏加载、未知/嵌套名字、shard_id 指错、tied embedding（含"跳过的那份不生效"）、真实 Qwen3-1.7B 分片加载 |
| `check_step57_model_logits.py` | 18 | 手写参考对照（GQA 敏感）、HF 逐位置 logits 对照 + 三次消融（qk norm / rope_theta / 扰动 KV head）、残差（最终 hidden + 用 HF 的 lm_head 复算）、两种 tie、full/chunked/decode 三切分一致、模型 forward 的签名边界 |

六个脚本全部通过（共 149 项）。数值容差按**实测**给：tiny 模型 FP32 下，与 HF 的逐位置
logits 最大差 ~5e-8（阈值 1e-4），三种切分之间 ~1e-7（阈值 1e-5），手写参考 ~6e-9（阈值 1e-5）。

真实模型短生成（`python step57/step57.py`，本机 Qwen3-1.7B，bf16，CUDA）：

```text
问: 用一句话解释什么是 KV cache。（19 个 prompt token）
答: KV cache 是在大模型推理过程中用于存储和复用键值对（Key-Value Pair）的缓
   结束原因 LENGTH
25 轮调度、48 个 token、2.32s（20.7 tok/s）
```

**没有跑**性能测试（本关不涉及）：20.7 tok/s 是逐请求 Torch attention + 无 CUDA Graph 的
参考实现的自然结果，不作为性能结论。

## 5. 接口变化与遗留

- 新增可导入的名字：`from step57 import GPUModelRunner, get_model, Qwen3ForCausalLM, Sampler`；
  子包 `step57.model_loader` / `step57.models` / `step57.layers` / `step57.attention` /
  `step57.sample` / `step57.worker` 各自导出本层的公开名。
- `Worker(vllm_config)` 现在**默认装真实模型**（原来是"没有 Runner 就报错"）；要继续用假执行
  就显式注入 `model_runner=FakeRunner(...)`（57A 的用法与用例不变）。
- `EngineCore` 在构造时就会调 `executor.initialize_kv_cache(cache_config)`——执行侧的 KV
  必须在这一步就位，否则第一轮就可能排出没有物理存储的块。
- `Attention` 由普通类变成 `nn.Module`（外部若 `isinstance(x, Attention)` 不受影响，
  但 `model.modules()` 会多出这些节点）。
- `RotaryEmbedding.forward` 的入参与返回都是**扁平** `[num_tokens, heads * head_dim]`
  （与 vLLM 的调用约定一致），内部按 `head_size` 折开再折回。
- **遗留**：字符串输入（tokenizer 在 `step57.py` 里用 transformers 代替，没有进引擎）；
  前缀缓存与抢占（57C）；采样与停止的完整实现（57D，本关的采样器遇到 top_k/top_p/惩罚项
  会明确报错）；CUDA Graph、异步、多进程、指标；多 KV group（本关所有地方都写死第 0 组）。
