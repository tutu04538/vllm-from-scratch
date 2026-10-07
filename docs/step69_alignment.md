# 第 69 关对齐记录：V1 投机输入 Padding 与编译 CUDA Graph

需求：`投机解码完整需求/069_V1投机输入Padding与编译CUDA_Graph.md`。
上游基线：本机 `vllm==0.28.0` 快照（2026-10-03）。执行路径：V1。

**一句话**：把"每轮形状都在变"的投机批补成**有限几种固定的形状**，再把这些形状的前向
**录成 CUDA Graph 重放**；同时保证补出来的假行/假请求**物理存在但绝不留痕**
（槽位哨兵、上下文长度 0、不写 KV、不参与采样）。

本文件按 §1 交付 → §2 路径与三态矩阵 → §3 关键机制 → §4 验收对照 → §5 实测 →
§6 差异账本 → §7 回归与缺口 组织。

---

## 1. 交付物

| 文件 | 内容 |
|---|---|
| `minivllm/forward_context.py`（新） | `BatchDescriptor` + `ForwardContext`（运行模式/批次键/slot_mapping）+ `set/get/override_forward_context` |
| `minivllm/cudagraph_dispatcher.py`（新） | `CudagraphDispatcher`：键表、补齐映射、`dispatch()`、`get_capture_descs()` |
| `minivllm/compilation/{__init__,monitor,cuda_graph}.py`（新） | 捕获窗口开关 + `CUDAGraphWrapper`（直通/捕获/重放、地址校验）+ `graph_capture` |
| `minivllm/config.py`（改） | `CompilationMode` / `CUDAGraphMode` / `CompilationConfig`；`ModelConfig.enforce_eager`；`VllmConfig._resolve_cudagraph_config()` / `_set_cudagraph_sizes()` |
| `minivllm/worker/gpu_model_runner.py`（改） | 静态输入工作区、`_determine_batch_execution_and_padding()`、`_prepare_inputs_padded()`、`_dummy_run()`、`_warmup_and_capture()`、`capture_model()`、`_run_model_padded()`、`initialize_cudagraph_capture()` |
| `minivllm/attention/{metadata.py,backends/torch_sdpa.py}`（改） | `uniform_query_len`（=1+K）；图内的固定形状注意力路径 `_forward_padded()`；写 KV 用 `clamp_min(0)` |
| `minivllm/core/{block_pool,kv_cache_manager}.py`、`minivllm/engine/core.py`（改） | **0 号块留白**（垃圾桶）：`reserve_null_block` |
| `minivllm/spec_decode/utils.py`（改） | `prepare_inputs_padded()`（上游 `eagle_prepare_inputs_padded_kernel` 的等价实现）、`eagle_step_update_slot_mapping_and_metadata()` |
| `minivllm/spec_decode/draft_model.py`（改） | 收下 padded 批的两个索引并逐值校验；AR 步的 device 侧算法复核；`seq_lens` 去掉被拒行；`initialize_cudagraph_keys()` |
| `minivllm/worker/worker.py`、`executor/uniproc_executor.py`（改） | `compile_or_warm_up_model()`：KV 绑定之后捕获 |
| `tests/step69/test_drafter_padding.py` | 9 项：padded 索引/上游内核逐值差分/工作区哨兵/seq_lens 修正/布局换算 |
| `tests/step69/test_spec_cudagraph.py` | 18 项：键与分派/配置解析/捕获与重放/污染 padding/profiler/无同步/68 关回归 |
| `benchmarks/check_step69_cudagraph.py` | **26 项**脚本式验收（A 配置 / B 键 / C 一致性 / D 留痕 / E profiler / F drafter padding） |
| `docs/results.json → step69` | 命令、依赖、设备、源码 hash、真实执行轨迹 |

包摘要（92 个 `*.py`）：`32be9781c5508f84f89a9c27ffbf74b1dd974ff927eac1a6d739e85318e2d7af`。

---

## 2. 路径映射与三态矩阵

| 本项目 | 上游（vllm 0.28.0） | 状态 |
|---|---|---|
| `minivllm/forward_context.py::BatchDescriptor` | `vllm/forward_context.py:29-58` | 逐字段一致（LoRA 两个字段保留但恒 False/0） |
| `ForwardContext` / `create_forward_context()` | 同文件 `:131-241` | **子集**：去掉 `no_compile_layers` / `dp_metadata` / `ubatch_slices` / `all_moe_layers`（本仓库没有编译期静态层表、DP、微批、MoE）；`set_forward_context()` 的第 2 个位置参数由上游的 `vllm_config` 换成 `num_tokens`（差异 §6.3） |
| `cudagraph_dispatcher.py` | `v1/cudagraph_dispatcher.py`（全文件） | 键构造/补齐映射/dispatch/get_capture_descs 逐行对齐；去掉 LoRA 特化与 breakable-cudagraph |
| `compilation/cuda_graph.py::CUDAGraphWrapper` | `compilation/cuda_graph.py:145-361` | 直通/捕获/重放/键缓存一致；去掉 offloader、gc patch、`compilation_counter`；**地址校验始终开启**（上游只在 DEBUG 日志级别下查） |
| `compilation/monitor.py` | `compilation/monitor.py` | 只保留 `set/validate_cudagraph_capturing_enabled`（没有编译路径，故不留 `set_torch_compile_enabled`） |
| `config.CompilationConfig` | `config/compilation.py:577-720,1119-1163,1519-1565` | **子集**：`mode` / `cudagraph_mode` / `cudagraph_num_of_warmups` / `cudagraph_capture_sizes` / `max_cudagraph_capture_size` / `compile_sizes` / `splitting_ops`；`adjust_cudagraph_sizes_for_spec_decode()` 逐行对齐（去掉 SP/TP 分支） |
| `CUDAGraphMode` / `CompilationMode` | `config/compilation.py:34-97` | 枚举与方法逐条一致（`decode_mode` / `mixed_mode` / `has_mode` / `separate_routine` / `valid_runtime_modes` / `__bool__`） |
| `GPUModelRunner._determine_batch_execution_and_padding()` | `v1/worker/gpu_model_runner.py:4055-4160` | 子集：去掉 cascade attn / DP 协调 / microbatch / encoder；保留"判统一 decode → 问分派器" |
| `_dummy_run()` / `_warmup_and_capture()` / `capture_model()` | 同文件 `:5940-6180`、`7067-7160`、`6949-7000` | 对齐形状分配与捕获顺序；**热身次数下限 1**（差异 §6.4） |
| `AttentionMetadataBuilder.build_for_cudagraph_capture()` | `v1/attention/backend.py:770-779` + 各后端实现 | 对齐语义（捕获期 `seq_lens.fill_(1)`，理由同上游注释） |
| `TorchAttentionImpl._forward_padded()` | —— | **本仓库自有**：教学后端没有分块 kernel，所以图内路径是"一次批量算完 + 固定形状 mask"。能力与显存边界见 §6.5 |
| `spec_decode/utils.prepare_inputs_padded()` | `spec_decode/utils.py:136-175`（kernel）+ `llm_base_proposer.py:1110-1172` | 逐值一致（CUDA 上与真 kernel 差分，`tests/step69` F2） |
| `spec_decode/utils.eagle_step_update_slot_mapping_and_metadata()` | `spec_decode/utils.py:88-133` | 逐值一致（同上，含 padding 行与越界行） |
| `SpecDecodeBaseProposer.initialize_cudagraph_keys()` | `llm_base_proposer.py:419-434` | 规则一致（draft 只在 PIECEWISE 下走图）；本仓库没有 PIECEWISE → 恒 NONE，并记下原因（§6.6） |
| `BlockPool(reserve_null_block=True)` | 上游 `null_block`（`v1/core/block_pool.py`） | 效应一致（0 号块永不分配）；触发条件不同（§6.7） |

**三态矩阵**（"支持 / 上游不支持 / 本项目未接线"，需求 §3.6 的纪律）：

| 能力 | 上游 0.28.0 | 本项目 |
|---|---|---|
| FULL decode 图（含投机统一 decode 批） | 支持 | **支持**（FULL_DECODE_ONLY，含 K=0 与 1+K 两种宽度） |
| PIECEWISE 图（混合 prefill/decode 批进图） | 支持（需 `mode=VLLM_COMPILE`） | **配置期明确拒绝**（`NotImplementedError`，指向 full_decode_only） |
| `torch.compile` / inductor 拆分 | 支持 | **配置期明确拒绝**（本仓库的注意力是逐请求 Torch 循环，编译只会到处 graph break） |
| LoRA 特化图 | 支持 | 不适用（没有 LoRA 实现；`has_lora=True` 直接报错） |
| drafter（EAGLE/MTP/draft_model）自己的图 | 仅 PIECEWISE | **未接线**：`cudagraph_mode` 记 NONE，回退 eager（不是"忘了"） |
| 图内 `compute_logits` / 采样 / logprobs / 语法掩码 | 图外（上游同样） | 图外（同一分层） |

---

## 3. 关键机制（为什么这么写）

### 3.1 图键就是"补齐后的批次形状"

`BatchDescriptor(num_tokens, num_reqs, uniform, has_lora, num_active_loras)`：图里的 kernel 网格、
workspace、注意力元数据形状都由这四个量决定。补齐规则（上游 `_compute_bs_to_padded_graph_size`）：

```
档位表 [1, 2, 4, 8, ...]：
  bs 正好等于档位   → 不补
  落在两档之间      → 补到**下一个**档位
  bs 超过最大档位   → 这一轮没有图（dispatch 直接返回 NONE）
```

统一 decode（`uniform=True`）的档位必须是 `1+K` 的倍数，否则 `num_reqs = 补齐行数 // (1+K)`
算不出整数。上游为此有 `adjust_cudagraph_sizes_for_spec_decode()`（把档位**向上取整**到 1+K 的倍数，
丢掉超过上限的），本项目逐行照抄（`tests/step69` A4）。**不做这一步的后果**：K=3 时默认档位里的 8
不是 4 的倍数 → 建键时整除断言炸（上游 issue #28207）。这是本关实现过程中真实踩到的一个坑。

### 3.2 补齐的三种"假东西"，各有各的挡法

| 假东西 | 挡法 | 破了会怎样 |
|---|---|---|
| 补齐的**行**（token/位置） | 槽位 = `PADDING_SLOT_ID(-1)` | 写进真实槽位 → 覆盖别人的 KV |
| 补齐的**请求**（seq_len=0） | 注意力 mask 整行掩掉（用 `finfo.min` 而非 `-inf`，避免 NaN） | 读到别的请求的 KV，输出垃圾（若含 NaN 会扩散） |
| 补齐请求的**块表行** | 每轮清零（指向 0 号块） | 读到被清零前那行的真实块 |

### 3.3 0 号块是垃圾桶（本关最容易被忽略的一条）

图内不能写 `if slot >= 0`（`bool(tensor)` 是一次 CPU 同步），所以写 KV 改成
`slot_mapping.clamp_min(0)`：padding 行的 `-1` 全部落到**0 号块**。于是 0 号块必须
**永远不被任何请求引用**（`BlockPool(reserve_null_block=True)`，`KVCacheManager` 由
`EngineCore` 按"最终解析出来的模式"传这个开关）。代价是可用块数 = `num_gpu_blocks - 1`。

> 64 关的待办就写着这一条（"69 关引入 CUDA Graph padding 行时必须同时把 0 号块留白"）。
> 本关兑现了：`tests/step69` 的 D1/D2 与 `test_padding_writes_land_only_in_the_blank_block` 盯着它。

### 3.4 forward 上下文持有"这一轮能不能走图"

`set_forward_context(..., cudagraph_runtime_mode=..., batch_descriptor=...)`：

* 图包装器读这两个字段决定 **直通 / 捕获 / 重放**（模式不匹配就直通，所以嵌套多层包装器也各管各的）；
* 注意力后端读运行模式决定走 `_forward_generic`（eager，逐请求）还是 `_forward_padded`
  （图内，固定形状）。这是"层按上下文执行"的落点。

### 3.5 捕获窗口与热身

`set_cudagraph_capturing_enabled(True)` 只在 `capture_model()` 期间打开；窗口外任何"顺手捕获"
都会被 `validate_cudagraph_capturing_enabled()` 拦下（否则服务中某一步会突然卡顿、显存悄悄涨）。
热身次数取 `max(1, cudagraph_num_of_warmups)`：cuBLAS 的 handle **不能在捕获里创建**
（实测报 `CUBLAS_STATUS_NOT_INITIALIZED ... cublasCreate(handle)`，随后整个捕获流作废）。
上游默认 0 次是因为它在定容阶段已经做过 profile run；本仓库没有显存 profiling，所以把
"至少一次 eager 前向"写成必需前置。

### 3.6 补齐后的行是紧凑布局的**尾巴**（所以采样/掩码/logprobs 的映射不用改）

统一 decode 批的真实行 = 每请求 `1+K` 行连续排布，与 57～68 关建立起来的紧凑布局**逐行相同**；
补齐只在尾部追加假行/假请求。于是：

* `logits_indices`（采样行）依旧是紧凑行号 → 仍然索引补齐布局的前缀；
* 68 关的语法掩码行映射（`_logit_row_of_req`）与 logprobs 的 `cu_num_generated_tokens` 不变；
* 图输出的 hidden 在进入消费者之前 `[:num_tokens]` 切一刀（去掉尾巴），消费者坐标系统一。

这条不变量是"图模式不必重写 68 关语义"的依据，`tests/step69::test_padded_rows_extend_the_compact_layout`
与 `test_graph_mode_keeps_spec_logprobs_and_grammar_working` 分别从行布局与 logprobs 两端钉它。

### 3.7 drafter 侧：padded 批的两个索引

上游 `prepare_inputs_padded()` 在 device 上算两个逐请求量（本关用 eager Torch 等价实现并在 CUDA
上与真 kernel 差分）：

```
token_indices_to_sample[i] = query_start_loc[i+1] - 1 - num_rejected[i]
num_rejected[i]            = (K_i + 1) - 有效采样数        （K_i = 0 时记 0）
```

它们的用途（上游同一处）：

1. **采样行**：padded 批里"该从哪一行取 hidden 采第一枚草稿"——不能再用"最后一行"（那是 padding）；
2. **上下文修正**：`seq_lens -= num_rejected`，把**没写过 KV 的被拒行**从 draft 的上下文里去掉。
   本仓库的 AR 步位置是直接算出来的，这个修正的作用是把 device 侧 `seq_lens` 的**起点摆正**
   （`eagle_step_update_slot_mapping_and_metadata` 是"原地 +1"的语义，起点错了后面步步错）。

**两套坐标系的换算**（本关最容易写错的地方，`tests/step69::test_draft_first_pass_sample_row_layouts` 钉住）：

```
draft_model：工作区 = [有效行][扩容行][被拒行]   → 采样行 = 块起点 + num_valid = target 行号 + 请求序号 + 1
EAGLE      ：行块 = target 的行块（扩容 token 打在最后一行）→ 采样行 = 块起点 + target_rows - 1
```

因此提议者里**不假设某一种布局**去比对 device 索引；它比的是"device 索引 vs 用 TargetRows 事实
按同一公式算出的 target 行号"（两个独立来源：协议的 `cu_num_draft_tokens`+有效采样数 vs
调度快照+记账结果），布局换算则由各自 `set_inputs_first_pass()` 的测试钉住。

---

## 4. 验收对照（需求 §4 逐条）

| 需求条目 | 落点 |
|---|---|
| eager/compile/Graph 同权重 greedy + 中间值对照，含首拒与全接受 | `check_step69` C1（K=None/1/3，两个请求）、C4（同一步 logits，max\|Δ\|=4.5e-08，fp32 tiny）；compile 一轴**配置期拒绝**（差异 §6.2） |
| B=1→3→1、图 bucket 边界、prefill/decode 混合、context 边界 | `test_batch_size_1_to_3_to_1_reuses_two_graphs`、B1/B2/B4、B3（混合批回退）、C3（prefill 回退 NONE） |
| 记录真实选中的 mode/key | `runner.cudagraph_selections`（每轮一条）+ C3 |
| 复用地址不变 | C2（静态缓冲 `data_ptr` 恒定、捕获次数不增） |
| 故意污染 padding buffer → 有效输出不变、不写错 KV slot | D3（输出逐 token 相同）、D4（真实块 KV **逐位**相同）、D2（0 号块不属任何请求） |
| profiler 看到真实 replay、热路径不夹带逐请求同步 | E1（`cudaGraphLaunch` 计数）、E2（`set_sync_debug_mode("error")` 下跑图区域） |
| 57 生命周期与 68 约束输出在受支持模式下回归 | 全量 `tests/step58..69`（532 项，含图模式默认开启）；`test_graph_mode_keeps_spec_logprobs_and_grammar_working` |
| 不支持的模式按规则降级/拒绝 | A5/A6（配置期 `NotImplementedError`）、B3（混合批回退 NONE = 上游 dispatch 规则） |
| 不放宽断言、未实测不写成通过 | 没有 CUDA 时 `check_step69` 打印"待验"并以失败退出；`requires_cuda` 明确 skip 理由 |

---

## 5. 实测记录

设备：RTX 5090 Laptop（WSL2）；python 3.12.14 / torch 2.13.0+cu130 / vllm 0.28.0；tiny Qwen3
（现场生成，2 层、hidden 32）、单个请求、greedy、`max_tokens` 见下。profiler 用
`torch.profiler`（CUDA activity），统计 `cudaLaunchKernel` 与 `cudaGraphLaunch`。

| 场景 | 路径 | kernel launch / step | cudaGraphLaunch | 墙钟 ms / step |
|---|---|---|---|---|
| 非投机（K=0） | eager | 121.0 | 0 | 4.545 |
| 非投机（K=0） | FULL 图 | **4.2** | 11 | 3.708 |
| 投机 K=3 | eager | 381.6 | 0 | 21.552 |
| 投机 K=3 | FULL 图（target 上图） | 296.4 | 3 | 18.218 |

捕获（`capture_model()` 返回值，实测）：

```
K=0：captured=2、0.272 s、sizes=[1,2,4]、图池 ≈ 50 MB
K=3：captured=2、0.027 s、sizes=[4,8,16]、图池 ≈ 6 MB
```

**怎么读这组数**（不设虚构加速倍数，AGENTS §8）：

* 非投机 decode 的 launch 次数降了两个数量级（121 → 4.2）：一轮前向从"逐 kernel 启动"变成
  "一次 `cudaGraphLaunch`"，这正是 CUDA Graph 要解决的那件事；墙钟只降 18%（tiny 模型 +
  单请求下 GPU 本来就闲，CPU 侧还有很多别的工作：采样、logprobs、记账）。
* **投机场景收益小得多**（381.6 → 296.4，18.2/21.6 ms）：图只覆盖 **target 的前向**；
  drafter 的提议循环仍是 eager（上游的 EAGLE 只在 PIECEWISE 下走图，本项目没有 PIECEWISE，
  见 §6.6），它贡献了剩下的 ~290 次 launch。要在这条路上继续降，需要 74 关（V2 speculator）
  或先补 PIECEWISE。
* 这些数字是**小规模测量**，不代表其它形状/模型的收益；本关不做统一性能矩阵（84 关才做）。

---

## 6. 差异账本（逐条，不写"简化了一些"）

### 6.1 默认解析成 `FULL_DECODE_ONLY`（上游默认 `FULL_AND_PIECEWISE`）

上游 `cudagraph_mode=None` 最终解析成"decode 用 FULL、混合批用 PIECEWISE"。本项目没有
PIECEWISE，所以解析成 `FULL_DECODE_ONLY`（decode 用 FULL、混合批回退 eager）——这是
**能力子集**上的最近等价模式，不是"把默认关掉"。CPU 设备与 `enforce_eager=True` 时解析成 NONE
（上游同款开关）。

### 6.2 没有 torch.compile 一轴（配置期拒绝）

需求 §3.5 要求"编译负责算图、图负责减少 launch，分别验收"。本项目的模型是逐请求 Torch 循环
（注意力里有 Python 层循环与动态形状），编译它只会到处 graph break，所以：

* `CompilationMode` 的四个取值照抄（共同的名字），但**除 NONE 之外全部在配置期报错**；
* `compile_sizes` 非空即报错（不接受"配了但不生效"）；
* 因此验收里的"compile"一轴**没有实测**，只有 eager 与 FULL 图两轴；这一条是缺口，不是通过。

### 6.3 `set_forward_context()` 的签名差异

上游第 2 个位置参数是 `vllm_config`（用来填 `no_compile_layers` / DP metadata）。本项目没有
编译期静态层表、没有 DP，所以签名是 `(attn_metadata, num_tokens=0, *, cudagraph_runtime_mode,
batch_descriptor, slot_mapping, additional_kwargs)`——**位置参数变了、关键字不变**，所有调用点
都是关键字调用（`tests/step64` 里那处历史调用也已随本关改为关键字）。

### 6.4 捕获前至少一次 eager 热身

`cudagraph_num_of_warmups` 默认 0（与上游一致），但实现里取 `max(1, ...)`。理由见 §3.5
（cuBLAS handle 不能在捕获里创建，实测报错）。这不改变图的语义。

### 6.5 图内注意力是"一次批量算完"，会物化 K/V gather

上游图内的注意力是 FlashAttention 类分块 kernel。本项目的教学后端没有 kernel，图内路径
（`_forward_padded`）用固定形状张量一次算完：

* 显存 ≈ `2 · B · P · Hk · D · 4` 字节（`P = 块表宽度 × block_size` = 上下文上限，fp32 计算）；
* 为提高数值一致性，K/V **升到 float32** 再算（与 eager 路径同精度）。这一条不是洁癖：
  一开始为了让显存减半而"直接在 bf16 上乘"，实测在**真实 EAGLE3（bf16）**上 greedy 输出的
  第 7 个 token 与 eager 分叉（`spec` 与 `plain` 两条路都开图时也分叉）——升 fp32 后逐 token 相同。
  这是"看起来等价、其实会翻 token"的典型例子，记在这里免得以后有人再去"优化"。
* 因此本关的图路径**不适合超长上下文 + 大 max_num_seqs**；能力边界写在这里，不假装支持。

### 6.6 drafter 侧不做图（上游只在 PIECEWISE 下做）

上游 `SpecDecodeBaseProposer.initialize_cudagraph_keys()`：`mixed_mode()` 是 PIECEWISE/FULL 时
给 draft 建 PIECEWISE 键，否则 NONE——EAGLE 系的自回归循环每步形状都不同，只有"注意力拆到图外"
的分段图能容纳。本项目没有 PIECEWISE → draft 恒 NONE：

* `proposer.cudagraph_mode is NONE`（可观测，测试盯着），`cudagraph_unsupported_reason` 记原因；
* draft 的第一遍仍是 **CPU 拼行 + 上传**（58 关的设计），所以 `prepare_inputs_padded` 的产物
  在本项目里的作用是**校验与对齐口径**（device 算法 vs CPU 事实逐值比），不是"取代 CPU 拼行"；
  这一步会产生一次很小的 D2H（3 个小张量）。上游把整条路放在 GPU 上是为了省掉它，
  等 74/75 关（V2 speculator / 分块验证）才有真正的收益。

### 6.7 0 号块"按需留白"而不是永远留白

上游的 `null_block` 是**常驻**的（`num_gpu_blocks - 1` 可用）。本项目只在"最终解析出的模式含
FULL 图"时留白：不开图时留白是纯损耗。区别可观测（`cache_config.num_gpu_blocks` 的容量口径），
所以它是显式参数 `reserve_null_block`，由 `EngineCore` 按解析后的模式传。

### 6.8 预填/混合批没有图

`FULL_DECODE_ONLY` 下混合批与 prefill 批都回退 eager（上游同款：mixed_mode() 为 NONE 时没有
混合键）。本项目**还多一层边界**：即便用户显式要 `cudagraph_mode="full"`（非分段，混合批也进图），
也会被拒绝——因为非统一形状的注意力在本项目只有逐请求实现（图内会夹带 `int(tensor)` 同步，
录进去以后重放读到的是录制时的数字，**静默算错**）。

### 6.9 静态工作区的容量口径

`input_ids/positions/slot_mapping` 按 `max_num_batched_tokens` 开，`query_start_loc/seq_lens`
按 `max_num_seqs(+1)` 开，块表复用 `InputBatch` 的镜像——**不新增旋钮**（AGENTS §8）。
图档位上限又被 `max_num_batched_tokens` 夹住，所以"工作区一定放得下补齐后的行数"。

---

## 7. 回归与已知缺口

**回归（本机实测）**：

```
pytest tests/step58..69           → 532 passed   （68 关基线 505 + 本关 27）
pytest tests/step69               → 27 passed    （9 + 18）
benchmarks/check_step69_cudagraph.py → 全部通过（26 项）
其余 check_step57_* / check_step58..68_* 全绿（15 + 14 个脚本）
```

**改过的旧期望（3 处，都是"值变了、断言强度不变"，AGENTS §6.3）**：

1. `tests/step58/test_draft_inputs.py::test_prefill_first_pass_layout`：槽位从 `range(n)` 改成
   "按块表第一块展开"——**0 号块留白**之后第一个物理块不再是 0（并新增断言 `first_block != 0`）。
2. `tests/step58/test_draft_inputs.py::test_rows_invariant_under_preemption`、
   `benchmarks/check_step57_draft_model.py`（第 7 项）、`benchmarks/check_step57_spec_lifecycle.py`：
   `blocks=3 → 4`（可用块数 = blocks − 1），预抢占前提不变。
3. `benchmarks/check_step58_draft_inputs.py` B2 同上（槽位按块表算）；另 `B4b` 的 `blocks 3 → 4`。

**已知缺口 / 下一步**：

* 没有 compile 一轴（§6.2）；没有 PIECEWISE 图（§6.1/§6.6/§6.8）；
* 图路径的注意力会物化 gather（§6.5）——长上下文下显存吃紧，上游 kernel 不会；
* drafter 侧仍 eager（§6.6），投机场景的 launch 只降了 ~22%；
* 未接线：跨进程/多卡、量化 KV、`logprobs=-1` 的入口归一化（68 关遗留）、异步调度（70 关）。
  70 关的 `prepare_next_token_ids_padded()` 与"图 + 异步"的地址协议本来就要用到本关的静态工作区。
