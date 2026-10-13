# 第七十四关：V2 自回归 Speculator 与多步融合 Graph

> 需求：`learning_notes/14_vllm_from_scratch/投机解码完整需求/074_V2自回归Speculator与多步融合Graph.md`
> 基线：本机 `vllm==0.28.0` 文件快照（2026-10-03）。执行路径：**V2**（73 关的 `minivllm/worker/gpu/`）。

## 0. 这次做了什么（两三句话）

73 关把 V2 的执行路径搭起来了，但草稿提议者每轮都在 eager 里跑：**K 个草稿步 = K 次 Python 侧
输入更新 + K 次 kernel 提交**。这一关给提议者补上两张 CUDA Graph，并把"K−1 个草稿步"融成**一张图
里的一次重放**（融合多步 decode）：

    prefill 图   草稿位置 0，每请求 `1+K` 行，复用 **target 本轮**的注意力元数据与 slot_mapping
    decode 图    草稿位置 >0，每请求 1 行，用草稿自己的元数据（`seq_lens + step`）

走融合还是逐步**由后端 capability 决定**（`supports_draft_decode_metadata_update`），不猜、也不
伪造：声明不支持的后端走"每步重建元数据"的回退路径（正确但慢）。

## 1. 路径映射（本项目 ↔ 上游）

| 本项目 | 上游 | 状态 |
|---|---|---|
| `minivllm/worker/gpu/cudagraph_utils.py` | `v1/worker/gpu/cudagraph_utils.py` | 核心子集：`BatchExecutionDescriptor` / `_is_compatible` / `CudaGraphManager`（候选表、捕获、分派、重放）/ `prepare_inputs_to_capture` / `AttentionState` |
| `minivllm/worker/gpu/spec_decode/autoregressive/cudagraph_utils.py` | `v1/worker/gpu/spec_decode/autoregressive/cudagraph_utils.py` | `SpeculatorCudaGraphManager`（逐行对齐，71 行 → 本仓库 78 行） |
| `minivllm/worker/gpu/spec_decode/autoregressive/speculator.py` | 同名文件 L29–L649 | `init_cudagraph_manager` / `capture` / `propose` / `_multi_step_decode` / `_fused_multi_step_decode` / `_generate_fused_drafts` / `_generate_draft` / `_configure_fused_multi_step_decode`；三个 kernel 未改 |
| `minivllm/worker/gpu/spec_decode/mtp/speculator.py` | `v1/worker/gpu/spec_decode/mtp/speculator.py` | `MTPSpeculator`（4 个 hook + `share_mtp_topk_indices` 探测） |
| `minivllm/worker/utils.py` | `v1/worker/utils.py` | 子集：`is_uniform_query_len` / `get_uniform_decode_token_count` / `AttentionGroup`（含 capability 转发） |
| `minivllm/attention/backend.py`、`metadata.py` | `v1/attention/backend.py` | `supports_draft_decode_metadata_update`（构造器类属性，默认 False）+ `update_draft_decode_metadata()`（默认 `NotImplementedError`） |
| `minivllm/attention/backends/torch_sdpa.py` | 上游 Triton 后端的同一条理由 | Torch 后端声明 **True**，`update_draft_decode_metadata()` 是空操作 |
| `minivllm/worker/gpu/{input_batch,block_table}.py` | 同名文件 | 新增 `InputBatch.make_dummy` / `BlockTables.get_dummy_block_tables` / `get_dummy_slot_mappings`（逐行对齐） |
| `minivllm/worker/gpu/model_runner.py` | `v1/worker/gpu/model_runner.py` | `initialize_cudagraph_capture`（能力协商）/ `capture_model`（捕获草稿图）/ `_build_capture_attn_metadata` / 元数据键集合扩展到草稿层 |
| `minivllm/testing/fake_attn_backend.py` | 无（测试替身） | "不支持原地更新"的后端替身（只给测试） |

## 2. 机制要点

### 2.1 两张图的形状从哪来

`CudaGraphManager._init_candidates()` 把档位表翻成图键，规则逐条对齐上游：

* **统一 decode 图**（`separate_routine() and decode_mode`）：档位向上取整到 `decode_query_len`
  的倍数，`num_reqs = 行数 / decode_query_len`；超过 `max_num_reqs × decode_query_len`、
  超过 `max_cudagraph_capture_size`、或请求数超过 `max_num_seqs` 的一律丢掉。
  草稿 prefill 图的 `decode_query_len = 1+K`，草稿 decode 图的 `decode_query_len = 1`。
* **运行时 token 数 → 候选图**：向上映射到"能装下它的最小捕获档位"；`dispatch()` 是查表 +
  逐条 `_is_compatible`，没有匹配就回退 `NONE`（**不猜**）。

`init_cudagraph_manager()` 把 target 的最终模式收敛成两档（上游规则）：`decode_mode() == FULL`
→ `FULL_DECODE_ONLY`，否则 `NONE`。**PIECEWISE 不支持草稿 decode**，所以不会把 target 的模式原样
强加给草稿（需求 §3.4）。

### 2.2 融合 vs 逐步：门与两条路

| | 逐步（fallback） | 融合（fused） |
|---|---|---|
| 元数据 | 每步在 CPU 侧重算 slot_mapping + 重建 metadata | 只在 `step=1` 建一次 |
| 图 | 每档位一张图，**每步重放一次** | 每档位一张图，**一次重放走完 K−1 步** |
| 步间的设备侧工作 | 无（Python 重建） | `compute_slot_mappings()` 原算 + `update_draft_decode_metadata()` |
| 选择条件 | 任一注意力组 `supports_draft_decode_metadata_update == False` | 全部支持，或 `advance_draft_positions == False`（强制），且 `K > 1` |

`update_draft_decode_metadata()` 的硬约束（照抄上游注释）：**图重放不会执行这个 Python 方法**，
所以实现只能做捕获安全的操作、把状态放在常驻缓冲里。Torch 后端的元数据里所有步相关字段
（`seq_lens` / `slot_mapping` / `query_start_loc` / `block_table`）**都是常驻缓冲的切片**，写这些
缓冲的内核已经把新值写进去了，所以那里是空操作——与上游 Triton 后端（`triton_attn.py:104`：
"Step-dependent fields reference persistent input buffers directly"）逐字同一条理由。

### 2.3 捕获顺序与"热身"

`CudaGraphManager.capture()`：**大档位先捕获**（小图复用大图占下的显存池）→ 每张图先**热身一次**
（eager，让 cuBLAS handle / Triton 自动调优在图外发生）→ FULL 图**重新准备一次输入**再录（热身
可能改过元数据）→ `torch.cuda.graph(graph, pool)`，池是**全进程共享**的那一个（与 69 关的
`graph_capture` 同一个）。

两处"先做的小事"都有实测理由：`capture()` 先把 `last_token_indices` / `idx_mapping` **清零**
（假输入会读它们，残留旧值就是越界索引）；`on_*_begin/end` 钩子在**捕获与运行两侧成对触发**
（MTP 的 `skip_topk` 靠这条保证"图里录的那段代码"与"重放前打开的状态"一致）。

### 2.4 捕获期假输入：写在谁的缓冲上

| 图 | `InputBuffers` | 元数据 | 理由 |
|---|---|---|---|
| 草稿 prefill | **target 的** | target 的（`_build_capture_attn_metadata`） | 运行期草稿 prefill 直接复用 target 本轮的元数据与 slot_mapping，捕获期必须写在**同一批**常驻缓冲上 |
| 草稿 decode | 草稿自己的 | 草稿自己的（`uniform_query_len=1`） | 每请求 1 行，草稿自己按 `seq_lens` 现算 |

假输入的安全值（`InputBatch.make_dummy` / `get_dummy_*`，逐行对齐上游）：`input_ids`/`positions`
清零（会被 embedding/RoPE 查表读到）、`seq_lens` 补齐行 = 0、`query_start_loc` 非递减、
块表补齐行清零（0 号块 = 垃圾桶）、slot_mapping **整块**填 `PAD_SLOT_ID(-1)`（内核可能访问到
请求范围之外）。

### 2.5 草稿 prefill 图为什么复用 target 的元数据

草稿层与 target 层在同一个 batch 形状下跑，KV 也按**同一套槽位号**写（草稿的块表就是 target 的
块表）——所以那份元数据对两边都成立。V2 Runner 的元数据字典因此必须覆盖 **target 层 ∪ 草稿层**
的层名（上游 runner 的 `init_attn_backend` 不带 `active_layer_names`，天然如此）；本仓库在
`initialize_kv_cache()` 里把 `self.speculator.draft_attn_layer_names` 并进去（键少了会在 forward
里 `KeyError`）。

### 2.6 MTP 的 hook 契约

`MTPSpeculator` 只覆写四件事（不复制 `propose`）：`on_prefill_begin`（**无条件**把
`skip_topk` 关掉——上一轮死在中间也不会残留）、`on_prefill_end`（K>1 时把 step 0 的 top-k 索引
压到每请求最后一行）、`on_multi_step_decode_begin`（打开复用）、`on_multi_step_decode_end`（关掉）。
`share_mtp_topk_indices` 的探测要三样齐全（checkpoint 声明 + `set_skip_topk` +
`compact_topk_indices`），缺一样就当不支持（不猜）。

### 2.7 V2 的图模式协商

`GPUModelRunner.initialize_cudagraph_capture()`（V2）：环境事实（没有 CUDA → NONE）→
`resolve_cudagraph_mode_and_sizes(min_cg_support, 后端名, 1+K)`（注意力后端能力决定 FULL 能不能
成立）→ 写回 `compilation_config.cudagraph_mode`。两条与上游 V2 的差别：

* **纯 PIECEWISE 配置期拒绝**（V2 的分段图未接，"请求了图却什么也不做"必须显式报错）；
* `FULL_AND_PIECEWISE` 收下，但 **PIECEWISE 那一半在 V2 未接**（混合批走 eager）——
  初始化时 warning + 本文记录，草稿的两张图只认 FULL 那一半。

配置期的档位取整也不再无条件做：`_set_cudagraph_sizes()` 现在**只对 V1** 调
`adjust_cudagraph_sizes_for_spec_decode(1+K)`（上游注释原话："MRV1 adjusts ... MRV2 handles
cudagraph capture sizing in `cudagraph_utils.py`"）。原因很具体：V2 两张草稿图的档位口径不同
（prefill 要 `1+K` 的倍数、decode 要 1..max_num_seqs 这些小档位），配置期按 `1+K` 取整会把小档位
抹掉——K=3、`max_num_seqs=3` 时档位表变成 `[4,8,16,24]`，decode 图需要的 ≤3 档位一个都不剩。

## 3. 差异账本（逐条）

| 编号 | 差异 | 为什么 | 影响 / 何时消除 |
|---|---|---|---|
| D-74-1 | 捕获时传给前向的是 `cudagraph_runtime_mode=FULL`（上游在 `torch.cuda.graph` 里传 `NONE`） | 本仓库的注意力**按 forward 上下文里的运行模式**决定走不走固定形状路径（`_graph_dispatched()`）；上游按元数据（`for_capture=True`）决定。传 `NONE` 会把逐请求 eager 路径录进图（那里有 `int(张量)` 同步，捕获当场就炸） | 只影响"录制时用哪条路径"，重放语义一致（图里记的就是要执行的算子）；测试 D1/E1 钉住两条路数值相同 |
| D-74-2 | 草稿 prefill 图**要求精确形状**（`desc.num_tokens == num_tokens and desc.num_reqs == num_reqs`）；上游允许 `>=` 并靠 target 侧补齐 | 上游 V2 的 **target 自带 FULL 图**并把整批补齐到图档位；本仓库 V2 的 target 图不属 74 关（target 不补齐）。若不补齐却让图服务更小的批，图里记的 `num_reqs` 会去读 target 缓冲里**没被写过**的行（旧块号/seq_len）→ 越界或读到别人的 KV，且不报错 | 批形状不在捕获表里时草稿 prefill 走 eager（decode 图不受影响：它补齐的是**自己**那份缓冲） |
| D-74-3 | `set_attn()` 用回调（`target_build_attn_metadata`）代替上游的 `model_state` / `kv_cache_config` / `target_attn_groups` | 本仓库只有一个后端、一个 KV group、没有 ModelState/KVCacheSpec 对象 | 调用点与上游同形；将来加后端时参数位已在（`_init_attn_groups()` 就是那个接缝） |
| D-74-4 | 草稿的两张图**只做 FULL** | V2 没有分段件（`CompilationMode` 只有 NONE，不进 `enable_piecewise_pieces`），混合批没有可重放的分段图 | 混合批（真 prefill）走 eager；纯 PIECEWISE 配置在配置期明确拒绝 |
| D-74-5 | 捕获期**不调** `build_for_cudagraph_capture()`（它把 `seq_lens` 全填 1） | 填 1 会让**热身的 eager 那一遍**出现"整行被掩掉"（`past_len = seq_len − query_len < 0`）→ `_attend` 的 `-inf` 掩码让 softmax 出 **NaN** → NaN 写进 KV 缓存与特征缓冲，之后每一次重放都跟着烂（实测：捕获后 `hidden_states[0]` 与 0 号块全 NaN、重放出的草稿 token 全是 0）。另外"填 1 更快"对本后端的图内路径不成立（它按块表宽度 gather 整段上下文） | 形状守门改成显式断言 `uniform_query_len` 已定死 |
| D-74-6 | `_attend`（eager 逐请求路径）的掩码值由 `-inf` 改成 `finfo.min` | 与图内路径（69 关）同一条规则：全掩行的 `softmax(-inf)` 是 NaN，而 NaN 会扩散；有效请求永远不会出现全掩行，所以只影响"本来就没人在读"的行 | 数值上只在全掩行有差别（NaN → 均匀垃圾）；69 关全部用例仍逐 token 相同 |
| D-74-7 | V2 不在配置期按 `1+K` 取整档位表 | 上游 MRV2 同样不取整（`resolve_cudagraph_mode_and_sizes(..., use_v2_model_runner=True)`）；V2 两张图的档位口径不同（见 §2.7） | V1 行为不变（`use_v2_model_runner=False` 时取整）；V2 的取整各自由 `_init_candidates()` 做 |
| D-74-8 | 裁剪：LoRA / varlen decode（79 关）/ DP 同步（`num_tokens_across_dp`）/ offloader / breakable CUDA graph / 多模态 / EPLB / local-argmax / 多模块 MTP（80 关）/ `skip_attn_for_dummy_run`（V2 没有 dummy run） | 本仓库没有对应能力（单卡、无编译路径、无多模态） | 逐项写在对应函数的 docstring 里；新增能力时按上游同名位置补 |
| D-74-9 | V2 + `method="mtp"` 要求**显式**给出派生好的 draft 配置；V1 仍由 Runner 用 `derive_mtp_draft_config()` 派生 | 上游在配置期派生（`SpeculativeConfig` 拿得到 target），本仓库的 `SpeculativeConfig` 拿不到 target——"什么时候派生"这件事在 V2 的接线属 80 关 | 缺 draft 配置时初始化期明确拒绝并给出用法；V1 不变 |
| D-74-10 | `draft_logits_spec()` 把 `ModelConfig.dtype` 过 `STR_DTYPE_TO_TORCH_DTYPE` 映射表 | 本仓库配置里 dtype 是**字符串**（上游是 `torch.dtype` 且读 `head_dtype`）；直接把 `"float32"` 交给 `torch.full(dtype=...)` 会 TypeError——这是 73 关遗留的真 bug（`draft_sample_method="probabilistic"` 那条路当时没有任何调用方） | 概率草稿路径现在真的能跑（D1/G2 的概率版用例） |
| D-74-11 | `_build_draft_attn_metadata()` 不再收 `seq_lens_cpu_upper_bound` | 本仓库的 Torch 后端只用**设备上的** `input_buffers.seq_lens`（它就是"target 上下文 + 已走步数"）；上游 FA3 用 CPU 上界避免一次 D2H 同步 | 不留没人读的参数；多模块 MTP 的非均匀 query 布局（80 关）到时再加参数 |
| D-74-12 | `SpeculatorCudaGraphManager.capture()` 没有 `num_tokens_across_dp` 参数位 | 单卡 | 与 D-74-8 同类 |
| D-74-13 | 真实 EAGLE3 的 `d2t`/`t2d` 放在**参数所在设备**（原来是 CPU 张量，每次 `compute_logits()` 现拷） | 开图必需：CUDA Graph 捕获**不允许**从非 pinned 的 CPU 张量拷贝（真实权重下直接报 `Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the CPU tensor is pinned`）；顺带省掉每次前向一次 H2D（32000×int64） | 上游这类 buffer 本来就跟模块走；`tests/step63/test_eagle3_real_e2e.py` 的期望值改成"显式拉回 CPU 再比"（断言强度不变） |
| D-74-14 | `graph_capture()` 在捕获窗口里 `gc.disable()`（退出时按原状态恢复） | 自动 GC 会在任意一次分配时触发；一旦它回收掉**别的** `torch.cuda.CUDAGraph`，那个对象的析构（`CUDAGraph::reset()`）就会在当前流上做捕获期不允许的调用 → **作废正在录的图**（CUDA 只给一句 warning `operation not permitted when stream is capturing (function reset)`），紧接着分配器内断言 `cudaStreamCaptureStatusInvalidated` 杀掉进程。症状是"单独跑没事、长会话必现"。上游在同一窗口里 patch 掉 `gc.collect`（注释写的是性能），同一类风险 | V1 的 `capture_model()` 用的是同一个上下文，所以这个修复对 V1/V2 都生效 |

## 4. 验证（实测）

命令与数字见 `docs/results.json` → `step74.results` / `step74.models`。摘要：

* `python -m pytest tests/step74 -q` → **42 passed**（融合/等价 20 + 管理器单元 12 + 边界 5 + MTP 5）；
  全量 `tests/step58…74` → **715 passed**（664 + 42 + 本关顺手补进 `tests/step73` 的 9 项用例：输入组装 1 + 验证分块 8）。
* **真实权重**（`Qwen3-1.7B` + `Qwen3-1.7B-eagle3`，K=3、bf16、48 token）：V1 eager / V2 eager / V2 图
  三条路 greedy **逐 token 相同**，`drafted/accepted/rounds` 也全等（120/7/40）；V2 图捕获 **4 张**
  （prefill 4、8 行；decode 1、2 行）/ 0.168 s / 池 76 MiB / `fused=True`。
* `python benchmarks/check_step74_fused_draft.py` → **23 项全过**，其中：
  * **融合门**（K=1/2/4 × 支持/不支持，两个后端各跑一遍）：K=1 恒 `fused=False`；
    K>1 时 `fused == 后端声明`。
  * **两条路逐轮等价**（3 条请求、10 轮）：草稿 token / positions / seq_lens / slot_mapping /
    草稿 KV 校验和**全等**。
  * **多步真的在图里**：融合路径一轮 decode 图重放 **1** 次、Python 前向 **0** 次、采样 **0** 次，
    而 `current_draft_step` 到 `K−1`、K 列草稿全写出来；逐步对照组同位置重放 `K−1` 次。
  * **端到端**：EAGLE3 与原生 MTP 下，`V2 图 == V2 eager == V1 eager`（greedy 逐 token）；
    含分块 prefill（budget=8）、抢占恢复（6 块）、模型长度边界（`max_model_len=12`）。
  * **profiler**（同一 workload、K=4、12 轮 decode）：融合 `3.03 ms/轮` 草稿耗时 vs 逐步
    `3.20 ms/轮`；CPU 侧算子派发 `12773` vs `13158`；草稿图重放 `13` vs `39` 次。
    ⚠️ 本机是 tiny 权重（模型计算量可忽略），差异主要来自"少提交 K−1 次"；**不给加速倍数**，
    真实权重下的数字见 results.json 的 `real_model` 段。

## 4.1 本关顺带修掉的真问题（都不是"顺手改"，是跑通图上路径必须解决的）

| 问题 | 症状（实测） | 修法 |
|---|---|---|
| 捕获期 `_attend` 的 `-inf` 掩码 | 假批里 `seq_lens < query_len` → 整行被掩 → `softmax(-inf)` = **NaN** → NaN 写进 KV 缓存与 `hidden_states`，之后每次重放都跟着烂（捕获完 `hidden_states[0]` 与 0 号块全 NaN、重放出的草稿 token 全是 0） | 与图内路径统一用 `finfo.min`（69 关已经写下的理由） |
| 捕获期自动 GC | `step67+step68+tests/step74` 连跑必现：捕获被作废 → 分配器内断言杀进程（`step74` 单独跑不现、别的组合也不现，取决于 GC 何时被触发） | `graph_capture()` 里 `gc.disable()`（见 D-74-14） |
| 概率草稿的 dtype | `draft_sample_method="probabilistic"` 一跑就 `TypeError: full(): argument 'dtype' must be torch.dtype, not str`（73 关留下的：那条路当时没有任何调用方） | `draft_logits_spec()` 过 `STR_DTYPE_TO_TORCH_DTYPE`（见 D-74-10） |
| 真实 `d2t`/`t2d` 在 CPU | 真实 EAGLE3 开图报 "Cannot copy between CPU and CUDA tensors during CUDA graph capture…" | 放到参数所在设备（见 D-74-13） |
| 热身的路径 | 捕获前用 `NONE` 热身时，图里那条（固定形状）路径是在**捕获窗口里第一次执行**——第一次要做的懒初始化落在捕获里就是"不允许的操作" | 热身也走 `desc.cg_mode` |
| 上游 kernel 一条细微护栏的可达性 | `_combine_sampled_and_draft_tokens_kernel` 的 `first_logit_seq_pos >= prefill_len` 容易被读成前一条早退的重复；实测（7 个场景、100 个"请求·轮"）今天走不到，但它是"**要写的那一行**还在历史里"的护栏（跨 prefill 边界的批） | 补 `tests/step73/test_v2_request_state.py` 的用例把三种情形并排钉住 + kernel 注释写明可达性与放开条件 |

## 5. 边界与遗留

* **V2 的 target 图**（`ModelCudaGraphManager` 那一半：target 前向进图 + 整批补齐）不在本关范围，
  所以 D-74-2 那条"精确形状"约束会一直生效到它落地；届时应改成上游的 `>=` + 补齐。
* **PIECEWISE 在 V2 未接**（混合批 eager），纯 PIECEWISE 配置期拒绝。
* 草稿 prefill 图只在"批形状精确命中档位"时生效（典型情况：全部请求都在 decode 且每请求 `1+K` 行）；
  decode 图会补齐自己的缓冲，覆盖面大得多。
* 真实 EAGLE3 权重下的接受长度差距仍是 63 关的已知项（`docs/step63_alignment.md` §8），
  本关不改变草稿质量。
* `logprobs=-1` 的入口归一化（68 关记的本项目待修项）仍未修，与本关无关。
* **改动过两条旧期望值**（断言强度不变，理由见 §4/§3）：`tests/step73/test_v2_spec_pipeline.py::test_v2_config_boundaries`
  与 `benchmarks/check_step73_v2_runner.py` 的 G1 —— "V2 的 CUDA Graph 在配置期被拒"改成
  "接受 FULL 那一半、仍拒纯 PIECEWISE"；`tests/step63/test_eagle3_real_e2e.py` 的 `d2t` 设备
  （D-74-13）；`minivllm/config.py::_set_cudagraph_sizes` 的档位取整只对 V1 做（D-74-7）。
