# 第七十三关：Runner V2 请求状态与投机执行路径迁移（`minivllm/worker/gpu/`）

> 需求：`learning_notes/14_vllm_from_scratch/投机解码完整需求/073_RunnerV2请求状态与投机执行路径迁移.md`
> 基线：本机 `vllm==0.28.0` 文件快照（2026-10-03）。上游 V2 路径在哪：`vllm/v1/worker/gpu/`。
> 执行路径：**V2 独立入口**（`VLLM_USE_V2_MODEL_RUNNER=1` 才走，默认仍是 V1，理由见 §5 差异账本 D1）。

## 1. 这一关做了什么（一句话）

把"**请求的身份 = 本轮的 batch 行号**"换成"**身份 = 常驻 slot，行号只是本轮的座位表**"：
`RequestState` 按 slot 存全部每请求状态（`all_token_ids` / `total_len` / `num_computed_tokens` /
`last_sampled_tokens` / `draft_tokens` / `prompt_len` / `prefill_len`），输入组装与采样/验证
全程用 `idx_mapping`（batch 行→slot）与 `expanded_idx_mapping`（logits 行→slot）寻址，
采样结果由 GPU 内核**按 slot 就地写回**。V1 那条路径原样保留（需求 073 §2：V1 支持的方法不因
引入 V2 而消失），两条路径**不互相调用对方的输入准备**。

## 2. 路径映射

| 本项目 | 上游 | 说明 |
|---|---|---|
| `minivllm/worker/gpu/model_runner.py` | `vllm/v1/worker/gpu/model_runner.py` | `GPUModelRunner`（V2）/ `ExecuteModelState` / `BatchReqState` / `sort_batch_req_ids` |
| `minivllm/worker/gpu/states.py` | `vllm/v1/worker/gpu/states.py` | `RequestState` |
| `minivllm/worker/gpu/buffer_utils.py` | `vllm/v1/worker/gpu/buffer_utils.py` | `UvaBuffer(Pool)` / `UvaBackedTensor` / `StagedWriteTensor` / `FusedStagedWriter` / `_apply_write_kernel` |
| `minivllm/worker/gpu/input_batch.py` | `vllm/v1/worker/gpu/input_batch.py` | `InputBuffers` / `InputBatch` + 7 个输入组装内核 |
| `minivllm/worker/gpu/block_table.py` | `vllm/v1/worker/gpu/block_table.py` | `BlockTables`（行 = 常驻 slot）+ gather/slot_mapping 内核 |
| `minivllm/worker/gpu/sample/` | `vllm/v1/worker/gpu/sample/` | `SamplingStates` / `PenaltiesState` / `Sampler` / `gumbel` / `min_p` / `logprob` / `output` |
| `minivllm/worker/gpu/structured_outputs.py` | `vllm/v1/worker/gpu/structured_outputs.py` | `StructuredOutputsWorker` + 掩码内核 |
| `minivllm/worker/gpu/spec_decode/` | `vllm/v1/worker/gpu/spec_decode/` | `init_speculator` / `BaseSpeculator` / `DraftModelSpeculator` / `AutoRegressiveSpeculator` / `EagleSpeculator` / `RejectionSampler` / `DraftTokensHandler` |
| `minivllm/worker/gpu/async_utils.py` | `vllm/v1/worker/gpu/async_utils.py` | `AsyncOutput`（侧流非阻塞 D2H）+ `async_copy_to_np` |
| `minivllm/utils/platform_utils.py` | `vllm/utils/platform_utils.py` + `platforms/{interface,cuda}.py` | `in_wsl` / `is_pin_memory_available` / `is_uva_available` |
| `minivllm/utils/torch_utils.py` | `vllm/utils/torch_utils.py` | `async_tensor_h2d` / `np_to_pinned_tensor` / `get_accelerator_view_from_cpu_tensor` / `STR_DTYPE_TO_TORCH_DTYPE` |
| `minivllm/utils/math_utils.py` | `vllm/utils/math_utils.py` | `cdiv` |
| `minivllm/triton_utils.py` | `vllm/triton_utils/__init__.py` | `HAS_TRITON` / `triton` / `tl` / `tldevice` |

## 3. 执行流程（两段式，与上游同序）

引擎侧协议不变（57 关定的）：`execute_model()` 返回 `None` → `sample_tokens(grammar_output)`
交回结果；V2 只是把"状态更新"与"采样"都搬到 slot 坐标系里：

```
execute_model(scheduler_output)
  ① finish_requests → free_states → add_requests → update_requests   （改常驻 slot 状态）
  ② block_tables.apply_staged_writes()                                （块表落盘）
  ③ 0 token → 返回空输出（不碰模型）
  ④ gather_batch_req_state   req_ids 排序 + idx_mapping_np/prefill_len/is_prefilling
  ⑤ prepare_inputs           idx_mapping / cu_num_logits / expanded 映射 /
                             prepare_prefill_inputs / prepare_pos_seq_lens /
                             combine_sampled_and_draft_tokens → logits_indices
  ⑥ prepare_attn             块表 gather + slot_mapping 现算 → 注意力 metadata
  ⑦ 前向（本关 eager）       留 hidden_states / aux_hidden_states 到 execute_model_state

sample_tokens(grammar_output)
  ⑧ compute_logits(logits_indices) → （有语法掩码就就地打掩码）
  ⑨ 采样：num_draft_tokens == 0 → Sampler；否则 RejectionSampler（标准验证）
  ⑩ AsyncOutput：侧流上非阻塞 D2H（句柄先交出去，`get_output()` 才等）
  ⑪ postprocess_sampled → post_update 内核按 slot 写回 all_token_ids/total_len/
     last_sampled/num_computed_tokens（+ 惩罚计数）
  ⑫ speculator.propose(...) → draft_tokens[idx_mapping] = 草稿（留到下一轮验证）
  ⑬ draft_tokens_handler.set_draft_tokens(...)（结构化输出时才真拷回 CPU）
```

## 4. 状态与 shape 对照

### 4.1 `RequestState`（`states.py`）

| 字段 | shape / dtype | 权威侧 | 谁改 | 备注 |
|---|---|---|---|---|
| `req_id_to_index` / `index_to_req_id` / `free_indices` | dict / list | CPU | Runner | 常驻 slot 的分配（`free_indices` 是**栈**，LIFO 复用） |
| `all_token_ids` | `[max_num_reqs, max_model_len]` int32 | **UVA**（host 常驻、GPU 可见） | `add_request` 走 staged write；采样后由 `post_update` **内核**追加 | 大表，显存放不下 → 上游同款 UVA |
| `prompt_len` | `[R]` int32 | UVA 后备 | `add_request` + `copy_to_uva` | 用户给的 prompt 长度 |
| `prefill_len` | `[R]` int32 | UVA 后备 | 同上 | 喂进 runner 的长度，恢复时 **> prompt_len** |
| `total_len` | `[R]` int32 | GPU（staged write） | `post_update` 累加 | prompt + output |
| `num_computed_tokens` | `[R]` int32 | GPU（staged write） | `update_requests` 校正 + `post_update` 加 delta | GPU 真值 |
| `num_computed_tokens_np` | `[R]` int32 numpy | CPU | `update_requests` | **乐观上界**（`seq_lens_cpu_upper_bound` 用它） |
| `num_computed_prefill_tokens` | `[R]` int32 numpy | CPU | `update_requests`（`min(computed, prefill_len)`） | 判断"还在 prefill" |
| `last_sampled_tokens` | `[R, 1]` int64 | GPU | `post_update` | 下一轮的输入 token 来源之一 |
| `draft_tokens` | `[R, K]` int64 | GPU | 提议者（`draft_tokens[idx_mapping] = ...`） | 下一轮验证的候选 |
| `next_prefill_tokens` | `[lookahead, R]` int32 | GPU | `prepare_prefill_inputs` 内核（prefill 期间） | chunked prefill 的下一枚 token |
| `max_seq_len` | `[R]` int32 numpy | CPU | `add_request` | `prompt_len + max_tokens` |

### 4.2 `InputBatch`（`input_batch.py`）

`idx_mapping [B]`（行→slot）、`idx_mapping_np`、`expanded_idx_mapping [num_logits]`（logits 行→slot）、
`expanded_local_pos [num_logits]`（该请求的第几行 logits）、`num_scheduled_tokens [B]`、
`query_start_loc [B+1]`、`seq_lens [B]`、`seq_lens_cpu_upper_bound [B]`、
`num_computed_tokens_np / prefill_len_np / num_computed_prefill_tokens_np / is_prefilling_np [B]`、
`has_prefill`、`num_draft_tokens` / `num_draft_tokens_per_req`、`input_ids / positions / is_padding`、
`logits_indices [num_logits]`、`cu_num_logits [B+1]`（**含前导 0**）、`cu_num_logits_np`、
`has_structured_output_reqs`。字段与上游逐个同名同义；本仓库裁掉的只有 DCP/PP/R-SWA 三项
（`dcp_local_seq_lens` / `max_seq_len_np` / `prompt_lens`，见 §5 D4）。

### 4.3 内核（逐行移植，除注明外）

| 内核 | 位置 | 做什么 |
|---|---|---|
| `_apply_write_kernel` | `buffer_utils.py` | staged write 批量落盘（支持多 group） |
| `_prepare_prefill_inputs_kernel` | `input_batch.py` | prefill 行从 `all_token_ids` 抄进 `input_ids`（含 lookahead） |
| `_prepare_pos_seq_lens_kernel` | `input_batch.py` | `positions` / `seq_lens`（末块顺手把 `seq_lens` 尾部清零） |
| `_combine_sampled_and_draft_tokens_kernel` | `input_batch.py` | 上次采样 + 本轮草稿写回 `input_ids`；算 `logits_indices` |
| `_get_num_sampled_and_rejected_kernel` | `input_batch.py` | `num_rejected = num_logits − num_sampled`（chunked prefill 一律 0） |
| `_post_update_kernel` | `input_batch.py` | 按 slot 追加 token、更新 `last_sampled` / `total_len` / `num_computed_tokens` / 惩罚计数 |
| `_expand_idx_mapping_kernel` | `input_batch.py` | logits 行 → slot + 行内序号 |
| `_gather_block_tables_kernel` / `_compute_slot_mappings_kernel` | `block_table.py` | 块表 gather、`slot = block * block_size + pos % block_size`、padding 行 `PAD_SLOT_ID(-1)` |
| `_prepare_prefill_inputs_kernel` / `_prepare_decode_inputs_kernel` / `_update_draft_inputs_kernel` | `spec_decode/autoregressive/speculator.py` | draft 第一遍的**左移 + 打补丁**、AR 步的输入推进 |
| `_flatten_sampled_kernel`、`rejection_sample` 全链路 | `spec_decode/{rejection_sampler,rejection_sampler_utils}.py` | 标准验证（V2 版内核，slot 寻址 + 每行可不同 K） |
| `_apply_grammar_bitmask_kernel` | `structured_outputs.py` | 语法掩码打到 logits（按 `cu_num_logits` 解析每行位置） |
| `_temperature_kernel` / `gumbel_sample` / `_min_p_kernel` / `_penalties_kernel` / `_bincount_kernel` / logprob 内核 | `sample/` | 采样处理链（惩罚 → 温度 → min_p → top-k/p → gumbel） |

### 4.4 自回归提议者（`AutoRegressiveSpeculator`）

V2 与 V1（`spec_decode/eagle.py` + `draft_model.py`）**算法相同、坐标系不同**：

| 关注点 | V1 | V2 |
|---|---|---|
| 输入行来源 | 提议者按 `TargetRows` **重建**输入缓冲（63 关的教训：必须与 target 同源） | 直接用 target 的 `input_batch.input_ids/positions` + `num_rejected`（**GPU 张量**）在 draft 缓冲上左移打补丁 |
| 谁决定锚点行 | `rows[i].start + target_rows − 1 − num_rejected`（CPU 算出） | 内核里 `query_len -= num_rejected; last_token_index = query_start + query_len - 1`（**同一语义，GPU 上算**） |
| 被拒行处理 | `is_rejected_token_mask` 掩码 + 哨兵槽位 | **不建掩码**：被拒行留在块里（下一轮必被重算），但只有 `query_len` 内的行参与左移 |
| 草稿的输出位置 | `pending_draft_token_ids`（CPU list） | `req_states.draft_tokens[slot]`（GPU，按 slot） |
| 每步 AR 输入 | 工作区 + `_upload()` 逐块 H2D | `prepare_decode_inputs` 内核改写：1 行/请求，`position+1`、`seq_len+1` |
| 特征 | `combine_hidden_states(cat(aux))`（在提议者里） | 同（`propose()` 开头），宽度用 **draft 配置的** `hidden_size` |

## 5. 差异账本（逐条：为什么、影响、归属）

| 编号 | 差异 | 为什么 | 影响 / 归属 |
|---|---|---|---|
| **D1** | **默认仍是 V1**：`use_v2_model_runner` 只在 `VLLM_USE_V2_MODEL_RUNNER` 显式给值时返回 True；上游默认 V2（少数配置除外） | 本仓库 V2 目前只覆盖 EAGLE/EAGLE3 + 标准验证；58–72 关的既有能力（ngram/suffix/medusa/extract/自定义 proposer/异步/PIECEWISE 图）都只在 V1 路径上，改默认值等于让它们"因为默认值变化而消失" | 环境变量**同名同义**，切换方式与上游一致；`validate_v2_model_runner()` 在不支持时报错而不是退回 V1。放开默认属 84 关的总验收 |
| **D2** | **UVA 视图自建**：上游用 C++ 算子 `torch.ops._C.get_cuda_view_from_cpu_tensor`；本仓库用 `cudaHostAllocMapped` + `cudaHostGetDevicePointer`（libcudart，ctypes）分配映射内存，再把设备指针包成 DLPack 张量 | 本仓库没有 C++ 扩展，也不想为这一个算子引入构建链 | 内存设计相同（host 常驻、GPU 可见、零拷贝、`all_token_ids` 不占显存）。实现细节两条：① DLPack 的 `deleter` 必须是 **NULL**——实测用 Python 回调时张量在解释器退出阶段析构会段错误（exit=139）；② 映射内存进程级不释放（`_MAPPED_KEEPALIVE`） |
| **D3** | **CPU 退化路径**：`UvaBuffer` 在纯 CPU 上让设备视图 = CPU 张量自身；`StagedWriteTensor.apply_write()` 在 CPU 上用 Torch 索引替代 Triton 内核 | CPU 上只有一份内存、也没有 Triton | 只影响无 CUDA 的机器；**V2 Runner 本身在无 CUDA 时直接 `NotImplementedError`**（输入组装/采样/验证全是内核） |
| **D4** | 裁掉 `InputBatch` 的 DCP/PP/R-SWA 三个字段与 `make_dummy` / `set_dummy_context`；`BlockTables` 的 CP 分支保留但恒 `CP_SIZE=1` | 本仓库单卡、无并行策略、无 R-SWA；图（`make_dummy` 的用武之地）属 74 关 | 字段级裁剪，未引入替代算法；`_compute_slot_mappings_kernel` 的 CP 分支逐行保留（只是常量传 1） |
| **D5** | `sample/sampler.py` 裁掉 `logit_bias` / `bad_words` / `thinking_budget` / `prompt_logprob` / flashinfer 后端 / `num_nans` 指标 | 本仓库这些字段在**请求期**就明确拒绝（68 关白名单）；无 flashinfer 依赖；无指标通路 | 不建"没有生产者"的状态类（AGENTS §8）。`num_nans` 恒 None；`use_flashinfer` 恒 False → 走上游的 Torch 分支 |
| **D6** | `sample/states.py` 保留 **`logprobs=-1 → vocab_size`** 的归一化 | ——（这条是上游行为） | ⚠️ 顺带说明：68 关记的"本项目未修"是 **V1 侧**；**V2 侧已对齐**（`SamplingStates.add_request`）。V1 侧的归一化仍是待办 |
| **D7** | `spec_decode/__init__.py` 只实现 EAGLE/EAGLE3 分支；dflash(76)/dspark(77)/多模块 MTP(80)/mtp(80) **明确 `NotImplementedError`** | 本机没有那些 checkpoint 可验收；"把普通 draft 自动转接成别的算法"是需求 073 §3 点名禁止的 | 报错信息里写明归属关卡；V1 侧的 mtp 路径不受影响 |
| **D8** | `AutoRegressiveSpeculator` 只做 **eager**：`init_cudagraph_manager` 在图模式 != NONE 时 `NotImplementedError`，`capture()` 空操作；`use_fused_multi_step_decode` 恒 False（走上游的"逐步重建 metadata"回退路径） | V2 图与融合属 74 关；本仓库注意力后端没有 `update_draft_decode_metadata` | 与上游"后端不支持融合就回退"的分支一致；配置期 `validate_v2_model_runner()` 也挡住非 NONE 的图模式 |
| **D9** | `model_runner.py` 裁掉 LoRA / 多模态 / PP / DCP / PCP / MoE+EPLB / KV connector / 池化 / adaptive verification / prompt logprobs / `_dummy_run`（显存 profiling） | 本仓库没有这些子系统；KV 容量由 `CacheConfig` 给定而不是 Runner 实测 | `get_kv_cache_spec()` / `capture_model()` 明确不实现（不是空壳）；`free_states()` 保留但为空（上游在那里释放多模态缓存） |
| **D10** | `AsyncOutput` 只搬 `sampled_token_ids` 与 logprobs；`num_nans` / `sampling_masks` / `routed_experts` / prompt logprobs / EP 故障检测不搬 | 对应子系统本仓库没有（见 D5/D9） | 侧流非阻塞 D2H + blocking 事件 + 一次性交付（`_delivered`）这些**关键性质**都保留 |
| **D11** | `spec_decode/rejection_sampler.py` 只接 **standard**；synthetic/block 断言拒绝（配置期也拒绝） | 属 75 关（块验证/synthetic） | 上游同位置的两条分支在 75 关补；`use_block_verification` 恒 False |
| **D12** | `use_fp64_gumbel` 恒 False；`use_local_argmax_reduction` 不支持（配置里没有这两项） | 本仓库配置无对应字段 | fp32 gumbel 与上游默认一致（上游 fp64 只在特定机型开）；`get_top_tokens` 未实现，开了也没有模型支持 |
| **D13** | Scheduler 侧 V2 只分叉两处（恢复请求按 `NewRequestData` 发 + 带 `prefill_token_ids`；续跑不再随包带 `all_token_ids`），**仍维护 `prev_step_scheduled_req_ids`** | 上游 V2 靠 worker 侧草稿 + `next_decode_eligible_step`，不再需要这份集合；本仓库的 Scheduler 仍持有 `spec_token_ids` 并据此判"草稿是否断代" | 多维护一份集合没有行为副作用（V1 语义不变）；等 74 关把草稿所有权也搬过去时可以删 |
| **D14** | `DraftTokensHandler.get_draft_tokens()` 在同步调度下交回 **-1 占位** | ——（上游行为） | 语义：V2 的草稿常驻执行侧，调度器只需要**宽度**；真 id 由 `combine_sampled_and_draft_tokens` 按 slot 读。这条是 V2 能异步的前提，探针 E3 钉住 |
| **D15** | `minivllm/outputs.py::LogprobsTensors` 补上 `to_cpu_nonblocking()` / `cat()`；`SamplerOutput` 的 V2 版放在 `worker/gpu/sample/output.py` | 68 关时这两个方法"没有调用方"被裁掉，V2 的 chunked 验证与侧流交付现在需要它们 | 与上游同名同语义；`filter()` 的断言（不能与 `cu_num_generated_tokens` 同用）保持 |

## 6. 实测（本机，RTX 5090 Laptop / WSL2）

命令与结果（全部真跑，`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`，V2 相关需要
`VLLM_WSL2_ENABLE_PIN_MEMORY=1`——与上游在本机的行为一致）：

| 命令 | 结果 |
|---|---|
| `python -m pytest tests/step73 -q` | **50 passed**（上游逐值差分 11 + logprob 差分 9 + 采样状态 17 + 常驻 slot/输入组装 5 + V1/V2 管道 8） |
| `python -m pytest tests/step58 … tests/step73 -q` | **659 passed**（609 + 50，无回归） |
| `python benchmarks/check_step73_v2_runner.py` | **23 项全部通过**（A 常驻 slot 3 / B 输入组装 6 / C 状态所有权 4 / D V1-V2 差分 3 / E 投机管道 3 / F logprobs 2 / G 边界 2） |
| `check_step58…check_step72` 18 个脚本 | 全部通过（58/59/60/61/62/63/64/65/66/67/68×2/69/70/71/72），`check_step73` 23 项 |
| `check_step57_*.py` 15 个脚本 | 全部通过（350 项） |

关键数字：

- **V1 与 V2 在同一 tiny 模型 + EAGLE3 上 greedy 输出逐 token 相同**（K=1、K=3；三条请求、
  混合 prefill 长度；以及 `num_gpu_blocks=6` 压出 **4 次抢占恢复**的场景）。
- **V2 真的在投机**：每轮草稿行数 = K（3），`num_rejected = num_logits − num_sampled` 逐轮自洽。
- **常驻 slot 例子**（需求 §4 第一条）：A=slot 5、B=slot 2、batch `[B,A]` →
  `idx_mapping=[2,5]`、`cu_num_logits=[0,3,6]`、`expanded_idx_mapping=[2,2,2,5,5,5]`、
  块表 gather 后第 0 行 = B 的块、第 1 行 = A 的块、`slot_mapping=[14,15,28,20,21,22]`（批尾 `-1`）。
- **状态跟着 slot 走**：`post_update` 在 batch 从 `[B,A]` 翻成 `[A,B]` 之后仍然：
  A 的历史追加在 slot 5、B 在 slot 2（`last_sampled` / `total_len` 同）。
- **`prompt_len` 不被 `prefill_len` 覆盖**：抢占恢复后 `prompt_len=4 < prefill_len=6`，
  `max_seq_len=12`。

### 6.1 真实权重对照（`models/Qwen3-1.7B` + `models/Qwen3-1.7B-eagle3`，K=2，bf16，greedy）

三条 prompt（`The capital of France is` / `1+1=2, 2+2=4, 3+3=` / `Write a haiku about the sea:`）
各生成 48 个 token，`max_num_seqs=1`、`cudagraph_mode=NONE`（V2 本关只做 eager）：

| 执行路径 | 生成 token | drafted | accepted | 轮次 | 耗时 |
|---|---|---|---|---|---|
| V1（`gpu_model_runner.py`） | `r0/r1/r2` 各 48 | 180 | 54 | 90 | 5.1 s |
| V2（`worker/gpu/model_runner.py`） | 同上 | 180 | 54 | 90 | 5.2 s |

- **逐 token 完全相同**（`IDENTICAL: True`），连接受计数也一致（drafted/accepted/轮次三项全等）
  ——这是本关最强的对齐证据：两条路径连"提了哪几枚草稿、被接受了几枚"都一样。
- 接受长度 `(轮次 + accepted) / 轮次 = (90 + 54) / 90 ≈ 1.60`（这三个 prompt 短、事实性强，
  比 63 关那条 128-token 续写 workload 的 1.2162 高，两者不可直接比）。
- 说明：V2 的 eager 路径没有 CUDA Graph，所以这里的耗时**不能**当作"V2 更快"的证据
  （V2 的图与融合属 74 关，性能矩阵属 84 关）。

## 7. 边界与待办（不假装通过）

1. **V2 的 CUDA Graph / 融合多步 decode** 属 **74 关**：本关 `init_cudagraph_manager()` 对
   非 NONE 模式直接报错，`validate_v2_model_runner()` 也在配置期挡住（可跑 `enforce_eager`）。
2. **块验证（block）与合成接受率（synthetic）** 属 **75 关**：配置期与 `RejectionSampler`
   双重拒绝。⚠️ 诚实交代：`rejection_sampler_utils.py` 里与 block verification 相关的三个内核
   （`_compute_cumulative_log_p_kernel` / `_compute_local_residual_mass_kernel` /
   `_compute_global_residual_mass`）是**逐行搬运但未被测试跑过**的（本关差分固定
   `use_block_verification=False`）；75 关接它们时要先补差分。
   另一个口径说明：`rejection_sample` 交回的 `sampled` 是 `new_empty`，某请求的草稿数少于 K 时
   其尾部**两个实现都不写**（是回收显存），所以差分断言写成"`num_sampled` 逐值相等 + 每请求
   `[:num_sampled[r]]` 前缀逐值相等，且写满时整张张量逐值相等"——不是放宽已定义的位。
3. **V2 只支持 EAGLE/EAGLE3**：DFlash(76)/DSpark(77)/多模块 MTP(80) 明确报错；原生 MTP 的 V2
   speculator 未接（本机无 checkpoint 可验收，V1 路径仍可用）。
4. **异步调度仍未放开**（70 关的边界）：V2 提供了"状态按 slot 落 GPU + 侧流交付"这两块地基，
   但引擎级的 `async_scheduling=True` 依旧 `NotImplementedError`。放开需要把输入组装也搬上
   GPU（74/75 关那条路）。
5. **草稿质量未在此关重新评估**：tiny 权重是随机的（接受率≈0 属正常）；真实 EAGLE3 权重的
   接受长度差距是 63 关的已知问题（`docs/step63_alignment.md` §8），V2 不改变它——
   V1/V2 在同一权重上**逐 token 相同**才是本关的判据。
