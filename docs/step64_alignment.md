# 64 关对齐记录：HiddenStateExtraction 的 CacheOnly 执行路径

需求：[`064_HiddenStateExtraction的CacheOnly执行路径.md`](../../vllm-omni/learning_notes/14_vllm_from_scratch/投机解码完整需求/064_HiddenStateExtraction的CacheOnly执行路径.md)
基线：本机 `vllm==0.28.0` 文件快照。

> **状态：已实现并实跑**（`tests/step64` 13 项 + `benchmarks/check_step64_hidden_cache.py` 18 项全 PASS；
> tiny 与**真实 Qwen3-1.7B**（fp16、辅助层 (2,14,25)）两条路径都做过物理 slot 逐值校验与 greedy 对照）。
> 未做项集中在 §5：CUDA Graph 的 padding 行与"0 号块留白"（69 关）、异步调度的
> `prepare_next_token_ids_padded`（70 关）、跨进程 KV/特征搬运（81 关）。

## 0. 一句话：这一关解决什么

**特征算出来了，但没有地方体面地存。** 63 关已经能让 target 顺带吐出辅助层特征 `[T, L, H]`，
可它只是当轮的中间量。想要它就得每步 `torch.save` / 塞进 Python 列表——那套做法没有寻址、
没有块生命周期、也不知道 prefix 命中时"这段已经算过、不会再算"：

| 项 | 数值（Qwen3-1.7B：H=2048、辅助层 (2,14,25) → L=3、fp16） |
|---|---|
| 每 token 特征 | 3 × 2048 × 2 B = **12 KB** |
| block_size=16 的一块 | 16 × 12 KB = **192 KB**（真实实测 196,608 B） |
| 32 请求 × 4096 token | 131072 × 12 KB ≈ **1.5 GB**，且请求结束后无释放时机 |
| prefix 命中 900 token | 天真做法直接**丢 900 × 12 KB = 10.8 MB/请求**（命中段不再前向） |

做法不是自造一套存储，而是**把特征伪装成 KV**：cache-only 层的缓存形状是
`[num_blocks, block_size, L, H]`——**L 当 head 数、H 当 head_size**，于是一个 token 恰好占一个
slot，和 target 的 KV 共用同一份 `slot_mapping`、同一张逻辑块表、同一套 prefix 复用与释放逻辑。

同时它**不猜 token**：`propose()` 返回 `sampled_token_ids[:, :1]`，即 target 本轮自己采出的第一列
当草稿，`num_speculative_tokens` 固定 1。所以本关不计入"投机加速算法"，也不宣称速度收益
（需求 §5 原话）；采样分布仍然精确是 target 的分布（草稿是点质量 q，走 59 关的 `NO_DRAFT_PROBS` 分支）。

## 1. 代码映射

| 本项目 | 上游参考 | 说明 |
|---|---|---|
| `spec_decode/extract_hidden_states.py::ExtractHiddenStatesProposer` | `v1/spec_decode/extract_hidden_states.py:28-405` | `propose()` 三步：`stack` 特征 → 建 drafting metadata → forward 写缓存；返回第 0 列 |
| `models/extract_hidden_states.py::CacheOnlyAttentionBackend` | `model_executor/models/extract_hidden_states.py:96-148` | 缓存形状公式 `(num_blocks, block_size, num_kv_heads, head_size)`，**没有 k/v 维** |
| `models/extract_hidden_states.py::CacheOnlyAttentionMetadata(MetadataBuilder)` | 同文件 `:151-187` | cache-only 层的元数据**只有 slot_mapping** |
| `models/extract_hidden_states.py::CacheOnlyAttentionImpl` + `basic_cache` | 同文件 `:44-91, 190-231` | 散射写缓存 `kv_cache[slot//bs, slot%bs] = to_cache`；`-1` → `clamp_min(0)` |
| `models/extract_hidden_states.py::CacheOnlyAttentionLayer` | 同文件 `:237-333` | 取 metadata → `do_kv_cache_update()`；`kv_cache` 由提议者绑定 |
| `models/extract_hidden_states.py::ExtractHiddenStatesModel` | 同文件 `:339-394` | 只有一个 cache-only 层的"模型"，层名 `cache_only_layers.{target 层数}`；`load_weights` 返回空集 |
| `config.py::SpeculativeConfig.uses_extract_hidden_states` / `_resolve_extract_hidden_states` / `derive_extract_hidden_states_config` / `extract_hidden_states_hf_config` | `config/speculative.py:850-874, 1495-1496`；`transformers_utils/configs/extract_hidden_states.py` | 方法识别（`model` 换成字面量）、K=1 与辅助层编号校验、把 target 配置派生为 cache-only 配置 |
| `worker/gpu_model_runner.py::_propose_extract_hidden_states` / `_padded_sampled_token_ids` | `v1/worker/gpu_model_runner.py:5233-5258`（`propose_draft_token_ids()` 的 extract 分支） | Runner 的**特殊协议分支**：把本轮采样摊成 `[B, K+1]`（无效 `-1`）→ 调 `propose()` → 只取第 0 列当草稿 |
| `worker/gpu_model_runner.py`（`load_model` / `initialize_kv_cache` / `ExecuteModelState.common_attn_metadata`） | 同文件 `:695-700, 5620-5636, 7800-7805` | 辅助层采集开关、`validate_same_kv_cache_group()`、把本轮元数据留给提议者 |
| `models/qwen3.py`（63 关就有的 `set_aux_hidden_state_layers` / 辅助层输出） | `model_executor/models/qwen3.py` 的 EAGLE3 接口 | extract 复用同一条采集路径（`capture_aux_hidden_states`） |

## 2. 设计要点（改动时不要破坏）

1. **槽位必须与 target 同源**：提议者用的是本轮 target 那份 `AttentionMetadata.slot_mapping`
   （`ExecuteModelState.common_attn_metadata` 传下去），**不是**按位置公式重算的。上游同样直接吃
   `common_attn_metadata.slot_mapping`。特征与 KV 因此落在同一个 slot 上——这是"读物理 slot 能校验"
   的前提。
2. **缓存张量独立、块编号共用**：cache-only 层有自己的物理张量（和 draft 的 KV 一样），但块编号来自
   target 的逻辑块表。于是 prefix 命中的块在两边指向同一段位置：特征与 KV 都是"token + 位置的确定
   函数"，共享块对任何同前缀请求都成立（`tests/step64/test_hidden_cache.py::test_prefix_hit_keeps_cached_features`）。
3. **层名 ↔ metadata**：`cache_only_layers.{target_num_hidden_layers}` 是 forward 上下文的键；
   提议者按层名收集（`_collect_attn_layers`），并断言**恰好一层**。
4. **`[T, L*H]` ↔ `[T, L, H]`**：本仓库 target 把辅助层拼在最后一维（63 关给 draft 的
   `combine_hidden_states` 切块用），提议者用 `view(T, L, H)` 重解释；与上游的
   `torch.stack(list, dim=1)` 逐值相同（有用例）。两种输入形态都接受。
5. **`-1` 哨兵落到 0 号块**：上游的块池把 0 号块留白当垃圾桶，所以 `clamp_min(0)` 是安全的。
   **本仓库的 0 号块会分配给真实请求**，因此 69 关引入 CUDA Graph padding 行时**必须同时把 0 号块
   留白**，否则 padding 行会覆盖真实请求的特征（现在 target 的 slot_mapping 从不产生 `-1`——
   越界会当场报错，见 `BlockTable.compute_slot_mapping`）。
6. **拒绝尾部**：K=1 时每请求一轮 2 行（b + 1 枚"草稿"），草稿行的位置下一轮必被重算
   （63 关不变量），所以它的特征槽位自然被正确 token 覆盖；用例断言"同一槽位被写多次、最终内容
   = 最后一次写入"。
7. **请求生命周期**：提议者**没有按请求的状态**（只有一块定长特征缓冲），所以不需要
   `remove_requests`；`validate_same_kv_cache_group()` 校验的是"缓存确实是按本轮规格分配的"
   （块数/块大小对不上就会把特征写进别的块，且不报错）。

## 3. 与上游的差异账本（逐条）

| 上游 | 本项目 | 为什么 / 影响 |
|---|---|---|
| `disable_padded_drafter_batch` 为真时报错 | 没有这个开关 | 本仓库没有 padded drafter batch 的异步路径（70 关），开关不存在 |
| `_get_slot_mapping()` 把 target 槽位拷进常驻 buffer、尾部补 `PADDING_SLOT_ID` | 直接用本轮 target 的同一份 `slot_mapping` | 那个 buffer 是给 CUDA Graph 的 padding 与稳定 `data_ptr` 用的（69 关）；本关没有 padding |
| `_determine_batch_execution_and_padding()` / `dummy_run()` / DP 协调 / EPLB | 无 | 本仓库没有 CUDA Graph、DP、EPLB |
| `prepare_next_token_ids_padded()`（第 0 列 + backup 回退 + 有效计数） | 不实现 | 它服务的是"异步调度下不落 CPU 的输入缓冲补齐"（`prev_sampled_token_ids`，70 关）；本仓库的下一轮输入来自已提交历史（`_bookkeeping_sync`），没有这个消费者。不写没有调用方的代码 |
| K 的约束是 `ExtractHiddenStatesProposer.__init__` 里的 `assert K == 1` | 配置期 `ValueError` | 约束相同，报错更早、信息更清楚（需求 §3.1 要求 config 处理 K 的约束） |
| 层从 `forward_context.slot_mapping`（dict）取槽位，并用 `unified_kv_cache_update()` 自定义算子保序 | 层从 `attn_metadata[layer_name]` 取槽位（与 `attention/layer.py::Attention` 同款） | 上游那层间接是给 torch.compile 保序 + 触发 KV connector；本仓库没有 compile（69 关）与 connector（81 关） |
| `@maybe_transfer_kv_layer` 的 `dummy_attention()` | 无 | KV connector 是 81 关 |
| `set_default_quant_scales` / `is_quantized_kv_cache` / `CacheDType` / "量化 KV 时把 cache_dtype 改成 auto" | 无 | 本仓库 KV 精度 = 模型精度，没有 `cache_dtype` 旋钮；保留"写进去的 dtype 必须与缓存一致"的断言 |
| `get_kv_cache_spec()` → `HiddenStateCacheSpec`（KV 规格类型树 + 单独 group） | 层给出形状公式，提议者按 `cache_config` 分配 | 本仓库只有一个 KV group、没有规格类型树；cache-only 层不走 `KVCacheManager` 的层列表（复用 target 逻辑块表，与 draft 同一套） |
| `ExtractHiddenStatesConfig`（在 `SpeculativeConfig.__post_init__` 里把 target 配置派生出来） | `extract_hidden_states_hf_config()` + `derive_extract_hidden_states_config()`，由提议者调用 | 上游的 `SpeculativeConfig` 持有 `target_model_config` 字段；本仓库的配置拿不到 target，所以派生放在同时持有两者的提议者里。另外多写两个字段（`cache_block_size` / `torch_dtype`），因为本仓库的模型只吃一个 config dict |
| `CacheOnlyAttentionImpl.forward()` 是空实现 | 调它直接报错 | "算了但没写缓存"在特征提取里是静默丢数据，报错更好查 |
| 上游注释称"草稿总能通过验证（always verify）" | 不照抄这句结论 | 按同一套协议在本仓库实测：草稿是**上一轮**采出的 token，下一轮验证时一般是**被拒**的（tiny：26 轮里 4 轮被接受；真实 1.7B 的 8 token 生成：7 轮全部 `num_accepted=0`）。被拒时第 0 列就是 target 自己采出的 token，输出仍逐 token 正确，所以"能不能一直通过"不影响正确性。另外按上游代码读，`valid_sampled_tokens_count = is_valid(sampled[:, 0])` 恒为 1（→ `num_accepted = 0`），在草稿**真的被接受**的那一轮会漏掉 bonus 那一列；这一点没有在上游引擎上实跑验证，本仓库的记账走 `RejectionSampler.parse_output` 的实际产出，不复制它 |

## 4. 实测

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/step64 -q          # 13 passed
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmarks/check_step64_hidden_cache.py   # 18 项 PASS
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/step58 tests/step59 tests/step60 \
    tests/step61 tests/step62 tests/step63 tests/step64 -q                    # 314 passed
```

设备 `cuda:0`（RTX 5090 Laptop / WSL2），torch 2.13.0+cu130，tiny target `tiny_gqa`
（2 层、H=32、vocab 11）+ 辅助层 `(0, 1)`。

**tiny 轨迹（关键数字）**

| 项 | 实测 |
|---|---|
| 缓存形状 | `(32, 4, 2, 32)` = `[blocks, block_size, L, H]`，fp32（tiny） |
| 首轮写入 | 两请求 prefill 共 12 行，槽位与 `slot_mapping` 逐行一致；`[T, L, H] = [12, 2, 32]` |
| 独立参考 | 同权重 + 同元数据重跑一次前向，与写进缓存的那份 **逐值相同**（`torch.equal`） |
| 层/请求区分力 | `stacked[:,0] != stacked[:,1]`、`stacked[0] != stacked[8]`（互换看得出来） |
| 特殊协议 | `sampled=[B,2]` → 返回 `[B,1]` 且 == 第 0 列；6 次调用全部成立 |
| 拒绝尾部 | 同一槽位最多被写 2 次（草稿行 → 下一轮重算的正确 token）；最终内容 = 最后一次写入 |
| chunked prefill | 预算 8、prompt 12 → 2 块（8+4），12 个位置与"另一个引擎一次装下"的参考一致（atol 1e-6） |
| prefix 命中 | 第二条请求起点 = 8（命中两个完整块）、本轮只排 4 个 token；命中段特征非空、新算段逐值正确 |
| 端到端 | greedy：`{'a': [4,4,4,6,0,6], 'b': [10,4,4,4,4,4]}` == 非投机；草稿真的进过调度器 |
| 接受率（tiny，26 轮 greedy） | 4 轮接受（产出 2 个 token）、22 轮被拒（产出 1 个） |

**真实权重轨迹（`models/Qwen3-1.7B`，fp16，辅助层 (2,14,25)）**

| 项 | 实测 |
|---|---|
| 缓存形状 / 每块 | `(64, 16, 3, 2048)` fp16，**196,608 B/块**（= 16 × 3 × 2048 × 2） |
| 首轮写入 | prompt 3 token → slots `[0, 1, 2]`，逐槽位 `torch.equal` 通过 |
| 独立参考 | 重跑前向对照 **max\|Δ\| = 0.0**（逐位相同） |
| 位置 0 的缓存值（层 0 前 4 维） | `[-4.04296875, -6.1640625, -4.546875, 6.0546875]` |
| greedy 8 token | 投机 `[264, 729, 315, 882, 11, 323, 1221, 1477]` == 非投机（逐 token） |
| 调度统计 | 7 轮 ×（1 枚草稿、0 枚接受）——与"草稿是上一轮 token"的语义一致 |

## 5. 未做 / 待验（不要当成已覆盖）

1. **CUDA Graph 的 padding 行**（69 关）：`PADDING_SLOT_ID` 的重定向已按上游实现并有单测，
   但"0 号块留白"这个前提要等 69 关引入 padding 时一起做（§2.5）。
2. **异步调度路径**（70 关）：`prepare_next_token_ids_padded` / `prev_sampled_token_ids` 那套
   输入补齐没有实现；本仓库的输入来自已提交历史。
3. **跨进程搬运**（81 关）：上游的 `ExampleHiddenStatesConnector` 把缓存里的特征写盘/传输；
   本关只保证"特征在缓存里、按 slot 可寻址"，不建传输服务（需求 §3.5 明确划界）。
4. **量化 KV cache**：上游支持 `cache_dtype=bfloat16` 等并对量化 KV 做特殊处理；本仓库没有
   `cache_dtype`（KV 精度 = 模型精度）。
5. **无 Graph vs 有 Graph 的数据一致性**（需求 §4 最后一条）：属于 69 关回归项。
6. **EPLB / DP 协调**：本仓库没有（上游 `set_eplb_state` / `coordinate_batch_across_dp`）。

## 6. 验证命令（复跑）

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/step64 -q
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmarks/check_step64_hidden_cache.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/step58 tests/step59 tests/step60 \
    tests/step61 tests/step62 tests/step63 tests/step64 -q
```
