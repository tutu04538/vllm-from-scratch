# 第 80 关对齐记录（80A / 80B / 80C / 80D）

> 本文件按子需求追加。当前完成 **80A 输入层**、**80B eager 执行/拒绝后 KV 修复**、
> **80C 多模块整段 CUDA Graph/采样生命周期** 与 **80D Q-only 共享 KV 的映射与只读
> 契约**；80E–80H 未开始。80D 只做机制（源层映射、只读、固定位置、所有权），不加载
> 真实 Gemma4；真实 assistant 结构与端到端接入属 80E。

## 0. 需求与边界

- 需求：
  - `投机解码完整需求/080A_多模块输入布局与跨轮缓存.md`
  - `投机解码完整需求/080B_多模块Eager执行与拒绝后KV修复.md`
  - `投机解码完整需求/080C_多模块Graph与采样生命周期.md`
  - `投机解码完整需求/080D_Q_only共享KV的映射与只读契约.md`
- 80A 做**输入准备和跨轮缓存布局**：把 target 本轮输入整理成 draft 的 packed 行，
  把窗口尾部 token/hidden/embedding 按持久 slot 缓存，定义窗口移动与尾部 padding。
- 80B 把 80A 接成 V2 的 `MultiModuleMTPSpeculator`：显式 `propose()`、整段 packed
  query 的模块循环、按 `spec_step_idx` 选模块、拒绝后 gap 在当前 query 窗口内重算。
- 80C 把 80B 已经正确的整段循环（`cache_inputs` → K 个 module forward → 采样 →
  hidden 回写/输入 shift）捕获成一张 FULL 图；`propose` 只做图外输入准备、分派和
  `run_fullgraph`，不在 replay 后补 CPU 计算。
- 80D 做 **Q-only 草稿层的源层映射 + 只读读路径**：按 target `layer_types` 与
  `num_kv_shared_layers` 选同类型最后一个非共享源层；draft 层只绑 target 的 cache，
  不计算/不写 K/V、不分配/不释放第二份 cache；同轮草稿步 positions/seq_lens 固定。
  映射用冻结上游方法逐例差分，真实 Gemma4 assistant 仍明确拒绝（80E）。
- 明确不做：Gemma4/Step3p5 模型结构、真实 MTP 权重、稀疏索引、性能矩阵。
  生产工厂把 `use_multi_module_mtp()` 分派到多模块类；Gemma4 分支明确拒绝。

## 1. 路径映射

| 本项目 | 冻结上游（vLLM 0.28.0 / 2026-10-03） |
|---|---|
| `minivllm/worker/gpu/spec_decode/multi_module_mtp/speculator.py` | `v1/worker/gpu/spec_decode/multi_module_mtp/speculator.py` 的输入层 |
| `minivllm/worker/gpu/input_batch.py::InputBuffers` | `v1/worker/gpu/input_batch.py::InputBuffers` |
| `minivllm/worker/gpu/input_batch.py::InputBatch` | 同上 `InputBatch` |
| `minivllm/worker/gpu/block_table.py::PAD_SLOT_ID` | `v1/attention/backends/utils.py::PAD_SLOT_ID` |
| `minivllm/worker/gpu/spec_decode/gemma4/kv_sharing.py` | `v1/worker/gpu/spec_decode/gemma4/speculator.py::Gemma4Speculator._setup_gemma4_kv_sharing` |
| `minivllm/attention/layer.py::Attention.bind_shared_kv_cache` / `forward` 的 `q_only` 分支 | `model_executor/models/gemma4_mtp.py::Gemma4MTPAttention` + `Attention` 的共享读路径 |
| `minivllm/attention/backends/torch_sdpa.py::forward_q_only` | 后端"只读 cache、不写 K/V"的读路径 |
| `minivllm/worker/gpu/model_runner.py::_allocate_draft_kv_caches` 的共享分支 | Runner 把 target cache 绑给 draft、跳过分配 |

本关实现的函数（均与上游同名）：

| 函数 | 作用 |
|---|---|
| `prepare_input_buffers` / `_prepare_input_buffers_kernel` | target 行 → draft packed 行；写 `last_token_indices`、`seq_lens`、overrides；chunked prefill 的 next token |
| `prepare_input_hidden_states_and_embeddings` / 对应 kernel | 把 target hidden 复制到 draft 窗口，并用同一缓存列补 gap 的 hidden/embedding |
| `cache_inputs` / `_cache_inputs_kernel` | 按持久 slot 快照窗口尾部 K−1 个 token/hidden/embedding |
| `pad_trailing_draft_slots` / 对应 kernel | 把每条请求 `last_token_index` 之后的槽位打成 `PAD_SLOT_ID(-1)` |
| `update_draft_inputs` / `_shift_input_ids_kernel`、`_shift_input_embeds_kernel` | 丢掉窗口最左一行，右端追加新候选 |

## 2. 三条不变量

| 编号 | 不变量 | 落地 |
|---|---|---|
| I1 | 跨轮缓存按**持久 request slot** 索引 | `cached_*[idx_mapping[req_idx], ...]`；不是 batch 行 `req_idx` |
| I2 | 缓存列保留**未截断的窗口起点** | `window_start = last - (K-1) + 1` 可为负；读时只把下界夹到 `query_start`，写列仍用 `i - window_start` |
| I3 | token/hidden/embedding 同源 | `cache_inputs` 从同一批 `input_ids` / `hidden_states` / `inputs_embeds` 行快照；refill 从同一缓存列补 |

另外两条协议边界：

- `K=1` 时缓存宽度就是 `K-1=0`，**不能**写 `max(1, K-1)` 改成 1 列。
- `pad_trailing_draft_slots` 必须收到**实际** `last_token_indices`；传 `query_end-1`
  会让循环为空，看起来通过但尾部没有屏蔽。

## 3. 固定例子：`[2,1]` 而不是 `[0,0]`

`max_num_seqs=4, B=1, K=3, idx_mapping=[2]`，`query_start_loc=[0,1]`，输入行 0
token=42、hidden=7，cache 初值 -9。

理想窗口宽度 `K-1=2`，窗口起点：

```text
window_start = last - (K-1) + 1 = 0 - 2 + 1 = -1
```

读时只能从行 `max(-1,0)=0` 读到行 0；但缓存列要用未截断的 `window_start`：

```text
cache_write_slot = 0 - (-1) = 1
cached_ids[2,1] = 42
cached_hidden[2,1] = 7
```

如果先 clamp 到 0，就会写成 `cached_ids[0,0]`：主键从持久 slot 2 变成 batch 行 0，
列从 1 变成 0。下一轮按 `[2,1]` 读到的是 -9 或别的请求的数据；不会抛异常，只会让
草稿质量变差。列 0 空着不是 bug：它表示窗口起点在请求开头之前；把 42 挤到列 0 会
改变窗口内的相对位置。

## 4. 各函数的精确口径

### 4.1 `prepare_input_buffers`

对每条请求：

```text
r            = num_rejected
num_repr     = max(0, r - 1)
num_input    = query_len - r
last         = query_start + num_repr + num_input - 1
seq_len      = target_seq_len - num_repr
```

- 输入 token：target 第 `1..num_input-1` 行复制到 draft 的
  `query_start+num_repr .. last-1`；target 第 0 行是 anchor，不复制。
- positions：target 第 `0..num_input-1` 行复制到 draft 的
  `query_start+num_repr .. last`。
- gap：`num_repr` 行从缓存列 `K-1-num_repr+i` 读取；位置为
  `first_position - num_repr + i`。
- `next_token`：`num_sampled>0` 用 `last_sampled[slot]`，否则用
  `next_prefill_tokens[0, slot]`。
- overrides：仅 `num_sampled==0` 时写入
  `next_prefill_tokens[i, slot]`（i=1..K-1），否则 -1；**按持久 slot 取**。
- 尾部补齐：超出真实请求数的 `query_start_loc` 写最后一条的 `query_end`，
  `seq_lens` / `last_token_indices` 写 0。

### 4.2 `cache_inputs`

```text
cache_window_size = K - 1
window_start      = last - cache_window_size + 1
for i in [max(window_start, query_start), last]:
    cache_write_slot = i - window_start
    cached_ids[slot, cache_write_slot] = input_ids[i]
    cached_hidden[slot, cache_write_slot, :] = hidden[i, :]
```

`idx_mapping=-1`（CUDA Graph 补齐请求）直接跳过，不能写 0 号 slot。

### 4.3 `update_draft_inputs`

对每条请求的 `[query_start, last]`：

```text
input_ids[query_start + i - 1] = input_ids[query_start + i]   (i=1..query_len-1)
input_ids[last]                = draft_tokens[req_idx]
```

embedding 同方向。这里只移动**行**，不改变绝对 position；positions 的推进属于
80B 的模型语义。

### 4.4 `pad_trailing_draft_slots`

槽位按“陈旧 token 位置”算出来后，`last_token_index+1 .. query_end-1` 这些行仍可能
带着合法块号；不覆盖就会读/写真实 KV。因此函数把这些行原地改成 `PAD_SLOT_ID`。
多个 KV group 都要处理。

## 5. 验证

测试文件：`tests/step80/test_multi_module_inputs_alignment.py`，**43 passed**（CUDA）。
每个输入函数都直接调用本机冻结 `vllm 0.28.0` 的同名函数，用 `torch.equal` 比较
**整个缓冲**（input_ids/positions/seq_lens/query_start_loc/last_token_indices/
overrides/cache/hidden/embedding/slot_mappings），不是只比 shape 或最终 token。

覆盖：

- 固定例子整块 cache：`cached_ids[2,1]=42`、hidden/embedding 同列；
- `idx_mapping=[3,1]` 再反转为 `[1,3]`：数据跟持久 slot 走，不跟 batch 行；
- `idx_mapping=-1` 补齐行不写 0 号 slot；
- `num_rejected=0..K` × `K=1/2/3/4`：逐值差分 + 手算 last/seq_len/输入复制/gap/positions；
- query 长度 `< = > K-1`（K=3 的 1/2/4 行）；尾部哨兵不被污染；
- chunked prefill overrides 按持久 slot 读，反例用 batch 行值区分；
- hidden/embedding 的 no-embeds 与有 embeds 两条分支；gap 的 hidden/embedding 同列；
- pad 的实际尾部屏蔽；`last=query_end-1` 的空操作单独对照，不能拿来冒充通过；
- `update_draft_inputs` 的 token/embedding 同步移动、请求边界不串；
- `K=1` 空窗口：cache 宽 0，不写 cache，不崩；另有纯 CPU 的宽度断言
  （`K-1`，不是 `max(1,K-1)`）。

全量回归（含 80A）：

```bash
PATH=/home/user/anaconda3/envs/vllm-omni-dev/bin:$PATH \
VLLM_WSL2_ENABLE_PIN_MEMORY=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m pytest tests/step58 … tests/step79 tests/step80 -q
# 1011 passed, 149 warnings in 91.87s
```

`PATH` 里要有 conda env 的 `ninja`；否则 step73 的 UVA C++ 扩展构建会报
“Ninja is required”，这是环境问题，不是本关代码回归。

## 6. 80A 已知边界与差异

1. **没有模型/Graph/性能**：80A 只测输入布局，不加载 target/draft 权重，不做 CUDA
   Graph，不做性能测试（需求明确“无需性能测试”）。
2. **没有真实多模态模型**：`cached_draft_input_embeds` 只用人工 tensor 做差分，
   不声称支持多模态 MTP。
3. **`next_prefill_tokens` 行数**：包装函数显式断言至少
   `num_speculative_steps` 行；80B 在 runner 里按 K 设置 `num_prefill_lookahead`。
   这是显式契约，不是改变算法。
4. **CPU**：Triton 输入 kernel 只在 CUDA 上跑；纯 CPU 只覆盖
   `K-1` 宽度协议。完整 CPU 等价实现不在本关范围。
5. **未对齐项**：80A 没有改 `RequestState` / `BlockTables` / attention；80B 只新增
   多模块自己的类与接线，不重写 Engine。

## 7. 80A 改动文件

- 新增 `minivllm/worker/gpu/spec_decode/multi_module_mtp/__init__.py`
- 新增 `minivllm/worker/gpu/spec_decode/multi_module_mtp/speculator.py`
- 新增 `tests/step80/test_multi_module_inputs_alignment.py`
- 更新 `docs/results.json` → `step80.subrequirements.80A`
- 新增本文件（当时只有 80A；80B 追加在下方）

## 8. 80B：eager 执行与拒绝后 KV 修复

### 8.1 类与调用链

`MultiModuleMTPSpeculator(DraftModelSpeculator)` 在同一个文件里新增，显式实现：

```text
propose
  → _copy_request_inputs
  → _prepare_inputs（80A 的 prepare_input_buffers / hidden）
  → compute_slot_mappings + pad_trailing_draft_slots + _build_draft_attn_metadata
  → _generate_drafts
      → cache_inputs（第一个模块之前）
      → for step in 0..K-1:
            _run_model(..., spec_module_idx=step)
            sample_draft（用 last_token_indices 选行）
            update_draft_inputs（丢最左一行，右边追加候选）
```

关键点：

- **不继承 `MTPSpeculator`**，不保留旧实验实现的
  `_maybe_reprefill_stale_modules()` 旁路、`args[5]/args[6]/args[8]` 猜参数、
  额外变长 batch + clone/restore。
- 第一遍/后续模块都跑**同一段 packed query**；模块自己的 `spec_step_idx` 协议选择层
  （`Qwen3MTP` 是 `% num_mtp_layers`），proposer 不自己改 K/模块数的算法。
- `_run_model()` 分开两个返回值：`last_hidden_states` 只用于 logits/采样，
  `hidden_states` 反馈给下一步。
- `cache_inputs()` 在第一个模块之前快照，不能在最后一个模块之后才写。
- `positions` 不随模块步进；`sample_positions` 只用于采样位置（源码同款）。

### 8.2 拒绝后的修复窗口（固定例子）

`K=4`、`query_len=5`、`num_rejected=3`：

```text
num_reprefill = max(3 - 1, 0) = 2
num_input     = 5 - 3 = 2
packed 有效行 = 2 + 2 = 4
第 5 行 = padding
seq_len: 8 -> 8 - 2 = 6
```

窗口：

| 行 | 内容 |
|---|---|
| 0 | 从上一轮 cache 补的 refill 行 |
| 1 | 从上一轮 cache 补的 refill 行 |
| 2 | 本轮有效 target 行（target 第 1 行） |
| 3 | next token（采样行） |
| 4 | padding，槽位 `PAD_SLOT_ID(-1)` |

这不是“再拼一个小 batch 调两次 forward”：补算行就在当前 packed query 里；
`seq_lens`、`last_token_indices`、pad slot 同步更新，然后所有模块按正常循环跑一遍，
各自覆盖自己的 KV。

### 8.3 验证

新增 `tests/step80/test_multi_module_reprefill_alignment.py`，**8 passed**。80B 完成时
加上 80A 的 43 项，`tests/step80` 共 **51 passed**（80C 后为 60）。

- **调用链/边界**（纯 CPU + CUDA）：工厂返回 `MultiModuleMTPSpeculator`；K=2/4、
  模块数=2；每轮 `spec_step_idx` 为 `0..K-1`；纯 CPU 用例检查
  `init_cudagraph_manager()` 的模式收敛（80C 起不再拒绝图；见 §9）。
- **greedy 烟测**：K=2/4 的投机输出与不投机逐 token 相同（只作为最后一项，不当主证据）。
- **独立重放**（主证据）：`RoundRecorder` 记录生产每轮的 target hidden/token/
  position、`num_rejected/num_sampled/last_sampled/next_prefill`、上一轮 cache 与各模块
  KV；测试内的 `ReferenceReplay` 用**自己的**持久 KV、80A 风格 cache 和手写布局
  （不调用生产 `_prepare_inputs` / `_generate_drafts` / `cache_inputs`）逐轮重放。
  比较：
  - 每轮开始前的 `cached_ids` / `cached_hidden` 与完整模块 KV；
  - 每步第一遍输入 IDs、positions、hidden、slot mapping；
  - 每步 logits；
  - 每步各模块 K/V 的有效槽位；
  - 最终 K 枚草稿 token。
- **覆盖的拒绝形态**：记录里同时有 `num_rejected=0`（第一遍/全接受）、
  `0 < num_rejected < K`（部分拒绝）、`num_rejected=K`（全拒）；K=2/4 都跑。
- **负向控制**：
  - 捕获正确的上一轮 cache 后把生产 cache 清零 → 生产输入与独立重放必须不同；
  - 用 step 0 的**输入** hidden 冒充后续 step 的反馈 hidden → 第二步输入必须不同。
- **混合调度**：3 条请求、`budget=8`、`blocks=6`、`enable_prefix_caching=True`；
  trace 里必须有 partial prefill、prefix 命中、抢占恢复；输出与不投机相同
  （`num_preemptions > 0` 有断言，不是空转）。

全量回归（80B 完成时）：

```bash
PATH=/home/user/anaconda3/envs/vllm-omni-dev/bin:$PATH \
VLLM_WSL2_ENABLE_PIN_MEMORY=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m pytest tests/step58 … tests/step79 tests/step80 -q
# 1019 passed, 149 warnings in 94.38s
```

包摘要：`minivllm/` 159 个 `.py`，sha256
`b05e374d48251904c0c430ad3a7f55cdc8dd98ab334e5ee4b0afcff711ae58ec`。

### 8.4 80B 的差异与边界

1. **80B 本身只验证 eager**：80C 已把 `init_cudagraph_manager`/`capture` 接到
   `SpeculatorCudaGraphManager`，并覆盖 q/RNG/slot 归属；Graph 的模式收敛、
   replay 对齐与图桶复用见 §9。
2. **只支持纯文本**：`inputs_embeds` 恒为 None；80A 的 embedding 分支不接生产。
   多模态 MTP 不在本关。
3. **只支持单卡**：不接 DP/EPLB/`num_tokens_across_dp`；上游这些参数本仓库没有对应
   组件，不伪造。
4. **专用模型仍拒绝**：Gemma4/Step3p5/Q-only/稀疏索引不在本类偷偷替换；对应
   80D–80G。
5. **tiny 模型是测试材料**：使用 `tiny_mtp_dir(..., num_mtp_layers=2)` 的
   `Qwen3MTP` 结构；它不声称支持任意真实 MTP checkpoint（例如 Qwen3-Next 的混合
   注意力/MoE）。
6. **`cached_positions` 不保留**：80A 的生产重建 positions 从本轮 target positions
   推导；没有第二套缓存位置权威。
7. **`num_prefill_lookahead`**：runner 对 `use_multi_module_mtp()` 设 K；其它提议者
   仍是 1。80A 的包装函数断言 `next_prefill_tokens` 至少 K 行。

### 8.5 80B 改动文件

- `minivllm/worker/gpu/spec_decode/multi_module_mtp/speculator.py`：新增
  `MultiModuleMTPSpeculator` 与 eager 调用链；更新模块 docstring。
- `minivllm/worker/gpu/spec_decode/multi_module_mtp/__init__.py`：导出类。
- `minivllm/worker/gpu/spec_decode/__init__.py`：`use_multi_module_mtp()` 分支从拒绝
  改为返回新类。
- `minivllm/worker/gpu/model_runner.py`：多模块 MTP 的 `num_prefill_lookahead=K`。
- `tests/step80/test_multi_module_reprefill_alignment.py`：新增 8 项。
- 本文件追加 80B 小节；`docs/results.json` → `step80.subrequirements.80B`。

## 9. 80C：多模块 Graph 与采样生命周期

### 9.1 范围与冻结入口

80C 不新增模型结构，只把 80B 已经跑通的整段多模块循环录成图。参考：

- 冻结 `v1/worker/gpu/spec_decode/multi_module_mtp/speculator.py` 的
  `init_cudagraph_manager()` / `capture()` / `propose()` / `_generate_drafts()`；
- `v1/worker/gpu/spec_decode/autoregressive/cudagraph_utils.py::SpeculatorCudaGraphManager`；
- 本仓库 74 的 `CudaGraphManager` / `BatchExecutionDescriptor` / 常驻输入缓冲。

和普通自回归路径最大的区别：**一个 manager 捕获整段 K 步循环**，不是 prefill/decode
两张图。`decode_query_len = K + 1`（每请求 anchor + K 个 draft 输入行）。

### 9.2 请求模式 → 草稿实际模式

冻结源码只对 PIECEWISE 降级，不把 FULL 降成 FULL_DECODE_ONLY：

| 请求的 `cudagraph_mode` | 实际草稿 manager 模式 | 行为 |
|---|---|---|
| `NONE` | `NONE` | 全 eager |
| `PIECEWISE` | `NONE` | 全 eager；log 记录 requested/actual |
| `FULL` | `FULL` | 源码原样保留 |
| `FULL_DECODE_ONLY` | `FULL_DECODE_ONLY` | decode 批 FULL，混合批 eager |
| `FULL_AND_PIECEWISE` | `FULL_DECODE_ONLY` | 丢掉 PIECEWISE 那一半 |

运行时真正捕获的图只有 FULL；multi-module 没有 PIECEWISE 分段图。冻结源码的
`# TODO: Support piecewise cudagraph for multi-module MTP` 原样保留为边界，不自己发明
一套分段图。

注意 `FULL` 在源码里不降级，但本仓库正常 V2 + Torch 后端路径会在 74 的
`resolve_cudagraph_mode_and_sizes()` 里先把 `FULL` 收成 `FULL_AND_PIECEWISE`（Torch
后端只支持 UNIFORM_BATCH），80C 再收成 `FULL_DECODE_ONLY`。直接对 speculator 调
`init_cudagraph_manager(FULL)` 时 manager 仍保留 FULL；这是源码语义与本地后端能力的
交界，不是 80C 偷偷改写模式。

### 9.3 捕获与重放边界

捕获对象就是 `_generate_drafts`：

```text
cache_inputs（第一个模块之前）
→ for step in 0..K-1:
      module forward(spec_module_idx=step)
      sample_draft（真实 q 缓存路径）
      hidden 回写 / update_draft_inputs
```

关键点：

1. **顺序与 eager 完全一致**。图里没有“采样后再用 CPU 修正”的第二套逻辑。
2. `spec_module_idx` 是 Python 循环变量，捕获时展开，决定每步用哪个 MTP module；
   `current_draft_step` 是 GPU 标量，`fill_(step)` 也被录进图，供采样/q 缓存使用。
   两者不能互相替代：不能用 `.item()` 从 GPU 标量选 module，也不能用 Python step
   冒充采样需要的 GPU 标量。
3. `capture()` 前清 `last_token_indices` / `idx_mapping` / `hidden_states`，避免 dummy
   批读到上一轮的越界 slot；上游只清 `last_token_indices`，这里多清两个是同一类防护。
4. 运行期 `propose()` 在图外做 `_prepare_inputs`、块表 gather、slot mapping/pad；
   选中 FULL 后只调 `run_fullgraph(desc)`，不执行 Python `_generate_drafts`。
5. 图桶比真实批大时：补齐的 input_ids/positions/hidden 清零，块表行清零，
   `seq_lens=0`，slot 写 `PAD_SLOT_ID(-1)`；0 号块留白。重放只写真实槽位。

### 9.4 q / RNG / slot 归属

- q 缓存列按 `current_draft_step` 落到 0..K-1；不是重放时都写同一列。
- temperature/seeds 是逐请求常驻缓冲；图里采样 kernel 读当前值，不会冻结捕获时的
  随机数。
- `idx_mapping` 每轮在 replay 前更新；图地址不变，但 q/候选/KV 归属跟随新的 batch
  顺序，不跟随捕获时 dummy 顺序。
- padding 请求 `idx_mapping=-1`：`cache_inputs` 跳过、采样忽略、槽位 PAD；不能写到
  0 号 slot 或有效请求的块。

### 9.5 验证

新增 `tests/step80/test_multi_module_graph_state.py`，**9 passed**：

1. `test_graph_modes_follow_frozen_downgrade`：五种请求模式的收敛表（纯 CPU）。
2. `test_k1_does_not_take_multi_module_path`：K=1 仍走普通 `MTPSpeculator`，不冒充
   多模块。
3. `test_graph_replay_matches_eager_q_and_module_kv`：K=3、probabilistic、B=2；
   replay 与 eager 比候选 token、q 的 K 列、跨轮 `cached_ids/cached_hidden`，以及
   每个模块在本轮 draft 窗口**同一逻辑位置**上的 KV（经各自 block table 映射，不能
   直接比物理块号）。
4. `test_multi_module_mtp_full_graph_matches_eager`：需求点名的旧测试名保留；断言
   一个 manager、捕获 2 个桶、selection 真正 replayed，不再维护旧实验实现里不存在的
   内部字段。
5. `test_graph_padding_bucket_with_one_real_request_is_safe`：只捕获 2 请求桶，用
   1 请求 replay；补齐行不能污染真实槽位/输出。
6. `test_graph_bucket_reuse_1_2_1_and_padding_safety`：B=1→2→1 复用同一批
   descriptor/graph 对象；0 号块留白，真实请求块号不为 0。
7. `test_graph_replay_segment_has_no_tensor_item`：在 `run_fullgraph()` 段内把
   `Tensor.item` 打成异常，确认 replay 没有 CPU 标量读回。
8. `test_graph_forced_rejection_and_acceptance[1]`：强制 `num_rejected=1`，图内仍
   按拒绝后的 gap/窗口修复。
9. `test_graph_forced_rejection_and_acceptance[0]`：强制全接受。

### 9.6 回归与 trace

```bash
PATH=/home/user/anaconda3/envs/vllm-omni-dev/bin:$PATH \
VLLM_WSL2_ENABLE_PIN_MEMORY=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m pytest tests/step80 -q
# 60 passed（80A 43 + 80B 8 + 80C 9）

PATH=... python -m pytest tests/step58 … tests/step79 tests/step80 -q
# 1028 passed, 149 warnings in 98.08s
```

CPU：

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest tests/step80 -q
# 6 passed, 53 skipped
```

小型 trace：K=2、B=2 的一次 decode `propose` 选中
`FULL num_tokens=6 / num_reqs=2`，`torch.profiler` 记录到 **2 次 `cudaGraphLaunch`**
（target 图 + 多模块草稿图各一次）；草稿段没有 Python `_generate_drafts`，也没有
`Tensor.item()`（由上面的异常测试钉住）。不跑吞吐矩阵。

包摘要：`minivllm/` 159 个 `.py`，sha256
`63767e3d1bdd228f4a59259ecd1b3e41376b71e70dd19f61ecfe00712bca8760`。

### 9.7 80C 的差异与边界

1. **不做 PIECEWISE 草稿图**：冻结源码只接整段 FULL；带 PIECEWISE 的请求按 §9.2
   降级，并记录 requested/actual。不是“选了 PIECEWISE 但什么都没捕获”。
2. **混合批仍走 eager**：`FULL_DECODE_ONLY` 只服务统一 decode；纯 prefill / 混合批
   `dispatch` 返回 NONE。直接 `FULL` 的混合桶虽能被源码 manager 捕获，但本机正常
   Torch V2 路径走不到；不把这条当生产能力。
3. **单卡、纯文本**：不接 DP/EPLB/`num_tokens_across_dp`/多模态；`inputs_embeds=None`。
4. **没有专用模型**：Gemma4/Step3p5/稀疏索引分别属 80E–80G；Q-only 共享契约已在
   §10（80D）落地，但真实 Gemma4 仍未接入。
5. **没有真实权重与性能结论**：tiny `Qwen3MTP` 是测试材料；不声称真实多模块 MTP
   的加速比。
6. **旧测试名保留、字段不保留**：`test_multi_module_mtp_full_graph_matches_eager`
   在当前回退后的树里原本不存在；80C 新增同名测试，按一个 manager 的冻结结构断言，
   不为了保留旧字段维持错误设计。

### 9.8 80C 改动文件

- `minivllm/worker/gpu/spec_decode/multi_module_mtp/speculator.py`：新增
  `init_cudagraph_manager()` 的模式收敛/manager 创建、`capture()`、
  `_dispatch_cudagraph()`、`_draft_attn_metadata_for_capture()`；`propose()` 增加
  FULL replay 路径与补齐缓冲；`_run_model()`/`_generate_drafts()` 增加
  `cudagraph_runtime_mode`。
- `minivllm/worker/gpu/spec_decode/__init__.py`：更新工厂注释（80B/80C）。
- `tests/step80/test_multi_module_graph_state.py`：新增 9 项。
- `tests/step80/test_multi_module_reprefill_alignment.py`：把旧的“图必须拒绝”断言
  改成模式收敛断言。
- `docs/step80_alignment.md`：本 §9；`docs/results.json` → `step80.subrequirements.80C`；
  `docs/README.md` / `minivllm/README.md` 同步 80C。

## 10. 80D：Q-only 共享 KV 的映射与只读契约

### 10.1 范围与冻结入口

80D 只做"Q-only 草稿层怎么正确读 target 的 KV"这一层协议：

- 参考冻结 `v1/worker/gpu/spec_decode/gemma4/speculator.py::_setup_gemma4_kv_sharing`
  的选源规则；
- 参考 `model_executor/models/gemma4_mtp.py::Gemma4MTPAttention` 的 `q_only` 语义：
  本层只算 Q，K/V 来自 target；
- 不实现 Gemma4 assistant 的模型结构/加载/真实接入（80E），也不把 tiny Qwen 层拼成
  "能服务的 Gemma4" 来冒充。

### 10.2 映射规则与固定例子

所有权与字段：

| 字段 | 来源/所有者 | 80D 的用法 |
|---|---|---|
| target `layer_types` | target 模型配置 | 建"类型 → 非共享层下标"表 |
| target `num_kv_shared_layers` | target 模型配置 | 尾部这么多层是消费者，不参与选源 |
| draft `layer_types` | assistant 配置 | 决定每个 draft 层选哪种类型 |
| `kv_sharing_target_layer_name` | speculator 写入、Runner 消费 | 形如 `<真实前缀>.<idx>.self_attn.attn` |
| 物理 KV | target 分配/写入，draft 只读 | draft 不分配、不写、不释放 |

规则（与冻结方法逐条一致）：

```text
num_non_shared = len(target_layer_types) - target_num_kv_shared_layers
候选 = target_layer_types[:num_non_shared] 中同类型的下标
draft 第 i 层 → 候选[-1]（同类型最后一个非共享源层）
draft 类型列表短了 → 默认 full_attention
没找到同类型候选 → 保留 kv_sharing_target_layer_name=None，告警
```

固定例子：target `[S,F,S,F,S]`、`num_kv_shared_layers=1`，尾部第 4 层是 S 类型的
**消费者**，只有前 4 层可以当源：

| draft 层 | 类型 | 同类型非共享候选 | 选中 |
|---|---|---|---|
| 0 | S | 0, 2 | **2** |
| 1 | F | 1, 3 | **3** |

所以结果是 `model.layers.2.self_attn.attn` / `model.layers.3.self_attn.attn`，不是
把所有 draft 层都指向第 4 层。真实层名前缀从 target 的注意力层名里取（例如
`language_model.model.layers.2.self_attn.attn`），不写死 `model.layers`。

`num_kv_shared_layers` 数的是 **target 内部复用别人 KV 的消费者层**，不是"被别人复用
的源层"；80D 只用它排除尾部层，不实现 target 内部的跨层共享执行。

### 10.3 只读契约与所有权

- `Attention.q_only=True` 时，`forward()` 只调后端的 `forward_q_only()`；后者
  直接 `_forward_from_cache()`，**不调用 `write_kv`**。后端没有 `forward_q_only`
  就明确 `NotImplementedError`，不退回普通 `forward`（那会把 dummy K/V 写进 target）。
- `Attention.bind_shared_kv_cache()` 只做检查与绑定：
  - KV heads、head_size 必须一致；
  - cache 形状必须是 `[2, blocks, block_size, kv_heads, head_size]` 且连续；
  - draft 浮点 dtype 必须与 target cache 一致；
  - **不要求 Q heads 相等**：GQA 下 draft 的 Q 头数可以与 target 不同；
  - 源层不能自己也是 `q_only`（禁止共享链），且必须真正持有传入的 cache 对象。
- Runner 的 `_allocate_draft_kv_caches()`：
  - 有 `kv_sharing_target_layer_name` → 从 `self.kv_caches` 取 target 的那份，验证
    block 数/block_size/层存在，然后 `bind_shared_kv_cache()`；**不新建 cache**；
  - `q_only=True` 但没有映射 → 抛出 `NotImplementedError`；**不静默分配第二份**；
  - 普通草稿层保持原有分配逻辑不变。
- 源 cache 的分配/写入/释放仍归 target；草稿层重建、slot 复用或请求结束时都不 free
  源 cache。

### 10.4 固定位置

Q-only 不追加 K/V，所以同一轮 K 个草稿步里 positions/seq_lens 必须保持不变。现有
`advance_draft_positions=False` 会强制融合多步；80D 的测试直接跑
`update_draft_inputs(..., advance_draft_positions=False)` 的 Triton 内核，确认它只改
`input_ids`/hidden 反馈，不改 positions/seq_lens。

### 10.5 验证

新增 `tests/step80/test_q_only_contract.py`，共 **26 passed**，覆盖：

1. 固定例子：`[S,F,S,F,S]` + `num_kv_shared_layers=1` → `[2,3]`；
2. 与已安装冻结 vLLM 的 `_setup_gemma4_kv_sharing` 逐例差分（5 组：正常、默认类型、
   缺类型、真实前缀、全部共享）；
3. 缺类型保留 `None` + 告警，Runner 对未接线 Q-only 层明确拒绝；
4. 源 KV 在 Q-only 前向（eager 与 FULL 读路径）前后逐字节一致；
5. 相同 Q/KV 与独立手写注意力数学逐值对照；
6. 无新增草稿 cache 分配（patch `torch.zeros` 为抛错，仍能绑定）；
7. KV heads/head_size/dtype/非连续布局/共享链的拒绝；
8. Q heads 不同但 KV 相同 → 允许；
9. 重新绑定/重建草稿层后源 cache 对象与内容不变；
10. `update_draft_inputs` 在 `advance_draft_positions=False` 下 positions/seq_lens 固定。

命令与结果：

```bash
PY=/home/user/anaconda3/envs/vllm-omni-dev/bin/python
PATH=/home/user/anaconda3/envs/vllm-omni-dev/bin:$PATH \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY -m pytest \
  tests/step80/test_q_only_contract.py -q      # 26 passed
PATH=... $PY -m pytest tests/step80 -q          # 86 passed（60 + 26）
CUDA_VISIBLE_DEVICES='' PATH=... $PY -m pytest \
  tests/step80/test_q_only_contract.py -q      # 25 passed, 1 skipped
```

全量回归结果（2026-10-25 本机）：**1054 passed, 149 warnings in 136.04s**（80C 后
1028 + 80D 26；无回归）。`tests/step80` 为 **86 passed**；CPU-only 跑本文件的
Q-only 测试为 **25 passed, 1 skipped**（Triton 位置用例明确 skip）。
包摘要：`minivllm/` **161** 个 `.py`，sha256
`65b6ec08f6fed84ddb607651d3fb47720ad6fd4ee9de853e304a28984410e56d`。

### 10.6 差异与边界

1. **只做 V2 机制**：`Attention`/Runner 的共享读路径是 V2 的；V1 的 Gemma4 proposer
   与 per-group metadata 属 80E，不在本关改。
2. **不实现真实 Gemma4**：没有 `Gemma4MTPAttention`/decoder/loader，也没有真实权重；
   工厂对 `model_type == "gemma4_mtp"` 明确拒绝并指向 80E，避免落到普通 mtp 分支。
3. **混合 SWA/full 不执行**：本关只验证选源（例如 `[S,F,S,F,S]`），不同窗口/不同
   KV group 的真实执行依赖 81 的多 group/backend；不能把"映射对"说成"混合执行完成"。
4. **非法共享层数明确报错**：冻结方法对 `num_kv_shared_layers > len(layer_types)` 会
   走 Python 负切片；本仓库在纯映射入口直接 `ValueError`，不静默少看层。合法配置
   行为与冻结方法一致。
5. **dtype 严格相等**：Q-only 读取按同一 cache dtype 检查；这是冻结路径的同 dtype
   前提。没有用"读到 float32 再转"来掩盖不一致。
6. **未做性能测量**：Q-only 的收益是省一份草稿 KV，不在本关给加速比。

### 10.7 全量回归与改动文件

- 全量回归：**1054 passed, 149 warnings in 136.04s**；`tests/step80` **86 passed**；
  CPU-only Q-only 测试 **25 passed, 1 skipped**；包摘要 161 个 `.py` /
  `65b6ec08f6fed84ddb607651d3fb47720ad6fd4ee9de853e304a28984410e56d`。
- 改动文件：
  - `minivllm/attention/layer.py`：`q_only`/`kv_sharing_target_layer_name`、
    `bind_shared_kv_cache()`、`forward()` 的 Q-only 分派。
  - `minivllm/attention/backends/torch_sdpa.py`：`forward_q_only()` 与
    `_forward_from_cache()`（普通 forward 行为不变）。
  - `minivllm/worker/gpu/model_runner.py`：`_allocate_draft_kv_caches()` 的共享绑定、
    只读检查、未接线拒绝、禁止二次分配。
  - `minivllm/worker/gpu/spec_decode/gemma4/kv_sharing.py`（新增）：纯映射与
    `setup_gemma4_kv_sharing()`。
  - `minivllm/worker/gpu/spec_decode/gemma4/__init__.py`（新增）。
  - `minivllm/config.py`：`use_gemma4_mtp()` 判据。
  - `minivllm/worker/gpu/spec_decode/__init__.py`：Gemma4 分支明确拒绝并指向 80E。
  - `tests/step80/test_q_only_contract.py`（新增，26 项）。
  - `docs/step80_alignment.md`、`docs/results.json`、`docs/README.md`、
    `minivllm/README.md`。
