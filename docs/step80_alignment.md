# 第 80 关对齐记录（80A / 80B / 80C / 80D / 80E.E1 / 80E.E2）

> 本文件按子需求追加。当前完成 **80A 输入层**、**80B eager 执行/拒绝后 KV 修复**、
> **80C 多模块整段 CUDA Graph/采样生命周期**、**80D Q-only 共享 KV 的映射与只读
> 契约**、**80E.E1 Gemma4 assistant 模型层** 与 **80E.E2 的 target 架构/多 group/
> V1/V2 接线**；80F–80H 未开始。
>
> **E2 已接真实权重**：本机从 HF 镜像下载并固定
> `google/gemma-4-E2B-it`（target，revision
> `3e22461f65e89153144f8adb70e3b8c2cc9845a7`，`model.safetensors` sha256
> `2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550`）和
> `google/gemma-4-E2B-it-assistant`（revision
> `2d874ef7d29f9a30599a1e4b3c1cbc9595f005df`，sha256
> `93682eb1c97639d18f007704dc880bd74cbe530adaf7b1bb561213863fdad2a6`）。
> target 文本 4.63B 参数（文件 10.25GB，含 vision/audio）；实测 target-only eager 与
> target+assistant 投机 graph 在 float32 下 8-token greedy 逐 token 相同。
> 多 group/SWA/full/head_dim 的 81 子集已独立通过（`tests/step81/`），Mamba/
> 抢占矩阵等 81 其余项未完成。

## 0. 需求与边界

- 需求：
  - `投机解码完整需求/080A_多模块输入布局与跨轮缓存.md`
  - `投机解码完整需求/080B_多模块Eager执行与拒绝后KV修复.md`
  - `投机解码完整需求/080C_多模块Graph与采样生命周期.md`
  - `投机解码完整需求/080D_Q_only共享KV的映射与只读契约.md`
  - `投机解码完整需求/080E_Gemma4Assistant结构与真实模型接入.md`（E1/E2）
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
- 80E.E1 做 **Gemma4 assistant 的模型层**：`Gemma4MTPAttention`（Q-only、q_norm、
  sliding/full 与 proportional RoPE）、`Gemma4MTPDecoderLayer`、
  `Gemma4MultiTokenPredictor`（`2*backbone → assistant` 与 `assistant → backbone`）、
  `Gemma4MTP`（lm_head、softcap、可选 centroid masked embedder）与权重路由。
  用冻结上游模型同 tiny 配置/权重做逐层 hidden、两个返回值与 logits 差分。
- 80E.E2：真实 Gemma4 target 架构/加载 + 多 KV group/SWA/full/backend；V2
  `Gemma4Speculator` 与 V1 `Gemma4Proposer` 均已接。真实 target/assistant 权重已下载，
  float32 下 8-token greedy 的 target-only eager == 投机 graph。
- 明确不做：Step3p5 模型、Mamba/抢占生命周期、稀疏索引、性能矩阵。

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
| `minivllm/models/gemma4_mtp.py` | `model_executor/models/gemma4_mtp.py`（E1 模型层） |
| `minivllm/layers/rotary_embedding.py` 的 `proportional`/`key=None` | `layers/rotary_embedding/gemma4_rope.py` + `base.py::forward_static` |

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

## 11. 80E.E1：Gemma4 assistant 模型层差分

### 11.1 范围

80E 分两段：E1 只做 **assistant 模型结构 + 权重加载**，E2 才接真实 target 与提议者。
本关不把完整 Gemma4 target 偷塞进来，也不下载/伪造真实权重。

E1 的冻结参考：

- `model_executor/models/gemma4_mtp.py` 的 `Gemma4MTPMaskedEmbedder`、
  `Gemma4MTPAttention`、`Gemma4MTPDecoderLayer`、`Gemma4MultiTokenPredictor`、
  `Gemma4MTP`；
- `model_executor/models/gemma4.py::Gemma4MLP`；
- `v1/worker/gpu/spec_decode/gemma4/speculator.py` 的 embedding 共享/归一化规则。

### 11.2 固定数据流：backbone=16、assistant=8

| 阶段 | 形状 | 说明 |
|---|---|---|
| token embedding | `[T, 16]` | 共享 target 的 backbone 宽 embedding，乘 `sqrt(16)` |
| target feedback hidden | `[T, 16]` | 上一轮 target/assistant 交回的 backbone hidden |
| concat | `[T, 32]` | `[embedding ‖ hidden]`，顺序不能反 |
| `pre_projection` | `[T, 8]` | `ColumnParallelLinear(32, 8, bias=False)` |
| decoder × N | `[T, 8]` | Q-only attention + Gemma4MLP + 四个 RMSNorm + `layer_scalar` |
| `draft_hidden` | `[T, 8]` | 过 final norm，给 `lm_head` 出 logits |
| `post_projection` | `[T, 16]` | 下一步反馈必须用它，不能把 8 宽的 draft hidden 喂回去 |

模型入口对两条宽度做显式检查：反馈 hidden 不是 16、或 embedding 不是 16，都直接
`ValueError`，不会让 `pre_projection` 去解释一个 24 宽的拼接。

### 11.3 关键结构

- `Gemma4MTPAttention`：只有 `q_proj`/`o_proj`/`q_norm`，没有 K/V 投影；`scaling=1.0`；
  per-head q_norm；RoPE 按层类型选 sliding/full，full 层用 proportional（频率分母是
  `head_dim`，未旋转维零填充），`key=None` 合法。
- `Gemma4MTPDecoderLayer`：显式 residual 相加 + 四个 RMSNorm + `layer_scalar`。
  这里**不能**复用 Qwen3 的 `RMSNorm(x, residual)` 融合顺序。
- `Gemma4MultiTokenPredictor`：`embed_tokens`、`pre_projection`、全部 assistant 层、
  `norm`、`post_projection`、`normalizer=sqrt(backbone)`。
- `Gemma4MTP`：draft `lm_head`（`tie_word_embeddings=True` 时与 embed 共享 Parameter）、
  `LogitsProcessor`（final softcap）、可选 `Gemma4MTPMaskedEmbedder`。
- `Gemma4MTPMaskedEmbedder`：centroid top-k 选簇 + 只对选中列做点积；未选列填
  `finfo.min`；`get_top_tokens` 不物化完整 `[T, V]`。

### 11.4 权重路由

检查点布局与模型的差异：

| 检查点 | 模型 | 处理 |
|---|---|---|
| `pre_projection.*` / `post_projection.*` | `model.pre_projection.*` / `model.post_projection.*` | 前缀加 `model.` |
| `*.mlp.gate_proj.*` / `*.mlp.up_proj.*` | `*.mlp.gate_up_proj.*` | 打包映射，shard 0/1 |
| `*.layer_scalar` | 同名 buffer | 直接拷贝 buffer（不是 Parameter） |
| `masked_embedding.token_ordering` | 同名 buffer | 直接拷贝；masked 模式缺它直接报错 |
| `lm_head.weight`（tied） | 与 `model.embed_tokens.weight` 共享 | 跳过，覆盖检查仍通过 |
| 其余未知名字 | — | `AutoWeightsLoader` 明确报错 |

`use_ordered_embeddings=True` 时，`token_ordering` 不再是"可选 buffer"：缺了以后 centroid
选簇会越界，所以加载阶段就拒绝。

### 11.5 验证

新增 `tests/step80/test_gemma4_model_alignment.py`，共 **12 passed**：

1. 参数形状 / 无 K/V 投影 / `scaling=1.0` / full 层异构 head_dim；
2. 权重路由：pre/post 前缀、gate/up 打包、norm/q_norm、`layer_scalar`、tied lm_head；
3. 缺参数/未知参数/token_ordering 缺失均报错；
4. `get_model()` 通过 registry 加载 `Gemma4MTPModel`；
5. Q 投影 + q_norm + RoPE（sliding 与 full/proportional 两层）与冻结模块逐值一致；
6. 全模型逐层 hidden、draft/backbone 两个返回值、logits 与冻结模型一致；
7. backbone/draft 宽度不可互换；
8. masked embedder 的稀疏 logits / 有限列 / argmax 与冻结模块一致；
9. final logit softcap 与手算一致；
10. 未适配的 RoPE 类型、centroid 配置、MoE/double-wide MLP、attention softcap 明确报错。

```bash
PY=/home/user/anaconda3/envs/vllm-omni-dev/bin/python
PATH=/home/user/anaconda3/envs/vllm-omni-dev/bin:$PATH \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY -m pytest \
  tests/step80/test_gemma4_model_alignment.py -q   # 12 passed
PATH=... $PY -m pytest tests/step80 -q              # 98 passed（60 + 26 + 12）
CUDA_VISIBLE_DEVICES='' ...                          # 12 passed（CPU-only）
```

全量回归结果与包摘要见 §11.7。

### 11.6 差异与边界

1. **E1 不接真实 target**：没有 Gemma4 target 架构/权重，也没有 V2/V1 提议者接线；
   `init_speculator()` 仍对 `gemma4_mtp` 明确拒绝。E2 需要真实 target + 81 的
   KV group/SWA/full/异构 head_dim/backend。
2. **attention core 在模型差分中被 identity 替换**：E1 比较的是 assistant 的胶水、q 投影/
   q_norm/RoPE、层结构、宽度与 logits；Q-only 的真实 cache 读取已由 80D 契约覆盖。
   E2 才把真实 target cache/attention group 接进来。
3. **未实现 attention logit softcap**：配置出现 `attn_logit_softcapping` 时明确
   `NotImplementedError`，不静默忽略；`final_logit_softcapping` 已实现并测试。
4. **只覆盖稠密 MLP**：`enable_moe_block` / `use_double_wide_mlp` / per-layer input
   embedding 明确拒绝；选中的 assistant 检查点必须走稠密结构。
5. **真实 Gemma4 权重不在本机**：E1 的 tiny 配置/权重是现场生成的教学材料，不是
   checkpoint 格式验证；真实权重加载覆盖率属 E2。
6. **V1 未改**：本关是模型层，V1 Gemma4 proposer/per-group metadata 留到 E2。

### 11.7 全量回归与改动文件

- 全量回归（2026-10-25 本机）：**1066 passed, 149 warnings in 96.87s**
  （80D 后 1054 + 80E.E1 12；无回归）；`tests/step80` **98 passed**；
  E1 CPU-only **12 passed**。
- 包摘要：`minivllm/` **162** 个 `.py`，sha256
  `f66af6fafc83941adb2fa88a53a8f6621041f98c61e07e658ebc8970f6d38108`。
- 改动文件：
  - `minivllm/models/gemma4_mtp.py`（新增）：E1 模型、masked embedder、权重路由。
  - `minivllm/models/__init__.py`、`minivllm/models/registry.py`：注册/导出。
  - `minivllm/layers/rotary_embedding.py`：`proportional` RoPE、`key=None`、
    `apply_to_query`；普通路径行为不变。
  - `tests/step80/test_gemma4_model_alignment.py`（新增，12 项）。
  - `docs/step80_alignment.md`、`docs/results.json`、`docs/README.md`、
    `minivllm/README.md`。

## 12. 80E.E2：Gemma4 target 架构、多 KV group 与 V1/V2 接线

### 12.1 范围与证据边界

E2 做四件事：

1. **Gemma4 文本 target**：`Gemma4ForCausalLM`（注册名同时接受
   `Gemma4ForConditionalGeneration` 的包装命名），包含 YOCO 跨层 KV 共享、PLE、
   double-wide MLP、sliding/full 异构 head_dim、proportional RoPE、final softcap。
2. **多 KV group 控制面/执行面**：真实 `KVCacheSpec`/`KVCacheGroup`/`KVCacheConfig`，
   `FullAttentionManager` + `SlidingWindowManager`，组间共同前缀命中边界。
3. **V2 Gemma4Speculator**：常量位置、真实 assistant 加载、embedding 共享、
   Q-only 源层绑定、逐组 draft metadata、centroid greedy。
4. **V1 Gemma4Proposer**：独立接入；逐组 target 块表工作区；Q-only 的常量位置
   AR 步；V1 多 group 图明确拒绝（先走 eager）。

**不冒充的部分**：

- **真实 target 与 assistant 都已下载并固定 revision**：
  - target 文件 10.25GB（文本 4.63B 参数 + vision/audio），本地
    `models/gemma-4-E2B-it`，revision/sha 见文件头；
  - assistant 77.7M，本地 `models/gemma-4-E2B-it-assistant`，sha 见文件头。
  - 真实 target 测试检查 35 层、12 sliding + 3 full owner、20 个 Q-only 尾层、
    assistant 3 层读 target 13、1 层读 target 14；target-only eager 与投机 graph
    8-token greedy 相同。
- 测试仍保留 tiny/full-width 合成 target：tiny 用于跨 block/window 的 HF 逐值差分，
  full-width 用于不依赖大文件下载时也能覆盖真实 assistant 的加载/组映射。下载权重
  不提交（`models/` gitignore），没有权重的机器该用例显式 skip。
- 81 只完成 E2 所需子集；Mamba 混合状态回滚、抢占三时刻矩阵、KV transfer/PD、
  多模态 M-RoPE 不在 E2 声明内。

### 12.2 路径映射

| 本项目 | 冻结上游（vLLM 0.28.0 / 2026-10-03） |
|---|---|
| `minivllm/models/gemma4.py` | `model_executor/models/gemma4.py` 的文本子集 |
| `minivllm/core/kv_cache_interface.py` | `v1/kv_cache_interface.py` 的 spec/group/config 子集 |
| `minivllm/core/kv_cache_coordinator.py::HybridKVCacheCoordinator` | `v1/core/kv_cache_coordinator.py::HybridKVCacheCoordinator` 的共同命中边界 |
| `minivllm/core/single_type_kv_cache_manager.py::SlidingWindowManager` | 同名 manager 的保留策略子集 |
| `minivllm/worker/gpu/spec_decode/gemma4/speculator.py::Gemma4Speculator` | `v1/worker/gpu/spec_decode/gemma4/speculator.py` |
| `minivllm/spec_decode/gemma4.py::Gemma4Proposer` | `v1/spec_decode/gemma4.py` |
| `minivllm/worker/gpu/model_runner.py` 的 `get_kv_cache_spec` / 多组 metadata | V2 `GPUModelRunner` 的 `init_attn_backend` / `prepare_attn` |
| `minivllm/worker/gpu_model_runner.py` 的 V1 多组块表/缓存 | V1 `GPUModelRunner` 的 `initialize_kv_cache` / `_build_attn_metadata` |

### 12.3 target 的固定结构

以真实 assistant 配置（`backbone=1536`、assistant hidden=256、4 层
`[S,S,S,F]`）配本机生成的 full-width target（4 层 `[S,F,S,F]`、2 个共享尾层）为例：

| 项 | sliding 组 | full 组 |
|---|---|---|
| 拥有 cache 的 target 层 | `model.layers.0` | `model.layers.1` |
| Q-only target 层 | `layers.2`（读 0） | `layers.3`（读 1） |
| head_dim / KV heads | 256 / 1 | 512 / 1 |
| window | 512 | 无 |
| assistant 层 | 0,1,2（读 target 0） | 3（读 target 1） |

真实的 35 层 target 会得到 13 个 sliding 拥有层 + 2 个 full 拥有层、20 个 Q-only
尾部层；同一套映射代码按 `layer_types`/`num_kv_shared_layers` 计算。

target 的关键执行顺序：

1. 前 15 层正常写 K/V：sliding 组写 `[2, blocks, 16, 1, 256]`，full 组写
   `[2, blocks, 16, 1, 512]`；
2. 后 20 层 `q_only=True`，只算 Q → q_norm → RoPE，读对应组源层的 paged cache；
3. PLE 每层额外注入 `gate(hidden)` × per-layer embedding，再接
   `per_layer_projection`/`post_per_layer_input_norm`；
4. 共享层 MLP 宽度翻倍（`use_double_wide_mlp`），`layer_scalar` 最后乘；
5. final norm 后走 tied `lm_head` + `final_logit_softcapping=30`。

### 12.4 assistant 的固定数据流（真实宽度）

| 阶段 | 形状 | 来源 |
|---|---|---|
| target embedding | `[T, 1536]` | 真实 `model.embed_tokens`（被 assistant 直接复用） |
| target feedback hidden | `[T, 1536]` | target 最后一层 hidden / 上一步 `post_projection` |
| concat | `[T, 3072]` | `[embedding ‖ hidden]` |
| `pre_projection` | `[T, 256]` | 检查点顶层 `pre_projection.weight` |
| assistant decoder ×4 | `[T, 256]` | Q-only + 稠密 MLP |
| `draft_hidden` | `[T, 256]` | 过 final norm，喂 centroid/lm_head |
| `lm_head` | `[T, 262144]` | assistant 自己的 256 宽 head（**不**共享 target head） |
| `post_projection` | `[T, 1536]` | 下一步反馈；不能把 256 宽 draft hidden 喂回 |

assistant 配置在 `SpeculativeConfig` 里归一化：`gemma4_assistant` /
`gemma4_unified_assistant` → `model_type=gemma4_mtp`、
`architectures=[Gemma4MTPModel]`、`n_predict=1`、嵌套
`text_config.num_kv_shared_layers=0`。不归一化会让 assistant 自己的 4 层又去做
跨层共享，读错源。

### 12.5 V1/V2 分工

| 路径 | 输入 | 元数据 | 图 |
|---|---|---|---|
| V2 | `InputBuffers` packed 行；常量位置 | 每 KV group 一张表；draft 层用 `draft::` 别名映射到源组 | target/草稿 prefill 图 + 融合多步 decode 图（`FULL_DECODE_ONLY` 实测） |
| V1 | CPU 行块；`Gemma4Proposer` 独立循环 | 每组一张工作区块表；Q-only AR 步位置/seq_len 固定 | 多 group 图明确 `NotImplementedError`，E2 先 eager |

V1 的 `_set_autoregressive_inputs` 不再 `anchor + k`：Q-only 每步都读 target 同一段
上下文，位置必须停在采样锚点；seq_lens 停在 `plan.seq_lens - num_rejected`。V2 用
`advance_draft_positions=False` + 融合多步图达到同一语义。

### 12.6 验证

新增 `tests/step80/test_gemma4_real_integration.py`，共 **12 passed**：

1. 真实 assistant 配置归一化，且不原地改用户对象；
2. tiny target 跨 sliding window/block 边界的 logits 与 HF **逐值**比较：
   `max diff 2.98e-7`，argmax/top-5 相同；
3. target 两个 group 的 spec（head_dim 256/512、window 512/None）与 Q-only 源层；
4. 真实 assistant（77.7M）配 full-width 合成 target（457M）：
   `pre_projection 3072→256`、`post_projection 256→1536`、
   `lm_head 262144×256`、centroid 结构、embedding 共享对象同一、
   assistant 三层读 sliding/一层读 full、`propose()` 前后 target cache 逐字节不变；
5. V1 与 V2 同 tiny 模型/同 greedy workload 输出一致；
6. V2 eager 与 `FULL_DECODE_ONLY` 图输出一致，prefill/decode/target 图都真实捕获；
7. V1 多 group 图明确拒绝；
8. 真实 target 包装命名 `model.language_model.*` 可加载文本栈；
9. request id 完成后复用，不继承旧 Gemma4 speculative 状态；
10. Gemma4 + 非 Torch 后端明确拒绝（Q-only 只在 Torch 后端实现，不能静默换后端）；
11. V1 真实 assistant 的 centroid greedy 小图真实捕获（1/2/4/.../64 档），并跑通生成；
12. **真实 target + 真实 assistant**：35 层/2 组/20 个 Q-only 尾层、assistant 源层
    13/14，float32 下 8-token target-only eager == 投机 `FULL_DECODE_ONLY` graph
    （`[107, 1, 106, 1, 106, 106, 106, 107]`），target/prefill/decode 图都真实重放。

另外 `tests/step81/test_spec_kv_groups.py` 4 项覆盖：

- 组块大小 4/8、hash 粒度 min、分配/命中块数逐组正确；
- 只删 sliding 组第一块 hash → 共同命中 0（不允许 full 组单独误命中）；
- 容量不足时原子失败，两张组块表都不留半成品；
- `SlidingWindowSpec` 缺 window 明确报错。

```bash
PY=/home/user/anaconda3/envs/vllm-omni-dev/bin/python
PATH=/home/user/anaconda3/envs/vllm-omni-dev/bin:$PATH \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY -m pytest \
  tests/step80/test_gemma4_real_integration.py -q     # 12 passed
PATH=... $PY -m pytest tests/step81/test_spec_kv_groups.py -q  # 4 passed
PATH=... $PY -m pytest tests/step80 -q                      # 110 passed
PATH=... $PY -m pytest tests/step58 … tests/step81 -q       # 1082 passed, 172.33s
```

### 12.7 差异与边界

1. **真实权重已验，但 bf16 有 near-tie**：float32 下 graph/eager 8-token 完全相同；
   bf16 真实权重在某个位置两个 token 的 logits 完全相等，graph/eager 的微小规约差
   会把 greedy 翻一枚（106 vs 107）。这是 bf16 数值 tie，不是组/位置/缓存逻辑错；
   生产若要"逐 token 与 target-only 完全一致"，需固定 precision/比较口径，不能拿
   bf16 的 near-tie 当逻辑通过。
2. **滑窗不回收窗口外块**：`SlidingWindowManager` 保留全历史、attention 按窗口掩码；
   正确但不省显存。上游 `remove_skipped_blocks` + null block 属 81 后续。
3. **不同 group block_size 只到控制面**：manager/coordinator/块表支持 4+8 的分配与
   命中；Runner 执行层仍要求所有组 block_size 相同（Gemma4 正好相同）。不同块大小的
   逐组 attention kernel 编译未接。
4. **V1 图未按组捕获**：多 group 时 V1 `cudagraph_mode != NONE` 明确拒绝；V2 已支持。
5. **V1 的 PLE/多模态**：V1 Gemma4 只走文本 target；多模态 inputs_embeds/M-RoPE 未接。
6. **centroid CUDA Graph**：V2 greedy 用 `get_top_tokens`（不物化完整词表），但没有像
   上游 `_setup_centroids_cuda_graphs` 那样为 1/2/4/... 捕获 centroid 小图；这是性能
   项，不是正确性项。V1 仍走完整 logits 的 argmax。
7. **未跑性能矩阵**：不声称任何加速比或真实草稿接受率。
