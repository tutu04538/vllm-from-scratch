# 第 80 关对齐记录（80A / 80B）

> 本文件按子需求追加。当前完成 **80A 输入层**与 **80B eager 执行/拒绝后 KV 修复**；
> 80C 的 Graph 与采样生命周期、80D–80H 均未开始。80B 只支持单卡、纯文本、eager；
> `init_cudagraph_manager()` 对非 NONE 模式明确拒绝。

## 0. 需求与边界

- 需求：
  - `投机解码完整需求/080A_多模块输入布局与跨轮缓存.md`
  - `投机解码完整需求/080B_多模块Eager执行与拒绝后KV修复.md`
- 80A 做**输入准备和跨轮缓存布局**：把 target 本轮输入整理成 draft 的 packed 行，
  把窗口尾部 token/hidden/embedding 按持久 slot 缓存，定义窗口移动与尾部 padding。
- 80B 把 80A 接成 V2 的 `MultiModuleMTPSpeculator`：显式 `propose()`、整段 packed
  query 的模块循环、按 `spec_step_idx` 选模块、拒绝后 gap 在当前 query 窗口内重算。
- 明确不做：Gemma4、Step3p5、Q-only 共享 KV、稀疏索引、CUDA Graph、真实 MTP 权重、
  性能测试。生产工厂现在会把 `use_multi_module_mtp()` 分派到 80B 的类；其它专用模型
  分支仍保持明确拒绝。

## 1. 路径映射

| 本项目 | 冻结上游（vLLM 0.28.0 / 2026-10-03） |
|---|---|
| `minivllm/worker/gpu/spec_decode/multi_module_mtp/speculator.py` | `v1/worker/gpu/spec_decode/multi_module_mtp/speculator.py` 的输入层 |
| `minivllm/worker/gpu/input_batch.py::InputBuffers` | `v1/worker/gpu/input_batch.py::InputBuffers` |
| `minivllm/worker/gpu/input_batch.py::InputBatch` | 同上 `InputBatch` |
| `minivllm/worker/gpu/block_table.py::PAD_SLOT_ID` | `v1/attention/backends/utils.py::PAD_SLOT_ID` |

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

新增 `tests/step80/test_multi_module_reprefill_alignment.py`，**8 passed**。加上 80A
的 43 项，`tests/step80` 共 **51 passed**。

- **调用链/边界**（纯 CPU + CUDA）：工厂返回 `MultiModuleMTPSpeculator`；K=2/4、
  模块数=2；每轮 `spec_step_idx` 为 `0..K-1`；`init_cudagraph_manager(FULL)` 与
  `capture()` 都明确抛 80C 的 `NotImplementedError`。
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

全量回归：

```bash
PATH=/home/user/anaconda3/envs/vllm-omni-dev/bin:$PATH \
VLLM_WSL2_ENABLE_PIN_MEMORY=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m pytest tests/step58 … tests/step79 tests/step80 -q
# 1019 passed, 149 warnings in 94.38s
```

包摘要：`minivllm/` 159 个 `.py`，sha256
`b05e374d48251904c0c430ad3a7f55cdc8dd98ab334e5ee4b0afcff711ae58ec`。

### 8.4 80B 的差异与边界

1. **只支持 eager**：`init_cudagraph_manager` 对 `FULL`/`FULL_AND_PIECEWISE` 等非
   NONE 模式明确拒绝；图捕获、q/RNG/slot 归属属 80C。
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
