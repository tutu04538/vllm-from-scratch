# 第 80 关对齐记录（80A：多模块输入布局与跨轮缓存）

> 本文件按子需求追加。当前只完成 **80A**；80B 的 Eager 执行/拒绝后 KV 修复、80C 的
> Graph 与采样生命周期、80D–80H 均未开始。不要把本文件里的“输入协议完成”理解为
> 多模块 MTP 已经能端到端跑起来。

## 0. 需求与边界

- 需求：`投机解码完整需求/080A_多模块输入布局与跨轮缓存.md`。
- 本关只做**输入准备和跨轮缓存布局**：把 target 本轮输入整理成 draft 的 packed 行，
  把窗口尾部 token/hidden/embedding 按持久 slot 缓存，定义窗口移动与尾部 padding。
- 明确不做：draft 模型前向、目标拒绝后重算后续模块 KV、Graph、专用模型、性能测试。
- 生产工厂 `init_speculator()` 仍然对 `use_multi_module_mtp()` 明确抛
  `NotImplementedError`；本关的函数没有被生产路径调用，80B 才接线。

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

## 6. 已知边界与差异

1. **没有生产接线**：`MultiModuleMTPSpeculator` 类、`propose()`、拒绝后的模块 KV
   修复都在 80B；当前工厂仍明确拒绝多模块 MTP。
2. **没有模型/Graph/性能**：本关不加载 target/draft 权重，不做 CUDA Graph，不做
   性能测试（需求明确“无需性能测试”）。
3. **没有真实多模态模型**：`cached_draft_input_embeds` 只用人工 tensor 做差分，
   不声称支持多模态 MTP。
4. **`next_prefill_tokens` 行数**：本关的包装函数显式断言至少
   `num_speculative_steps` 行；上游 multi-module runner 正是按 K 设置
   `num_prefill_lookahead`。这是显式契约，不是改变算法。
5. **CPU**：Triton 输入 kernel 只在 CUDA 上跑；纯 CPU 只覆盖
   `K-1` 宽度协议。完整 CPU 等价实现不在本关范围。
6. **未对齐项**：本关没有改 `RequestState` / `BlockTables` / attention；这些是已有
   组件。80B 接线时需要删掉旧实验实现里重复的 `_snapshot_cached_inputs()` /
   Python 循环 pad（当前 HEAD 上这些旧代码已回退，不在本关新增）。

## 7. 本轮改动文件

- 新增 `minivllm/worker/gpu/spec_decode/multi_module_mtp/__init__.py`
- 新增 `minivllm/worker/gpu/spec_decode/multi_module_mtp/speculator.py`
- 新增 `tests/step80/test_multi_module_inputs_alignment.py`
- 更新 `docs/results.json` → `step80.subrequirements.80A`
- 新增本文件
