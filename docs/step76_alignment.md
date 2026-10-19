# 第七十六关：DFlash 上下文 KV 与并行 Query 执行

> 需求：[`076_DFlash上下文KV与并行Query执行.md`](../../vllm-omni/learning_notes/14_vllm_from_scratch/投机解码完整需求/076_DFlash上下文KV与并行Query执行.md)
> 基线：本机 `vllm==0.28.0` 的源码快照（`v1/spec_decode/dflash.py`、
> `v1/worker/gpu/spec_decode/dflash/{speculator,cudagraph}.py`、`model_executor/models/qwen3_dflash.py`）
> 状态：V1 与 V2 两条路径都已实现并自检；⚠️ **本机没有真实 DFlash 权重**，草稿质量与加速比未验
> （本关按要求只做 context/query 的 trace 验证，不把"少调 target"写成"更快"）。

## 0. 一句话

DFlash 的草稿模型**不读 token 序列**：target 每步把若干辅助层的 hidden states 交出来，草稿用
`fc` 把它们投到自己的宽度，再**一次性**翻译成所有层的 K/V 写进自己的缓存（`precompute_and_store_context_kv`）；
真正过草稿模型的只有每请求 `1 + K` 行 query（1 个 bonus anchor + K 个 mask），一次前向出 K 枚草稿。
它的注意力**默认非因果**——mask 行是"槽位"，同块互相可见。

本关就是把这条"异构批 + 翻译式 KV"的路径在 V1（`DFlashProposer`）与 V2（`DFlashSpeculator`）
里各实现一遍，并用**逐值差分**钉住行布局、槽位、采样映射与缓存写入。

---

## 1. 为什么需要这一关（痛点）

1. **上下文不该由草稿再读一遍**：PARD/EAGLE3 的草稿 KV 是草稿自己跑 token 得到的（每层完整前向）；
   DFlash 只让上下文经过**一次跨层融合的 K/V 投影**（跳过 Q/O、MLP 与注意力本身）。长 prompt 下
   这是数量级差别（对 PARD 尤其明显，对只有 1 层的 EAGLE3 head 小一些）。
2. **批布局不再是"一行一个 token"**：`context 行`（只贡献 KV、不出 logits）+ `query 行`
   （`1 + K`，只有 mask 行被采样）+ 被拒尾部（不写 KV）混在一起。任何一行错位都**不报错**，
   只是草稿质量静默崩掉。
3. **注意力形状不同**：DFlash 的 query 层可以是**非因果**的；把它当普通因果 LM 跑，输出是另一个
   模型的输出。

---

## 2. 数据流

### 2.1 V1（`minivllm/spec_decode/dflash.py`）

```
target 本轮（1 + K 行/请求）
   │  token/positions/hidden(aux)  ─────────────────────────────┐
   ▼                                                           │
expand_dflash_inputs()  ← 上游 copy_and_expand_dflash_inputs_kernel 的逐行等价
   ├─ context 行：位置 = target 的位置（含被拒尾部），槽位 = 块表查找
   └─ query 行  ：[bonus, mask×K]，位置 = last_valid_pos+1+off，槽位 = 块表查找
                                                               │
propose() ── set_inputs_first_pass()（写 staging、算槽位、筛采样行）◄─┘
   │
   ├─ _forward()
   │    ├─ precompute_and_store_context_kv(hidden, ctx_pos, ctx_slots)   ← 图外
   │    │     hidden_norm → 融合 GEMM(L 层) → 分组 K-norm → RoPE → do_kv_cache_update
   │    └─ 模型前向：只喂 1+K 行 query（非因果 metadata）
   │
   └─ parallel_drafting 早退分支：在 K 个 mask 行上各采一枚 → 直接返回
```

### 2.2 V2（`minivllm/worker/gpu/spec_decode/dflash/speculator.py`）

```
propose(input_batch, attn_metadata, slot_mappings, hidden, aux_hidden, ...)
   │
   ├─ combine_hidden_states(cat(aux)) → hidden[:num_target_tokens]   （与 target 同源）
   ├─ prepare_dflash_inputs()  ← 上游 _prepare_dflash_inputs_kernel（Triton，12 个输出）
   │     query 行写入常驻输入缓冲；context 位置/槽位写入独立缓冲；
   │     采样映射 sample_indices/sample_pos/sample_idx_mapping；补齐行映射 −1
   ├─ precompute_and_store_context_kv(...)                           ← 图外（形状每步变）
   ├─ query 图分派：FULL → run_fullgraph(desc)；否则 eager _generate_draft(...)
   └─ 返回 draft_tokens[:num_reqs]
```

**为什么上下文预计算在图外**（需求 076 §3.3）：它的行数 = 本轮 target 处理了多少 token，每步都变，
录不进固定形状的图。这**不是**可以从"少一张图"换来的简化——上游同样把这一步留在图外。

---

## 3. 输入布局（本关最核心的表）

以 B=2、K=3、请求 0 的 target 块 4 行（位置 7..10，被拒 2 行）、请求 1 的 target 块 2 行
（位置 20..21）为例（测试 `tests/step76/spec76_helpers.py::layout_case` 就是这组数字）：

| 输出 | 请求 0 | 请求 1 | 说明 |
|---|---|---|---|
| `context_positions`（V1） | `7 8 9 10` | `20 21` | 上游 V1 内核按 `is_ctx` 整段写位置 |
| `context_slot_mapping`（V1） | `7 8 9 10` | `84 85` | 块表查表（请求 1 的块从 16 开始，位置 20 → 块 21 → 槽位 84） |
| `context_positions`（V2） | `7 8 0 0` | `20 21` | V2 内核把被拒尾部写 0 |
| `context_slot_mapping`（V2） | `7 8 −1 −1` | `84 85` | 被拒尾部**不写草稿缓存** |
| `query_positions` | `9 10 11 12` | `22 23 24 25` | `last_valid_pos + 1 + off` |
| `input_ids` | `11 m m m` | `22 m m m` | 第 0 行 bonus、其余 mask（`m = mask_token_id`） |
| `sample_indices`（全局行号） | `1 2 3` | `5 6 7` | 只采 mask 行 |
| `sample_idx_mapping`（V2，slot） | `0 0 0` | `5 5 5` | 指向请求的常驻 slot |
| `seq_lens` | `13` | `26` | `last_valid_pos + 1 + (1+K)` |

补齐（`max_num_reqs=8`、`max_num_tokens=64`）：`query_start_loc = [0,4,8,8,…,8]`、
`seq_lens[2:] = 0`、`sample_idx_mapping[6:] = −1`、`query_slot_mapping[8:] = −1`。
**补齐采样行映射写 0 是致命的**：图重放会把补齐的草稿写进 0 号请求的槽位（上游注释点名的坑）。

---

## 4. 实现对照

| 本项目 | 上游 | 状态 |
|---|---|---|
| `minivllm/spec_decode/dflash.py::DFlashProposer` | `v1/spec_decode/dflash.py::DFlashProposer` | 语义逐条对齐（接口适配 V1 的 staging 布局） |
| `minivllm/spec_decode/utils.py::expand_dflash_inputs` | `v1/spec_decode/utils.py::copy_and_expand_dflash_inputs_kernel` | Python 逐行等价；与上游内核 6 个输出逐值相等 |
| `minivllm/worker/gpu/spec_decode/dflash/speculator.py::DFlashSpeculator` | `v1/worker/gpu/spec_decode/dflash/speculator.py::DFlashSpeculator` | 语义对齐（裁剪：DP/EPLB/多模态/多 KV group） |
| 同上 `_prepare_dflash_inputs_kernel` | 同名内核 | 12 个输出逐值相等（去掉 `SAMPLE_FROM_ANCHOR` 分支） |
| `SpeculatorCudaGraphManager`（复用 74 关的类） | `v1/worker/gpu/spec_decode/dflash/cudagraph.py::DFlashCudaGraphManager` | 上游因为 `CudaGraphManager.capture` 的签名要单独一层；本仓库的 `capture(build_attn_metadata, ...)` 已经是这个形状，**不重复造一个类** |
| `minivllm/models/qwen3_dflash.py` | `model_executor/models/qwen3_dflash.py` | 子集：去掉 SWA/attention sink/独立 mask embedding/Laguna |
| `minivllm/testing/torch_dflash.py` | —（本仓库的冻结参考） | 逐层显式复算上下文 KV 与 query 前向 |
| `minivllm/testing/tiny_models.py::tiny_dflash_dir` | —（本仓库的测试权重生成器） | 支持 `causal`、`midlayer` 命名、缩小词表、无 aux |

**注意力后端**：`AttentionMetadata` 新增 `causal` 字段（默认 True），`TorchAttentionImpl` 的两条
掩码路径都只在 `causal=True` 时加"未来不可见"那一项；`do_kv_cache_update()` 是 DFlash 写上下文
K/V 的入口（与 `write_kv()` 同一套散射，但槽位由参数给、跳过哨兵）。

---

## 5. 差异账本（逐条）

| 编号 | 差异 | 为什么不改 | 影响 |
|---|---|---|---|
| D-76-1 | `sample_from_anchor=True` 在**配置期**拒绝（上游 V2 的 `DFlashSpeculator.__init__` 才 raise，V1 会静默按 anchor=bonus 跑） | 需求 076 §3.1 明说本机不接受；静默跑会产出位置语义错误的草稿 | 77 关（DSpark）才放开 |
| D-76-2 | 混合滑窗/全注意力（`use_swa` / `layer_types`）在配置期拒绝 | 需要多 KV group；本仓库 V2 只有单 group（上游要求 V2 runner） | 81 关（多 KV group）才放开 |
| D-76-3 | 独立的 `mask_embedding.pt` 不支持；权重里出现 `mask_hidden` 直接拒绝 | 本仓库加载器只读本地 safetensors，不做 HF 文件下载；两者都是"静默用错 embedding"的来源 | 82 关（权重加载边界）复核 |
| D-76-4 | `is_neox_style` 不从 target 拷贝 | 本仓库的 RoPE 只有 `rotate_half`（neox）一种风格，没有 interleaved 实现 | 与 D-76-3 同档；遇到 interleaved target 会数值错，需在 82 关加显式检查 |
| D-76-5 | 上游 `DFlashCudaGraphManager` 是一个类；本仓库复用 `SpeculatorCudaGraphManager` | 本仓库的捕获接口本来就是"传一个造元数据的回调"，与上游那个类做的事完全一样（少 110 行重复） | 无行为差异 |
| D-76-6 | V1 上下文行**整段写**位置/槽位（含被拒尾部），V2 把被拒尾部写 0/−1 | 两条都逐值对齐各自的上游内核；V1 的写入落在请求自己的 lookahead 槽位上，`seq_lens` 已把被拒行减掉 | 无行为差异（V2 的口径更严格） |
| D-76-7 | V1 不实现 `sample_from_anchor`；V2 的 `SAMPLE_FROM_ANCHOR` 分支未移植 | 配置期已拒绝，生产路径不可能走到 | 77 关补 |
| D-76-8 | 没有真实 DFlash 权重，端到端只用 tiny（随机但确定） | 本机只有 Qwen3-1.7B 与它的 EAGLE3 草稿；不伪造下载 | 草稿质量/加速比"集成待验" |
| D-76-9 | `DFlashSpeculator.set_attn()` 不调基类、不保存 `target_input_buffers` / `target_build_attn_metadata` | 这两个是 AR 提议者捕获草稿 prefill 图用的；DFlash 的 query 前向用自己的输入缓冲和自造元数据，上游 DFlash 分支同样没读它们 | 无行为差异（少两处"存了没人用"的引用） |

---

## 6. 验证

```bash
PY=/home/user/anaconda3/envs/vllm-omni-dev/bin/python
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY -m pytest tests/step76 -q
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY benchmarks/check_step76_dflash.py   # 27 项
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    $PY -m pytest tests/step58 tests/step59 ... tests/step76 -q      # 全量回归
```

### 6.1 逐值差分（上游内核）

| 对象 | 结果 |
|---|---|
| V1 `expand_dflash_inputs` vs 上游 `copy_and_expand_dflash_inputs_kernel` | **6/6 输出逐值相等**（`input_ids` / context 位置 / query 位置 / context 槽位 / query 槽位 / 采样行） |
| 本仓库 V2 `_prepare_dflash_inputs_kernel` vs 上游同名内核 | **12/12 输出逐值相等**（含 `query_start_loc`、`seq_lens`、`sample_pos`、`sample_idx_mapping`、温度/seed 拷贝与补齐行） |

### 6.2 模型与 KV（冻结参考）

| 对象 | 结果 |
|---|---|
| 融合上下文 KV vs 逐层参考（`reference_context_kv`） | K/V **逐值相等**（`atol=1e-6`；CPU fp32 上 max diff 0.0） |
| query 前向 vs `reference_forward`（因果 / 非因果） | max diff **0.0 / 0.0** |
| 因果开关是否真的起作用 | 两种模式输出 max diff **1.375**（不是"两边都没跑对"） |
| 显式槽位写入 | 写到 `slots` 指定位置；其他槽位不动 |

### 6.3 端到端（tiny）

| 项 | 结果 |
|---|---|
| V1 e2e | 2 条请求 × 8 token 正常出字 |
| V1 vs V2（贪心） | **逐 token 相同** |
| V2 图 vs eager | **逐 token 相同**；捕获 `FULL_DECODE_ONLY` 图、分派记录里同时出现"精确命中"和"补齐重放" |
| 被拒尾部 | 每轮 `context_slot_mapping` 的哨兵数 = 被拒行数，且哨兵落在每请求**最后** `num_rejected` 行 |
| probabilistic 草稿 | V1/V2 两条路径都能跑 |

### 6.4 回归

| 命令 | 结果 |
|---|---|
| `pytest tests/step76 -q` | **24 passed**（布局 13 + KV/模型/端到端 11） |
| `pytest tests/step58 … tests/step76 -q` | **796 passed**（= 75 关的 772 + 本关 24） |
| `benchmarks/check_step76_dflash.py` | **27/27** |
| `check_step59 / 71 / 72 / 73 / 74 / 75` | 21 / 24 / 18 / 24 / 23 / 20 项全部通过 |
| 15 个 `check_step57_*` + step58/60~70 探针 | 全部通过 |

⚠️ **顺带修掉一个既有的收集顺序问题**：UVA 判定被 `functools.cache` 缓存，如果在
`tests/step73` 的用例运行前被别的模块以"环境变量没设"的状态调用，会有 3 项上游差分
（`test_v2_logprob_diff` / `test_v2_sample_states`）失败——在提交 `1f62833` 上可稳定复现，
与 76 关本身无关。新增 `tests/conftest.py` 在**任何测试模块 import 之前** `setdefault`
`VLLM_WSL2_ENABLE_PIN_MEMORY=1`，于是不 export 环境变量也能 796 passed
（显式设成 0 的人仍然说了算）。

真实执行轨迹见 `docs/results.json` 的 `step76.results.execution_trace`。

---

## 7. 已知边界

1. **没有真实 DFlash checkpoint**：端到端只验证协议与数学，草稿接受长度与加速比"集成待验"。
2. **单 KV group**：混合滑窗/全注意力 checkpoint 在配置期拒绝（81 关）。
3. **`is_neox_style` 不拷贝**：遇到 interleaved-RoPE 的 target，草稿数值会错且不报错（D-76-4）。
4. **V1 的上下文行整段写**（D-76-6）：与上游 V1 内核一致，但比 V2 宽松；被拒尾部靠 `seq_lens` 排除。
5. **图只覆盖 query 前向**：上下文 KV 预计算每轮 eager（上游同款，形状每步变）。
6. **`sample_from_anchor` / DSpark**：77 关。
