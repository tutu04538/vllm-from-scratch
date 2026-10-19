# 第七十八关：DFlash2 候选关联选择与 LiLiCorr 版本边界

> 需求：[`078_DFlash2候选关联选择与LiLiCorr版本边界.md`](../../vllm-omni/learning_notes/14_vllm_from_scratch/投机解码完整需求/078_DFlash2候选关联选择与LiLiCorr版本边界.md)
> 基线：本机 `vllm==0.28.0` 的源码快照（`v1/worker/gpu/spec_decode/dflash2/speculator.py`、
> `model_executor/models/qwen3_dflash2.py`、`v1/worker/gpu/spec_decode/__init__.py:8-52`、
> `config/vllm.py:639-677`）
> 状态：**只有 V2 一条执行路径**（上游 `use_v2_model_runner` 为 DFlash2 强制 V2），已实现并自检；
> ⚠️ **本机没有真实 DFlash2 权重**，草稿质量与加速比未验；LiLiCorr 的对应关系见 §8（**不声称相同**）。

## 0. 一句话

DFlash2 = **DFlash 的并行主干** + **候选关联 walk**：每个草稿位置先取词表 top-k 候选
（`compute_candidates`），再用"前一个候选的码本行 · 当前位置 hidden · 当前候选的码本行"
给候选图的每条边打分（`candidate_selector`），然后沿图从左到右条件采样
（`_selector_walk_kernel`），最后把**实际走过的那条路径的分数**写进 `draft_logits`
（`_cache_draft_logits_kernel`）交给验证器。

本关就是把这条路径在 V2 里实现，用**与上游内核逐值差分**钉住 walk/缓存两个内核，用
**上游真实模块 + 显式循环参考**钉住分组卷积与边分，用**图/eager 对照 + greedy 无损**钉住端到端。

---

## 1. 为什么需要这一关（痛点）

1. **每位置有 k 个候选时，"取 unary 的 top-1"不是模型真正的选择**：DFlash2 的候选之间有关联
   （前一个候选影响下一个候选的得分）。用原始 unary logits 当验证分布，等于按一个**根本没用来
   采样**的分布算接受概率——不报错，只是交付分布不再等价于 target 的分布。
2. **候选集是稀疏的**：真实 q 只在 k 个候选列上有值。用整词表 logits 当 q，验证器会以为模型
   给过其他 token 概率（那些概率其实从未参与提议）。
3. **q 必须按持久 slot 复用**：`draft_logits` 是常驻缓冲。slot 换请求、候选集换掉之后，只写新
   候选列会让旧列留下有限值 → q 多出"幽灵候选"。清理依据必须是上一轮**实际写过**的列。
4. **上下文必须调制边分**：同一个 (前一个词, 当前候选) 在不同位置可以得不同分——这就是
   `hidden` 那一项的作用，也是它与 DSpark（位置无关的低秩偏置）的本质差别。

---

## 2. 数据流

```
target 本轮（num_tokens 行）
   │  aux hidden ──combine_hidden_states(fc)──► draft hidden
   │                        precompute_and_store_context_kv（图外 eager，写草稿 KV）
   ▼
prepare_dflash_inputs（GPU 内核，DFlash 的 1+K 布局；DFlash2 不改布局）
   │  query 行（1+K）+ 采样映射（sample_indices / sample_pos / sample_idx_mapping）
   ▼
query 前向（FULL 图；非因果注意力；每层挂 attention_conv / mlp_conv 两个分组卷积）
   │  [num_query_tokens, hidden]
   ▼
_generate_draft()（图里也录这一段）
   │  hidden = last_hidden[sample_indices].view(B, K, hidden)
   │  candidate_ids, unary = compute_candidates(hidden.flatten(0,1))
   │      └─ lm_head → top-k（checkpoint 的 selector_top_k）→ output_multiplier / softcap
   │  scores = candidate_selector(candidate_ids, unary, hidden, anchor_token_ids)
   │      └─ [B, K, k, k]；l=0 的前驱是 anchor（bonus token），l≥1 是上一位置的候选
   │  _selector_walk_kernel(...)
   │      └─ 每步在上一列选中的那片 [k] 里做 Gumbel-max；噪声按 **candidate token id** 取
   │  _cache_draft_logits_kernel(...)（只有 probabilistic）
   │      └─ 清旧候选列 → 写新候选列 = 实际路径分数
   ▼
draft_tokens [B, K]（target 空间的 id）+ draft_logits [max_num_reqs, K, V]（fp32，其余列 -inf）
```

---

## 3. 数据布局与不变量

### 3.1 候选图

- `candidate_ids` / `unary_logits`：`[B, K, selector_top_k]`（每个草稿位置 k 个候选）。
- `scores`：`[B, K, k, k]`，`[b,l,p,c]` 的语义是"第 l 步、**上一步选中候选 p**、当前候选 c"。
  `l=0` 时 p 维全部指向 anchor（`_score_edges` 的 `expand(-1, 1, top_k)`）。
- walk 的 `previous` 是**列号**（0..k−1），不是 token id；第二步只读 `scores[b, l, previous, :]`。
- `selector_top_k` 来自 checkpoint 的 `dflash_config.selector_top_k`；**不是**采样器的
  `top_k`/`top_p`。

### 3.2 realized q

| 量 | 布局 | 谁写 | 不变量 |
|---|---|---|---|
| `_selector_scores` | `[max_num_reqs, K, k]`（**请求行**索引） | `_selector_walk_kernel` | 每 (row, step) 存实际走到的那片 `[k]` 分数 |
| `draft_logits` | `[max_num_reqs, K, V]`（**持久 slot** 索引） | `_cache_draft_logits_kernel` | fp32；只有这一轮候选列有限；旧列 `-inf` |
| `_cached_candidate_ids` | `[max_num_reqs, K, k]` | 同上 | 下一轮清旧列的依据；初值 0（首轮清 0 号列，`draft_logits` 本来就是 -inf） |
| `sample_idx_mapping` | `[max_num_reqs*K]` | `prepare_dflash_inputs` | `-1` = 补齐/惰性行，两个内核都**不写** |

**为什么 `_selector_scores` 按请求行、`draft_logits` 按 slot**：walk 的输入是"这一轮第 b 条请求"，
输出 q 要按常驻 slot 被下一轮验证器读取；`_cache_draft_logits_kernel` 的 `req_state` 就是
`sample_idx_mapping`，两个索引空间在这一步换轨。

### 3.3 分组卷积

`block_size = 1 + K`（每请求 query 行数），位置取模后只回看块内；`taps = conv_kernel_size`，
`num_groups = hidden_size / conv_group_size`。卷积系数 = 静态 `base_kernel` + 由 hidden 现算的
`kernel_projection` 增量；注意力与 MLP 两侧各一份（`attention_conv` / `mlp_conv`）。

---

## 4. 实现对照

| 本项目 | 上游 | 状态 |
|---|---|---|
| `minivllm/worker/gpu/spec_decode/dflash2/speculator.py::DFlash2Speculator` | `v1/worker/gpu/spec_decode/dflash2/speculator.py` | `__init__` / `draft_logits_spec` / `_sample_path` / `_cache_draft_logits` / `_generate_draft` 逐条对齐；`propose()` 继承 DFlash 的 |
| 同上 `_selector_walk_kernel` / `_cache_draft_logits_kernel` | 同名内核 | 内核体逐行照抄（只改 import 前缀）；与上游**逐位相等** |
| `minivllm/models/qwen3_dflash2.py` | `model_executor/models/qwen3_dflash2.py` | `_grouped_conv` / `DFlashGroupedConv` / `DFlash2Qwen3DecoderLayer` / `_score_edges` / `CandidateSelector` / `DFlash2Qwen3Model` / `DFlash2Qwen3ForCausalLM` |
| `minivllm/layers/logits_processor.py::LogitsProcessor.get_top_k_tokens` | 同名方法 | TP=1 等价物；top-k 选择在 dtype 内做，之后 `float()` → scale → soft cap（顺序照抄） |
| `minivllm/layers/logits_processor.py::soft_cap` | 上游 `LogitsProcessor.soft_cap` | 新增可选参数；`forward()` 内部顺序"先 soft cap、再 scale"也照抄 |
| `minivllm/models/registry.py` | `registry.py:628` | 注册名 `DFlash2DraftModel`（checkpoint `architectures` 字符串），类名 `DFlash2Qwen3ForCausalLM` |
| `minivllm/worker/gpu/spec_decode/__init__.py::init_speculator` | 同名函数 | 同一 `method="dflash"` 下按 architecture 分派 `DFlash2Speculator` |
| `SpeculativeConfig.uses_dflash2()` / `_resolve_dflash2()` | `VllmConfig._is_dflash2_draft` + 模型构造期取值 | 判据相同（architecture）；校验提前到配置期，见 D-78-8 |
| `VllmConfig.validate_v2_model_runner()` | `use_v2_model_runner` 强制 V2 | 本仓库默认 V1 → 配置期明确拒绝（D-78-9） |
| `minivllm/testing/torch_dflash2.py` | —（本仓库的冻结参考） | 卷积/边分/top-k/greedy walk/解码器层，全部显式循环 |
| `minivllm/testing/tiny_models.py::tiny_dflash2_dir` | —（测试权重生成器） | 卷积、码本、hidden 投影、K、softcap、output_multiplier |

**图**：主干前向 + `compute_candidates`（含 `torch.topk`）+ `candidate_selector`（含 `einsum`）+
两个 Triton 内核全部录进 FULL 图。walk 的 grid 是 `(num_reqs,)`，重放时按捕获时的大小跑；补齐行
的 `sample_idx_mapping=-1` 让两个内核跳过它们（上游同款）。

---

## 5. 差异账本（逐条）

| 编号 | 差异 | 为什么不改 | 影响 |
|---|---|---|---|
| D-78-1 | 只实现 **Qwen3** 基座（`DFlash2DraftModel`）；上游还有 MuseGlimmer / Laguna 等 DFlash 变体 | 本机没有权重、也没验证过它们的 `dflash_config` 与卷积布局 | 加载/配置期明确报错，不猜基座 |
| D-78-2 | 分组卷积的 `block_size` 取 draft hf_config 里的 `num_speculative_tokens`（配置期由 `_resolve_dflash2()` 写回） | 本仓库 `get_model()` 只拿得到 `hf_config`，没有 `vllm_config.speculative_config` 这一轴；上游从后者读同一个值 | 语义相同；写回保证"配置写的 K"与"跑的 K"是同一个数 |
| D-78-3 | `ReplicatedLinear` 用 `ColumnParallelLinear(..., return_bias=False)` 代替 | 本仓库 TP=1、没有 `ReplicatedLinear`；两者在 TP=1 下是同一个算子（形状/权重名/加载路径一致） | 数值等价（B2/B4 实测 `max abs(Δ)=0`） |
| D-78-4 | **缩减词表（d2t）在 DFlash2 上明确拒绝** | `compute_candidates()` 用 lm_head 的 draft 宽度 id，而 q 按 target 词表宽度索引；上游这条路径没有散射，本仓库不自行补"看起来能用"的映射 | 配置期 `NotImplementedError`；要缩减词表请走 DSpark（77 关已实现） |
| D-78-5 | 没有 `set_model_tag("dflash2_candidate_selector")` 的独立 compile 缓存 | 本仓库 `CompilationMode` 除 NONE 外配置期拒绝（69 关边界：没有 torch.compile 这一轴） | 纯性能项，无行为差异 |
| D-78-6 | 配置期校验比上游**更早**（上游在模型构造时 `dflash_config["selector_top_k"]` 才 `KeyError`） | 早报错能把"不是 DFlash2 权重"与"字段写错"分开；检查项与模型真正读的字段一一对应 | 报错更明确；合法权重行为不变 |
| D-78-7 | `VllmConfig.validate_v2_model_runner()` 对 DFlash2 要求 `VLLM_USE_V2_MODEL_RUNNER=1` | 上游默认 V2 并强制；本仓库默认 V1。V1 的 `DFlashProposer` 不调用候选选择器，会**静默退化成 DFlash1** | 配置期 `NotImplementedError`，不静默降级 |
| D-78-8 | `get_top_k_tokens()` 是 TP=1 版本（无 all-gather/分片 padding） | 本仓库没有第二条 rank；保留"先 top-k 再 scale/softcap"的顺序 | 与上游 TP=1 路径逐值一致（B4 实测 0） |
| D-78-9 | 没有 `head_dtype` / fp32 lm_head / `mask_embedding.pt` 通路 | D-78-5 同类的"本仓库没有那一轴"；`mask_embedding.pt` 是 76 关边界（加载器只读本地 safetensors） | 无行为差异；缺文件时不会静默替换 |
| D-78-10 | 只做 `qwen3_dflash2` 的 Qwen3 子集；混合滑窗/多 KV group、DP/EPLB、量化不接 | 76 关的既有边界（单 KV group、TP=1） | 配置期/加载期明确报错 |

---

## 6. 验证

```bash
PY=/home/user/anaconda3/envs/vllm-omni-dev/bin/python
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY -m pytest tests/step78 -q
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY benchmarks/check_step78_dflash2.py    # 31 项
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    $PY -m pytest tests/step58 … tests/step78 -q                             # 全量回归
```

### 6.1 内核与公式（上游 / 冻结参考）

| 对象 | 结果 |
|---|---|
| 分组卷积 vs 逐 token 参考（`block_size=4/5`） | 逐值相等（`max abs(Δ)=0`） |
| 分组卷积 vs 上游 `DFlashGroupedConv`（同一份权重） | `max abs(Δ)=0.00e+00`（探针 B2） |
| `_score_edges` vs 逐元素参考（B=3,K=4,k=3,r=5） | `max abs(Δ)=1.19e-07`（fp32 einsum 舍入；探针 B3） |
| `CandidateSelector` vs 上游模块（同一份码本/投影） | `max abs(Δ)=0.00e+00`（探针 B4） |
| `_selector_walk_kernel` vs 上游（greedy） | token 与 realized scores **逐位相等**（K=1/2/3/5 × top-k=1/3，slot 重排） |
| `_selector_walk_kernel` vs 上游（probabilistic，temp=0/0.7） | 同上，**逐位相等**（共 16 组参数） |
| `_cache_draft_logits_kernel` vs 上游（同 slot 两轮、候选集换掉） | `draft_logits` 与 `_cached_candidate_ids` **逐位相等** |

### 6.2 realized q（需求 078 §3）

| 对象 | 结果 |
|---|---|
| q 初值 | fp32、全 `-inf`；概率草稿才有 q（greedy 是点质量 `None`） |
| 每步有限列数 | 恰 `selector_top_k`；列集合 == 这一轮候选；值 == 实际路径的 realized scores |
| 新旧候选集**完全不相交** | 旧列恢复 `-inf`，新列有限；`_cached_candidate_ids` 更新 |
| 新旧候选集**有交集** | 交集列更新为新分数（先清后写），非交集旧列清空 |
| slot 复用 / B 重排 | 同 slot 连续两轮与上游逐位相等；B=3、slot=`[2,0,1]` 时 token 与分数都正确 |
| 补齐行（`mapping=-1`） | token 与分数都保持哨兵值，不被写 |
| 2000 个种子的经验分布 vs `softmax(q/temp)` | **TV=0.0177**（阈值 0.1；探针 C7） |
| 噪声 key | 同一组 token 换列排布，20 个种子下选出的 token 序列全部相同 |

### 6.3 模型四层

| 层 | 结果 |
|---|---|
| **forward** | 解码器层（残差/两侧卷积/norm/MLP）vs 显式公式：`max abs(Δ)=0.00e+00`（探针 D1）；`input_embedding_scale` 只乘输入嵌入 |
| **selector** | 与上游模块 `max abs(Δ)=0.00e+00`；`l=0` 的 k 行完全相同（共用 anchor）；`hidden=0` 时退化成 unary |
| **logits** | `compute_candidates` 与参考 top-k 逐值相等（id 完全相等；含 `output_multiplier` 与 soft cap 的**先乘后 tanh** 顺序） |
| **verifier** | **未改动**：V2 验证器与 75 关是同一份代码。这里只证明交给它的 q 是 fp32、合法分布、只在候选列有值；端到端无损由 §6.4 的 greedy 对照与 75 关的验证器差分共同覆盖 |

### 6.4 端到端（tiny）

| 项 | 结果 |
|---|---|
| greedy 投机 vs 不投机的 greedy | 输出完全相同（`{'a': [4,4,6,0,6,0], 'b': [10,4,4,4,4,0]}`） |
| 图 vs eager（probabilistic，温度 0.7，同 seed） | **逐轮草稿 token 相同**（6 轮）；捕获 `captured=1`、分派记录 5 条 |
| greedy vs probabilistic（温度 0） | 输出相同 |
| slot 复用（3 条请求、`max_num_seqs=1`、K=2） | 输出与不投机 greedy 对照完全相同 |

### 6.5 回归

| 命令 | 结果 |
|---|---|
| `pytest tests/step78 -q` | **65 passed**（walk 29 + realized q 8 + 模型 9 + 边界 15 + 端到端 4） |
| `benchmarks/check_step78_dflash2.py` | **31/31** |
| `pytest tests/step58 … tests/step78 -q` | **901 passed**（= 77 关的 836 + 本关 65；无回归） |
| `benchmarks/check_step76_dflash.py` / `check_step77_dspark.py` | 27/27 / 28/28（shared `DFlashModel` 只加了 `decoder_layer_cls`，行为不变） |

---

## 7. 已知边界

1. **没有真实 DFlash2 checkpoint**：只验证协议/数学/图；草稿质量、接受长度与加速比"集成待验"
   （需求 078 §5 明确要求 tiny 与真实权重分开报告）。
2. **只有 Qwen3 基座**（D-78-1）；其他 DFlash2 基座的 checkpoint 会在注册表/配置期明确报错。
3. **缩减词表不支持**（D-78-4）；带 `d2t` 的 DFlash2 权重会配置期拒绝。
4. **LiLiCorr 未纳入本机基线**（§8）：本机快照里没有 `lilicorr` 模块；不声称 DFlash2 与
   LiLiCorr 是同一个实现，也不启用 LiLiCorr+adaptive 之类未验证组合。
5. **`selector_top_k` 的代价**：真值掉出候选就永远抽不到；本关只保证"候选图内的 q 与实际路径
   一致"，不评估掉出率。
6. **图捕获的 grid**：walk 的 grid 是捕获时的 `num_reqs`；重放更大 batch 需要更大的捕获档位
   （与 DFlash/DSpark 同一套 `SpeculatorCudaGraphManager` 规则）。

---

## 8. LiLiCorr ↔ DFlash2 版本对应表（**不声称相同**）

本机 `vllm==0.28.0` 快照里**没有** `lilicorr` 模块（`grep -ri lilicorr vllm` 为空），实现入口叫
DFlash2。官方文档只用于确认"LiLiCorr 是另一个已存在的特性"
（[docs.vllm.ai/.../lilicorr](https://docs.vllm.ai/en/latest/features/speculative_decoding/lilicorr/)），
**不能**据此把两者等同。下表左列是本机能核对的事实，右列是"官方 LiLiCorr"的待核对项。

| 维度 | 本机 0.28.0 的 DFlash2 路径（已核对） | 官方 LiLiCorr（本机无法核对） | 结论 |
|---|---|---|---|
| 执行路径/文件名 | `v1/worker/gpu/spec_decode/dflash2/`，类 `DFlash2Speculator` | 官方文档标题为 LiLiCorr；本机快照无对应模块 | **名称不同**；未验证是否同一实现 |
| 配置入口 | `method="dflash"` + `architectures=["DFlash2DraftModel"]`；没有 `method="dflash2"`/`"lilicorr"` | 未核对（本机配置枚举里没有 lilicorr） | 本关只按本机入口实现 |
| checkpoint 字段 | `dflash_config.selector_top_k / selector_rank / conv_kernel_size / conv_group_size / input_embedding_scale / final_logit_softcapping / output_multiplier` | 未核对 | 字段名以本机源码为准 |
| selector 结构 | 两张码本 `[V, rank]` + hidden 投影；边分 `unary + pred·hidden·succ` | 未核对 | 结构以 `qwen3_dflash2.py` 为准 |
| 采样协议 | 候选图上的条件 Gumbel-max walk；噪声按 candidate token id 取；q = 实际路径分数（fp32、其余列 -inf） | 未核对 | 协议以 `dflash2/speculator.py` 为准 |
| 与 adaptive 的组合 | 本仓库**未启用**；`enable_adaptive_verification` 属 79 关且当前配置期拒绝 | 文档未给出可直接照抄的组合 | 不启用未验证组合（需求 078 §4） |

**版本扩展待办**：若以后把基线升级到含 `lilicorr` 模块的 vLLM，需要**单开一关**重新做
architecture / checkpoint / selector / 采样协议的逐项差分，不能把本关的 DFlash2 结论直接平移；
升级基线本身也要单独记录差异（总纲 §4）。

---

## 9. 产物

- `minivllm/models/qwen3_dflash2.py`、`minivllm/worker/gpu/spec_decode/dflash2/speculator.py`
- `minivllm/testing/torch_dflash2.py`、`minivllm/testing/tiny_models.py::tiny_dflash2_dir`
- `tests/step78/{dflash2_helpers,test_selector_walk,test_realized_draft_logits,test_dflash2_model,test_dflash2_boundaries,test_dflash2_e2e}.py`
- `benchmarks/check_step78_dflash2.py`
- 本文件
