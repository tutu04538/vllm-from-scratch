# 65 关对齐记录：原生 MTP 模型加载与通用迭代提议

需求：[`065_原生MTP模型加载与通用迭代提议.md`](../../vllm-omni/learning_notes/14_vllm_from_scratch/投机解码完整需求/065_原生MTP模型加载与通用迭代提议.md)
基线：本机 `vllm==0.28.0` 文件快照。

> **状态：已实现并实跑**（`tests/step65` 47 项 + `benchmarks/check_step65_mtp.py` 17 项全 PASS）。
> 与上游 **真实 `Qwen3NextMTP`** 的数值对照：胶水 forward 与 `compute_logits` 都是 **max|Δ| = 0.0**
> （逐位相同）。**顺带修掉了 63 关的一个真 bug**（见 §5：CUDA 上特征从没上传到 device 缓冲）。
> 未做项集中在 §6：其它 23 个别名（缺 MLA/MoE/混合层）、多模块调度（80 关）、真实 MTP checkpoint
> （本机没有权重）。

## 0. 一句话：这一关解决什么

MTP（Multi-Token Prediction）是**训练时**让 target 学会"一次预测未来 n 个 token"而多长出来的一层
预测头，**权重就在 target 自己的 checkpoint 里**（排在最后一层之后）。于是推理时想拿它当草稿机，
有三条诱惑的错路：

| 做法 | 代价（61 层、hidden 7168 的 DeepSeek 风格 target） | 结果 |
|---|---|---|
| 把 target 再当一次 draft（`draft_model` 路径） | 参数 **+100%**（671B → 再 671B）、每枚草稿 61 层 | 正确但荒谬 |
| 拿同一个 hidden 连调 K 次 lm_head | 0 参数、0 层 | greedy 下 K 枚草稿全是**同一个 argmax 复读** |
| 每枚草稿都从 target 最后一层重跑 | 每枚草稿 61 层 | 等于没省 |
| **MTP（本关）** | 胶水 `2H×H` + **1 层**（1.7B 上胶水 ≈ 8.4M ≈ +0.5%） | 第 2 枚草稿真的条件在第 1 枚之上 |

MTP 每一步吃的是 **`(上一枚 token 的 embedding, 上一步的 hidden)`**：

```
hidden_t = norm( block( fc( [ pre_fc_norm_embedding(embed(token_t)) ‖ pre_fc_norm_hidden(hidden_{t-1}) ] ) ) )
```

第一遍的 token/hidden 来自 target 本轮喂进去的行与**最后一层 hidden**（与 EAGLE 的
`(h_i, t_{i+1}) → t_{i+2}` 配对同构）——这正是"通用迭代提议"能复用的原因：**不用为 MTP 另写一个主循环**。

## 1. 代码映射

| 本项目 | 上游参考 | 说明 |
|---|---|---|
| `models/qwen3_mtp.py::Qwen3MTP` | `model_executor/models/qwen3_next_mtp.py::Qwen3NextMTP` | 顶层：predictor + 自己的 `lm_head` + `compute_logits(hidden, spec_step_idx)` |
| `models/qwen3_mtp.py::Qwen3MultiTokenPredictor` | `qwen3_next_mtp.py::Qwen3NextMultiTokenPredictor` | `embed_tokens` / `pre_fc_norm_{embedding,hidden}` / `fc` / `layers[]` / `norm`；`mtp_start_layer_idx = num_hidden_layers` |
| `models/utils.py::get_spec_layer_idx_from_weight_name` | `model_executor/models/utils.py:496` | "这个名字属于哪个 spec 层"（绝对层号命名那一派） |
| `models/utils.py::skip_spec_layer_weights` | `deepseek_v2.py:1575-1577`（target 侧两行） | 加载 **target** 时把这些名字丢掉 |
| `models/qwen3.py::Qwen3ForCausalLM.load_weights` 的 `skip_prefixes=["mtp."]` | `qwen3_next.py:846` | 另一派命名（`mtp.*`）的跳过规则 |
| `models/qwen3_mtp.py::Qwen3MTP._spec_weights` | `qwen3_next_mtp.py::load_weights` 的 `remap_weight_names` | `mtp.` → `model.`；只留 spec 层与共享 `embed_tokens`/`lm_head`，其余整份丢掉 |
| `models/qwen3_mtp.py::Qwen3MTP._rewrite_spec_layer_name` | `deepseek_mtp.py:513-546`（`_rewrite_spec_layer_name`） | 绝对层号 → 相对层号；胶水权重提到顶层 |
| `config.py::MTP_MODEL_TYPES` + `__post_init__` 的归一 | `config/speculative.py:37-61` + `:748-756` | 24 个别名一律归一为 `method="mtp"` |
| `config.py::_resolve_mtp` / `derive_mtp_draft_config` | `config/speculative.py:759-772` + `hf_config_override` + `:1046-1084` | "目录 = target 目录"、`n_predict`、`architectures`、K 的整除约束 |
| `spec_decode/eagle.py::EagleProposer`（复用） | `v1/spec_decode/eagle.py` + `llm_base_proposer.py` | MTP 走的就是这个类（上游 `use_eagle()` 把 mtp 算进来） |
| `spec_decode/draft_model.py::_upload` 的特征上传 | 上游 `hidden_states.copy_to_gpu()` 的对应位置 | 65 关修掉的 bug，见 §5 |

## 2. 别名表（需求 §5 的交付物）

上游 `MTPModelTypes` 有 24 个名字，归一后都是 `method="mtp"`；差别只在**模型结构**与**权重命名**。
本仓库的适配状态（`benchmarks/check_step65_mtp.py` 会原样打印这张表）：

| 别名 | 归一结果 | 上游 draft 架构 | 本仓库状态 |
|---|---|---|---|
| `mtp` | `mtp` | 由 target 家族决定 | ✅ **实现**（以 Qwen3 稠密 MTP 为对齐对象，架构名 `Qwen3MTPModel`） |
| `qwen3_next_mtp` | `mtp` | `Qwen3NextMTP` | ⚠️ 归一/加载规则已实现，但**块结构不同**（Qwen3-Next 的混合注意力 GatedDeltaNet 未实现）→ 真 checkpoint 会在加载期报"没有这个参数"，**不会静默跑错** |
| `deepseek_mtp` / `kimi_k3_mtp` / `pangu_ultra_moe_mtp` | `mtp` | `DeepSeekMTPModel` / `KimiK3MTPModel` | ⏳ 未适配：缺 MLA/MoE；两者的 glue 差异（`enorm/hnorm/eh_proj`+`shared_head`、返回 `(pre_norm, post_norm)` 两个 hidden、position-0 掩码）已记录在下表 §3 |
| `dots3_note_mtp`, `mimo_mtp`, `mimo_v2_mtp`, `glm4_moe_mtp`, `glm4_moe_lite_mtp`, `glm_ocr_mtp`, `ernie_mtp`, `nemotron_h_mtp`, `exaone_moe_mtp`, `exaone4_5_mtp`, `qwen3_5_mtp`, `longcat_flash_mtp`, `bailing_hybrid_mtp`, `bailing_hybrid_v3_mtp`, `minimax_m3_mtp`, `step3p5_mtp`, `hy_v3_mtp`, `gemma4_mtp`, `inkling_mtp` | `mtp` | 各自的 MTP 类 | ⏳ 未适配（缺对应块结构）；**别名归一与 K 校验已生效**，因此不会静默退化成 `draft_model` |

## 3. 与上游的差异账本（逐条）

| 上游 | 本项目 | 为什么 / 影响 |
|---|---|---|
| 以 `DeepSeekMTP`（MLA + MoE + `shared_head` + 返回两个 hidden + position-0 掩码）为一种代表；Qwen3-Next 用 `Qwen3NextMTP`（稠密 Glue + 混合层 + 单 hidden）为另一种 | 实现 **Qwen3 家族**那一套 glue（`fc` + `pre_fc_norm_*` + `norm` + 单 hidden），块用**稠密 Qwen3 decoder layer** | 本仓库没有 MLA/MoE/GatedDeltaNet；把 Qwen3-Next 的混合层换成稠密层后，**MTP 特有的胶水逐位可比**（§5 实测 max\|Δ\|=0）。因此架构名用我们自己的 `Qwen3MTPModel`，**不冒充** `Qwen3NextMTP` |
| `spec_step_idx` 在通用 V1 路径里恒为 0（只有 step3p5 的专用提议者递进） | 同：`spec_step_idx % num_mtp_layers` 选层，通用路径调用时不传 → 恒选第 0 个模块 | 多模块（`min(n_predict, K) > 1`）的调度/状态行为属 **80 关**；`use_multi_module_mtp()` 判定已实现并在文档里标出边界 |
| MTP 的 hidden 由 `get_mtp_target_hidden_states()`（DeepSeek-V4 的 pre-hc_head 残差）可覆盖 | 不实现 | 那是 DeepSeek-V4 专用路径；本仓库的 MTP 直接吃 target 最后一层 hidden |
| `Qwen3NextRMSNorm` = `GemmaRMSNorm`（`x * (1 + w)`，weight 初始为 0） | 本仓库是 Qwen3 式 `RMSNorm`（`x * w`） | **家族差异，不是 MTP 差异**；数值对照时把权重按两种语义对齐（两边实际乘的系数相同），比的仍是胶水结构 |
| 词表补齐到 64 的倍数（`vocab_size_padded`） | 不补齐 | 对照测试里把词表取成 64 的倍数，避免拿"补齐规则"的差异冒充 MTP 差异 |
| 别名归一时打一条 deprecation warning | 不打（本仓库不引 logger） | 记在这张表里代替 |
| 加载时 `maybe_fuse_shared_experts`（MoE shared expert 融合） | 无 | 稠密 Qwen3 没有 MoE |
| `disable_padded_drafter_batch` / CUDA Graph / DP / EPLB / `prepare_next_token_ids_padded` | 无 | 与 63/64 关同一批"本仓库没有的基建"（69/70/分布式关卡） |

## 4. 设计要点（改动时不要破坏）

1. **同一个 checkpoint 里两套权重**：target 加载时跳过 spec 权重（`mtp.*` 用 `skip_prefixes`、
   `model.layers.{N+i}.*` 用 `get_spec_layer_idx_from_weight_name`），MTP 加载时只挑 spec 层与共享的
   `embed_tokens`/`lm_head`。**两边都不许静默忽略未知名字**：认不出的参数名（例如 MLA 的 `q_a_proj`）
   必须当场报错——那是"这个家族的块没实现"的唯一可靠信号。
2. **`mtp_start_layer_idx = num_hidden_layers`**：spec 层的**绝对**层号 = `num_hidden_layers + i`；
   判定必须用带点号的前缀（`model.layers.61.` 不能命中 `model.layers.6.`）。
3. **胶水权重提到顶层、块权重改成相对层号**（`_rewrite_spec_layer_name`）：与上游的分法一致，
   只是本仓库的模块路径是相对的，所以多一次减法。
4. **第一遍的 hidden 是 target 的最后一层**（不是辅助层）：`capture_aux_hidden_states` 保持 False，
   Runner 额外留一份 `target_hidden_states`；判据用 `use_eagle()`（含 mtp），不能用
   `capture_aux_hidden_states`（MTP 不吃辅助层，但仍需要本轮原始输入行做左移打补丁）。
5. **迭代提议复用 `EagleProposer`**：`pass_hidden_states_to_model=True`、
   `model_returns_tuple()` 对 MTP 返回 **False**（Qwen3 家族只返回一个 hidden）。
   **单返回值时，上一步的返回 hidden 就是下一步要回灌的那份**——不记下来会静默退化成
   "第 2 枚起不再条件于第 1 枚"，用例用"冻结回灌"的反证盯着。
6. **K 与 `n_predict`**：`K > n_predict` 时必须整除（模块复用），否则配置期报错。

## 5. 顺带修掉的一个 63 关真 bug（重要）

**症状**：CUDA 上 EAGLE3 / MTP 的 draft **完全没吃到 target 的 hidden**。

**根因**：`SpecDecodeBaseProposer._upload()` 上传了 `input_ids / positions / slot_mapping /
mask / query_start_loc / seq_lens / block_table`，**漏了 `hidden_states`**。
CPU 上 staging 与 device 是同一份张量（`_buffer()` 里 `return cpu, cpu`），所以模型照样拿到正确特征；
CUDA 上是两份，模型拿到的是**从没被写过的 device 缓冲（全零）**。

**为什么 63 关没发现**：当时的用例比的是 **CPU staging 缓冲**（`hidden_states_cpu`）与特征对齐，
以及"greedy == 非投机"——后者与草稿质量无关。**没人看过"模型实际收到的是什么"。**

**修复**：`_upload()` 在 `pass_hidden_states_to_model` 时补上
`self.hidden_states[:num_tokens].copy_(self.hidden_states_cpu[:num_tokens])`。

**回归用例**（`tests/step65/test_feature_upload_regression.py`）：EAGLE3 与 MTP 各一条，spy 住
`proposer.model.forward`，在**调用当场**比"模型收到的特征 == staging 里的那份"，并断言非全零；
外加一条把"CPU 上两者是同一对象、CUDA 上不是"钉在真实构造路径上。

**发现方式**：65 关的"错位反证"用例（`test_misaligned_target_hidden_changes_the_drafts`）第一次
跑就报 logits 差 0.0——于是回头查"特征到底有没有进模型"。

## 6. 实测

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/step65 -q            # 47 passed
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmarks/check_step65_mtp.py       # 17 项 PASS
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/step58 ... tests/step65 -q   # 见 §7
```

设备 `cuda:0`（RTX 5090 Laptop / WSL2），torch 2.13.0+cu130，tiny target `tiny_gqa`（2 层、H=32、vocab 11）。

| 项 | 实测 |
|---|---|
| 与上游 `Qwen3NextMTP` 的**胶水 forward** | **max\|Δ\| = 0.0**（逐位相同；两侧都把 decoder 层换成直通，比的是 MTP 特有的那部分） |
| 与上游的 `compute_logits` | **max\|Δ\| = 0.0** |
| 上游配置归一对照 | 同输入下上游也把 `qwen3_next_mtp` → `mtp`、`n_predict=1`、`architectures=["Qwen3NextMTP"]`；K 不整除时两边都报错（错误信息都含 "divisible"） |
| 加载数量（两派命名） | 各 **14** 个参数（MTP 自己的 12 个 + 共享 `embed_tokens`/`lm_head`），覆盖检查全过；target 侧 19 个参数 |
| spec 层归属反证 | MTP 的 `layers.0.*` ≠ target 第 0 层的同名权重（搞混层号会相等） |
| 端到端 greedy（2 派命名 × K=1/2） | 4 组全部 == 非投机，且草稿都进过调度器 |
| 胶水顺序反证 | 拼接顺序反 / 少归一化 / 错位 hidden → 输出都变（`not allclose`） |
| 回灌反证 | 冻结回灌后草稿 logits max\|Δ\| = **3.8e-1** |
| 错位反证 | 错位 target hidden 后草稿 logits max\|Δ\| = **4.8e-1** |
| 抢占/恢复 | 块很少（`num_gpu_blocks=6`）触发抢占后 greedy 仍与非投机一致 |
| 特征上传回归 | 3 次 draft 前向，模型收到的特征逐位 == staging（修复前是全零） |

## 7. 未做 / 待验（不要当成已覆盖）

1. **其它 23 个别名**：缺 MLA/MoE/混合层等块结构（§2 表）；别名归一与 K 校验已生效，加载期会明确报错。
2. **多模块 MTP**（`n_predict > 1` 且 `K > 1`）的调度/状态行为 → **80 关**（本关把判定与
   `spec_step_idx % num_mtp_layers` 的选层实现好了，但通用路径恒用第 0 个模块）。
3. **真实 MTP checkpoint**：本机没有（`models/` 只有 Qwen3-1.7B 与它的 eagle3 draft），
   所以"真实权重下的 greedy 对照"未跑；tiny 与上游类的对照不能替代它。
4. **与上游 `DeepSeekMTP` 的 forward 对照**：需要 MLA + MoE（本仓库没有），未做；
   其 glue 差异（`enorm/hnorm/eh_proj`、`shared_head`、两个 hidden、position-0 掩码）记录在 §3。
5. **index sharing**（`set_skip_topk` / `compact_topk_indices`）：只在有稀疏 attention 的
   DeepSeek-V3.2 上适用；按需求 §3.5 留到 80 关。
6. **CUDA Graph / 异步调度下的 MTP**：与 63/64 关同一批（69/70 关）。

## 8. 验证命令（复跑）

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/step65 -q
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmarks/check_step65_mtp.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/step58 tests/step59 tests/step60 \
    tests/step61 tests/step62 tests/step63 tests/step64 tests/step65 -q
```
