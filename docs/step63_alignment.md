# 63 关对齐记录：EAGLE 与 EAGLE3 的特征传递和位置对齐

需求：[`063_EAGLE与EAGLE3的特征传递和位置对齐.md`](../../vllm-omni/learning_notes/14_vllm_from_scratch/投机解码完整需求/063_EAGLE与EAGLE3的特征传递和位置对齐.md)
基线：本机 `vllm==0.28.0` 文件快照。

> **状态：阶段 A/B/C 已实现并实跑（第一遍输入已严格对齐上游不扩容分支）**，并完成**与上游真实实现的数值对照**：`combine_hidden_states` 逐位相同
> （max|Δ|=0）、映射回 target 词表的 logits max|Δ|=7.0e-4（容差 5e-3）、复合两步 argmax 逐行相同。
> 唯一待验项：**真实 checkpoint 的完整生成**（依赖 67 关的异构词表采样空间语义）。
> 本文件按阶段更新。**不要**把本关当成"已通过"——下面 §5 明确列了未做的部分。

本关解决什么痛点（一句话）：**draft 的输入从"只有 token"变成"(token, target 特征) 对"以后，
输入必须逐请求错开一格、特征与 positions 不许动、每请求的最后一格必须正好是那条请求新采出的 token**；
错一格**不报错**，只是草稿全废（接受率崩），所以必须精确到行号而不是"看起来 shape 对"。

## 1. 阶段划分与当前进度

| 阶段 | 内容 | 状态 |
|---|---|---|
| A | **第一遍输入对齐** | ✅ 已完成：默认 EAGLE 通路**严格照抄上游不扩容分支**（行数 = target 行数、整体左移 + 打补丁、positions/特征逐行原样，Runner 交本轮原始 token/positions）。验收改为**只测生产路径**（`benchmarks/check_step63_eagle_inputs.py` 8 项 + `tests/step63/test_eagle_e2e.py`）；**扩容分支（并行提议）留到 72 关**，届时重建与上游内核 `shift_input_ids=True` 的逐值差分 |
| B | **EAGLE3 模型适配**（`Eagle3Qwen3ForCausalLM`/`Eagle3LlamaForCausalLM` + target 辅助层输出 + `combine_hidden_states` + d2t/t2d 词表映射） | ✅ 已完成：模型 + **真实 checkpoint 张量全部落位** + 与上游真实实现的数值对照（`combine_hidden_states` 逐位相同、映射回 target 词表的 logits max\|Δ\|=7.0e-4，`tests/step63/test_eagle3_upstream_diff.py` 4 项）；draft **解码层**的逐值对照要等 69/70 的 forward 上下文基建（§5.2） |
| C | **提议者与 Runner 接线 + 端到端** | ✅ 已完成：`EagleProposer`（复用基类 KV/AR/工作区）、Runner 设辅助层并把本轮 hidden 交给提议者、EAGLE 第一遍对齐、greedy == 非投机（K=1/2/4） |

## 2. 阶段 A：第一遍输入对齐（已实现）

### 2.1 代码

| 本项目 | 上游参考 | 说明 |
|---|---|---|
| `spec_decode/eagle.py::EagleProposer.set_inputs_first_pass` | `llm_base_proposer.py:846-872`（`set_inputs_first_pass` 的 **no-extra-slots 通路**） | 整体左移一格 + 按 `query_start_loc[1:] - 1` 打补丁；**特征与 positions 都不动**；输入直接吃 Runner 交来的**本轮原始缓冲**（`target_token_ids` / `target_positions`），不自己重建行 |
| `spec_decode/utils.py::compute_new_slot_mapping` | `v1/spec_decode/utils.py::compute_new_slot_mapping` | 逐行同公式；本通路 `num_new_tokens=0`（不额外加行），被拒行**不掩码**（它的 KV 下一轮必被重算覆盖） |

> 历史说明：阶段 A 期间曾有**两个只被测试调用的参考实现**（`eagle_first_pass_input_ids` /
> `expand_eagle_inputs_shifted`）与对应的 12 项差分用例；按用户要求"参考实现只放 `minivllm/testing/`、
> 生产包不留没有调用方的代码"删除了。现在这一关的验收**只测生产路径**（`benchmarks/check_step63_eagle_inputs.py`
> 8 项 + `tests/step63/test_eagle_e2e.py` 4 项 + 模型/上游对照 17 项）。

需求 §3 的两请求例子已逐元素固定：

```
target tokens: [a1,a2, b1,b2,b3]        query_start_loc = [0, 2, 5]
next tokens:   [a3,b4]
→ draft 输入:  [a2,a3, b2,b3,b4]        采样行 = [1, 4]
```

### 2.2 实测证据（`benchmarks/check_step63_eagle_inputs.py` 8 项 + `tests/step63/test_eagle_e2e.py`）

现在**只测生产路径**：脚本直接驱动 tiny 引擎，对着提议者写进工作区的内容做检查。

- 需求例子逐元素一致（`[a2,a3,b2,b3,b4]`、采样行 = 每请求最后一行）；
- positions 与特征**逐行原样**（第 i 行配第 i 行的特征，扩容行 = 采样行）；greedy 端到端 == 非投机；
- **"被拒位置下一轮必被重算"**：实测 `next_start == start + num_valid`（7/8 轮出现被拒行），
  这是"把被拒草稿也喂进 draft"无害的前提（KV 下一轮必被覆盖）；
- 阶段 A 期间还做过**与上游内核的逐值差分（CUDA）**：`copy_and_expand_eagle_inputs_kernel(
  shift_input_ids=True, num_padding_slots_per_request=1)` 的 `input_ids`/`positions`/`is_rejected`/
  `new_token_indices` 与当时的参考实现逐值一致（含带被拒行的用例；58 关差分的是 `False` 分支）。
  该差分随参考实现一起删除，**结论保留、命令不再可复跑**——上游行为现在由
  `tests/step63/test_eagle3_upstream_diff.py`（直接实例化上游类做数值对照）继续盯着。

### 2.3 两条通路的**物理布局差异**（实测发现，写清楚免得当 bug）

内核路径会为每条请求补 `num_rejected` 个 padding 行（"被拒行也占工作区"），**后面的请求整块后移**；
no-extra-slots 通路不带这些行（被拒位置本来就是本轮 target 查询行的一部分，原地留着）。
实测（A=[11,12] 被拒 1 行、B=[21,22,23]）：

```
内核:      [12, 13, PAD, 22, 23, 24]      is_rejected = [0,0,1,0,0,0]
无扩容:    [12, 13, 22, 23, 24]
```

所以不变量是"**每条请求的 [shift 后有效行 + 扩容行] 相同**"，不是"整段扁平数组相同"。
默认 EAGLE（非并行提议）`net_num_new_slots_per_request == 0` → 走 no-extra-slots 通路；
内核通路要等并行提议（72 关 P-EAGLE / DFlash）。

## 3. 真实权重 manifest（阶段 B 用；权重不入库）

本机已下载（`models/` 在 `.gitignore` 里）：

| 项 | 值 |
|---|---|
| target | `models/Qwen3-1.7B`（Qwen3ForCausalLM，28 层，bf16） |
| draft | `models/Qwen3-1.7B-eagle3`（HF `AngelSlim/Qwen3-1.7B_eagle3`，下载时间 2026-10-04） |
| draft config | `architectures=["LlamaForCausalLMEEagle3"]`、`model_type="llama"`、`num_hidden_layers=1`、`hidden_size=2048`、`head_dim=128`、`num_attention_heads=16`、`num_key_value_heads=8`、`rope_theta=1e6`、`tie_word_embeddings=true` |
| 权重文件 | `pytorch_model.bin` 274,132,734 B（15 个张量，fp16） |
| 辅助层数 | **3**（`fc.weight` 形状 `(2048, 6144)` = `hidden × 3`；对应上游 `fc_input_size = target_hidden_size * num_aux_layers`） |
| 词表 | draft `draft_vocab_size = 32000`，target `vocab_size = 151936` → checkpoint 带 `d2t (32000,)` / `t2d (151936,) bool` 映射（TLI，67 关主题；模型适配要用到这两个张量） |
| 权重名 | `midlayer.*`（→ `layers.0.*`，q/k/v → qkv、gate/up → gate_up）、`fc.weight`、`norm.weight`、`lm_head.weight`、`d2t`/`t2d`（无 `embed_tokens` → 与 target 共享 embedding） |

**注意**：draft 的 config 里**没有** `eagle_aux_hidden_state_layer_ids`；上游此时回落到
`model.get_eagle3_default_aux_hidden_state_layers()`（阶段 B 要用上游同一套默认值，不能自己拍）。

## 4. 顺带修掉的一个测试顺序问题（与本关实现无关）

`tests/step59/test_rejection.py::test_mixed_greedy_and_random` 原来把注入的均匀随机数写成
`[0.5, 0.9]`，隐含"random 行一定读到第二个元素"这个**与行序有关**的假设；在整套 `tests/` 一起跑时
（新增 `tests/step63` 后暴露）会把"拒绝"翻成"接受"。改成两行都给 `0.9`（greedy 行根本不读 u），
**断言强度不变**（random 行仍是 `u=0.9 > p/q=0.889` → 拒绝）。修完 `pytest tests/step58..63` 连跑两次
均 292 passed。

## 5. 还没做的部分（不得当作已通过）

阶段 A/B/C 都已实现并实跑；下面这些是**明确留到后续关卡**的，不要当成 63 关已经覆盖：

1. ~~**真实 EAGLE3 checkpoint 的完整生成**~~ **已于 66 关收尾时补上**：`compute_logits()` 现在按
   `d2t` 把 draft 词表的 logits **scatter 回 target 宽度**（上游 `llama_eagle3.py:339-356` 同款），
   提议者不必自己映射；`tests/step63/test_eagle3_real_e2e.py` 用真实
   `models/Qwen3-1.7B` + `models/Qwen3-1.7B-eagle3` 跑通 greedy 端到端（16 token 逐 token 相同）
   并断言草稿 id 全部落在 `t2d` 集合里（含"不映射就会大量落在集合外"的反证）。
   **注意**：这条是"draft 缩小词表 + 偏移映射"（同一个 tokenizer）；67 关的 TLI 是**两套 tokenizer
   的 token 级交集**（`VocabMapping`），两件事分开做、分开测（需求 067 §3.6）。
   映射放在 `compute_logits()` 里还顺带解决了 `q`（草稿概率）的宽度问题：映射发生在 **softmax 之前**，
   所以 `softmax(32000 个值)` 与"scatter 到 151936 宽再 softmax"逐位相同（非 d2t 位置是 `-inf` →
   概率恰好 0），`draft_probs` 天然就是 target 宽度的、**不需要事后换算**；实测
   `pending_draft_probs.draft_probs.shape == (2, 151936)`，贪心行的 q 是点质量（一行一个非零）。
   （若把映射放在 id 阶段，就得把 32000 宽的 q 重排成 151936 宽：拒绝采样内核按 **target 词表**步长
   索引 `draft_probs`，宽度不符是**越界读**——内核只断言了 ndim。）

   **顺带记一条给 67 关的结论**（"TLI 为什么不能照 d2t 这么做？"）：**能做**。把"交集掩码 + scatter"
   放到 softmax **之前**，与"draft 空间 mask→softmax→把 probs 搬过去"逐位等价（实测 max|Δ|=6e-8、和=1；
   argmax 也可交换：draft 空间 argmax 3 → 经映射表 = target 7 = target 空间 argmax 7），greedy 下这就是
   上游 `_greedy_sample()` 现在的做法。上游只是**没做**概率那条路：`compute_probs_and_sample_next_token()`
   只用了 `temperature`（NOTE：忽略其它采样参数不影响最终分布），`use_heterogeneous_vocab` 在配置期强制
   `draft_sample_method="greedy"`，代码里留着 TODO "remap draft_probs to target-vocab space"；本仓库按
   需求 067 §3.5 照抄这条边界。**真正的坑是顺序**：先对整份 draft 词表 softmax、再掩码搬运而不重新
   归一化时，q 的和 ≠ 1（实测 0.9548）→ q 与实际提议分布不一致 → 采样分布不再精确等于 target 的分布。
2. **草稿质量（接受长度）还没对齐**：真实权重下实测 K=2、单个 prompt 的接受长度 ≈ **1.07**
   （14 个请求·轮 drafted=28 / accepted=1），而官方模型卡（`AngelSlim/Qwen3-1.7B_eagle3`）在
   Qwen3-1.7B 上写的是 **2.13~2.2**。id 空间已经正确（第 1 条钉住了），所以差距更可能出在
   **draft 解码层本身**，也就是下面第 3 条那条待办。本机跑不了上游引擎做对照
   （`LLM(...)` 在 WSL2 上直接 `RuntimeError: UVA is not available`），只能与模型卡比。
3. **draft 解码层（attention/MLP）与上游的逐值对照**：需要 69/70 的 forward 上下文与 CUDA Graph 基建
   才能把两边的前向放到同一条件下比；本关比的是 `combine_hidden_states` 与最终 logits（max|Δ|=7.0e-4）。
   第 2 条的接受长度差距大概率要在这里定位。
4. **M-RoPE 未接**：上游 `_raise_if_mrope` 的对应检查在配置期报错，三路位置没有实现。
5. **`lm_head` 共享的另一套条件**：上游 `_maybe_share_lm_head` 在 draft 词表 == target 词表时共享
   lm_head；本关只按"检查点缺 `embed_tokens`"共享嵌入。
6. **扩容分支（并行提议）**：默认 EAGLE 走 no-extra-slots 通路；`shift_input_ids=True` 的扩容分支
   要等 **72 关**（P-EAGLE/DFlash），届时重建与上游内核的逐值差分。

## 6. 验证命令（实测，2026-10-05）

```bash
python -m pytest tests/step63 -q                                   # 24 passed（含 66 关收尾补的真实 checkpoint 端到端 3 项）
python -m pytest tests/step58 tests/step59 tests/step60 tests/step61 \
                 tests/step62 tests/step63 tests/step64 -q         # 314 passed（含 64 关）
python benchmarks/check_step63_eagle_inputs.py                     # 8 项 PASS（只测生产路径）
```

---

## 7. 2026-10-08 复核修复：自回归步的起点必须与上游同源

> 本轮由用户在需求 72 之前点名复核："EAGLE3/MTP 自回归提议步的 positions/seq_lens 是否偏离上游"。
> 结论：**偏离属实，已修**；同时量到一个重要的**否定结论**——63 关一直挂着的接受长度差距
> **不是**这里造成的。

### 7.1 问题（复核前）

提议循环里自回归步的位置/上下文长度是**反推**出来的：

```
position = history_end + 已提枚数 − 1
seq_len  = start + num_valid + 1          （在 `_apply_num_rejected_to_seq_lens()` 里）
```

上游不是反推，而是**从第一遍的采样行直接取**（`llm_base_proposer.py:634` + `:698`）：

```
positions = self.positions[token_indices_to_sample]      # 采样行自己的 position
每步：positions = _update_positions_dependent_metadata(positions)   # 内核里 position + 1、seq_len + 1
seq_lens 起点 = 第一遍 seq_lens − num_rejected_tokens_gpu           # :654-660
```

反推公式对 **draft 布局**恰好等价（它的采样行就是尾部扩容行，位置 = `history_end − 1`），
但对 **EAGLE 布局**（采样行 = target 行块的**最后一行**，token 内容被换成新采样的 token）会差
`1 − 被拒数` 格：

| 场景（3 token prompt、K=3） | 修复前 | 上游规则 | 后果 |
|---|---|---|---|
| 第 1 轮（被拒 0） | positions 4、5 | 3、4 | 位置整体偏 +1 |
| 第 2 轮（被拒 3） | positions 5、6 | 7、8 | **落回第一遍刚写过的行**，把刚写的 KV 覆盖掉 |

EAGLE 与 draft 的采样行**不是同一行**，所以任何"用历史长度反推"的公式都只对其中一种布局成立——
这正是当初写错的原因。

### 7.2 修法（对齐上游）

* `FirstPassPlan` 增加两个字段：`sample_positions`（采样行自己的 position）与 `seq_lens`（第一遍的
  逐请求长度）；`DraftModelProposer.set_inputs_first_pass()` 与 `EagleProposer.set_inputs_first_pass()`
  各自按自己的布局填。
* 提议循环：`position = sample_position[req] + 已提枚数`；
  上下文 `= (第一遍 seq_lens − 被拒数) + 已提枚数`（`_apply_num_rejected_to_seq_lens()` 里把设备侧
  `seq_lens` 也改成修正后的值——上游就是在 device 批量上做 `-=`，自回归步的内核再逐步 +1）。
* draft 布局的数值**逐值不变**（回归由 `test_draft_model_autoregressive_steps_follow_the_same_rule` 盯着）。

### 7.3 验证

| 检查 | 结果 |
|---|---|
| `tests/step63/test_eagle_ar_alignment.py`（新，2 项） | R1：`位置 == 采样行位置 + k`；R2：`上下文 == (第一遍长度 − 被拒) + k`。EAGLE3 与 draft_model 各一项，且要求"被拒>0 / 被拒=0"两种情形都出现 |
| 判别力（故意把公式改回旧的再跑） | EAGLE3 那一项**失败**、draft_model 那一项仍通过 → 测试确实钉住了这个差异 |
| `benchmarks/check_step63_eagle_inputs.py` | 8 → **11 项**（新增 F1/F2/F3） |
| 全量回归 | `pytest tests/step58..71` → **579 passed**（577 + 2）；16 个 `check_step58..71` + 15 个 `check_step57` 全绿 |

### 7.4 否定结论：接受长度**没有**因为这个修复而变化

同一份真实权重（`models/Qwen3-1.7B` + `Qwen3-1.7B-eagle3`）、K=2、5 个 prompt × 128 token：

| | drafted | accepted | acc_len（1+acc/draft） | 每轮产出 token | 第1枚被接受的比例 | 第2枚被接受的比例 |
|---|---|---|---|---|---|---|
| 本仓库·修复前 | 1024 | 124 | **1.1211** | 1.2500 | — | — |
| 本仓库·修复后 | 1022 | 124 | **1.1213** | 1.2524 | 115/511 = 22.5% | 9/511 = **1.8%** |
| **上游引擎**（同机、同权重、同 prompt、同 K） | 816 | 230 | **1.2819** | 1.5686 | 170/408 = 41.7% | 60/408 = 14.7% |

（上游那行是本次实跑：`VLLM_WSL2_ENABLE_PIN_MEMORY=1 VLLM_ENABLE_V1_MULTIPROCESSING=0`，脚本与原始
数字记在 `docs/results.json` → `step63_ar_positions`。第 2 枚的"被接受比例"含"第 1 枚也被接受"这个前提。）

**两条差距都不在自回归步的坐标上**：

* 第 2 枚（自回归产出的那一枚）：1.8% vs 14.7% —— 把坐标改对之后**没有改善**，说明自回归步的
  **条件本身**（喂进去的 hidden / 上一步写下的 KV / 注意力上下文）还有别的问题；
* 第 1 枚（第一遍产出、与自回归无关）：22.5% vs 41.7% —— **第一遍就已经差了一半**，这是更大的一块。

本轮顺手量到的一条线索（写进 `vllm_bugs/`，供后续复核）：在
`test_eagle3_real_e2e.py` 那个 prompt 上，真实 draft head 的 **draft 空间 argmax 有 22/32 行落在
id < 124**（其中多数是 id=1），也就是草稿几乎退化成同一个 token。这不像"坐标差一格"能解释的，
更像 draft 前向的输入（特征拼接 / prenorm / 注意力上下文）仍有偏差——**这与 §5 第 3 条（draft 解码层
逐值对照）是同一件事**，仍待定位；本文件不声称 BUG-5 已解决。

> ⚠️ 由此修正一条旧记录的措辞：§5 第 2 条里"本机跑不了上游引擎"**已过时**——2026-10-06 复核澄清，
> 加 `VLLM_WSL2_ENABLE_PIN_MEMORY=1 VLLM_ENABLE_V1_MULTIPROCESSING=0` 就能跑上游引擎
> （`vllm_bugs/VERIFICATION_REPORT_20261006.md`），本次对照就是这么做的。
