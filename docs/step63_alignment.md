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
