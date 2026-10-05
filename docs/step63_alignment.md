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
| B | **EAGLE3 模型适配**（`Eagle3Qwen3ForCausalLM`/`Eagle3LlamaForCausalLM` + target 辅助层输出 + `combine_hidden_states` + d2t/t2d 词表映射） | 🟡 **模型与真实权重加载 ✅**（`minivllm/models/qwen3_eagle3.py`、`tests/step63/test_eagle_model.py` 11 项）；**逐层对照（与上游实现比容差）待做** |
| C | **提议者与 Runner 接线 + 端到端** | ✅ 已完成：`EagleProposer`（复用基类 KV/AR/工作区）、Runner 设辅助层并把本轮 hidden 交给提议者、EAGLE 第一遍对齐、greedy == 非投机（K=1/2/4） |

## 2. 阶段 A：第一遍输入对齐（已实现）

### 2.1 代码

| 本项目 | 上游参考 | 说明 |
|---|---|---|
| `spec_decode/utils.py::eagle_first_pass_input_ids` | `llm_base_proposer.py:846-872`（`set_inputs_first_pass` 的 **no-extra-slots 通路**） | 整体左移一格 + 按 `query_start_loc[1:] - 1` 打补丁；**特征与 positions 都不动** |
| `spec_decode/utils.py::expand_eagle_inputs_shifted` | `v1/spec_decode/utils.py::copy_and_expand_eagle_inputs_kernel`（`shift_input_ids=True`） | 扩容分支的等价展开：`num_valid = query_end - query_start`（比 `False` 少 1）、`input_offset = 1`、`output_start = query_start + i*(slots-1)`；positions 不跟着移 |

需求 §3 的两请求例子已逐元素固定：

```
target tokens: [a1,a2, b1,b2,b3]        query_start_loc = [0, 2, 5]
next tokens:   [a3,b4]
→ draft 输入:  [a2,a3, b2,b3,b4]        采样行 = [1, 4]
```

### 2.2 实测证据（`tests/step63/test_eagle_inputs.py` 12 项 + `benchmarks/check_step63_eagle_inputs.py`）

- 需求例子逐元素一致；三请求不同长度、单 token 请求、全单 token 批各自只拿自己的新 token；
- **反证**：把补丁下标算成 `query_start_loc[1:]`（下一条请求的第一格）→ A 的最后一格留成 B 的第一个 token
  （实测 `[12, 21, ...]`，而正确是 `[12, 13, ...]`）——需求原文点名的坑；
- **positions/特征不动**：`expand_eagle_inputs_shifted` 给出的 positions 与 target 逐行相同，
  扩容行位置 = 该请求最后一行的位置（内核注释 "Positions are NOT shifted"）；
- **与上游内核逐值差分（CUDA）**：`copy_and_expand_eagle_inputs_kernel(shift_input_ids=True,
  num_padding_slots_per_request=1)` 的 `input_ids`/`positions`/`is_rejected`/`new_token_indices`
  与我们逐值一致（含带被拒行的用例）；58 关差分的是 `shift_input_ids=False` 分支，本关补上 `True`。

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

## 5. 未做（阶段 B/C）——不得当作已通过

1. **EAGLE3 模型适配**：`models/qwen3_eagle3.py`（`Eagle3Qwen3ForCausalLM`：fc 投影 + layer0 的
   `cat([input_layernorm(embeds), hidden_norm(hidden)])` + `norm` 返回 `(hidden, prenorm)` + lm_head）
   与 `llama_eagle3.py` 对应类（真实 checkpoint 是 Llama 风格：`midlayer` 名字映射、无 qk-norm）；
   target 侧 `set_aux_hidden_state_layers` / 返回辅助层 hidden states。
2. **逐层对照**（需求 §4）：tiny 权重上比较 target 辅助输出、draft forward、logits 并记录容差；
   真实 checkpoint 加载后的 forward/logits 对照。
3. **提议者与端到端**：`EagleProposer(SpecDecodeBaseProposer)`（`pass_hidden_states_to_model=True`）、
   `build_model_inputs_first_pass` 传 `hidden_states`、`prepare_next_token_ids_padded` /
   `prepare_inputs_padded`、`num_rejected_tokens_gpu` 修正、后续自回归步的 token/hidden/positions/
   seq_lens 同步、`model_returns_tuple()`、`_maybe_share_embeddings/_maybe_share_lm_head` 条件、
   prefix 命中/抢占/批重排/结束清理、greedy == 非投机。
4. **不支持的组合**：M-RoPE（上游 `_raise_if_mrope`）、backend、padded 开关组合的限制触发。
5. **d2t/t2d 词表映射**：本关只记录 manifest；完整"异构词表 draft 采样空间"是 67 关。

## 6. 阶段 A 的验证命令

```bash
python -m pytest tests/step63 -q                      # 12 项（含 CUDA 上的内核差分）
python -m pytest tests/step58 tests/step59 tests/step60 tests/step61 tests/step62 tests/step63 -q
python benchmarks/check_step63_eagle_inputs.py        # 10 项
```
