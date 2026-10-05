# 67 关对齐记录：异构词表 TLI 与 Draft 采样空间

需求：[`067_异构词表TLI与Draft采样空间.md`](../../vllm-omni/learning_notes/14_vllm_from_scratch/投机解码完整需求/067_异构词表TLI与Draft采样空间.md)
基线：本机 `vllm==0.28.0` 文件快照。执行路径：**V1**。

> **状态：已实现并实跑**（`tests/step67` **23 项** + `benchmarks/check_step67_vocab_mapping.py` **21 项**全 PASS）。
> 与上游真 `VocabMapping` 的差分：三张量表 + 两个方向的 map + 约束后的 logits **逐位相同**。
> 集成：一对真正的 tiny 模型（同 KV 规格、vocab 11 vs 13、**两套真 tokenizer 文件**）跑通 greedy。
> 明确不做（照抄上游边界）：概率草稿的 TLI —— 配置期直接拒绝（§3.5）。

## 0. 一句话：这一关解决什么

**target 的 token 17 和 draft 的 token 17 未必是同一段文字。** 拿另一个训练好的模型当 draft 时，
两套 tokenizer 的 id 空间毫无关系；只比 `vocab_size` 相等也挡不住（不同 tokenizer 完全可能撞上同一个
词表大小）。实测（本机 Qwen3-1.7B 的 tokenizer vs gpt2）：

```
同一段文字 " hello world, the quick brown fox"
    target(Qwen3) ids = [23811, 1879, 11, 279, 3974, ...]
    draft(gpt2)   ids = [23748,  995, 11, 262, 2068, ...]
把 target 的 id 直接喂给 draft：gpt2.decode(...) = ' Ellen Car, pipp Rouueless'
```

**不报错**（id 都在范围内），只是 draft 读到的历史全是别的字 → 草稿几乎必被拒。TLI（token-level
intersection）就是为这件事做的：初始化按 **token 字符串**建一张交集表，运行期把 id 在两个空间之间
搬运，**不重新分词**（位置数、KV 槽位、块表一律不变）。

本机的真实规模（记录用，`check_step67` 的 E5 会打印）：

| 这一对 | target(Qwen3-1.7B) | draft(gpt2) | 交集 |
|---|---|---|---|
| 词表 | 151936 | 50257 | **42257** |
| 覆盖率 | 27.8% | 84.1% | —— |

也就是说：**target 有 72% 的 token 是 draft 说不出来的**（进 draft 空间时填 unk），draft 有 16% 的列
永远说不出来（logits 被掩成 `-inf`）。拒绝采样对任意 `q` 都无损，所以这只是"接受率低"，不是"输出错"。

## 1. 代码映射

| 本项目 | 上游参考 | 说明 |
|---|---|---|
| `spec_decode/vocab_mapping.py::VocabMapping` | `v1/spec_decode/vocab_mapping.py::VocabMapping`（L68-154） | 三张表 + 两个方向的 map + `constrain_draft_logits()` |
| `vocab_mapping.py::_detect_space_prefix` | 同名函数（L10-32） | 用 `encode(" a")` 探"词首空格"标记（Ġ / ▁）；探不到就两种都试 |
| `vocab_mapping.py::_normalize_token` | 同名函数（L35-39） | 把 `Ġfoo` / `▁foo` 还原成 `" foo"` |
| `vocab_mapping.py::_get_unk_token_id` | 同名函数（L42-65） | `unk → eos → 报错`（**0 是合法 unk**，不能用 `unk or eos`） |
| `vocab_mapping.py::load_tokenizer` | 上游 `get_tokenizer()` | 读 tokenizer 目录（本仓库第一次在提议者里读 tokenizer）；读不到明确报错 |
| `spec_decode/draft_model.py::DraftModelProposer.__init__` 里的建表 | `v1/spec_decode/draft_model.py:36-59` | `use_heterogeneous_vocab` 时建映射表，否则走"词表必须相等"的校验 |
| `SpecDecodeBaseProposer._to_draft_space` / `_to_draft_space_tokens` | `llm_base_proposer.py:841-845` / `:698-699` | 第一遍的历史行 + 扩容行、自回归步的上一枚草稿 → 映射到草稿空间 |
| `SpecDecodeBaseProposer._sample_draft_tokens` 的 TLI 分支 | `llm_base_proposer.py:436-445`（`_greedy_sample`） | 掩码 → argmax → 映回 target 空间；q 是点质量（不带 `draft_probs`） |
| `config.py` 的 `use_heterogeneous_vocab` / `draft_sample_method` + 两条校验 | `config/speculative.py:153-157`、`:1387-1396` | 只支持 `draft_model`；只支持 greedy 草稿 |
| `config.py::ModelConfig.tokenizer` / `tokenizer_path` | `ModelConfig.tokenizer` | TLI 要知道两边分别去哪读 tokenizer |
| `testing/tiny_models.py::tiny_hetero_pair` / `write_wordlevel_tokenizer` | 无（测试用） | 一对 tiny 模型（同 KV 规格、vocab 11 vs 13）+ 两套**真** tokenizer 文件（▁ / Ġ 两族） |

## 2. 四条路径（与上游逐条对应）

```text
① 第一遍：target 的历史行 + 扩容行（新采出的 token）→ map_target_to_draft_ids → 喂进 draft
② 自回归：上一枚草稿是 **target id**（要交回调度器）→ map_target_to_draft_ids → 再喂回 draft
③ 采样：  草稿 logits 先 constrain_draft_logits（非交集列 -inf）→ argmax（TLI 只支持 greedy）
④ 交回：  采出的草稿 id → map_draft_to_target_ids → 调度器/target 验证**全程用 target id**
```

**只在 id 上换标号，不改行数/位置**：`start` / `history_end` / `num_rejected` / `slot_mapping` /
块表全部照旧——这是 token 级 TLI 与"重新分词桥接"的根本区别（需求 §5 明确不做后者）。

## 3. 与上游的差异账本（逐条）

1. **map 方法的设备处理**：上游默认索引在 GPU 上的表（`self.target_to_draft_ids[target_ids]`）。
   本仓库的第一遍输入是在 **CPU** 上组织的（`input_ids` 是 Python 列表 / CPU 缓冲），所以 map 方法会把
   表挪到输入所在的设备（`.to(ids.device)`；已经同设备时是空操作）。语义与上游逐位一致。
2. **日志 → 结构化字段**：上游用 `logger.info` 打 target/draft 词表与交集大小、交集 < 100 时 warning。
   本仓库不引 logger，改成 `self.stats`（含 `intersection_size` / 两侧覆盖率）+ `warnings.warn`
   （阈值同样取 100，消息里写明"上游同样只告警"）。需求 §5 要的"记录交集大小"由它提供。
3. **`draft_sample_method` 的落地范围**：本仓库把它做成配置项并落实上游的两条校验（TLI 必须是
   `draft_model` + `greedy`）。**没有**改基类既有的草稿采样路径——63/65/66 关的 draft 在请求开温度时
   本来就"在自己的分布上采样并把 q 交出去"（上游把这叫 probabilistic drafting）。TLI 这条**独立**走
   `_greedy_sample` 语义（掩码 → argmax → 点质量 q），因为概率草稿的 TLI 上游未实现（见 §5）。
   差异记在这里：`draft_sample_method="greedy"`（默认）在**非 TLI** 路径上还不改变 63/65/66 的行为。
4. **`pytest` 里的假 tokenizer**：上游只对真 tokenizer 测；本仓库额外允许任何提供
   `get_vocab/encode/convert_ids_to_tokens/unk_token_id/eos_token_id` 的对象，测试用假对象造形状
   （同词不同 id、重复规范化、越界 id、空交集），**集成**那条用真 tokenizer 文件。
5. **`constrain_draft_logits` 的输入**：上游在 `_greedy_sample` / `_sample_draft_tokens` 里都调用它；
   本仓库在 TLI 分支里调用一次（greedy 是 TLI 唯一允许的路径，不需要第二个调用点）。

## 4. 设计要点（改动时不要破坏）

1. **只换 id、不重新分词**：映射前后**位置数不变**，所以 `slot_mapping`/块表/attention 元数据与 target
   完全一致。任何"重新切词/拼字符串"的做法都会让行数变化，进而破坏 KV 槽位对齐。
2. **交集外的历史 token 填 `draft_unk_token_id`**（不是 `-1`）：`-1` 是哨兵，喂进模型会变成"最后一个
   token"，是静默错。反过来 `map_draft_to_target_ids` 填 `target_unk_token_id`。
3. **`unk_token_id = 0` 是合法的**：判空只能用 `is not None`；写成 `unk or eos` 会在 unk=0 的 tokenizer
   上悄悄换成 eos（需求 §3.4 专门点名）。
4. **草稿 logits 必须先掩码**：交集外的列置 `-inf` 之后，argmax/采样永远选不到它——"交出去的 id 一定
   在 target 空间有对应物"这条前提才成立。
5. **交出去的草稿是 target id，q 是点质量**：draft 空间的概率宽度与 target 词表不符，**绝不能**当作
   `draft_probs` 交给验证器（拒绝采样内核按 target 词表步长索引它）。所以 TLI 路径 `draft_probs=None`
   （59 关的 `NO_DRAFT_PROBS` 分支，点质量语义）。
6. **不让 target 模型改自己的词表**（需求 §2）：映射只存在于提议者这边；target 全程用它原来的 id 空间。
7. **`use_heterogeneous_vocab` 必须显式开启**：不开时"词表不一致"照旧明确报错（TLI 不是静默生效的兜底）。

## 5. 概率草稿的 TLI：为什么只支持 greedy（需求 §3.5）

上游把 `use_heterogeneous_vocab` 限制成 `draft_sample_method="greedy"`，并在
`llm_base_proposer.py` 里留了 TODO：`remap draft_probs to target-vocab space for lossless
probabilistic rejection sampling with heterogeneous vocabularies`。本仓库照抄这条边界。

补一句给后来人的账（66 关收尾时实测过）：**这件事在数学上可行**——把"交集掩码 + scatter"放到
**softmax 之前**，与"draft 空间 mask→softmax→把 probs 搬过去"逐位等价（实测 max|Δ|=6e-8、和为 1；
argmax 也可交换）。上游没做的原因是它的代码结构是"先在 draft 空间采样、再映射 id"，于是 q 留在
draft 空间，要重排就得额外铺一张 target 宽（151936）的 q。**真正的坑是顺序**：先对整份 draft 词表
softmax、再掩码搬运而不重新归一化时，q 的和 ≠ 1（实测 0.9548）→ q 与实际提议分布不一致 → 采样分布
不再精确等于 target 的分布（静默错）。要放开这条限制，必须先有一个**真正实现它**的上游提交可对照。

## 6. 实测

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/step67 -q                  # 23 passed
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmarks/check_step67_vocab_mapping.py   # 21 项 PASS / 0 FAIL
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/step58 ... tests/step67 -q # 438 passed
```

设备 `cuda:0`（RTX 5090 Laptop / WSL2），torch 2.13.0+cu130。tiny 对由
`tiny_hetero_pair()` 现场生成：target vocab 11（▁ 系 tokenizer）、draft vocab 13（Ġ 系），**同一个词的
id 不同**，交集 10 个 token（`▁onlyt` 是 target 独有、`Ġonlyd`/`Ġx`/`Ġy` 是 draft 独有）。

| 项 | 实测 |
|---|---|
| 与上游 `VocabMapping` 的差分（2 组用例） | 三张量表、两个方向的 map、`constrain_draft_logits` 输出**逐位相同** |
| 真实 tokenizer 规模（Qwen3-1.7B × gpt2） | 交集 **42257**（target 27.8% / draft 84.1%）；target 侧无 unk → 退回 eos（告警，上游同款） |
| tiny 集成 greedy | 开了 TLI 与非投机**逐 token 相同**；6 轮草稿，草稿 id 全部落在交集像内 |
| 点质量 | 所有轮次 `draft_probs is None`（TLI 只支持 greedy 草稿） |
| 反「假接线」 | 第一遍/自回归的 target id **真的**过了 `map_target_to_draft_ids`（12 次调用，入参含 target 独有 token 时出参变成 draft unk=0） |
| 不重新分词 | `[2, 5, 6] → [6, 3, 4]`：长度不变、逐位置"规范化后的 token 字符串"不变 |
| 掩码 | 给 draft 独有列灌 100.0 的 logits，掩码后 argmax 依然落在交集里（`[5, 5]`） |
| 空交集 | 只告警；`intersection_size=0`、mask 全 False、`constrain_draft_logits` 全 `-inf`、两个 map 全回退 unk（按源码行为记录） |

## 7. 未做 / 待验（不要当成已覆盖）

1. **概率草稿的 TLI**：上游未实现，本仓库配置期拒绝（§5）。要放开得先有真正支持它的上游提交。
2. **真实异构词表 target/draft 的端到端**：本机只有一个 target（Qwen3-1.7B，Qwen tokenizer）与它的
   eagle3 draft（同词表），**没有**"另一套 tokenizer 的独立小模型"；所以集成用的是 tiny 对（两套真
   tokenizer 文件）。gpt2 × Qwen3 的**交集规模**已实测记录（§6），但"gpt2 词表的 draft 模型"本机没有
   → 真实权重的端到端**待验**。
3. **跨 tokenizer 的字符串桥接**（重新分词对齐）：需求 §5 明确不做；本关只做 token 级交集。
4. **交集外的历史 token 造成的草稿质量损失**：本关只保证"不报错、语义有据"（填 unk），
   量化它对接受率的影响不在本关范围。
5. **`draft_sample_method="probabilistic"` 在非 TLI 路径上的语义**：配置项与校验已就位，但基类的草稿
   采样行为没有按它切换（§3.3 记着这条差异）。
6. **CUDA Graph / 异步调度下的 TLI**：与 63/64 关同一批（69/70 关）。

## 8. 验证命令（复跑）

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/step67 -q
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmarks/check_step67_vocab_mapping.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/step58 tests/step59 tests/step60 \
    tests/step61 tests/step62 tests/step63 tests/step64 tests/step65 tests/step66 tests/step67 -q
```

`C1`（上游差分）需要本机装有 `vllm==0.28.0`；`E5`（真实 tokenizer 规模）需要
`models/Qwen3-1.7B` 与 HF 缓存里的 gpt2，缺任一项会跳过并写明原因（**不是"通过"**）。
