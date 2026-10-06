# minivllm：对齐 vLLM V1 架构的文本生成子集

这是仓库里**唯一的实现**（原 `step57/`，含 204/205 两轮验收修复，以及第五十八～六十六关：draft 输入/双预算、GPU 批量拒绝采样、CPU/GPU ngram、Suffix Decoding、自定义 Proposer 接入、EAGLE/EAGLE3、HiddenStateExtraction、原生 MTP、Medusa 多头提议）。旧的 `stepNN/` 代码目录已经
删除，历史记录留在 `docs/` 与 git 历史里；后续改动只在这个包上做。

它不自己发明协议，而是做一个**能逐层映射到本机 vLLM（0.28.0）的、可运行的文本生成子集**。
每一层都要能回答：谁拥有这份状态？谁可以改它？模块之间传什么（而不是偷偷共享什么）？
对应 vLLM 哪个类、哪个方法？省略了哪些条件？

包名不叫 `vllm` 是刻意的：`benchmarks/compare_step57_vllm.py` 等对照脚本要在**同一个进程**里
`import vllm`（site-packages 的真 vLLM），同名会静默遮蔽。

设计与差异账本见
[`docs/step57a_skeleton.md`](../docs/step57a_skeleton.md)（骨架）、
[`docs/step57b_real_model.md`](../docs/step57b_real_model.md)（模型、loader、Attention、Runner）、
[`docs/step57c_kv_and_prefix.md`](../docs/step57c_kv_and_prefix.md)（块池、前缀缓存、抢占恢复）、
[`docs/step57d_sampling_and_stop.md`](../docs/step57d_sampling_and_stop.md)（采样、惩罚、停止、增量输出）、
[`docs/step57e_speculative.md`](../docs/step57e_speculative.md)（投机验证、草稿时序、draft 模型）；
57F 的两篇对照见 [`docs/step57_architecture.md`](../docs/step57_architecture.md)（与真实 vLLM 的
结构/数值对照）与 [`docs/step57_alignment.md`](../docs/step57_alignment.md)（差异账本）；
第五十八/五十九关见 [`docs/step58_alignment.md`](../docs/step58_alignment.md)（draft 输入与双预算）、
[`docs/step59_alignment.md`](../docs/step59_alignment.md)（GPU 批量拒绝采样、投机元数据口径、接受率统计）、
[`docs/step60_alignment.md`](../docs/step60_alignment.md)（CPU/GPU ngram 提议与历史增量维护）、
[`docs/step61_alignment.md`](../docs/step61_alignment.md)（Suffix Decoding：请求内 + 跨请求后缀树，外部依赖见
[`docs/results.json`](../docs/results.json)（`step61.dependencies`））、
[`docs/step62_alignment.md`](../docs/step62_alignment.md)（自定义 Proposer 与分派边界）、
[`docs/step63_alignment.md`](../docs/step63_alignment.md)（EAGLE/EAGLE3）、
[`docs/step64_alignment.md`](../docs/step64_alignment.md)（cache-only 特征提取）、
[`docs/step65_alignment.md`](../docs/step65_alignment.md)（原生 MTP）、
[`docs/step66_alignment.md`](../docs/step66_alignment.md)（Medusa 多头提议与 MLP 版本缺口）、
[`docs/step67_alignment.md`](../docs/step67_alignment.md)（异构词表 TLI 与 draft 采样空间）、
[`docs/step68_alignment.md`](../docs/step68_alignment.md)（采样约束、Logprobs 与结构化输出的投机语义）。

| 层 | 文件 | 对应 vLLM |
|---|---|---|
| 配置 | `config.py` / `sampling_params.py` | `vllm/config/*`、`vllm/sampling_params.py` |
| 请求状态 | `request.py` | `v1/request.py` |
| 协议 | `outputs.py`、`core/sched/output.py` | `v1/engine/__init__.py`、`v1/outputs.py`、`v1/core/sched/output.py` |
| KV 控制面 | `core/kv_cache_manager.py` → `core/kv_cache_coordinator.py` → `core/single_type_kv_cache_manager.py` → `core/block_pool.py` → `core/kv_cache_utils.py` | `v1/core/` 下同名文件（这条链一层一个问题） |
| 调度 | `core/sched/{scheduler,request_queue,utils}.py` | `v1/core/sched/*` |
| 编排 | `engine/{core,core_client,output_processor,llm_engine}.py`、`engine/logprobs.py` | `v1/engine/*` |
| 结构化输出 | `structured_output/{__init__,backend_types,backend_xgrammar,request,utils}.py` | `v1/structured_output/*` |
| logprobs 容器 | `logprobs.py`、`tokenizer_utils.py` | `vllm/logprobs.py`、`vllm/tokenizers/` |
| 执行部署 | `executor/uniproc_executor.py` | `v1/executor/uniproc_executor.py` |
| 执行端 | `worker/worker.py` | `v1/worker/gpu_worker.py` |
| 一轮怎么跑 | `worker/gpu_model_runner.py` | `v1/worker/gpu_model_runner.py` |
| 批状态缓冲 | `worker/gpu_input_batch.py`、`worker/block_table.py` | 同名文件 |
| 权重加载 | `model_loader/*` | `model_executor/model_loader/*` + `models/utils.py` |
| 模型 | `models/qwen3.py` | `model_executor/models/{qwen2,qwen3}.py` |
| 层与注意力 | `layers/*`、`attention/*` | `model_executor/layers/*`、`attention/*` |
| 采样 | `sample/{metadata,sampler}.py`、`sample/ops/*` | `v1/sample/{metadata,sampler}.py`、`v1/sample/ops/*` |
| 投机验证 | `sample/rejection_sampler.py`（Triton 批量内核） | `v1/sample/rejection_sampler.py` |
| 投机提议 | `spec_decode/{metadata,metrics,ngram_proposer,ngram_proposer_gpu,draft_model,suffix_decoding,custom_class_proposer,eagle,extract_hidden_states,medusa,vocab_mapping}.py` | `v1/spec_decode/*` |
| 测试替身 | `testing/fake_runner.py`、`testing/tiny_models.py`、`testing/torch_rejection_sampler.py`、`testing/spec_metadata.py` | 无（只给测试；tiny 模型现场生成，不提交权重） |

## 怎么用（本地模型短生成）

```bash
python minivllm/demo.py                                     # 默认 models/Qwen3-1.7B + 默认问题
python minivllm/demo.py --max-new-tokens 64 --trace "问题"    # --trace 打印第一轮喂给模型的数字
```

`--trace` 那段输出是本关的核心证据之一：**协议只给了 `num_scheduled_tokens`，其它数字
（input_ids / positions / slot_mapping / 块表）全是执行侧自己算的**。

## 怎么用（注入假 Runner，不装真实模型）

给 `Worker` 注入 `testing.FakeRunner` 就不装真实模型，协议/调度用例靠它跑得又快又确定。

```python
from minivllm import (CacheConfig, LLMEngine, ModelConfig, SchedulerConfig,
                     SamplingParams,
                    UniProcExecutor, VllmConfig, Worker)
from minivllm.testing.fake_runner import FakeRunner

config = VllmConfig(model_config=ModelConfig(model="dummy", max_model_len=64),
                    cache_config=CacheConfig(block_size=4, num_gpu_blocks=4),
                    scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=2))
runner = FakeRunner(tokens={"r1": [11, 12]})          # 只给测试用；生产路径不会退回假执行
engine = LLMEngine(config, UniProcExecutor(config, Worker(config, model_runner=runner)))
engine.add_request("r1", [10, 11, 12], SamplingParams(max_tokens=2, eos_token_id=99))
while engine.has_unfinished_requests():
    for out in engine.step():
        print(out.request_id, out.token_ids, out.finished)
```

## 调度器可打印的轨迹（57C 交付物）

```bash
python minivllm/demo.py --scheduler-trace "同样的问题" "同样的问题"
```

采样参数（57D）：`--temperature/--top-k/--top-p/--seed`、三种惩罚、`--min-tokens`、`--ignore-eos`；
默认贪心、不开任何筛选。

```text
step scheduled                    hits           preempted    running            waiting        free cached
   1 {'q0': 19}                                               ['q0']             ['q1']           10      0
   2 {'q0': 1, 'q1': 3}           {'q1': 16}                  ['q0', 'q1']       []                9      1
```

读法：第 1 轮只排得下 q0 的 prompt；第 2 轮 q1 **命中 16 个 token**（用小预算逼它晚一轮进来），
所以只需要再算 3 个。抢占会出现在 `preempted` 那一列，块占用看最后两列。不运行模型也能看懂调度器
在做什么，`check_step57_preemption.py` 里也是靠它做断言。

## 第五十八关：draft 第一遍输入与双预算（热点）

- **Scheduler 两份预算**：`token_budget`（本轮 target 能算多少 token）与 `input_budget`
  （draft 第一遍工作区能放多少行）；普通 draft 每条被调度请求多占
  `max_num_new_slots_for_drafting = 1` 行。抢占时两份一起返还。
- **draft 第一遍**：每条请求的物理行 = `[有效行] + [1 行扩容行] + [被拒行]`，
  被拒尾部用 mask + `slot=-1` 屏蔽；展开规则与上游
  `copy_and_expand_eagle_inputs_kernel(shift_input_ids=False)` 逐值一致（`tests/step58` 里有 CUDA 差分）。
- **prefix 命中不再重算**：起点用调度快照里的 `num_computed_tokens`（含命中起点），
  命中段直接复用共享块里 draft 各层已经写好的 KV。
- **固定工作区**：输入缓冲按 `max_num_batched_tokens` / `max_num_seqs` 开一次、每轮只覆盖有效前缀，
  `data_ptr()` 稳定（69 关的编译/CUDA Graph 地基）。

设计与对照见 [`docs/step58_alignment.md`](../docs/step58_alignment.md)，
实测记录见 [`docs/results.json`](../docs/results.json)（`step58.results`）。

## 第六十一关：Suffix Decoding（请求内 + 跨请求历史）

- **外部依赖，不自研**：候选来自 `arctic_inference.suffix_decoding.SuffixDecodingCache`
  （Snowflake ArcticInference 的官方实现）；上游 vLLM 也是 import 它。没装时
  `SpeculativeConfig(method="suffix")` 在配置期直接 `ImportError`，不会退回别的提议者。
- **两棵树**：每条请求一棵 prompt 树（`start_request` 时把 prompt 全量入树），
  外加一棵**跨请求全局树**（请求的输出按 FIFO 缓存，`max_cached_requests` 条，0 = 关掉）。
  A 学到的"…1 2 3 → 4 5"能被 B 直接用来猜——这是 ngram 提议者（只看本请求历史）拿不到的。
- **候选长度动态**：`speculate()` 从 1 开始递增地匹配 context 后缀（**全长 context 不参与匹配**，
  最后一个 token 只当锚点），按 `match_len * max_spec_factor` 限长、按频次概率过 `min_token_prob`；
  同一批里每条请求的候选长度可以不同（ngram 是等宽 K）。
- **生命周期**：本轮空采样（中间 prefill）跳过；离开 input batch 的活跃请求在 `propose` 末尾
  `stop_request`（≠ Scheduler 的 FINISHED）；请求结束时 Runner 调 `remove_requests` 补齐
  "批为空"那一轮；同 ID 重用先 `evict_cached_response` 再 `start_request`。

设计与对照见 [`docs/step61_alignment.md`](../docs/step61_alignment.md)，
依赖锁定记录见 [`docs/results.json`](../docs/results.json)（`step61.dependencies`），
实测记录见 [`docs/results.json`](../docs/results.json)（`step61.results`）。

## 第六十二关：自定义 Proposer（插件接口与分派边界）

- **换候选算法不用动 Engine**：`SpeculativeConfig.model` 给一个 `module.Class` 点号路径，
  `method` 会被推成 `custom_class`（**推断只在配置期做一次**，Runner 只按 `method` 分派，
  不再自己判第二遍）。
- **插件契约**：`__init__(vllm_config)` + `propose(sampled_token_ids, num_tokens_no_spec,
  token_ids_cpu, slot_mappings=None) -> list[list[int]]`。行数必须等于批行数（未采样的行给空列表）；
  每行枚数随意（0..K，可变长）；候选内容错了不影响正确性（target 逐个验证）。
  两个缓冲是**定长**的（`max_num_reqs` / `max_num_reqs × max_model_len`），只有前 N 行有效。
- **越权边界**：插件只拿到只读的 `VllmConfig` 与三个 CPU 缓冲，拿不到 Scheduler 的 `Request`、
  `KVCacheManager` 或 `InputBatch`，改不了状态；`remove_requests` 是可选的（有才调）。
- **错误在启动期分类**：无点号 / 模块不存在 / 类不存在 / 构造失败 / `propose` 缺失或不可调用，
  五类各报各的（保留原始异常链），不静默回退成 ngram。

示例插件见 [`examples/custom_proposer.py`](../examples/custom_proposer.py)，
设计与差异见 [`docs/step62_alignment.md`](../docs/step62_alignment.md)，
实测记录见 [`docs/results.json`](../docs/results.json)（`step62.results`）。

## 第六十三关：EAGLE/EAGLE3（特征传递与位置对齐）

> **阶段 A/B/C 已实现并实跑**：输入对齐、EAGLE3 draft 模型（真实 checkpoint 张量全部落位）、
> 提议者/Runner 接线与 greedy 端到端（K=1/2/4）都过了，并完成与上游真实实现的数值对照
> （`combine_hidden_states` 逐位相同、logits max|Δ|=7.0e-4）。**真实 checkpoint 的完整生成已在 66 关
> 收尾时补上**：`compute_logits()` 按 `d2t` 把 draft 词表（32000）的 logits scatter 回 target 宽度
> （151936），真实权重下 greedy 端到端跟非投机逐 token 相同，草稿 id 与**草稿概率 q** 都钉在 target
> 空间（q 的宽度 = 151936，因为映射在 softmax 之前做，不需要事后换算）
> （`tests/step63/test_eagle3_real_e2e.py`）。剩余项（**草稿接受长度还没对齐**：实测 ≈1.07 vs 模型卡
> 2.13~2.2 → 疑似 draft 解码层差异、draft 解码层逐值对照 → 69/70、M-RoPE）见
> [`docs/step63_alignment.md`](../docs/step63_alignment.md) §5。

EAGLE 的 draft 不只吃 token，还吃 target 本轮算出的 hidden states；于是第一遍输入要满足：

- **token 逐请求错开一格**：整体左移 + 每条请求的最后一格换成这条请求新采出的 token
  （`query_start_loc[1:] - 1` 是补丁下标）；算错一位就会让 A 的最后一格留着 B 的 token；
- **特征与 positions 不动**（`(h_i, t_{i+1}) → t_{i+2}` 的配对），扩容行用该请求最后一行的特征/位置；
- 被拒行仍占工作区（padding + mask），默认 EAGLE 通路不需要额外槽位（`net_num_new_slots == 0`）。

## 第六十四关：HiddenStateExtraction 的 cache-only 执行路径

63 关已经能让 target 顺带吐出辅助层特征，但特征是当轮的中间量：想留下来只能自己存。这一关把
"存特征"接进 KV 的寻址与生命周期——**不做投机，只借投机框架跑一趟**：

- **特征即 KV**：cache-only 层的缓存形状是 `[num_blocks, block_size, L, H]`（L = 辅助层数当 head 数、
  H = hidden_size 当 head_size），一个 token 占一个 slot；写缓存就是一次散射
  `kv_cache[slot // bs, slot % bs] = to_cache`，槽位用的是**本轮 target 那份 `slot_mapping`**；
- **prefix 命中不会丢数据**：命中段不重算，但它的特征已经在复用的块里（特征与 KV 一样是
  "token + 位置的确定函数"，同前缀共享块成立）；
- **草稿是 target 自己采出的 token**：`propose()` 返回 `sampled_token_ids[:, :1]`，K 固定 1；
  被拒时第 0 列就是 target 自己的 token，所以输出逐 token 正确，但**这不是加速特性**，
  不计入"投机加速算法"，也不宣称速度收益；
- **验收靠读物理 slot**：两请求 × 两辅助层塞可辨认值，逐槽位对照"同权重 + 同元数据重跑一次"的
  独立参考；真实 Qwen3-1.7B（fp16、辅助层 (2,14,25)、每块 196,608 B）上 max|Δ|=0.0、greedy 逐 token 一致。

跑一段（真实 Qwen3-1.7B，辅助层 (2,14,25)，每块特征 192 KB）：

```bash
python minivllm/demo.py --device cuda --max-new-tokens 8 --spec-method extract_hidden_states \
    --spec-k 1 --spec-aux-layers 2,14,25 "The capital of France is"
```

设计与差异（含"0 号块留白"这个 69 关前提）见 [`docs/step64_alignment.md`](../docs/step64_alignment.md)，
实测记录见 [`docs/results.json`](../docs/results.json)（`step64.results`）。

## 第六十五关：原生 MTP（多 token 预测）

MTP 是模型**训练时**多长出来的一层预测头，权重就排在 **target checkpoint 的最后一层之后**。
这一关把它接成草稿机，三件事：

- **别名归一**：`deepseek_mtp` / `qwen3_next_mtp` / `glm4_moe_mtp` … 24 个名字一律归一到
  `method="mtp"`（与上游 `MTPModelTypes` 逐项一致）——它们在引擎里是同一件事；
- **同文件、两套权重**：target 加载时跳过 spec 层（`mtp.*` 用 `skip_prefixes`、
  `model.layers.{N+i}.*` 用 `get_spec_layer_idx_from_weight_name`），MTP 加载时只挑 spec 层 + 共享的
  `embed_tokens`/`lm_head`；认不出的参数名（例如 MLA 的 `q_a_proj`）**当场报错**，不静默跳过；
- **通用迭代提议**：MTP 复用 `EagleProposer`（上游 `use_eagle()` 就把 `mtp` 算进来）。它吃的是 target 的
  **最后一层 hidden**（不是辅助层），每一步 `hidden_t = norm(block(fc([norm_emb(embed(token_t)) ‖
  norm_hidden(hidden_{t-1})])))` —— 所以第 2 枚草稿真的条件在第 1 枚之上。

```bash
python -m pytest tests/step65 -q                      # 47 项
python benchmarks/check_step65_mtp.py                 # 17 项（含与上游 Qwen3NextMTP 的逐值对照）
```

> 与上游真实 `Qwen3NextMTP` 的胶水 forward / `compute_logits`：**max|Δ| = 0.0**（两侧 decoder 层都换成直通，
> 比的是 MTP 特有的那部分）。本仓库的 MTP 块是**稠密 Qwen3**，所以架构名用我们自己的 `Qwen3MTPModel`，
> 不冒充 `Qwen3NextMTP`；其余 22 个别名（缺 MLA/MoE/混合层）在加载期明确报错。

设计与差异（含别名表与"顺带修掉的 63 关 bug"）见 [`docs/step65_alignment.md`](../docs/step65_alignment.md)，
实测记录见 [`docs/results.json`](../docs/results.json)（`step65.results`）。

## 第六十六关：Medusa 多头提议（与 MLP 支持缺口）

Medusa 是挂在 target 旁边的 **N 个纯 MLP head**（没有 attention、没有 KV）：所有 head 读**同一份**
target hidden，各自 argmax 出一枚草稿，`stack` 成 `[B, K]`。与 EAGLE/MTP 的**自回归**候选不同，
这 N 枚候选互相"听不见"（并行省时间，第 2 枚的接受率天然低一档）。这一关做三件事：

- **K 就是 head 数**：旧 FasterDecoding 的 `config.json` 只有 `medusa_num_heads/medusa_num_layers`
  （连 `model_type`/`vocab_size`/`architectures` 都没有），上游把 `num_heads` 改写成 K、并把
  `vocab_size`/`truncated_vocab_size` 对齐到 target——本仓库同样在配置期归一；
- **线性链，不是树**：`propose()` = `model(hidden)` → 每个 head 的 `compute_logits` → 每个 head 一个
  `argmax` → `[B, num_heads]`；argmax 是点质量提议，所以 `draft_probs=None` 是**正确**的 q。
  论文里的 tree attention（`max_paths`/`topk`）在本机 V1 里没有读取点；
- **取对 hidden 行**：一轮验证的 query 是 `[b][d1]…[dK]`，采样后序列最后一个 token（bonus）本轮没算过，
  所以要用"产出 bonus 的那一行" = 块内第 `采样数 - 1` 行。上游的 stride 是 `num_draft + 1`
  （混合 prefill 批会错位），本仓库用调度快照的 `num_scheduled_tokens`。

```bash
python -m pytest tests/step66 -q                  # 51 项
python benchmarks/check_step66_medusa.py          # 28 项（含与上游 Medusa/MedusaProposer 的逐值对照）
```

> 与上游真实 `Medusa` + `MedusaProposer`：每个 head 的 blocks/logits **max|Δ| = 0.0**、候选**列顺序**相同。
> 旧 checkpoint 是 `.pt`（本仓库只读 safetensors，会明确报错），本机也没有 LLaMA/Vicuna 基座，
> 所以真实权重下的端到端**待验**。
>
> **`mlp_speculator` 是版本缺口**（需求 §3）：本机 0.28.0 里配置层认得它、注册表那行却被注释成
> "Temporarily disabled"，模型类连 `forward` 都没有、Runner 也没有分派。本仓库**不写自创实现**，
> 而是在配置期明确报错（`tests/step66/test_mlp_support_boundary.py` 用四份证据钉住）。

设计与差异（含真实旧 checkpoint 的实测形态、行选择错位反证、加载严格度）见
[`docs/step66_alignment.md`](../docs/step66_alignment.md)，实测记录见
[`docs/results.json`](../docs/results.json)（`step66.results`）。

## 第六十七关：异构词表 TLI（token 级交集）

拿**另一个训练好的模型**当 draft 时，两套 tokenizer 的 id 空间毫无关系——target 的 token 17 和 draft 的
token 17 未必是同一段文字，直接互换 id 不会报错、只会让草稿全错（实测同一段文字：Qwen3 给
`[23811, 1879, 11, ...]`、gpt2 给 `[23748, 995, 11, ...]`）。这一关按 **token 字符串**建交集表：

- `VocabMapping`：`draft_to_target_ids` / `target_to_draft_ids` / `intersection_mask_draft` 三张表 +
  两侧 unk 兜底（`unk→eos→报错`；**0 是合法 unk**，不能写 `unk or eos`）；空格标记两族（Ġ / ▁）先归一化；
  同一个 tokenizer 内规范化后重名的只留第一个；超出模型 `vocab_size` 的条目不入表；
- 四条路径：第一遍的历史行与扩容行、自回归步的上一枚草稿 → `map_target_to_draft_ids`；草稿 logits →
  `constrain_draft_logits`（非交集列 `-inf`，永远选不到）→ argmax → `map_draft_to_target_ids`；
  **交出去的草稿一定是 target 空间的 id**，q 是点质量（TLI 只支持 greedy 草稿）；
- **只换 id、不重新分词**：行数/位置/`slot_mapping`/块表一律不变（这是它和"字符串桥接"的根本区别）；
- 边界照抄上游：`use_heterogeneous_vocab` 只支持 `method="draft_model"`，且概率草稿的 TLI 在**配置期
  直接拒绝**（把 q 从 draft 空间搬到 target 空间上游还没做，需求 §3.5 要求不得自行放开）。

```bash
python -m pytest tests/step67 -q                     # 23 项
python benchmarks/check_step67_vocab_mapping.py      # 21 项（含与上游 VocabMapping 的逐位差分）
```

> 集成：一对真正的 tiny 模型（同 KV 规格、vocab 11 vs 13、两套**真** tokenizer 文件）跑通 greedy，
> 与非投机逐 token 相同；真实规模记录：Qwen3-1.7B × gpt2 的交集 **42257**（target 27.8% / draft 84.1%）
> ——target 有 72% 的 token draft 说不出来（填 unk），只影响接受率、不影响输出分布。

设计与差异（含设备处理、日志→字段、`draft_sample_method` 的落地范围、概率草稿那条 TODO 的账）见
[`docs/step67_alignment.md`](../docs/step67_alignment.md)，实测记录见
[`docs/results.json`](../docs/results.json)（`step67.results`）。


## 第六十八关：采样约束、Logprobs 与结构化输出的投机语义

投机让"每个位置该看哪份分布"变成一个**索引问题**，而索引错了不会报错——只会悄悄给出错的数或坏的 JSON。
这一关把两件事在投机路径上做对：

- **logprobs（四种模式）**：`raw_logprobs` / `raw_logits` / `processed_logprobs` / `processed_logits`
  （由 `ModelConfig.logprobs_mode` 选）。一轮里交付的位置可能是"接受的候选 + 恢复 token + bonus"，
  规则是**第 j 个位置读第 j 行**：接受候选读候选行、恢复 token 读**同一个位置**的行、bonus 读 bonus 行；
  被拒的候选位照样算一份，但 `parse_output` 用**同一张 valid_mask**把 token 与 logprobs 一起滤掉
  （停止 token 截断处同样同步截断）。`frequency_penalty` 是"按出现次数减"的那一项，所以
  "历史里算不算草稿"会直接改变 bonus 行的分布（差 1.0 个 logit 足以翻转采样）。
- **结构化输出（只接 xgrammar）**：请求期把 `choice` 改写成 EBNF 并校验 schema（xgrammar 不支持的
  关键字直接报错，不静默忽略）；每轮每请求生成 `1 + K` 行掩码（K 个候选位 + 1 个 bonus 位），
  候选位靠"假设前面都被接受"地**逐步试走**得到，离去前统一 `rollback`；**只有 Scheduler 在真正提交
  token 之后**才 `accept_tokens` 永久推进 FSM；草稿在收下时先过 `validate_tokens` 预筛。

```bash
python -m pytest tests/step68 -q                     # 67 项
python benchmarks/check_step68_logprobs.py           # 23 项（含与上游 Sampler / _get_logprobs_tensors 的逐值差分）
python benchmarks/check_step68_grammar.py            # 30 项（规格/掩码/状态/应用/端到端/后端边界）
```

> 真实权重实测：Qwen3-1.7B 的 `raw_logprobs` 与 `transformers` **逐值相同**（含 `decoded_token`）；
> `--json-schema` 下产出符合 schema 的 JSON；`--logprobs 3` 的候选带名次与解码文本。
> 三态矩阵（支持 / 上游不支持 / 本项目尚未接入）见 `docs/step68_alignment.md` §4：
> 六项未接入字段与三个未接入后端**请求期明确拒绝**，绝不静默忽略参数。
> 处理顺序表见 §2.9：**同一个约束在 bonus 行与候选行上可能不一样**——上游的 `min_p`
> 只作用在 bonus 行（候选验证行不掩码），本项目照抄并钉住（实测 `min_p=1.0` 时投机仍有 rank>1 的 token 被交付）。
> ⚠️ 2026-10-06 独立复核：`logprobs=-1` 上游在引擎入口把它归一化成 `vocab_size`，本项目缺这一步（待修）。

设计与差异（四种模式的行索引、掩码试走/回滚、采样约束处理顺序表、三态矩阵与复核后的更正）见
[`docs/step68_alignment.md`](../docs/step68_alignment.md)，实测记录见
[`docs/results.json`](../docs/results.json)（`step68.results`）。

## 明确不做

EAGLE/MTP、异步与多进程、指标、白名单/bad words/思考预算等未接入的采样字段、KV 连接器、多 KV group。
另外两条容易误以为已经具备的能力：

- **只支持 TP=1**：`layers/linear.py` 里的名字（`QKVParallelLinear` 等）是为了与 vLLM 源码
  一一对应，**没有实现通信**，改 `tp_size` 不会跑起来；
- **只读本地 safetensors**：单文件或带 index 的分片都行，不做 HF hub 下载、不支持 `.bin`/量化。

## 验证

```bash
python benchmarks/check_step57_engine_protocol.py    # 23 项：协议与数据契约
python benchmarks/check_step57_request_progress.py   # 29 项：请求进度与停止判定
python benchmarks/check_step57_scheduler_basic.py    # 25 项：统一预算调度
python benchmarks/check_step57_runner_inputs.py      # 30 项：输入打包（198 §4 逐值）、批状态、入口边界
python benchmarks/check_step57_weight_loading.py     # 25 项：权重读取/打包路由/覆盖检查/tied embedding
python benchmarks/check_step57_model_logits.py       # 18 项：GQA+qk norm+RoPE 参考对照、HF 对照、三种切分
python benchmarks/check_step57_block_pool.py         # 28 项：空闲队列/引用计数/同 hash 多块/发布/失败原子性
python benchmarks/check_step57_prefix_cache.py       # 25 项：hash 链、发布边界、命中粒度、共享不覆写、开关一致
python benchmarks/check_step57_preemption.py         # 23 项：victim 选择、计划撤销与预算退回、恢复整表替换、端到端一致
python benchmarks/check_step57_sampler.py            # 29 项：混批分流、min_tokens 屏蔽、三种惩罚、top-k/p 边界、分布统计
python benchmarks/check_step57_stop_and_outputs.py   # 20 项：五条停止规则、min_tokens 两处职责、增量输出、seed 可复现
python benchmarks/check_step57_spec_metadata.py      # 13 项：两个坐标系的索引数学（vLLM 算例逐值）
python benchmarks/check_step57_rejection_sampler.py  # 18 项：greedy/random 验证、恢复分布、与上游/参考实现差分
python benchmarks/check_step57_spec_lifecycle.py     # 12 项：提议与采用的时序、K 裁剪、进度回退、抢占清草稿
python benchmarks/check_step57_draft_model.py        # 32 项：draft 规格校验、KV 独立、端到端、逻辑上限与请求生命周期（205 回归）
python benchmarks/check_step58_input_budget.py       # 10 项：双预算（token + input）与抢占返还
python benchmarks/check_step58_draft_inputs.py       # 11 项：第一遍输入逐值 + prefix 复用 + 与上游 kernel 差分
python benchmarks/check_step58_workspace.py          # 15 项：固定工作区（地址稳定/只读有效切片）与端到端
python benchmarks/check_step59_rejection.py          # 21 项：元数据口径、内核语义、上游/参考差分、分布、统计、profiler
python benchmarks/check_step60_ngram.py              # 17 项：CPU/GPU ngram 与上游逐值差分、显存历史增量、哨兵不出门、端到端
python benchmarks/check_step61_suffix.py             # 27 项：依赖接入、请求内/跨请求候选、容量与 FIFO、同 ID 重用、调用顺序、参数生效、端到端
python benchmarks/check_step62_custom_proposer.py    # 34 项：方法推断、接口、错误分类、Runner 接线、行为等价、demo 端到端
python benchmarks/check_step63_eagle_inputs.py       # 8 项：EAGLE 第一遍输入对齐（只测生产路径）
python benchmarks/check_step64_hidden_cache.py       # 18 项：cache-only 路径（物理 slot、协议、chunked/prefix/拒绝尾部/复用、端到端）
python benchmarks/check_step65_mtp.py                # 17 项：MTP 别名/加载（两派命名）/胶水与上游逐值对照/端到端反证
python benchmarks/check_step66_medusa.py             # 28 项：Medusa 配置/加载/行选择/端到端 + 上游逐值对照 + MLP 缺口证明
python benchmarks/check_step67_vocab_mapping.py      # 21 项：TLI 构造/映射/上游逐位差分/配置边界/异构词表集成
python -m pytest tests/step58 -q                     # 41 项：step58 的单测 + 集成（总纲要求的入口）
python -m pytest tests/step59 -q                     # 52 项：step59 的单测 + 集成（总纲要求的入口）
python -m pytest tests/step60 -q                     # 95 项：step60 的单测 + 集成（总纲要求的入口）
python -m pytest tests/step61 -q                     # 55 项：step61 的单测 + 集成（含与上游 proposer 的逐事件 trace 差分）
python -m pytest tests/step62 -q                     # 37 项：step62 的接口/错误分类/接线/行为等价
python -m pytest tests/step63 -q                     # 21 项：step63（EAGLE3 模型 + 端到端 + 与上游的数值对照）
python -m pytest tests/step64 -q                     # 13 项：step64（cache-only 层/提议者/Runner 接线与物理 slot 校验）
python -m pytest tests/step65 -q                     # 47 项：step65（MTP 配置/加载/前向/端到端 + 特征上传回归）
python -m pytest tests/step66 -q                     # 51 项：step66（Medusa 配置/模型/加载/行选择/端到端 + MLP 支持缺口）
python -m pytest tests/step67 -q                     # 23 项：step67（TLI 构造/映射/上游差分/配置边界/异构词表集成）
python -m pytest tests/step68 -q                     # 67 项：step68（logprobs 四种模式/投机行索引/约束/语法掩码/端到端）
```

## 与真实 vLLM 的对照（需要 GPU）

```bash
# WSL2 必须带这两个环境变量（否则 vLLM 引擎起不来，见 docs/step57_architecture.md §4）
VLLM_WSL2_ENABLE_PIN_MEMORY=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    python benchmarks/compare_step57_vllm.py            # logits / 增量位置 / 拒绝采样 / 端到端
VLLM_WSL2_ENABLE_PIN_MEMORY=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    python benchmarks/check_step57_vllm_boundaries.py   # 两条边界在真实 vLLM 上核实
python benchmarks/trace_step57.py                       # 一条覆盖五种事件的状态轨迹（CPU）
```

数值对照用的外部参照是 **transformers 的 Qwen3**（同一份 tiny 权重）与一份**按公式手写**的
注意力参考实现；权重与 tokenizer 用本机 `models/Qwen3-1.7B` 与**现场生成**的 tiny 模型
（`minivllm/testing/tiny_models.py`，固定 seed、写临时目录，不下载也不提交权重）。
