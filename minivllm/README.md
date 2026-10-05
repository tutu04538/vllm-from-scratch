# minivllm：对齐 vLLM V1 架构的文本生成子集

这是仓库里**唯一的实现**（原 `step57/`，含 204/205 两轮验收修复、第五十八关的 draft 输入/双预算对齐、第五十九关的 GPU 批量拒绝采样、第六十关的 CPU/GPU ngram 提议、第六十一关的 Suffix Decoding、第六十二关的自定义 Proposer 接入）。旧的 `stepNN/` 代码目录已经
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
[`docs/step62_alignment.md`](../docs/step62_alignment.md)（自定义 Proposer 与分派边界）。

| 层 | 文件 | 对应 vLLM |
|---|---|---|
| 配置 | `config.py` / `sampling_params.py` | `vllm/config/*`、`vllm/sampling_params.py` |
| 请求状态 | `request.py` | `v1/request.py` |
| 协议 | `outputs.py`、`core/sched/output.py` | `v1/engine/__init__.py`、`v1/outputs.py`、`v1/core/sched/output.py` |
| KV 控制面 | `core/kv_cache_manager.py` → `core/kv_cache_coordinator.py` → `core/single_type_kv_cache_manager.py` → `core/block_pool.py` → `core/kv_cache_utils.py` | `v1/core/` 下同名文件（这条链一层一个问题） |
| 调度 | `core/sched/{scheduler,request_queue,utils}.py` | `v1/core/sched/*` |
| 编排 | `engine/{core,core_client,output_processor,llm_engine}.py` | `v1/engine/*` |
| 执行部署 | `executor/uniproc_executor.py` | `v1/executor/uniproc_executor.py` |
| 执行端 | `worker/worker.py` | `v1/worker/gpu_worker.py` |
| 一轮怎么跑 | `worker/gpu_model_runner.py` | `v1/worker/gpu_model_runner.py` |
| 批状态缓冲 | `worker/gpu_input_batch.py`、`worker/block_table.py` | 同名文件 |
| 权重加载 | `model_loader/*` | `model_executor/model_loader/*` + `models/utils.py` |
| 模型 | `models/qwen3.py` | `model_executor/models/{qwen2,qwen3}.py` |
| 层与注意力 | `layers/*`、`attention/*` | `model_executor/layers/*`、`attention/*` |
| 采样 | `sample/{metadata,sampler}.py`、`sample/ops/*` | `v1/sample/{metadata,sampler}.py`、`v1/sample/ops/*` |
| 投机验证 | `sample/rejection_sampler.py`（Triton 批量内核） | `v1/sample/rejection_sampler.py` |
| 投机提议 | `spec_decode/{metadata,metrics,ngram_proposer,ngram_proposer_gpu,draft_model,suffix_decoding,custom_class_proposer}.py` | `v1/spec_decode/*` |
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
> （`combine_hidden_states` 逐位相同、logits max|Δ|=7.0e-4）。剩余项（真实 checkpoint 完整生成 →
> 67 关、draft 解码层逐值对照 → 69/70、M-RoPE）见 [`docs/step63_alignment.md`](../docs/step63_alignment.md) §5。

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

## 明确不做

EAGLE/MTP、异步与多进程、指标、logprobs、KV 连接器、多 KV group。
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
python -m pytest tests/step58 -q                     # 41 项：step58 的单测 + 集成（总纲要求的入口）
python -m pytest tests/step59 -q                     # 52 项：step59 的单测 + 集成（总纲要求的入口）
python -m pytest tests/step60 -q                     # 95 项：step60 的单测 + 集成（总纲要求的入口）
python -m pytest tests/step61 -q                     # 55 项：step61 的单测 + 集成（含与上游 proposer 的逐事件 trace 差分）
python -m pytest tests/step62 -q                     # 37 项：step62 的接口/错误分类/接线/行为等价
python -m pytest tests/step63 -q                     # 21 项：step63（EAGLE3 模型 + 端到端 + 与上游的数值对照）
python -m pytest tests/step64 -q                     # 13 项：step64（cache-only 层/提议者/Runner 接线与物理 slot 校验）
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
