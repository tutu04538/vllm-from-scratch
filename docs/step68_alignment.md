# 68 关对齐记录：采样约束、Logprobs 与结构化输出的投机语义

需求：[`068_采样约束Logprobs与结构化输出的投机语义.md`](../../vllm-omni/learning_notes/14_vllm_from_scratch/投机解码完整需求/068_采样约束Logprobs与结构化输出的投机语义.md)
基线：本机 `vllm==0.28.0` 文件快照。执行路径：**V1**。

> **状态：已实现并实跑**（`tests/step68` **65 项** + `benchmarks/check_step68_logprobs.py` **23 项** +
> `benchmarks/check_step68_grammar.py` **30 项**全 PASS）。
> 与上游同层对象的逐值差分：`Sampler.forward`（4 种模式）、`RejectionSampler._get_logprobs_tensors`、
> `xgrammar` 的 `accept/validate/rollback/fill_bitmask` 全部对齐。
> 真实权重实测：Qwen3-1.7B 的 `raw_logprobs` 与 `transformers` **逐值相同**；JSON schema 约束下正常产出 JSON。
> 明确不做（照抄上游或本项目尚未接入）：见 §4 的三态矩阵。

## 0. 一句话：这一关解决什么

**投机让"每个位置该看哪份分布"变成了一个索引问题，而这个索引错了不会报错。**

    提问 1：用户要 logprobs（"模型对每个字有多大把握"），投机一次验证 K 枚草稿 →
            一条请求一轮要交付 0 ~ K+1 个位置，每个位置该读**哪一行**的 logits？
    提问 2：用户要结构化输出（"只能输出合法 JSON"），K 个候选位**同时待定**、
            各自面对不同的"假设历史" → 掩码该按哪个状态填？

答错这两问的后果不是崩溃，而是**悄悄给出错的数/坏的 JSON**：

```
frequency_penalty=0.5，token「kv」在已提交历史里 0 次、在本轮草稿里 2 次
    正确（bonus 行历史含全部草稿）：logit(kv) -= 0.5 × 2 = 1.0
    漏算草稿：                    logit(kv) -= 0            ← 差 1.0 个 logit
差 1.0 足以翻转 argmax/采样结果 → p 与用户看到的 logprobs 都不再是那个分布
（repetition_penalty 是"出现过就缩放一次"、presence 也是"出现过减一次"，
 所以**只有 frequency_penalty 是按次数**的——想举数字例子时别用 repetition 乘次数）
```

结构化输出这边更直白：需求 §1 的小例子——要求输出 `{"x":整数}`，候选在冒号前提出字母。
"生成之后删掉字母"会（a）改变分布、（b）让 grammar 的 FSM 走错位置，后面每一格都在用错误的状态判断。

这一关就是把这两件事在**投机路径**上做对：logprobs 按"第 j 个位置读第 j 行"索引、按同一张
valid_mask 裁掉尾巴；语法掩码按候选**逐步试走**再回滚，只有真正提交的 token 才推进 FSM。

## 1. 代码映射

| 本项目 | 上游参考 | 说明 |
|---|---|---|
| `minivllm/logprobs.py::Logprob` / `append_logprobs_for_next_position` | `vllm/logprobs.py`（同名） | 用户可见容器：`{token_id: Logprob(logprob, rank, decoded_token)}` |
| `outputs.py::LogprobsTensors` / `LogprobsLists` | `v1/outputs.py:94` / `:30` | 执行侧（张量）/ 跨边界（numpy）两层容器 + `filter`/`cat`/`slice_request` |
| `sample/sampler.py::forward`（4 种模式） | `v1/sample/sampler.py:73-142` | `raw_logprobs` / `raw_logits` / `processed_logprobs` / `processed_logits` |
| `sample/sampler.py::gather_logprobs` / `batched_count_greater_than` | `:306` / `sample/ops/logprobs.py` | 第 0 列 = 实际采到的 token，后面 top-k；rank = "≥ 选中 logprob 的个数" |
| `sample/sampler.py::apply_min_p` | `sample/logits_processor/builtin.py::MinPLogitsProcessor.apply` | 同一算式、同一位置（温度之后、top-k 之前），本项目内联（没有插件框架） |
| `sample/rejection_sampler.py::forward`（bonus 用 `logprobs_mode_override`） | `v1/sample/rejection_sampler.py:130-176` | bonus 行强制交回 **logits**，与候选行拼一起再算 |
| `sample/rejection_sampler.py::_get_logprobs_tensors` | `:190-240` | `final_logits` 铺候选行 + bonus 行；`cu_num_sampled_tokens` 给每请求起点 |
| `sample/rejection_sampler.py::parse_output`（logprobs 过滤） | `:243-285` | token 与 logprobs **同一张 valid_mask** + `cu_num_tokens` |
| `sample/metadata.py` 的 `max_num_logprobs` / `logprobs_mode` / `min_p` | `v1/sample/metadata.py`（同名） | 整批宽度取逐请求最大值，逐请求裁剪在输出处理 |
| `engine/logprobs.py::LogprobsProcessor` | `v1/engine/logprobs.py:22` | 逐位置累计 + `cumulative_logprob` + UTF-8 解码修正 |
| `structured_output/backend_types.py` | `v1/structured_output/backend_types.py` | `StructuredOutputGrammar` / `StructuredOutputBackend` 两个 ABC |
| `structured_output/backend_xgrammar.py` | `v1/structured_output/backend_xgrammar.py` | 本仓库**唯一**接入的后端（选本机已有 backend，不重造 JSON parser） |
| `structured_output/utils.py` | `v1/structured_output/utils.py` | Lark→EBNF、`choice_as_grammar`、正则超时、`apply_grammar_bitmask` |
| `structured_output/__init__.py::StructuredOutputManager` | `v1/structured_output/__init__.py:35` | `grammar_init` / `grammar_bitmask` / `should_fill_bitmask` / `should_advance` |
| `structured_output/request.py::StructuredOutputRequest` | `v1/structured_output/request.py` | 请求级 FSM 状态（挂在 `Request` 上） |
| `core/sched/output.py::GrammarOutput` | `v1/core/sched/output.py:287` | 掩码 + "按这个顺序排列"的请求 ID 列表 |
| `core/sched/scheduler.py::get_grammar_bitmask` / `update_draft_token_ids` / `update_from_output` | `:1709` / `:2237` / `:1893` | 挑请求、预筛草稿、提交后推进 FSM、按提交数切 logprobs |
| `worker/gpu_model_runner.py::_apply_grammar_bitmask` / `_logit_row_of_req` / `_logprobs_by_request` / `_concat_logprobs_in_req_order` | `v1/worker/gpu/structured_outputs.py` + `v1/structured_output/utils.py::apply_grammar_bitmask` | 打掩码 + 行映射 + 按请求重排 logprobs |
| `tokenizer_utils.py::cached_tokenizer_from_config` | `vllm/tokenizers/__init__.py` | 唯一 tokenizer 入口（67 关 TLI 与 68 关 xgrammar 共用；**模块名不叫 `tokenizers`**，见 §3.7） |
| `testing/tiny_models.py::tiny_structured_dir` / `write_plain_wordlevel_tokenizer` | 无（测试用） | tiny 模型 + JSON 记号词表（`{`/`}`/`"`/`x`/`:`/数字），让"只能输出 `{"x":1}`"可端到端验证 |

## 2. 数据流与设计要点（改动时不要破坏）

```text
调度      Scheduler.schedule()                → SchedulerOutput（has_structured_output_requests）
掩码      Scheduler.get_grammar_bitmask()     → GrammarOutput（每请求 1+K 行掩码）
          └ 管理器对每个候选位：先填掩码 → 试走 accept_tokens → 全部填完 rollback
执行      Runner._apply_grammar_bitmask()     → 原地把非法 token 打成 -inf
采样      Sampler / RejectionSampler          → token + logprobs（4 种模式）
提交      Scheduler.update_from_output()      → 只有**真正提交**的 token 才 accept_tokens（永久推进）
          └ logprobs.slice_request(req_index, len(new_token_ids))（停止 token 截断处同步截断）
交付      OutputProcessor + LogprobsProcessor → RequestOutput.logprobs / cumulative_logprob
```

1. **行索引是这一关的核心**：掩码与 logprobs 都按"请求 → 该请求的第一行"展开；请求在
   `input_batch` 里的行号 ≠ logits 里的行号（本仓库非投机路径只为**要采样的行**算 logits）。
   实测：把掩码打到别的行上**不报错**，只会让模型在该填数字的地方写出字母。
2. **同一张 valid_mask**：`parse_output` 用一份 mask 同时裁 token 与 logprobs，所以"被拒候选位
   多算的那一份"不可能漏给用户（068 §3.5）。
3. **试走与推进分离**：掩码阶段只 `accept_tokens` + `rollback`；永久推进只在 Scheduler 提交之后
   （068 §2：不要让 proposer 永久推进 grammar）。
4. **掩码在采样器之前打**：所以 `raw_*` 与 `processed_*` **两份都带掩码**——不存在"交付的 logprobs
   认为非法 token 还有概率"这种自相矛盾。
5. **草稿先过 `validate_tokens`**：Scheduler 收下草稿时就裁掉不合语法的尾巴（试走、不推进）；
   不裁的话掩码阶段会因为"草稿没被预筛过"直接断言失败（上游同款）。
6. **`-1`（padding 占位）语义**：掩码在判 `-1` **之前**填（所以那一位仍有掩码），判完之后
   `apply_bitmask=False`（之后不再填、不再推进），整段结束统一 rollback。逐行对齐上游。
7. **交出去的 logprobs 与 token 逐位置对齐**：`RequestOutput.logprobs[i]` 对应 `token_ids[i]`；
   `cumulative_logprob` 加的是**交付的那一份**（raw 或 processed，由 `ModelConfig.logprobs_mode` 决定）。
8. **停止 token 之后**：grammar 走完（`}`）会只剩 eos 可选；不设 `stop_token_ids` 时 eos 全程被掩掉，
   请求会被 `max_tokens` 截断。

## 3. 与上游的差异账本（逐条）

1. **tokenizer 懒加载**：上游 `StructuredOutputManager.__init__` 里就加载 tokenizer；本仓库在**第一次
   `grammar_init`** 时才加载（按目录缓存一份，与 67 关 TLI 共用）。理由：tiny 模型目录常常没有
   tokenizer 文件，而没有结构化请求时读它纯属浪费；语义（用哪份 tokenizer 编译 grammar）不变。
2. **同步编译**：上游默认把 grammar 编译丢进线程池（`Future`），Scheduler 轮询
   `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR`；本仓库**同步编译**，编译失败在 `add_request` 处直接抛。
   这是上游在 `distributed_executor_backend == "external_launcher"` 下用的那条同步分支；
   **异步编译属"本项目尚未接入"**（三态矩阵）。副作用（有意）：语法写错的请求不会先排队再失败。
3. **掩码串行填充**：上游在"批 > 128 且无投机"时用线程池并行填掩码；本仓库一律串行（同一批结果
   逐位相同，只是 CPU 并行度；本项目批很小）。
4. **`_get_logprobs_tensors` 与"全词表"**：`logprobs=-1` 在投机下**上游运行期报错**
   （把 -1 直接交给 `torch.topk` → `RuntimeError: selected index k out of range`，实测见 §5）。
   本仓库在进 topk 之前明确拒绝（`NotImplementedError`，消息里说明"上游同样不支持"）——
   失败点更靠前、原因更直白，但**判定一致**：这条组合不支持。
5. **非投机的 `logprobs=-1`**：采样器交出整份分布（上游同款），输出处理阶段按 top-k 那一列解释 →
   实测最终交付为**空**且 `cumulative_logprob` 保持 0（上游 `LogprobsProcessor` 的 zip 在空数组上
   截断，差分测试逐值复现）。本仓库照抄，不在这里"顺手修好"（上游行为疑似 bug，记在 §5）。
6. **上游疑似 bug：`processed_logprobs` 模式下 bonus 位的数字来自 raw**。上游
   `RejectionSampler.forward` 给 bonus 行传 `logprobs_mode_override="processed_logits"`，但
   `Sampler.sample()` 用的是**引擎级模式**，于是 `processed_logprobs` 模式下 bonus 采样器交回的
   其实是 logprobs，`_get_logprobs_tensors` 又对它做了一次 `log_softmax` → bonus 位置的数字退化成
   "**raw** 那一份的 logprob"（惩罚/温度都没算进去）。逐值证据：
   `tests/step68/test_spec_logprobs.py::test_processed_mode_bonus_row_comes_from_raw_upstream_quirk`
   （手算 processed 与实测差 > 1.0，实测值 == 手算 raw）。本仓库**照抄**这个行为（"不静默改写规范"），
   并在差分用例里把两边钉住。
7. **模块名 `tokenizer_utils` 而不是 `tokenizers`**：`python minivllm/demo.py` 会把 `minivllm/` 放进
   `sys.path[0]`，那样 `from tokenizers import ...`（transformers 内部要用的第三方包）会解析到本包的
   文件并 ImportError（实测）。上游没这个问题（`vllm/tokenizers/` 在包里、入口脚本目录是仓库根）。
8. **`apply_grammar_bitmask` 的签名**：上游自己从 `InputBatch` + `SchedulerOutput` 算"请求 → logits 行"；
   本仓库的 Runner 算好 `logit_index_of_req` 传进去（非投机路径只为要采样的行算 logits，只有 Runner
   知道行号）。掩码顺序与"每请求 1+K 行"的算术与上游逐行相同。
9. **`batched_count_greater_than` 不用 `torch.compile`**：上游包了 `@torch.compile`；本仓库直接写
   `(x >= values).sum(-1)`（同一算式、逐位相同的结果，差的是编译开销）。CUDA Graph 属 69 关。
10. **logprobs 容器只实现 list 那一支**：上游还有 `FlatLogprobs`（由 `SamplingParams.flat_logprobs`
    选择）；本仓库只实现 list 支、**不提供** `flat_logprobs` 开关（没有开关就没有"收下参数却按另一种
    结构返回"的静默差异）。
11. **没有 prompt logprobs / `logprob_token_ids`**：两者在请求期明确拒绝（§4 矩阵），
    `LogprobsProcessor` 因此只保留 `_update_sample_logprobs` 那一半。
12. **没有思考模式（reasoning parser）**：`should_fill_bitmask` 恒为 `True`、`should_advance` 只看
    "有没有结构化输出"——正是上游在**没有 reasoner** 时的返回值。
13. **`disable_any_whitespace=True` 的实测行为**（xgrammar 0.2.3）：JSON 里 `:` 之后**必须**有一个
    空格才允许数字（`{"x":1}` 被拒、`{"x": 1}` 通过）；默认（`False` = `any_whitespace=True`）两种都
    允许。配置项照抄上游，行为由后端决定，这里只记录实测。

## 4. 三态矩阵（支持 / 上游不支持 / 本项目尚未接入）

| 能力 | 上游实现路径 | 本项目状态 | 未接入时的表现 |
|---|---|---|---|
| temperature、top-k、top-p | `apply_sampling_constraints`（候选行按请求展开） | ✅ 支持（59 关起） | —— |
| `min_p` | `MinPLogitsProcessor.apply`（argmax 不变处理器） | ✅ 本关接入（同一算式/位置，内联） | —— |
| repetition / presence / frequency penalty | `RejectionSampler.apply_penalties`（按"假设历史"逐行） | ✅ 支持（59 关起） | —— |
| `min_tokens` | `MinTokensLogitsProcessor.apply_with_spec_decode`（屏蔽前 n_mask 行） | ✅ 支持（59 关起） | —— |
| logprobs（4 种模式、投机下按行索引） | `Sampler.forward` + `_get_logprobs_tensors` | ✅ 本关 | —— |
| 结构化输出：JSON schema / json_object / regex / grammar / **choice** / structural tag | `XgrammarBackend`（choice 在请求期改写成 EBNF） | ✅ 本关（只接 xgrammar） | 其它 backend 请求期 `NotImplementedError` |
| 语法掩码：试走 / 回滚 / 只有提交才推进 | `StructuredOutputManager.grammar_bitmask` + Scheduler `accept_tokens` | ✅ 本关 | —— |
| `logprobs=-1`（全词表）+ **投机** | 上游把 -1 交给 `torch.topk` → 运行期 `RuntimeError` | ⛔ **上游不支持**（实测） | 本仓库提前 `NotImplementedError`（同一判定，更早失败） |
| `logprobs=-1` + **非投机** | 采样器交全词表，输出处理 zip 截断成空（实测） | ⚠️ 上游行为疑似 bug，本仓库照抄 | 交付为空（`cumulative_logprob` 保持 0） |
| `processed_logprobs` 的 bonus 位 | 上游 double-log_softmax（§3.6） | ⚠️ 上游疑似 bug，本仓库照抄 + 差分钉住 | 数字等于 raw 那一份 |
| `allowed_token_ids`（白名单） | `allowed_token_ids_mask` | ❌ 本项目尚未接入 | `SamplingParams` 构造时 `NotImplementedError` |
| `bad_words` | `apply_bad_words_with_drafts` | ❌ 本项目尚未接入 | 同上 |
| `logit_bias` | `LogitBiasLogitsProcessor` | ❌ 本项目尚未接入 | 同上 |
| `thinking_token_budget` | `thinking_budget_state_holder` | ❌ 本项目尚未接入 | 同上 |
| `logprob_token_ids` | `Sampler.gather_specific_token_logprobs` | ❌ 本项目尚未接入 | 同上 |
| `prompt_logprobs` | 上游 prefill 逐行留 logprobs | ❌ 本项目尚未接入 | 同上 |
| 自定义 logits processor 插件 | `logitsprocs` 注册表 | ❌ 上游有注册表、本项目无（故**没有**对应字段可传） | —— |
| 异步编译 grammar | `grammar_init` + `Future` + `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR` | ❌ 本项目尚未接入（§3.2） | 同步编译：失败在 `add_request` 抛出 |
| 掩码并行填充（>128 批、无投机） | `ThreadPoolExecutor` | ❌ 本项目尚未接入（§3.3） | 串行填充，结果相同 |
| 思考模式（reasoning parser / `enable_in_reasoning`） | `_get_reasoner` / `trim_reasoning_for_advance` | ❌ 本项目尚未接入 | 无该字段；`should_fill_bitmask` 恒 True |
| 其它 backend（guidance / outlines / lm-format-enforcer） | 各自 `*Backend` | ❌ 本项目尚未接入 | 配置/请求期 `NotImplementedError` |
| `flat_logprobs`（摊平容器） | `FlatLogprobs` | ❌ 本项目尚未接入 | 无该字段；只有 list 支 |

## 5. 上游行为证据（可复跑）

```bash
# ① logprobs=-1 + 投机：上游失败路径的异常类型
python -c "import torch; torch.topk(torch.randn(1,4), -1, dim=-1)"
#    → RuntimeError: selected index k out of range

# ② 非投机 logprobs=-1：上游 LogprobsProcessor 把全词表那一份截成空
#    见 tests/step68/test_spec_logprobs.py::test_full_vocab_mode_shape_matches_upstream
#    （采样器形状与上游逐值相同；输出处理 zip 在 (0,) 上停住 → logprobs=[]、cumulative=0.0）

# ③ processed_logprobs 的 bonus 位 = raw 那一份（手算差 > 1.0）
#    见 test_processed_mode_bonus_row_comes_from_raw_upstream_quirk

# ④ 真实权重上 raw_logprobs 与 transformers 逐值相同（Qwen3-1.7B、prompt 末位）
#    HF:   [(151667, -0.0001, '<think>'), (151668, -11.2501, '</think>'), ...]
#    本仓库: [(151667, -0.0001, '<think>'), (151668, -11.2501, '</think>'), ...]
```

## 6. 实测

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/step68 -q                     # 65 passed
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmarks/check_step68_logprobs.py           # 23 项 PASS / 0 FAIL
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmarks/check_step68_grammar.py            # 30 项 PASS / 0 FAIL
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/step58 ... tests/step68 -q    # 503 passed
# 真实权重（Qwen3-1.7B、CUDA）：JSON schema + logprobs
python minivllm/demo.py --device cuda --max-new-tokens 20 --logprobs 3 --json-schema \
  '{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},
    "required":["name","age"],"additionalProperties":false}' "给一个虚构人物：名字和年龄，用 JSON"
```

设备 `cuda:0`（RTX 5090 Laptop / WSL2），torch 2.13.0+cu130，xgrammar 0.2.3。关键实测：

| 项 | 实测 |
|---|---|
| 4 种 logprobs 模式 vs 上游 `Sampler.forward` | token、列序、rank 相同；`logprobs` `allclose(atol=1e-6)` |
| 投机 `_get_logprobs_tensors` vs 上游 | 逐值相同（含被拒候选位与 bonus 位） |
| "第 j 个位置读第 j 行" | 手算 `log_softmax(第 j 行)[token]` 全中；读错行差 > 1.0（反证用例） |
| 真实权重 raw_logprobs vs transformers | 5 个位置逐值相同（`-0.0001 / -11.2501 / -12.3751 / -13.2501 / -13.2501`） |
| 语法掩码端到端（tiny + `const` schema） | 随机初始化模型也吐出 `{"x": 1}`（无约束时吐不出合法 JSON，反证区分度） |
| 投机 + 语法（ngram K=3、CUDA） | 16 轮里 4 轮带草稿；掩码行数 == Σ(1+K)；输出与每轮草稿都能被语法重放接受 |
| 三态矩阵里的"未接入项" | 6 个字段 + 3 个后端在请求期/配置期明确拒绝（各有用例） |

## 7. 遗留与已知未做

1. **异步 grammar 编译**（`WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR`）与**掩码并行填充**未接入（§3.2/§3.3）。
2. **思考模式**（reasoning parser、`enable_in_reasoning`、`trim_reasoning_for_advance`）未接入；
   接入前"结构化输出 + 思考"这条组合不存在（字段都没有，不会静默忽略）。
3. **白名单 / bad words / logit_bias / 思考预算 / logprob_token_ids / prompt_logprobs** 六个字段只有
   "请求期拒绝"，实现它们是独立的工作量（bad words 在投机下要按草稿前缀逐行判）。
4. **上游两处疑似 bug**（`logprobs=-1` 的输出处理截断、`processed_logprobs` 的 bonus 位）本仓库照抄并
   钉住；等上游修复后要重新对照（修复会改变"对齐基线"）。
5. **`-1` 占位草稿**在生产路径上目前不会出现（60 关的 GPU 提议者已在交接处裁掉无效草稿），
   所以 §2.6 的语义只有单元测试覆盖；异步调度（70 关）会真正用到它。
6. **异步调度 / V2（70、73 关）**：本关的用例是同步 V1 的；需求 §4 要求那两关回归同一批测试。
7. **未做性能测量**：`batched_count_greater_than` 没编译、掩码串行填充、`_concat_logprobs_in_req_order`
   在 CPU 上做 numpy 拼接——都只记录口径，不设虚构加速倍数（84 关才做统一性能矩阵）。
