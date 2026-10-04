# 62 关对齐记录：自定义 Proposer 接入与配置分派边界

需求：[`062_自定义Proposer接入与配置分派边界.md`](../../vllm-omni/learning_notes/14_vllm_from_scratch/投机解码完整需求/062_自定义Proposer接入与配置分派边界.md)
基线：本机 `vllm==0.28.0` 文件快照。

本关解决什么痛点（一句话）：**换一个候选来源，不应该动 Engine**。在这之前，"用哪种投机"这一个事实散在 5 处
（config 白名单、config 的三个派生谓词、Runner 工厂、Runner 的调用参数分支、Scheduler 的 lookahead 特例），
每加一个候选算法都要复制一遍接线，而真正与算法无关的三方（Scheduler、KV manager、拒绝采样 verifier）却被牵连；
同时"方法是什么"还可能被 CLI 和 Runner **各判一次**，两边不一致就静默走错路径。

本关做完之后，候选算法变成一个可替换零件：插件只实现一个 `propose(...)`，返回 `list[list[int]]`
（可空、可变长、可全错），其余全不用改。实测（`benchmarks/check_step62_custom_proposer.py`）：
同一个 8-token prompt 下，**常规 / 全空 / 全错 / 变长**四类插件的 greedy 输出与"不开投机"逐 token 相同
（`{'a': [4,4,4,6,0,6], 'b': [10,4,4,4,4,4]}`），而草稿枚数分别是 15 / 0 / 27 / 10 —— 候选真的走过链路，
只是最终答案由 target 的接受判定决定。

## 1. 路径与职责对照

| 本项目 | 上游参考 | 状态 |
|---|---|---|
| `minivllm/spec_decode/custom_class_proposer.py::create_custom_proposer` | `vllm/v1/spec_decode/custom_class_proposer.py:12-73` | 逐条对齐（含 5 类错误与 `raise ... from e` 异常链、`getattr(..., None)` 取类、返回实例本身不加套壳） |
| `config.py::SpeculativeConfig.method: str \| None`、`model: str \| None` | `vllm/config/speculative.py:85-1523`（`model` / `method` / `SpeculativeMethod` 字面量在 L70-79） | 字段与语义照抄；`method` 默认由 `"ngram"` 改成 `None`（见 §3.6） |
| `config.py::SpeculativeConfig._resolve_method` | `speculative.py:741-756` | 逐条照抄：点号路径 → `custom_class`；`model in ("ngram","[ngram]")` → `ngram`；其余 → `draft_model` |
| `config.py::SpeculativeConfig._is_custom_proposer_path` | `speculative.py:721-730` | 逐条照抄：`http(s)://`/`file://` 前缀、含 `/`、分段非标识符 都不算类路径 |
| `config.py::SpeculativeConfig._resolve_custom_class` | `speculative.py:787-793` | 只校验 `model` 非空（点号/导入/构造/`propose` 都由工厂在启动期分类报错） |
| `gpu_model_runner.py::_build_proposer` 的 `custom_class` 分支 | `gpu_model_runner.py:645-651` | 同序（放在最前），构造参数只有 `VllmConfig` |
| `gpu_model_runner.py::_propose_draft_tokens` 的 `custom_class` 分支 | `gpu_model_runner.py:5151-5159` | 调用参数与上游一致：`propose(sampled_token_ids, num_tokens_no_spec, token_ids_cpu, slot_mappings=...)` |
| `_update_states` 里对 `remove_requests` 的"有才调"守卫 | 上游：对 drafter 的通用契约只有 `propose` | 见 §3.5 |
| `examples/custom_proposer.py` | 上游无对应文件（需求 §3.2 要求提供） | 教学示例：确定性、非性能算法 |
| `demo.py --spec-method/--spec-model/--spec-k` | 上游 CLI 用 `--speculative-config '{"method": "custom_class", "model": "..."}'` | 差异见 §3.7 |

## 2. 插件接口契约（写清楚给插件作者）

```python
class MyProposer:
    def __init__(self, vllm_config):        # 只能收 VllmConfig；不能再要别的
        ...                                  # 拿不到 Request / KVCacheManager / InputBatch
    def propose(self, sampled_token_ids, num_tokens_no_spec, token_ids_cpu, slot_mappings=None):
        ...
        return drafts                        # list[list[int]]，逐行对齐
```

- **行数必须等于批行数**：`sampled_token_ids[row]` 是本轮第 row 行采到的 token，空列表表示这行本轮没采样
  （中间 prefill 块）。返回也必须一行不少（`len(drafts) == len(sampled_token_ids)`）。
- **两个缓冲是定长的**：`num_tokens_no_spec` 长度 = `max_num_reqs`，`token_ids_cpu` 形状 =
  `[max_num_reqs, max_model_len]`；**只有前 `len(sampled_token_ids)` 行是这一轮的**。按
  `range(len(num_tokens_no_spec))` 遍历会读到上一轮的脏行（上游 vLLM 的 InputBatch 同样是定长缓冲）。
  某行的历史 = `token_ids_cpu[row, :num_tokens_no_spec[row]]`；某行的 prompt 长度本关不给（上游也不给）。
- **每行枚数随意**：0 枚（不投机）、1 枚、K 枚、不同行不同长度都行；**内容错了不影响正确性**——
  草稿只是候选，下一轮由 target 逐个验证，错的丢掉。
- **可选钩子**：`remove_requests(req_ids)` 可以没有（Runner 按"有才调"处理，见 §3.5）；
  `load_model()` 不会被调用（上游同款：`create_custom_proposer` 只构造 + 校验 `propose`）。
- **不能越权**：插件只拿到只读配置 + 三个 CPU 缓冲，无法改 Scheduler 的 `Request`、KV 映射或输出。

## 3. 临时差异（逐条）

1. **日志**：上游在成功加载后打 `logger.info("Loaded custom proposer class '%s' with
   num_speculative_tokens=%d")`；本仓库没有 logger 设施，改成注释 + `demo.py` 打印一行
   （`投机：method=... 提议者=...`）。不影响行为。
2. **缓冲类型**：上游 `InputBatch` 把 `token_ids_cpu`/`num_tokens_no_spec` 以 **int32 numpy 视图**暴露，
   本仓库是 **int64 torch 定长张量**（与 61 关记录的同一条差异）。插件按本仓库类型写；
   示例里演示 torch 用法（只读，不要求插件改）。dtype/后端不同不影响语义。
3. **`slot_mappings` 恒为 `None`**：本仓库还没有 `slot_mappings` 对象（69 关做 CUDA Graph / 编译时才需要）。
   参数**留着**是为了签名与上游一致——插件照上游签名写，将来不用改。
4. **新增行数校验（有意的偏离）**：上游不检查返回行数；本仓库在插件↔Runner 边界上按"协议违约立刻报错"
   处理（`RuntimeError: ... 必须按批行逐行返回 ...`）。理由：外部插件少给几行是**静默**的
   （那几条请求只是没草稿，看起来"能用"），报错比让人误以为写对了更省事。
   注意这**不是**在防错配：我们的草稿交接是按 `req_id` 做 `zip`（`DraftTokenIds`），
   即使不检查也不会把 A 的草稿记到 B 上——检查的目的是尽早暴露插件 bug。
5. **可选的 `remove_requests`**：上游对 `propose` 之外没有任何契约，因此 Runner 不能假定它存在。
   本仓库把 57E 的结束清理钩子改成"有才调"（`getattr(proposer, "remove_requests", None)`）。
   不守卫的后果是实测过的：请求结束那一轮会 `AttributeError`（61 关的 suffix 提议者有这个方法，
   62 关的示例插件没有）。
6. **`method` 默认值从 `"ngram"` 改为 `None`**：照上游（`method: str | None = None`），
   `__post_init__` 里一次性推断。本仓库**没有**上游那种"从 target 模型推断 draft 模型"的能力，
   所以 `method=None` 且 `model=None` 会推成 `draft_model`，随后在 `DraftModelProposer.__init__`
   里明确报错要求 `draft_model_config`（57E 就有的检查，不是静默降级）。
   `num_speculative_tokens` 仍是 `int`（`0` = 没设），与 `method` 的 `None` 语义分开。
7. **CLI 形式**：上游用 `--speculative-config '{"method": "custom_class", "model": "pkg.Class", ...}'`
   一个 JSON 参数；本仓库 demo 加三个平铺参数 `--spec-method/--spec-model/--spec-k`（本关只做教学入口，
   不引 JSON 配置解析）。**方法推断仍然只发生在 `SpeculativeConfig` 里**：只给 `--spec-model` 的点号路径时，
   它会被推成 `custom_class`（`demo.py` 不做第二次判断）。
8. **不做**：插件发现平台、新服务框架、给所有 proposer 套统一抽象基类（源码没承诺签名一致：
   ngram 多一个 `num_speculative_tokens`、ngram_gpu 收显存张量、draft_model 收 `TargetRows`）。
   **异步调度组合**：本仓库还没有异步调度（70 关），所以没有"需要拒绝的 async 组合"；
   70 关接入 `use_async_scheduling` 时要在这里补一条拒绝/禁用检查。

## 4. 验收对照（需求 §4）

| 需求条目 | 用例 | 实测 |
|---|---|---|
| 合法类实例就是 Runner 持有的对象 | `test_custom_proposer.py::test_runner_holds_the_plugin_instance`、`test_happy_path_returns_instance_directly`、`check_step62` B1/D1 | `type(runner.proposer) is RepeatLastTokenProposer`（无套壳） |
| 调用参数与冻结 Runner 分支一致 | `::test_call_args_match_frozen_runner_branch`、`check_step62` D2/D3/D5 | 3 个位置参数 + `slot_mappings=`；两个缓冲与 InputBatch **身份相同**；行数 == 调用当刻批行数；冻结源码片段含同名参数序列 |
| 无点号路径 / 模块不存在 / 类不存在 / 构造失败 / `propose=123` 分别触发预期错误 | `::test_error_*`（6 条）、`check_step62` C 段 | `ValueError` / `ImportError` / `AttributeError` / `RuntimeError` / `AttributeError`，且 `__cause__` 保留原始异常 |
| 空候选、全部错误候选、变长候选仍保持正确目标输出 | `::test_candidates_do_not_change_greedy_output`（4 个参数）、`check_step62` E 段 | 四类插件下 greedy 输出与非投机逐 token 相同；草稿枚数 15/0/27/10 |
| 插件没有获得 Scheduler Request/KVManager 的越权引用 | `::test_constructor_receives_only_vllm_config`、`check_step62` B2 | 构造参数只有 1 个 `VllmConfig`；它上面没有 `scheduler`/`kv_cache_manager`/`requests`/`input_batch`/`worker` |
| 未支持的方法/Runner 组合在启动阶段拒绝 | `::test_unknown_method_rejected_at_config_time`、`::test_custom_class_requires_model_path`、`check_step62` A3/A4 | 配置期 `ValueError`；不回退成 ngram |
| 方法推断只在配置期判一次 | `::test_runner_does_not_re_guess_the_method`、`check_step62` A5 | `_is_custom_proposer_path` 只出现在 `config.py`；Runner 只比较 `config.method` |
| 教学入口传入类路径并进入原有链路 | `check_step62` F1/F2（真跑 `demo.py`，1.7B 模型） | 日志：`method='custom_class' model='examples.custom_proposer.RepeatLastTokenProposer' K=3，提议者=RepeatLastTokenProposer` |
| 中间 prefill 行 | `::test_mid_prefill_row_is_skipped` | 预算 4 时同一请求分多轮 prefill：中间块那行给空候选 |

## 5. 已知限制 / 未做

- 示例插件是"重复最后一个 token"，**明确不是性能算法**：它的价值是让接口与错误分类可验证。
  真实收益要等 63 关起的 draft 架构（EAGLE3/MTP/…）或 61 关的 suffix decoding。
- `propose` 的 `slot_mappings` 参数目前恒为 `None`；CUDA Graph（69 关）与异步调度（70 关）
  接入后要回头补：那时插件的入参才真正完整。
- 本关没有跑真实权重的性能数字（不涉及吞吐目标）；`demo.py` 的 1.7B 用例只验证"能接进真实链路"。
