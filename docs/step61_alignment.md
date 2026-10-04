# 61 关对齐记录：Suffix Decoding 的请求内与跨请求历史

需求：[`061_SuffixDecoding的请求内与跨请求历史.md`](../../vllm-omni/learning_notes/14_vllm_from_scratch/投机解码完整需求/061_SuffixDecoding的请求内与跨请求历史.md)
基线：本机 `vllm==0.28.0` 文件快照；依赖：`arctic_inference==0.3.0`（见 §2 偏差说明与 `docs/step61_dependencies.json`）。

本关解决什么痛点（一句话）：**ngram 只看当前这条请求自己的历史**——A 已经证明过 "1 2 3 后面跟 4 5"，
B 的结尾也是 "1 2 3" 却拿不到任何候选；suffix decoding 用一棵**跨请求全局树**把 A 的输出也纳入匹配，
同时按"匹配到的后缀有多长、这个分支历史上出现过多少次"给每条请求**不同长度**的候选。

实测证据（同一份历史 `prompt [7,8,1,2,3] + 本轮采样 [4]`）：

| 提议者 | 输入视野 | 候选 |
|---|---|---|
| `NgramProposer`（60 关，min=max=3 或 5/5） | 只有 B 自己的历史 | `[]` |
| `SuffixDecodingProposer`（61 关，全局树里已有 A 的响应 `[4,5]`） | B 的历史 + A 的全局序列 | `[5]` |
| 同上，但 `suffix_decoding_max_cached_requests=0`（关掉全局树） | 只有 B 自己的历史 | `[]` |

→ `tests/step61/test_suffix_config.py::test_ngram_cannot_reuse_another_requests_pattern` /
`::test_suffix_decoding_reuses_another_requests_pattern`、`benchmarks/check_step61_suffix.py` 项 C1/C2/C3。

## 1. 路径与职责对照

| 本项目 | 上游参考 | 状态 |
|---|---|---|
| `minivllm/spec_decode/suffix_decoding.py::SuffixDecodingProposer` | `vllm/v1/spec_decode/suffix_decoding.py:9-103` | 逐行对齐（`__init__` / `propose` / 空 `load_model`），另加 `propose_drafts`、`remove_requests` 两个本仓库协议入口（§3） |
| `minivllm/config.py::SpeculativeConfig.method="suffix"` + 4 个 `suffix_decoding_*` 字段 | `vllm/config/speculative.py:195-210` | 字段名/默认值/注释照抄（24 / 10000 / 1.0 / 0.1） |
| `minivllm/config.py::SpeculativeConfig._resolve_suffix_decoding` | `vllm/config/speculative.py:1146-1181::_validate_suffix_decoding` | 缺包 ImportError + 4 条取值校验逐条照抄（文案相同）；`num_speculative_tokens` 的"没设"判定见 §3 |
| `minivllm/config.py::has_arctic_inference` | `vllm/utils/import_utils.py:542` | 同义（`importlib.util.find_spec`） |
| `minivllm/worker/gpu_model_runner.py::_build_proposer` / `_propose_draft_tokens` / `_update_states→remove_requests` | Runner 的分派与生命周期钩子 | 接入（suffix 分支用 `propose_drafts(..., sampled_by_row=...)`） |
| 后缀树 / 匹配 / 淘汰 | `arctic_inference.suffix_decoding.SuffixDecodingCache`（**外部包，未自研**） | 直接使用，见 `docs/step61_dependencies.json` |

## 2. 依赖接入与版本偏差（需求 §2）

- 源码钉 `arctic-inference==0.1.1`；0.1.1 的**构建系统**把 torch 钉成 `torch == 2.7.0`（`pyproject.toml:11`），
  本机 torch 是 `2.13.0+cu130` → 装不上。改装 **0.3.0**（构建系统 `torch>=2.10.0`），用本机 torch 现场编译 `_C` 扩展。
- **等价性证据（逐文件 diff 两个 sdist）**：`suffix_decoding/{__init__,cache,simulator}.py`、
  `csrc/suffix_decoding/{suffix_tree.cc,suffix_tree.h,bindings.cc,int32_map.h,CMakeLists.txt}` **全部逐字节相同**；
  `cache.py` 两版 sha256 都是 `91c48e6c…`。差异只在构建依赖与包内其它模块（不在本关路径上）。
- sdist sha256 `7f22e3e1…` 与 PyPI 官方 digest 一致；安装命令、扩展 `.so` 的 sha256、用到的 API 面全部记在
  `docs/step61_dependencies.json`。
- 未安装时：`SpeculativeConfig(method="suffix")` 在**配置构造期**抛 `ImportError`（附 `pip install arctic-inference==0.1.1`
  与本次偏差说明），运行期不会退回别的提议者。

## 3. 临时差异（逐条）

1. **`num_speculative_tokens` 的"没设"**：上游字段是 `int | None`（`None` → 取 `suffix_decoding_max_tree_depth` 并打 warning）；
   本仓库是 `int`（`0` = 没设），所以 `0` 当"没设"处理，默认取树深。**显式传 `0` 想"一枚都不猜"时请直接别开 suffix**。
2. **CPU 缓冲类型转换**：上游把 vLLM `InputBatch` 的 **int32 numpy 视图**直接交给依赖包（零拷贝重载）；
   本仓库 `InputBatch` 的 `token_ids_cpu`/`num_tokens_no_spec`/`num_prompt_tokens` 是 **int64 torch**，
   故在 `_as_int32()` 里转成"1 维、C 连续、int32"再传。**只改容器类型，不改任何算法与候选**（差分测试逐步比对）。
3. **多两个协议入口**：
   - `propose_drafts(rows, all_token_ids, input_batch, sampled_by_row=...)`：把 Runner 的"逐行事实"摊成上游
     `propose` 要的 `sampled_token_ids`（`{批行: 采样结果}` → 逐行 list，未采样行给空）。
     `input_batch` 必须给：不像 ngram 能从 `all_token_ids` 现场拼缓冲（拼出来的 `num_prompt_tokens` 会与真实
     prompt 边界不一致 → 改变建树范围 → 改变候选，属于静默偏离），缺失时显式 `ValueError`。
   - `remove_requests(req_ids)`：Runner 在请求**结束**时的统一钩子（与 draft_model/ngram 提议者同款）。
     它做的是"对仍在活跃集合里的 ID 调 `stop_request`"，对已停/不活跃的 ID 是安全空操作。
4. **请求结束的清理路径**：上游只有 `propose` 末尾那次 `active_requests - req_id_to_index.keys()` 扫描；
   本仓库额外在 Runner 侧收 `finished_req_ids`（`remove_requests`）。原因：批为空时 Runner 不会调提议者
   （`_propose_draft_tokens` 提前返回），最后一条请求结束后扫描就跑不到，同 ID 复用时依赖包会抛
   "already active"。两条路径的**等价性有专门用例**：`test_suffix_lifecycle.py::test_remove_requests_matches_empty_batch_sweep`。
   §3 第 6 步的语义（"离开 input batch 的活跃请求按源码 stop"）原样保留，且**不等于** Scheduler 的 FINISHED。
5. **日志**：上游在补默认 K 时打 warning；本仓库不引 logger，改为写进本文件（§3.1）。
6. **不做的**：tree attention（需求 §5 明确不引入）、本关吞吐评测（需求 §5）、
   `SuffixDecodingDraft` 的 `parents/probs`（本关只取 `token_ids`；跨请求概率口径不属于 59 关的 q 对齐）、
   `use_tree_spec` 分支。

## 4. 依赖包的匹配语义（实测 + 读 `csrc/suffix_decoding/suffix_tree.cc:593-622`，写清楚免得误读候选）

- `speculate(context)` 让 `match_len` 从 **1 递增**地拿 context 的**后缀**去树里找，某个长度找不到就**停**
  （后缀树性质：更长的也一定找不到）→ 最终用的是"能匹配上的最长后缀"，但**全长 context 永不参与匹配**
  （循环上界 `match_len < context.size()`），最后那个 token 只当**锚点**。
- 每个匹配的得分是路径概率之和：`_speculate_path` 逐 token 乘"子节点频次 / 父节点频次"，
  低于 `min_token_prob` 就停；token 数上限 `min(K, match_len * factor + offset)`。
- 所以"上下文正好等于历史末尾"时候选常为空（末尾那次出现没有后继）；要出候选，**最近几个 token 必须在更早的
  位置带着同样的后继出现过**——这正是重复性 prompt / 跨请求共享模式能命中的原因，也是本关测试用例的构造依据。
- `start_request` 自己也会 evict 已缓存的同 ID 响应；上游 proposer 里那句显式 `evict_cached_response` 是**防御性重复**
  （只在 `max_cached_requests != 0` 时才有意义），本关**照抄保留**并在调用顺序用例里断言 evict 先于 start。

## 5. 验收对照（需求 §4）

| 需求条目 | 用例 | 实测 |
|---|---|---|
| 第一次请求建立 prompt 树；输出追加仅一次；第二条相似请求命中全局历史 | `test_suffix_lifecycle.py::test_single_request_five_steps_matches_upstream`、`test_global_tree_gives_cross_request_drafts`、`check_step61` B1-B5/C1 | 五步候选 `[[2,3,4,1]]→[[3,4,1,2]]→[[4,1,2,3]]→[[1,2,3,4]]→[[2,3,4,1,2]]`；B 命中 A 的 `[5]` |
| `max_cached_requests=0`、容量溢出、FIFO 淘汰、同 ID 重用 | `::test_disabling_global_tree_kills_cross_request_drafts_but_keeps_prompt_tree`、`::test_fifo_eviction_by_capacity`、`::test_capacity_paths_match_upstream`、`::test_id_reuse_evicts_previous_response_from_global_tree` | 容量 2 → C 拿不到 `[5]`（A 被 FIFO 淘汰）、容量 3 → 拿得到；不 evict 的反证给出 `[[5]]` |
| partial prefill、batch 暂停/重新进入、max length；逐事件 trace 对照冻结 proposer 和依赖 | `::test_partial_prefill_row_is_skipped_without_side_effects`、`::test_row_reordering_and_absence_match_upstream`、`::test_max_model_len_row_returns_empty_and_is_not_started` | 每条 trace 逐步比对 `drafts` + `active/cached_requests`；prefill 行"依赖包上一个方法都不调" |
| 不同请求不同候选长度经 target 验证后归属正确；错误候选不能改变 greedy 最终答案 | `::test_drafts_follow_request_identity_not_row_order`、`test_suffix_engine.py::test_greedy_output_matches_non_speculative`、`check_step61` F1 | 行序反转后候选跟着请求身份换（`[长,空]` ↔ `[空,长]`）；greedy 投机/非投机输出逐 token 相同 |
| spec factor、概率阈值和树深变化确实影响候选 | `::test_max_spec_factor_is_applied`、`::test_min_token_prob_is_applied`、`::test_max_tree_depth_changes_context_and_match`、`::test_num_speculative_tokens_caps_draft_length` | factor 1.0/0.5/0.1 → 4/2/0 枚；prob 0.5/0.51 → `[5]`/`[]`；depth 24/3 → 4/1 枚；K 1/2/8 → 1/2/4 枚 |
| §3 六步调用顺序 | `::test_call_order_and_arguments_follow_requirement_section_3`、`::test_partial_prefill_does_not_call_the_cache_at_all` | `start_request → add_active_response → speculate`；空采样零调用；离批末尾 `stop_request`；同 ID 时 `evict_cached_response` 在 `start_request` 之前；`speculate` 的 `max_spec_tokens == min(K, max_model_len - num_tokens - 1)` |

## 6. 生命周期与 Scheduler 的边界（需求 §3 最后一段）

- `suffix_cache` 只存在于 proposer 里，**不借着 KV BlockPool 存历史文本**；两个 "cache" 的淘汰含义不同：
  依赖包按"请求条数 FIFO"淘汰，KV 池按"物理块 + 前缀命中"管理。
- "离开 input batch" ≠ "请求 FINISHED"：被抢占、预算不够没排上、暂停后重进的请求都只是**离开批**，
  依赖包按 §3 第 6 步把它们停掉（局部 prompt 树丢弃，**已进全局树的响应保留**，重新进入时按"cached → evict → start"
  重建自己的树）；Scheduler 的 FINISHED 才走 Runner 的 `remove_requests`。两条路径的差异只有"早一步/晚一步"，
  在"批为空"的边界上由 `remove_requests` 补齐（§3.4）。

## 7. 实测命令与结果

见 `docs/step61_results.json`（命令、设备、依赖版本、包摘要 hash、逐项 passed/failed、真实执行轨迹）。

## 8. 已知限制 / 未做

- 只在 tiny Qwen3 上做了端到端（`tests/step61`、`benchmarks/check_step61_suffix.py` F 段），**没有跑真实权重**：
  本关不要求吞吐评测（需求 §5），候选质量随语料重复度变化很大，不做加速倍数声称。
- `speculate(use_tree_spec=True)`（论文里的 tree attention 分支）未接；`SuffixDecodingDraft.probs/parents` 未使用。
- 依赖包 0.3.0 的 `speculate(max_spec_tokens=None)` 分支引用 `self.max_depth`（该属性在 cache.py 里不存在，
  只有 `max_tree_depth`）——本关**永远显式传值**（上游同样传 `min(K, max_model_len - num_tokens - 1)`），
  所以碰不到这个潜在 bug；记录下来以免后续关卡改成默认值。
