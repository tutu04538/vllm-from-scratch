# 108：TP 投机执行状态与拒绝后修复

状态：本关限定的多 rank EAGLE3 eager 执行及验收完成。Graph 与真实多卡模型矩阵留给 109。

本关包 hash：`a7b3f8a919cd6343c8ed6d0740860b12a40b9ac261061f1d5d85b6c9a8e310b3`，含测试专用 worker；任何生产包代码改动后需重算。

## 冻结职责、时序和布局

| 项目入口 | 冻结 vLLM 0.28.0 | 职责、时序与布局 |
|---|---|---|
| `worker/gpu/model_runner.py::execute_model/sample_tokens` | `v1/worker/gpu/model_runner.py::execute_model/sample/sample_tokens` | 所有 TP rank 消费同一调度快照、执行 target 前向并采样；每个 rank 维护本地镜像，Executor 只把输出 rank 的结果交给 Scheduler。 |
| `worker/gpu/spec_decode/autoregressive/speculator.py::propose/prepare_prefill_inputs` | 同名冻结方法 | target 真正输入的 token、位置及辅助 hidden 形成第一遍 draft 输入；保留 target 扁平 batch 的行布局，拒绝行仍占位但不能作为有效位置写入或提交。 |
| `worker/gpu/spec_decode/autoregressive/speculator.py::update_draft_inputs` | 同名冻结方法 | 按请求更新草稿 step、token、position、seq_len 和下一次 hidden；输出为 `[B,K]` target token id，缩减词表映射发生在 logits、softmax 之前。 |
| `worker/gpu/spec_decode/rejection_sampler.py` | `v1/worker/gpu/spec_decode/rejection_sampler.py` 及其 utils | 按请求和固定随机源验证草稿，返回 `num_sampled[B]` 与 `num_rejected[B]`；每 rank 状态和结果必须一致。 |
| `core/sched/scheduler.py`、`engine/core.py::post_step` | `v1/core/sched/scheduler.py::update_from_output/update_draft_token_ids` | Scheduler 独占 Request、逻辑 KV 和提交；只消费一份权威输出和一份 draft，不把 TP 数乘到计数。 |

固定例子 TP=2、B=2、K=2：A 接受 0 个草稿、正常提交 1 个修正 token；B 接受 2 个草稿、正常提交 2 个候选和 1 个 bonus。预留及 padding 行不属于已计算位置；下一轮每请求满足 `next_start=start+num_valid`，终止截断沿用 Scheduler。两张卡的请求对应关系、候选、采样和长度须一致，草稿 KV 分别由本 rank 持有。

本关仅考虑 V2、单副本、同 TP 的 Qwen3 + EAGLE3 eager；Graph 开放及真实模型矩阵由 109 负责，PP/DP、异步调度、其他投机算法及多卡结构化输出不在范围内。

## 阶段检查

- 在 TP2 的 tiny 引擎中，K=1/2/4、B=1/2/4 和每个强制接受前缀的首轮矩阵，以及 TP4 的混合接受场景共 35 passed；此轮期间测试探针仍有修改，不能作为最终版本的完整验收。
- 额外场景初次实测 10 passed：TP2/4 概率草稿缩减词表映射、真实 rank seed/长度扰动诊断、EOS、开启或关闭 prefix 的取消/slot 复用与抢占、rank1 提议失败。测试专用快照区分每请求的有效采样前缀和被拒尾部，不比较未初始化的 padding；记录 target 与 draft 的独立 KV，并对相同请求历史的 target 有效 KV 做独立非投机重放。
- 一个 TP2、B2、K2 的混合例子中，强制 A 接受 0、B 接受 2；每 rank 的请求镜像、输出与 next_start 一致，Scheduler 的 drafted=4、accepted=2，逐位置接受数为 `[1,1]`，没有乘 TP。
- 冻结 tiny TP2 逐轮对照的首个分歧是辅助层边界 0：上游保存 embedding 引用，CUDA fused RMSNorm 后续原地更新残差，引用实际变为最后一层残差；项目之前使用新张量，保存的是原 embedding。已先在私有 `vllm_bugs/eagle3_boundary_zero_alias.py` 建上游最小复现并记录疑似问题，尚不判断上游是否有意。按上游重构本地 `RMSNorm` CUDA 融合路径的输入/残差原地语义：连续张量由融合 kernel 直接写回；非连续布局计算后写回原视图，CPU native 仍为功能式参考。FP32/BF16 连续和带 stride 布局与冻结 CUDA 算子检查通过。修复后固定同一历史、同 TP2/K2 的 target/draft 有效 KV、目标输入、草稿层/hidden/logits 与冻结参考按预登记阈值对照通过（`benchmarks/results/step108/compare32_tp2_k2_inplace2.json`）；日志中先前 600 项失败和原因保留。
- 最终版本独立测试：`CUDA_VISIBLE_DEVICES=2,3,0,1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TRITON_F32_DEFAULT=ieee python -m pytest tests/step108 -q -ra`，61 passed、0 failed、0 skipped，日志 `benchmarks/results/step108/acceptance_final_runs/tests.log`。包含 TP2/4、K1/2/4 与同 TP 冻结逐轮差分，以及先接受再拒绝的固定历史对照。
- 验收入口 `CUDA_VISIBLE_DEVICES=2,3,0,1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmarks/check_step108_acceptance.py --case all --timeout 1800 --output benchmarks/results/step108/acceptance_final.json`：36 个子步骤全部通过，其中冻结采样 10 passed；六组同 TP 逐轮参考先重复登记阈值再比较项目。产物及每条子命令见该 JSON。
- 完整验收后补强测试：`CUDA_VISIBLE_DEVICES=2,3,0,1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TRITON_F32_DEFAULT=ieee python -m pytest tests/step108/test_step108_distributed.py -q -ra -k 'padding_poison or (speculative_lifecycle and preempt)'`，4 passed、42 deselected。新增两项分别在 TP2/4 每 rank 检查被污染的 padding 输入被生产 Triton KV 写入内核跳过，同时检查跨 block 的有效写入；两个既有抢占场景复验，prefix 关闭时明确断言同一批含 prefill 与 decode。后者的 valid KV/输出仍由前述真实引擎用例比较；内核测试不替代完整 Graph padding 验收。
- 当前包 58—88 回归 1289 passed、0 failed、23 skipped（`benchmarks/results/step108/legacy_a7b.log`）；`CUDA_VISIBLE_DEVICES=2,3,0,1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest $(printf 'tests/step%03d ' $(seq 89 107)) -q -ra` 为 241 passed、0 failed、0 skipped（`benchmarks/results/step108/parallel_a7b.log`）。106/107 真实权重复验见各自报告。跳过项不计为实测通过。
- 真实 EAGLE3 多卡完整矩阵、target/draft Graph 捕获和重放由 109 验收；PP/DP、异步、动态 K 和多卡结构化输出仍明确拒绝。106 stream 竞争及 105 曾有的关闭超时根因仍未确定。

## 2026-11-17 边界 0 上游修复的跟进（工作树，待 GPU 验收）

上游 [PR #66546](https://github.com/vllm-project/vllm/pull/66546) 已修复边界 0 引用被原地 residual 更新覆盖的问题；项目在选择边界 0 时 clone embedding，保留 CUDA RMSNorm 本身的原地语义。现有冻结源码未修改；测试侧 `--reference-boundary-zero-fix` 将同一修复应用于参考 Worker，登记的合同和 trace 明确标记该语义，禁止混用历史原始参考；历史逐值比较仍属于当时源码，不能作为当前修复的通过证据。

当前包 hash `5a738fc095c544e59baf458a04cb0cf2beb25e89e4fe6e83e01c8911573b4565`。定向 CPU 回归 29 passed、3 skipped、1 deselected（CUDA 用例未运行，三个 skip 为旧 1.7B 权重缺失）；命令与首次隐藏全部 GPU 导致配置测试 8 passed/2 failed 的记录见 `docs/results.json → step108.boundary_zero_fix_20261117`。四卡均被其他计算进程占用，CUDA 专项、TP 差分及完整 GPU 回归均尚未执行；不能将本次修复标为已验收。
参考补丁及合同来源检查的独立 CPU 测试为 4 passed、1 deselected；其中检查不同边界 0 语义的参考不能混合比较。


## 2026-11-19 修复后重验（未完全收口）

`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TRITON_F32_DEFAULT=ieee python -m pytest tests/step106/test_step106_aux_boundaries.py tests/step108/test_step108_oracle_fix.py tests/step108/test_step108_reference.py -q -ra`：12 passed，覆盖 CUDA 边界快照、TP2/TP4、K=1/2/4 及强制接受后拒绝；参考是测试进程内应用上游 #66546 的冻结版本。

58—88 首次回归 1288 passed、1 failed、23 skipped：旧 `test_first_pass_inputs_follow_eagle_alignment` 依赖自然 draft 接受序列，边界 0 修复后不再触发所需场景。以相同 target 的非投机结果强制第一枚接受、第二枚拒绝，并同步点质量 q 后，发现旧断言把锚点之后的移位当作原位副本；按实际整体移位修正并保留逐行、位置和特征断言。定向测试 1 passed。随后同一 58—88 命令完整重跑：1289 passed、23 skipped。跳过项主要是缺失旧真实模型权重，不计为通过。日志见 `benchmarks/results/step108_boundary_zero_20261119/`。

089—112 回归排除了已在前述定向测试运行的 `test_step108_reference.py`；进行中出现 1 个失败，尚未输出失败堆栈时其他用户训练占用四卡，为避免争用而中断。pytest 进度显示 154 个通过标记、1 个失败标记，但中断无最终统计，不能视为整组通过。根据收集顺序，失败可能是 `test_step091_parallel.py::test_default_draft_tp_follows_target`；该测试单独 1 passed，其所在文件单独 56 passed，step090 模型文件加该文件 69 passed，但均不能替代整组重现。另有部分回归与其他用户 GPU 任务重叠，需在空闲时重跑，定位并修复该失败，再完成 089—112 回归。105 历史关闭超时风险仍未解决。
