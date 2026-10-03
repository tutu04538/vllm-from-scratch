# step59 对齐记录：GPU 批量拒绝采样与投机元数据

- 对应代码：`minivllm/sample/rejection_sampler.py`（新增，生产路径）、
  `minivllm/sample/sampler.py`、`minivllm/spec_decode/metadata.py`、`minivllm/spec_decode/metrics.py`（新增）、
  `minivllm/worker/gpu_model_runner.py`、`minivllm/core/sched/scheduler.py`、`minivllm/config.py`、
  `minivllm/testing/{torch_rejection_sampler,spec_metadata}.py`（参考实现/测试辅助）、`minivllm/outputs.py`
- 参考基线：本机 `vllm==0.28.0`（2026-10-03 文件快照），关键位置
  `v1/sample/rejection_sampler.py`（L38–L953）、`v1/spec_decode/metadata.py`（L10–L66）、
  `v1/spec_decode/metrics.py`（L18–L49）、`v1/worker/gpu_model_runner.py::_calc_spec_decode_metadata`（L2918）、
  `v1/core/sched/scheduler.py::make_spec_decoding_stats`（L2624）
- 需求：`投机解码完整需求/059_GPU批量拒绝采样与投机元数据.md`
- 目录映射：本项目顶层 `minivllm/` ↔ 上游 `vllm/v1/`；`minivllm/sample/` ↔ `vllm/v1/sample/`；
  `minivllm/spec_decode/` ↔ `vllm/v1/spec_decode/`
- 交付：`tests/step59/`（52 项 pytest）、`benchmarks/check_step59_rejection.py`（21 项）、
  `docs/step59_results.json`、`docs/step59_models.json`

## 1. 痛点与结论（实测）

验证是**每一步都要付**的固定开销。旧实现是 Python 逐候选循环 + 标量取值（每个候选位 4~5 次
D2H），批越大、K 越大、接受率越高（= 越该赢）它越慢。同一批张量（V=1000，全部在 GPU 上）实测：

| 场景 | 旧实现（=现在的 Torch 参考） | 本关新路径 | 上游 `rejection_sample` |
|---|---|---|---|
| B=32 K=5 random 全接受 | 11.428 ms / **385 次 D2H** | 0.582 ms / **0 次** | 0.314 ms |
| B=32 K=5 random 随机草稿 | 13.496 ms / 419 次 | 0.431 ms / 0 次 | 0.316 ms |
| B=32 K=5 greedy 全接受 | 2.705 ms / 73 次 | 0.094 ms / 0 次 | 0.145 ms |
| B=3 K=3 random 全接受 | 2.176 ms / 33 次 | 0.512 ms / 0 次 | 0.308 ms |

结论按证据强度写：**逐候选 D2H 从 O(B·K) 变成 0**（这次测量里 B=32/K=5 → 385 次降到 0 次），
内核启动数不随批大小增长（B=1/K=1 与 B=32/K=5 都是 10 个 CUDA 事件/步）。上表里的毫秒数
只是同一台机器同一次测量的对照，不当作"加速倍数"结论（性能矩阵在 84 关）。

## 2. 逐项对照

### 2.1 `v1/spec_decode/metadata.py::SpecDecodeMetadata`

| | 上游 | 本项目 |
|---|---|---|
| 字段 | `draft_token_ids [P]`、`num_draft_tokens [B]`、`cu_num_draft_tokens [B]`、`cu_num_sampled_tokens [B]`、`target_logits_indices [P]`、`bonus_logits_indices [B]`、`logits_indices [P+B]` | 同左（逐字） |
| `max_spec_len` | `__post_init__` 里 `max(num_draft_tokens)` | 同左 |
| dtype/设备 | `cu_*` 是 GPU **int32**、累积和**不带开头 0**；`draft_token_ids` int32 | 同左（Runner 里 `to_device(..., torch.int32)`） |
| `make_dummy` | 有（profiling/哑输入） | **没有**：本关没有 dummy run（CUDA Graph 是 69 关），不写没人用的构造器 |
| 构造位置 | `GPUModelRunner._calc_spec_decode_metadata`（L2918） | 同左（`GPUModelRunner._calc_spec_decode_metadata` + `_get_cumsum_and_arange` + 预分配 `_arange_np/_arange_scratch`） |
| 草稿 token 来源 | `input_ids.gpu[logits_indices][target_logits_indices + 1]` | 同一表达式，但对**同一份 CPU 输入行**求值（本机还没有常驻 GPU 输入缓冲）；多一条"输入行 vs 协议"的一致性检查 |
| 请求 → 行 | 元数据不带请求信息（行序 = 批行序） | 同左（**删掉了旧版的 `req_ids` 字段**；q 对齐改由 Runner 按批行序做） |

### 2.2 `v1/sample/rejection_sampler.py`

| | 上游 | 本项目 |
|---|---|---|
| 类/函数 | `RejectionSampler.forward/_get_logprobs_tensors/parse_output/apply_logits_processors/apply_penalties/_combine_outputs_with_spec_tokens`；`rejection_sample`、`apply_sampling_constraints`、`expand_batch_to_tokens`、`generate_uniform_probs`、`sample_recovered_tokens` | 同左（同名同顺序；`_get_logprobs_tensors` 保留边界但**调用即报错**，logprobs 是 68 关） |
| 内核 | `rejection_greedy_sample_kernel`、`rejection_random_sample_kernel`、`expand_kernel`、`sample_recovered_tokens_kernel`（Triton） | 逐行对应，去掉 `SYNTHETIC_MODE` / `use_fp64_gumbel` 两个分支（75 关与实验开关） |
| bonus | 走 `Sampler`，`predict_bonus_token=True`、`max_num_logprobs=-1`、`logprobs_mode_override` | 走 `Sampler`、`predict_bonus_token=True`；后面两个参数本关不存在（没有 logprobs） |
| 约束 | `apply_sampling_constraints`（温度 0→1、top-k/top-p 用 `expand_kernel` 展开） | 同左 |
| 输出 | `[B, max_spec_len+1]` **int32**、无效位置 `PLACEHOLDER_TOKEN_ID(-1)` | 同左 |
| CPU parse | `parse_output`：`cpu().numpy()` 一次 + mask（`!= -1` 且 `< vocab_size`）+ `discard_req_indices` | 同左（Runner 用它替掉了旧的 Python 列表推导） |
| 设备 | 只有 GPU 路径（Triton） | 同左：CPU 张量直接 `NotImplementedError`，提示用 `minivllm/testing/torch_rejection_sampler.py` |
| 方法校验 | `rejection_sample_method ∈ {standard, synthetic, block}` | 只接 `standard`；`synthetic`/`block` 在 `SpeculativeConfig` 与 `RejectionSampler.__init__` **两处**明确报错（75 关） |

### 2.3 `v1/spec_decode/metrics.py::SpecDecodingStats`

| | 上游 | 本项目 |
|---|---|---|
| 字段 | `num_spec_tokens/num_drafts/num_draft_tokens/num_accepted_tokens/num_accepted_tokens_per_pos/num_draft_tokens_per_pos` | 同左（逐字） |
| `new/observe_draft` | `new(K)` 开 K 长直方图；`observe_draft` 累加 + `assert accepted ≤ K` | 同左 |
| 记账点 | Scheduler 的 `make_spec_decoding_stats`（L2624），只统计**已验证**的候选 | 同左（`Scheduler.make_spec_decoding_stats`，调用点在取到本轮结果之后） |
| 汇总去向 | `SchedulerStats` → `EngineCoreOutputs` → 前端 / prometheus | **本项目没有指标前端**：Scheduler 只留最近一步的 `self.spec_decoding_stats` 给测试/demo 读（差异见 §3） |

### 2.4 采样器侧（`v1/sample/sampler.py`）

| | 上游 | 本项目 |
|---|---|---|
| `forward` | `(logits, sampling_metadata, predict_bonus_token=False, logprobs_mode_override=None)` | `(logits, sampling_metadata, predict_bonus_token=False)`（没有 logprobs 模式） |
| `apply_logits_processors` | 白名单 → bad words → 非 argmax 不变处理器（min_tokens）→ 惩罚 | 惩罚 → min_tokens（顺序与上游一致；惩罚从 `sample()` 挪进来） |
| `_combine_outputs_with_spec_tokens` | Sampler 版：`[*out, *spec] if spec else out`（行数不变，给 bonus 行） | 同左 |
| min_tokens 的投机版 | `MinTokensLogitsProcessor.apply_with_spec_decode`：按请求屏蔽**前 n_mask 行** | `Sampler.apply_min_tokens_for_spec_decode`（同样的 `n_mask = clamp(min_tokens - len(out), 0, K_i)`），没有处理器框架 |
| min_tokens 的整批施加 | `index_put_` 一次 | 同左（旧实现是逐行 `logits[row, stop_ids] = -inf`，每行一次 H2D） |

## 3. 临时差异（逐条写清，不留"简化了一些"）

1. **没有 logprobs**：`_get_logprobs_tensors` 只保留职责边界，被调用时 `NotImplementedError`
   （68 关）。`SamplerOutput.logprobs_tensors` 字段已按上游形状留好，本关恒为 `None`。
2. **没有 `synthetic` / `block`**：内核里没有 `SYNTHETIC_MODE` 分支，`rejection_sample` 不收
   `synthetic_mode` / `synthetic_conditional_rates` / `use_fp64_gumbel` 这三个参数（75 关）。
   传了也不生效的参数比没有参数更容易骗人，所以直接不写。
3. **没有 bad words / 白名单 / thinking budget / logits processor 插件框架**：本项目的
   `SamplingMetadata` 里没有这些字段（57D 的范围），`apply_logits_processors` 只有惩罚 + min_tokens。
4. **历史是 Python list 而不是 token 张量**：上游用 `prompt_token_ids[repeat_indices]` 索引进
   GPU 张量；本项目 `SamplingMetadata` 把历史存成 list（57D 的结构），所以逐行历史在 CPU 侧
   重复。**行序与展开规则与上游一致**（每请求 K_i 行、按草稿前缀），惩罚值仍走
   `expand_batch_to_tokens` 内核展开。
5. **`num_invalid_spec_tokens` 未实现**：上游统计时会扣掉"排了草稿但结果整批被丢弃"的请求；
   本项目没有那条路径（被丢弃的行走 `discard_req_indices`，且这些请求本就不带草稿）。
6. **`_get_spec_decode_draft_probs` 缺 q 时报错而不是 warning 后回退**：上游 `logger.warning`
   然后返回 `None`（= 把 q 当点质量，接受率虚高）。本项目按约定明确报错，不静默降级。
7. **CPU 不支持投机验证**：与上游一致（Triton 内核只有 GPU 路径）。CPU 上的算法语义由
   `minivllm/testing/torch_rejection_sampler.py` 覆盖；`rejection_sample` 遇到 CPU 张量直接报错
   并指向该文件。**这条会改旧测试/脚本的行为**，见 §5。

## 4. 验证

| 命令 | 结果 |
|---|---|
| `python -m pytest tests/step59 -q` | **52 passed** |
| `python -m pytest tests/step58 -q` | **41 passed**（跑真引擎的用例改为 CUDA，见 §5） |
| `python benchmarks/check_step59_rejection.py` | **21 PASS / 0 FAIL** |
| `check_step58_{input_budget,draft_inputs,workspace}.py` | 10 + 11 + 15 = **36 PASS** |
| 15 个 `check_step57_*.py` | **350 PASS / 0 FAIL**（脚本改动见 §5） |

关键用例（都在 `tests/step59/`，脚本版在 `check_step59_rejection.py`）：

- A/B/C 逐值：`drafts=[[1,2,3],[],[4]]` → `[[1,0,-1,-1],[3,-1,-1,-1],[4,1,-1,-1]]`；
  `logits_indices=[0..6]`、`target=[0,1,2,5]`、`bonus=[3,4,6]`、`cu_draft=[3,3,4]`。
- 全接受 / 首拒绝 / 中拒绝 / B=1 / D=0 / ragged K / 混合 greedy+random / 拒绝后尾部不泄漏。
- 随机语义（注入随机数逐值）：`p/q=0.5` 时 u=0.49 接受、0.51 拒绝；`p/q>1` 时接受概率封顶 1；
  `q[d]=0` 防御性拒绝；点质量 q（ngram）退化成 `p[d] >= u`。
- 分布（需求 §4 指定）：p=`[0.2,0.3,0.5]`、q=`[0.6,0.3,0.1]` → 实测输出频率
  `[0.1978,0.3048,0.4974]`（解析 p，N=20000，阈值 0.02 ≈ 5.7σ）；
  恢复分布 ∝ `max(p-q,0)`：p=`[0.3,0.3,0.4]`、q=`[0.6,0.1,0.3]` → 实测 `[0, 0.6665, 0.3335]`
  （解析 `[0, 2/3, 1/3]`，N=30000，阈值 0.015 ≈ 5.5σ）。
- 差分：greedy 与上游逐值一致；random 在同一 seed 下与上游逐值一致；注入同一组 u/recovered 时
  **我们的内核 / 上游内核 / Torch 参考三者一致**；`expand_batch_to_tokens` 与上游 `expand_kernel` 一致。
- 端到端：tiny 模型（draft_model 与 ngram 两条提议路径）greedy 投机 == 非投机；
  统计 `轮数=4、验证=12、接受=12`（位置接受率 `4/4`）。
- profiler：内核路径 D2H = **0 次/步**（B=1/K=1 与 B=32/K=5 相同），CUDA 事件数不随 B×K 增长。

## 5. 与 57/58 的行为差异（改了旧测试/脚本的地方）

| 改动 | 原因 | 影响 |
|---|---|---|
| `minivllm/spec_decode/rejection_sampler.py` 删除 → `minivllm/sample/rejection_sampler.py` | 上游把这个文件放在 `v1/sample/` 下 | 导入路径变化；`minivllm.spec_decode` 不再导出 `RejectionSampler` |
| `SpecDecodeMetadata.from_scheduled` / `req_ids` / `req_draft_slice` / `num_draft_tokens_total` / `batch_size` 删除 | 上游字段就是那 7 个；请求 → 行是 Runner 的事 | `benchmarks/check_step57_spec_metadata.py` 改为调 Runner 的方法（13 项不变，其中 1 项换成新 API 的等价失败模式） |
| `expand_batch_to_tokens(x, num_tokens_per_req)` → `(x, cu_num_tokens, num_tokens, replace_from, replace_to)`（Triton `expand_kernel`） | 上游签名 | `check_step57_rejection_sampler.py` 重写（22 → 18 项：注入方式从 `uniforms=`/`recoveries=` 参数改成 monkeypatch 模块函数，冗余用例合并，完整用例集在 `tests/step59`） |
| `Sampler.forward` 多一个 `predict_bonus_token` 参数；惩罚移进 `apply_logits_processors` | 上游签名/顺序 | `check_step57_sampler.py` 第 7 项断言随之更新（仍然不接 Request） |
| 投机**验证**只在 CUDA 上可用 | 上游同款（Triton） | `tests/step58/helpers.py` 与 `benchmarks/check_step57_{draft_model,spec_lifecycle}.py`、`check_step58_{draft_inputs,workspace}.py`、`trace_step57.py` 的默认设备改为 `cuda`（本机有 GPU；纯算法语义仍在 CPU 上由参考实现覆盖） |
| `check_step57_rejection_sampler.py` 的项数 22 → 18 | 见上 | 15 个 57 脚本合计 354 → **350** 项（其它脚本一项未改） |

已知未修（与本关无关）：`benchmarks/compare_step57_vllm.py` 在**基线提交上同样失败**
（vLLM 用 `spawn` 启 EngineCore，脚本没有 `__main__` 守卫会被子进程重跑 → 初始化失败）。
本关只更新了它内部调用拒绝采样的签名（`build → verify` 三段仍走同一套 metadata 索引）。

## 6. 留待后续（按总纲顺序）

- **68 关**：`_get_logprobs_tensors` 的真实实现（接受行的 logprobs 要按 `cu_num_sampled_tokens`
  重新对齐）、`allowed_token_ids` / bad words / 结构化输出与投机的交互。
- **69 关**：固定形状的 AR 步进（`B × (K-1)` 行 + padding 掩码）与 CUDA Graph；届时
  `SpecDecodeMetadata.make_dummy` 才会有人用。
- **75 关**：`synthetic` 与 V2 `block` 验证（内核里的 `SYNTHETIC_MODE` 分支、配置校验与互斥）。
- **84 关**：统一性能矩阵（本关只报"逐候选 D2H = 0"与同机对照毫秒数）。
