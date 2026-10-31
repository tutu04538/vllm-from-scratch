# 第八十八关：单卡执行公平对照与收尾

> 需求：`单卡执行优化需求/088_单卡执行公平对照与收尾.md`
> 前置：087（Graph 接入与 GPU 元数据边界）
> 状态：**实现与实跑完成；本文件记录矩阵、配置身份、指标口径和已知差异。主报告见
> `docs/step88_execution_report.md`。**

## 1. 这一关要解决什么

83 解决了“测量隔离”，但没有回答“隔离之后还剩多少差距”；84–87 又把 RoPE、融合算子、
统一分页 attention 和 target/draft Graph 接进了执行路径。088 不再新增模型能力，只做三件事：

```text
同一硬件、同一 workload、同一执行模式
  → 跑有限的 minivllm × vLLM 矩阵
  → 用输出身份/接受计数证明两边比的是同一件事
  → 用相同边界的 profiler 解释剩余差距
```

因此本关的通过标准不是“某次 tok/s 超过 vLLM”，而是：

- 每个优化为什么存在、在当前路径是否真的生效；
- 同一模式下的差距有多大，短/中上下文、B=1/4 分开说；
- 剩余瓶颈有 trace 或机制证据，不把多个收益相乘，也不把 BF16 差异说成严格等价。

## 2. 交付文件与职责

| 文件 | 职责 |
|---|---|
| `benchmarks/step88_common.py` | 纯逻辑：16+4 矩阵、配对交替、配置身份、token 差分、指标换算、汇总；不 import torch/vLLM |
| `benchmarks/step88_runtime.py` | vLLM EngineCore 侧只读 RPC：显存、KV storage、Graph manager、attention backend、capture counter |
| `benchmarks/compare_step88_execution.py` | 正式矩阵：父进程编排、一 case 一子进程、mini/vLLM worker、输出身份与汇总 |
| `benchmarks/trace_step88_execution.py` | 6 个代表配置的稳态 decode profiler；mini 用 `torch.profiler`，vLLM 用 `start_profile/stop_profile` |
| `tests/step88/test_case_matrix.py` | 16+4 覆盖、配对顺序、配置身份、缺失格子 |
| `tests/step88/test_metrics_contract.py` | committed 吞吐/TTFT/TPOT、vLLM Prometheus 差分、失败不伪装成功 |
| `tests/step88/test_compare_dry_run.py` | 正式脚本 dry-run 的 20 个 case 与无 CUDA 初始化 |
| `tests/step88/test_runtime_snapshot.py` | YOCO 共享 KV 的 storage 去重 |
| `tests/step88/test_trace_contract.py` | 6 个 trace case、Chrome/torch 事件归一、阶段标注不混进 kernel |
| `tests/step88/test_real_assistant_rejection.py` | 真实 Gemma4 target/assistant 拒绝后逐轮 eager vs Graph 差分 |
| `docs/step88_execution_report.md` | 一页结论 + 数据附表 |
| `docs/results.json` → `step88` | 命令、版本、设备、hash、结果、执行轨迹 |

## 3. 有限矩阵

### 3.1 主表（短 prompt，输出 128 token）

```text
2 引擎 × B{1,4} × K{0,2} × {eager, FULL decode Graph} = 16 case
```

- 短 prompt 固定为 26 个 token 的中文问题（与 82/83 同一 chat template）；
- B=4 时四条请求使用**同一组 prompt token IDs**，避免 82 的“长 prompt + chunked prefill”
  把 B=4 的 prefill 工作量混进 decode 对照；
- K=0 是 `speculative_config=None`，不是“同一个 proposer 改宽度”；
- 主表结束后才跑上下文补充点，不加入 K=4/prefix/dynamic 笛卡尔积。

### 3.2 上下文补充（B=1，K=2，~1024 token）

```text
2 引擎 × {eager, Graph} = 4 case
```

用真实 tokenizer 生成 3769 个 token 的固定对话，再取前 1024 个 token IDs；两边传给模型的
是同一组 IDs。1024 + 128 输出 + 2 草稿位 < `max_model_len=2048`。

### 3.3 配对顺序

20 个 case 串行；每个 `(B,K,mode,workload)` 是一对。偶数对先 mini、奇数对先 vLLM，
避免笔记本热状态只偏向一边。每个 case 单独子进程，预热后至少测 3 次。

## 4. 配置身份

`check_identity()` 比较 worker **实际生效**的字段，而不是回显请求参数：

| 字段 | mini | vLLM | 是否要求相等 |
|---|---|---|---|
| dtype | bfloat16 | bfloat16 | 是 |
| max_model_len | 2048 | 2048 | 是 |
| block_size（控制面） | 16 | 16 | 是 |
| max_num_seqs | B | B | 是 |
| max_num_batched_tokens | 128 | 128 | 是 |
| prefix caching | False | False | 是 |
| speculative tokens | K | K | 是 |
| max_new_tokens | 128 | 128 | 是 |
| temperature / ignore_eos | 0 / True | 0 / True | 是 |
| attention backend | `TRITON_ATTN` | `TRITON_ATTN` | 是 |
| Graph 模式 | `FULL_DECODE_ONLY` / `NONE` | `FULL_DECODE_ONLY` / `NONE` | 是 |
| 编译模式 | `NONE` | `NONE` | 是 |
| KV block 数 / 物理字节 | 512 / 144 MiB | 1536 / 144 MiB | **否，单独记录** |

vLLM 内部可能把 sliding/full 组拆成 block_size 32/16；这是正确布局差异，不强行抹平。
`num_gpu_blocks` 不同也不代表容量不同：vLLM 的 1536 个块对应的可容纳 token 数、每块字节与
mini 的 512 个块不是同一坐标系。

vLLM 侧通过 `LLM` 的 `CompilationConfig` 显式设置：

```python
CompilationConfig(
    mode="NONE",
    cudagraph_mode="FULL_DECODE_ONLY" if graph else "NONE",
    cudagraph_capture_sizes=[1, 2, 3, 4, 6, 8, 12, 16] if graph else [],
    max_cudagraph_capture_size=16 if graph else 0,
)
```

主对照**不引入 torch.compile**；vLLM 的 `enforce_eager=True` 只用于 eager 列。Graph 列从
worker 的 `cudagraph_manager` / `compilation_counter` 读取实际模式与捕获数，而不是只看
`enforce_eager=False`。

## 5. 指标口径

两边都通过“手动 add_request + 逐步 step”采集，vLLM 不用 `generate()` 的一次性 wall time：

```text
committed_output_tokens = 最终 token_ids 的总长度
output_tokens_per_s     = committed_output_tokens / wall_time_s
TTFT                    = 第一次有输出 - 请求提交
inter_burst_latency_s   = (last_output - first_output) / (输出事件数 - 1)
tpot_s                  = (last_output - first_output) / 首 burst 之后的 committed token
```

- 单 burst 的 TPOT 是 `None`，不填 0；
- vLLM 的 `RequestOutput` 每个 `step()` 返回累计 `token_ids`，用与 mini 相同的
  `RequestTiming.observe()` 记账，因此 burst 口径一致；
- vLLM 的接受计数来自 `get_metrics()` 的 Prometheus 快照差分：
  `num_drafts` / `num_draft_tokens` / `num_accepted_tokens` /
  `num_accepted_tokens_per_pos`；
- vLLM 只暴露 accepted-per-pos，不暴露 draft-per-pos；报告里 `draft_per_pos_available=False`，
  位置接受率按上游日志口径 `accepted[j] / num_drafts`，不伪造 actual 分母。

性能之外的记录：

```text
build_s / graph_capture_s / warmup_s
KV bytes、num_gpu_blocks、block_size、KV group 数
target/draft 图数量、捕获 token 档位、测量期间是否又捕获
worker allocated/reserved/peak、设备总量
```

## 6. 输出身份与数值边界

`build_identity_report()` 分开两类：

1. **mini 的 eager vs Graph**：必须逐 token 相同，K=2 接受计数也必须相同；这是硬门禁。
   **vLLM 的 eager vs Graph**先按同样口径记录；如果 B=1 不同，或 B>1 的 K=2 不同但缺少
   同 batch/workload 的 K=0 控制项，则 `vllm_boundary_gate=False`，矩阵不通过。
   B>1 的 K=0/K=2 都有分叉时，允许作为 layout/BF16 数值边界，但 `strict_equivalent=False`，
   不进入严格等价表。
2. **跨引擎 mini vs vLLM**：记录逐请求首分歧，不要求 BF16 下逐 token 相等；接受计数相同
   是最强对照，不同只解释，不篡改。

### 6.1 本次实际发现

- mini 的 B=1/B=4 三次重复输出与接受计数全部稳定；mini 的 eager/Graph 全部逐 token 相同。
- vLLM B=1 三次重复稳定，eager/Graph 逐 token 相同。
- vLLM **B=4 run-to-run 不稳定**：同一 case 的 3 次重复输出 sha256 不同；K=0 没有草稿也
  复现，因此不是 assistant/投机路径。B=4 的 eager/Graph 首分歧在 token 90，但这是非确定性
  的一次采样，不是稳定的 Graph 数值差。
- 单次 `check_step88_numerics.py` 的 B=4 K=0 eager/Graph 恰好 128/128 相同；这只能说明
  “单次对照不能证明确定性”，不能反推稳定。
- 因此主报告对 vLLM B=4 用 `strict_equivalent=False`，性能比值只作同模式观察；B=1 和
  跨引擎 B=1 K=0 才是强对照。

83 的固定历史（中文 prompt、64 token、K=0/K=2、FP16/BF16/FP32）已有结论：
BF16 的第一分歧是 target near-tie，FP32 可复现；85 改变归约顺序后该次固定复放不再分叉，
但 position 70 的 margin 仍可为 0，不能宣称“BF16 位等价”。

88 另补真实 Gemma4 target/assistant 的拒绝后差分：固定随机 seed，eager 与 FULL Graph 都
记录 target 每步 hidden/logits、assistant 每轮 propose 的输入 hidden/positions/input IDs、
`draft_tokens` 与 `num_rejected`；至少出现一次拒绝。该测试是对 82 遗留“真实 assistant 逐轮
差分”的收口，不把最终 token 相同当作 assistant 路径等价的唯一证据。

## 7. Graph 可比性

| 项 | mini | vLLM |
|---|---|---|
| 主对照编译 | 无（`CompilationMode.NONE`） | 无（`CompilationMode.NONE`） |
| 图模式 | `FULL_DECODE_ONLY` | `FULL_DECODE_ONLY` |
| prefill | eager | eager |
| target 图 | `ModelCudaGraphManager`，K=0/K=2 各捕获实际 bucket | worker `cudagraph_manager`，capture sizes 1..16 |
| draft 图 | assistant prefill/decode 两张 `SpeculatorCudaGraphManager`（K=2） | Gemma4 speculator 的 prefill/decode 两张图 |
| 稳态 capture | 测量前后图数不变 | 测量前后图数不变 |
| 实际重放 | `num_replays` 增量 | trace 里的 `cudaGraphLaunch` 计数 |

如果 vLLM 在本机把 `FULL_DECODE_ONLY` 降级/禁用，`_run_vllm_case()` 会直接失败，不会静默
把 Graph 列换成 eager。vLLM 的 `get_metrics()`/profiler 需要 EngineCore 跨进程；
`VLLM_ALLOW_INSECURE_SERIALIZATION=1` 只用于本地 benchmark 的只读 RPC，不进入生产路径。

## 8. KV 物理布局

Gemma4 有 YOCO 跨层 KV 共享：最后 20 层只算 Q，读前面同类型源层的 KV。mini 的
`kv_caches` 只保留 15 个源层；vLLM 的 `kv_caches` 会包含 Q-only aliases，因此第一次快照
如果逐 tensor 相加会把 144 MiB 误读成 1.76 GiB。`step88_runtime.py` 改为按底层 storage
去重后，两边物理 KV 字节一致，记录：

```text
mini: 512 blocks × 16 slots，15 个源层，144 MiB
vLLM: 1536 个逻辑块，block_size 32/16 的 sliding/full 组，144 MiB
```

两边可容纳 token 数不同（vLLM 的布局更细），报告分别记录“同一 workload 可容纳”和
“消耗字节”，不把 block 数当容量。

## 9. 已知边界

1. 真实 DFlash/DSpark/DFlash2/多模块 MTP 权重不在本机；088 不补这些模型的草稿质量。
2. 上下文补充只到 ~1024 token，不外推到 8k/32k。
3. 主表固定一个中文短 prompt；B=4 是同一 prompt 的 4 条请求，不覆盖不同 prompt 长度。
4. vLLM 的 EngineCore 在独立进程；Graph 实际重放只能通过 profiler 的
   `cudaGraphLaunch` 计数证明，主矩阵只能记录图数与捕获计数。
5. vLLM 的 `get_metrics()` 是累计 Prometheus 计数器；每次测量做差分，不能把 warmup
   计入样本。
6. 跨引擎输出不要求 BF16 逐 token 相等；同引擎 eager/Graph 才作为数值门禁。
7. `limit_mm_per_prompt={"image":0,"video":0}` 只用于跳过纯文本 case 的多模态 processor
   warmup；模型权重/文本路径未改。
8. vLLM 0.28 在本机 B=4 greedy 存在 run-to-run 非确定性；本关已把重复 hash 写入
   `docs/step88_execution_report.md` §4，并记录疑似问题到本地 `vllm_bugs/`（gitignored）。
9. 本关不增加新模型能力、不改 Scheduler/BlockPool/拒绝采样算法。

## 10. 后续 microbenchmark

88 收尾后补做了 `benchmarks/check_step88_attention_micro.py`，直接测 `unified_attention`：
q_len=3/context=1024 的 2D 只有 2 个 program、每 program 约 33 个 KV tile；强制 3D
q_len>1 在 1024/2048 的单 kernel 约 2.7–4.6×，128/256 反而更慢。方法、正确性与生产化边界见
`docs/step88_attention_micro.md`；生产 `can_use_3d()` 仍拒绝 q_len>1，未改执行路径。

## 11. 一句话

**088 用 16 个主 case + 4 个 1024-token 上下文 case，在无 compile、同 attention backend、
同 Graph 模式的条件下重新测量 minivllm 与 vLLM；同时把配置身份、KV 字节、Graph 捕获/重放、
输出首分歧和 profiler 边界分开审核，回答“还剩多少差距、差在哪”，而不是制造一条好看的加速曲线。**
