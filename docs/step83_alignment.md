# 第八十三关：基准隔离、资源释放与第一分歧诊断

> 需求：`单卡执行优化需求/083_基准隔离与引擎生命周期.md`
> 日期：2026-10-29。本关**不优化算子、不换 attention、不改调度器**。
> 结论先行：旧的跨配置基准不可直接比较；本关已把正式测量改成“一配置一子进程”，
> 补了幂等 shutdown 与指标口径，并把 64-token 第一分歧定位到 **target BF16 near-tie**，
> 不是 assistant/拒绝采样状态错误。BF16 数值边界本身仍未消除。

## 1. 这关解决什么痛点

旧基准在同一个进程里依次跑 K=0、K=2、K=4。前一个引擎即使 `shutdown()` 也没有真正
释放，后一个引擎会背着前一个的 CUDA allocator/对象继续跑。验收方复现过：

```text
旧循环 K=2：6.49 tok/s
释放旧对象、清 allocator 后单独跑：19.19 tok/s
```

同时旧结果里 K=0 与 K=2 的 128-token 输出在第 45 个 token 分叉，但当时没有共同历史
的数值快照，无法区分“测量错误、状态错误、浮点误差”。

本关只做三件事：

```text
A. 测量隔离：一个配置一个子进程，父进程不先初始化 CUDA
B. 生命周期：shutdown 真正释放本引擎的模型/KV/图，并阻止关闭后调用
C. 指标与诊断：吞吐/TPOT 口径分开；第一分歧用共同历史 teacher forcing
```

目标不是让数字变好看，而是让下一个优化关卡拿到可信的尺子。

## 2. 路径与上游映射

| 本项目 | 参考 | 本关职责 |
|---|---|---|
| `benchmarks/benchmark_runner.py` | 无直接上游文件 | `run_case` / `run_matrix`：新解释器、结果文件、失败传播、串行 GPU |
| `benchmarks/check_step82_real_spec.py` | 82 基准脚本 | 父进程只编排；preflight/correctness/perf/supplement 每个配置一个子进程 |
| `benchmarks/compare_step82_vllm.py` | vLLM 0.28 `LLM` | target-only / K=2 各自独立子进程，父进程不 import vllm |
| `minivllm/engine/llm_engine.py` | `v1/engine/llm_engine.py` | 用户侧关闭态与幂等 `shutdown()` |
| `minivllm/engine/core.py` | `v1/engine/core.py` | 关闭态检查、执行端关闭 |
| `minivllm/executor/uniproc_executor.py` | `v1/executor/uniproc_executor.py` | 初始化失败也回滚；关闭后禁止执行 |
| `minivllm/worker/worker.py` | `v1/worker/gpu_worker.py` | 资源所有权转交、Runner 引用断开 |
| `minivllm/worker/gpu/model_runner.py` | `v1/worker/gpu/model_runner.py::shutdown` | V2 自己释放图/模型/KV/workspace |
| `minivllm/worker/gpu_model_runner.py` | V1 runner | V1 对应释放与关闭态 |
| `minivllm/compilation/cuda_graph.py` | `compilation/cuda_graph.py` | `_all_instances` 改 WeakSet；按实例 `close()` |
| `minivllm/worker/gpu/cudagraph_utils.py` | V2 graph manager | manager 自己的 `shutdown()`，不全局清图 |
| `benchmarks/spec_decode_metrics.py` | 无直接上游文件 | committed 吞吐、burst 间隔、新 TPOT、分阶段显存 |
| `benchmarks/check_step83_numerics.py` | 无（诊断脚本） | 共同历史 teacher forcing、逐层 hidden、assistant 输入/草稿、FP32 对照 |

**与上游的差异**：上游 `_all_instances` 是强引用集合；本仓库改成 WeakSet，并增加按实例
`close()`，因为学习项目里引擎的创建/销毁发生在同一进程，强引用会让已关闭引擎的图无法
回收。这个差异只影响生命周期，不影响图内的捕获/重放语义。

## 3. A：一配置一子进程

### 3.1 父进程

`benchmarks/benchmark_runner.py` 的 `run_case()`：

```text
父进程写 case.json
  -> subprocess.run([python, worker_script, --case-json ..., --result-json ...])
  -> 子进程写 result.json
  -> 父进程读结果；退出码/超时/缺结果/status=failed 都是失败
  -> 下一个 case
```

约束：

- 用新解释器，不在父进程 `fork()` 后继承 CUDA context；
- 父进程不 import `minivllm` / `torch` 来建引擎；manifest 的 GPU 信息用 `nvidia-smi`；
- 子进程只建一个 engine，预热、测量、保存、`shutdown()`、退出；
- GPU case 串行，父进程不并发启动多个 GPU 子进程；
- 子进程失败时把 `result.json` 的 traceback 带回父进程；不能返回 0 tok/s 装成功；
- `PATH` 补解释器同目录（`ninja` 用于 UVA `load_inline`），不静默切换后端。

`check_step82_real_spec.py` 的 `perf` / `correctness` / `supplement` 现在都走这一层；
`preflight` 也单独跑一个子进程，父进程不用 CUDA 初始化来读 GPU 信息。

### 3.2 子进程记录

每个 perf case 记录：

```text
target/assistant revision、包 hash、prompt token IDs、dtype、attention backend、
Graph 模式、K、batch、上下文/token 预算、KV block、seed/线程数（args 整体）
build_s、load_s_estimate、graph_capture_s、warmup_s
```

加载/捕获与正式测量分开：

- `build_s` 包含构图；
- `graph_capture_s` 来自 runner 的 `cudagraph_capture_stats.capture_seconds`；
- `load_s_estimate = build_s - graph_capture_s`（无图时就是 build_s）；
- `warmup_s` 单列；
- 正式计时前完成对应 shape 的预热；profiler 另跑。

### 3.3 正反顺序小检查

固定中文 prompt、64 token、eager、B=1，K=0/K=2 各独立进程，正反顺序各跑一次：

| 顺序 | K=0 tok/s | K=2 tok/s | K=0/K=2 首分歧 | K=0 加载 allocated/reserved | K=2 加载 allocated/reserved |
|---|---:|---:|---:|---:|---:|
| K0 → K2 | 12.31 | 18.46 | 第 45 token | 13.989 / 20.959 GiB | 14.762 / 20.967 GiB |
| K2 → K0 | 13.00 | 17.77 | 第 45 token | 13.989 / 20.959 GiB | 14.762 / 20.967 GiB |

同一配置的加载 allocated/reserved 在不同顺序下逐字节同值；正式测量 peak 也分别为
14.025/14.799 GiB。这里**不做稳定性能排名**（各一次、没有重复），只验证：

```text
没有跨 case 的模型/图对象重叠
没有出现“后跑的配置背着前一个配置的显存”
```

### 3.4 vLLM 对照也隔离

`compare_step82_vllm.py` 现在 target-only 与 K=2 各一个子进程。隔离后两次 128-token
单样本：

```text
第一次：target-only 28.08 tok/s（4.558 s）；K=2 45.44 tok/s（2.817 s），1.62×
第二次：target-only 31.33 tok/s（4.085 s）；K=2 44.32 tok/s（2.888 s），1.41×
```

两次都没有改 workload，但 target-only 本身就有约 11% 的波动；这再次说明 83 只能给出
“隔离后的可复现协议”，不能拿一次单样本当稳定性能排名。

这**不替换** 82 的历史数字，也不把差异归因于投机算法：

- 82 的 28.53/37.36（1.31×）是旧同进程顺序测量；
- 83 的 28.08/45.44（1.62×）与 31.33/44.32（1.41×）是新隔离协议；
- 两者测量协议不同，不能混在同一张性能表里比较；
- 两次都只有 1 个样本，prefix cache 仍是 vLLM 默认开，正式公平矩阵属 88。

它只说明：**隔离后 vLLM 的 K=2 数字本身也会变化**，旧成绩单不能直接用于算法结论。

## 4. B：shutdown 释放的是所有权

### 4.1 关闭链

```text
LLMEngine.shutdown()
  -> InprocClient.shutdown()
  -> EngineCore.shutdown()
  -> UniProcExecutor.shutdown()
  -> Worker.shutdown()
  -> GPUModelRunner.shutdown()（V1 或 V2）
```

每一层：

- `_shutdown` 标记先置位，重复调用直接返回；
- 关闭后 `add_request` / `step` / `execute_model` / `sample_tokens` 明确报
  “已经 shutdown”，不返回空结果；
- `EngineCore` 在调度之前检查，避免关闭后还推进 Scheduler 状态；
- `Worker` 在 Runner 关闭后把 `model_runner=None`；
- `UniProcExecutor` 初始化半途失败时调用 `shutdown()` 回滚；
- `LLMEngine` 构造 EngineCore 失败时回滚已经装上的 executor。

### 4.2 Runner 释放

V2 `GPUModelRunner.shutdown()` 的顺序：

```text
1. 阻止后续执行（_shutdown=True）
2. torch.cuda.synchronize()（等本引擎已提交工作）
3. 丢弃 target/speculator 的 graph manager 图
4. 丢弃 model（若是 CUDAGraphWrapper，调用 close() 注销全局注册）
5. 清 sampler/rejection sampler/adaptive/execute state/draft handler
6. 清 kv_caches/packed_backing/block_tables/input_buffers/req_states/stream
```

V1 `GPUModelRunner.shutdown()` 做同义清理，额外清 `_padded_buffers` / `_padded_metadata` /
`cudagraph_dispatcher` / `cudagraph_logging`。

关键边界：

- **不调用 `clear_all_graphs()`**：那会清掉另一个仍存活引擎的图；
- 每个 graph manager 只清自己的 `graphs`；
- `CUDAGraphWrapper._all_instances` 改 WeakSet，`close()` 清本实例图并从注册表移除；
- 不销毁全局分布式组；本仓库单卡也不存在“别的引擎的全局资源”；
- `gc.collect()` / allocator 清理只在退出边界，不在每个 step；
- 不要求显存归 0：CUDA context/allocator 会保留；测试检查弱引用、独立 storage、
  allocated 趋势和循环三次不增长。

### 4.3 已有测试的调整

旧测试有 33 处在 `engine.shutdown()` 之后读 `runner.speculator` / `runner.target_cudagraph_manager`。
现在关闭真的释放了这些对象，测试改为：

- 先复制轻量诊断（`selections` 用 `list(...)`，因为 manager.shutdown 会 clear 原列表）；
- 或先在引擎存活时完成断言，再显式 `shutdown()`；
- 需要独立重放的测试，先构造 `ReferenceReplay`，再关闭引擎。

**没有删断言、没有把失败改成 skip**；断言强度不变，只是把“读诊断”放在释放之前。

## 5. C：指标口径

### 5.1 吞吐

```text
output_tokens_per_s = committed_output_tokens / wall_time_s
```

分母只数最终提交的输出 token，不数草稿/验证候选。`AcceptanceAggregate` 的
`num_draft_tokens` / `num_accepted_tokens` 仍是验证阶段口径，单独记录，不和最终
截断后的输出 token 混用。

### 5.2 burst 间隔与 TPOT

旧 `tpot_s` 的分母是“输出事件数 − 1”，实际测的是相邻两次交付之间的平均间隔。83 起：

```text
inter_burst_latency_s = (last_output_at - first_output_at) / (num_events - 1)
tpot_s                = (last_output_at - first_output_at) / tokens_after_first_burst
```

手算例子：

```text
t=1s 交付 3 token，t=2s 交付 2 token
committed_output_tokens = 5
output_tokens_per_s     = 5 / 1s = 5
inter_burst_latency_s   = 1s / (2-1) = 1s
tokens_after_first      = 5 - 3 = 2
tpot_s                  = 1s / 2 = 0.5 s/token
```

单 burst（没有首 burst 之后的 token）：

```text
tpot_s = None（不可定义），不填 0
undefined_requests 计数
```

不能从这两个数推断同一个 burst 内部每个 token 的真实到达时间。

### 5.3 分阶段显存

每个 perf case 记录：

```text
load_capture：build/capture 后 allocated、reserved、peak、device_total
measure_start：warmup 后、reset_peak 后
measure_end：正式测量后 allocated/reserved
measure_peak：正式测量阶段 peak_allocated/peak_reserved/device_total
```

`reserved` 是 allocator 缓存池，不是物理驻留量；`measure_peak` 是从预热后重置峰值到
测量结束的最大值。历史 `benchmarks/results/step82/*.json` **不重写**，它们的 `tpot_s`
是旧口径；新结果带 `metric_schema = "step83"`。

## 6. D：第一分歧定位

### 6.1 复现条件

```text
benchmarks/check_step83_numerics.py
target/assistant 固定 revision
eager、B=1、budget=128、blocks=512（BF16）/64（FP32）
中文 prompt（26 token），max_new_tokens=64
```

两个配置各自子进程；诊断 hook 只在本脚本安装，正式基准不启用。

### 6.2 BF16 结果

```text
first_diff = 45（0-based）
prompt_len = 26
predict_position = 26 + 45 - 1 = 70（request 内绝对 position）
common_output_tokens = 45
common_history_sha256 = affa116f88406f3c8e18217b67594bd005…

position 70 的输入 token：k0=237075，k2=237075（相同）
```

target 在 position 70 的候选：

| 配置 | top1 | top1 logit | top2 | margin |
|---|---:|---:|---:|---:|
| K=0 | 57208 | 15.5625 | 25352 | **0.0625** |
| K=2（step 24） | 25352 | 15.8125 | 57208 | 0.25 |

> 表中 top1 logit 以 trace 的 BF16 值为准；核心事实是 top1 不同，K=0 的 margin 只有 1/16。
> 输出第 45 个 token 分别提交 57208 / 25352，和 target 候选一致。

逐层 hidden 相对 L2（同一 position 70）：

```text
model.layers.0   6.31e-4
model.layers.1   4.97e-3   <- 第一个 > 1e-3 的层
...
final hidden     1.92e-2
```

说明不是“assistant 提错草稿”或“拒绝采样把 target 的候选提交错”：

- 两侧在 position 70 看到的历史 token 完全相同；
- 分歧前的 target top1 在共同 history 上 45/46 个 position 相同；
- 第 46 个 position（predict 70）target 自己已经给出不同 top1；
- assistant 首轮与首次拒绝后的输入也被保存：
  - 首轮 target_step=0、draft_step=0、position 25，采样 57137；
  - 首次拒绝后 target_step=1、draft_step=0、position 27，输入
    `[57137, 47910, 107]`，采样 1018，`num_rejected=[1]`；
  - draft 层到 target 源层映射：layer 0/1/2 → target 13，layer 3 → target 14。

### 6.3 FP32 对照

完整 FP32 前向（不是只换 lm_head）在同一脚本、`--dtype float32 --blocks 64` 下运行成功：

```text
K=0 与 K=2 的 64 个输出 token 完全相同
共同 position 上 target top1 64/64 相同
final hidden 最大相对 L2 = 1.87e-6
position 70：两配置 top1 都是 25352，margin 0.065395
top values：15.6187 / 15.5533
```

BF16 的 `57208` 在 FP32 下不是 top1。BF16 的 margin 只有 `0.0625`，而两种批布局
（K=0 单行 decode vs K=2 1+K 行验证）会改变 GEMM/softmax 的归约顺序；hidden 从 layer 1
开始已有可测差异，误差传到最终 logits 时把 near-tie 翻了。

### 6.4 最终投影控制

`--fp32-final-control` 把**同一份 BF16 hidden** 的最终投影换成 FP32：

```text
position 70：k0 top1 仍 57208，k2 top1 仍 25352
```

所以不能只怪最后一层 lm_head：BF16 hidden 本身已经不同。这个控制项排除的是
“仅最终 GEMM 舍入”；完整 FP32 前向才是更强的对照。

### 6.5 归因与边界

```text
归因：target 的 BF16 near-tie / 布局相关数值差异
不是：assistant 状态错误、拒绝采样提交错误、KV 路由错误、权重加载错误
未消除：BF16 下这枚 token 的翻转仍然存在
```

因此：

- 这个 case **不进入严格等价性能表**；
- 84（RoPE/融合）和 85（Norm/激活/QKV）可以改变舍入，但不能承诺“逐 token 等价”；
- 未解释的差异没有宣布正常；这里给出的是固定历史、逐层 hidden、FP32 对照三类证据。

## 7. 测试与实测

### 7.1 新增测试

| 文件 | 数量 | 覆盖 |
|---|---:|---|
| `tests/step83/test_metrics_contract.py` | 4 | committed 吞吐、burst 间隔、新 TPOT、单 burst 不可定义 |
| `tests/step83/test_benchmark_isolation.py` | 6 | case 子进程、非零退出、异常/缺结果、超时、矩阵失败隔离 |
| `tests/step83/test_shutdown_chain.py` | 6 | 双重 shutdown、关闭后调用、活引擎隔离、Runner 释放、半初始化回滚、图注册弱引用 |
| `tests/step83/test_gpu_runner_lifecycle.py` | 3 | 真实 tiny V1 Graph 注销、V2 三次建销毁、两个活引擎互不清 |
| `tests/step83/test_numerics_compare.py` | 3 | position 偏移、teacher-forcing 证据、缺失 logits 不当相等 |

`tests/step83` 合计 **22 passed**。

### 7.2 实跑命令

```bash
PY=/home/user/anaconda3/envs/vllm-omni-dev/bin/python
export PATH=/home/user/anaconda3/envs/vllm-omni-dev/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# 正反顺序隔离检查
$PY benchmarks/check_step82_real_spec.py --perf \
  --batch-sizes 1 --ks 0 2 --max-new-tokens 64 --repeats 1 \
  --output-dir benchmarks/results/step83/order_forward
$PY benchmarks/check_step82_real_spec.py --perf \
  --batch-sizes 1 --ks 2 0 --max-new-tokens 64 --repeats 1 \
  --output-dir benchmarks/results/step83/order_reverse

# BF16 第一分歧 + 最终投影控制
$PY benchmarks/check_step83_numerics.py --max-new-tokens 64 \
  --output-dir benchmarks/results/step83/bf16 --fp32-final-control

# 完整 FP32 对照
$PY benchmarks/check_step83_numerics.py --max-new-tokens 64 \
  --dtype float32 --blocks 64 \
  --output-dir benchmarks/results/step83/fp32

# vLLM 隔离对照
$PY benchmarks/compare_step82_vllm.py

# 全量回归
$PY -m pytest tests/step58 … tests/step83 -q
```

### 7.3 结果

```text
tests/step83：22 passed
全量：1154 passed, 149 warnings in 161.01s
```

（全量命令实际包含 `tests/step58`–`tests/step83`；无 failed/skipped。退出码 0。）

结果产物在 `benchmarks/results/step83/`（`benchmarks/results/` 已 gitignore，不提交）：

```text
order_forward/perf.json
order_reverse/perf.json
bf16/trace_k0.pt、trace_k2.pt、fp32_top_k0/2.json、numerics_summary.json
fp32/trace_k0.pt、trace_k2.pt、numerics_summary.json
```

## 8. 已知边界/未跑项

1. **只测一个中文 prompt、64 token**；其他 prompt/长度可能有不同的首分歧位置，不能外推。
2. **BF16 数值差异未修**；83 只定位。84/85 是算子和 RoPE 优化，不承诺逐 token 等价。
3. **没有跑 B=1/4 × K=0/2/4 完整矩阵**；公平对照属 88。
4. vLLM 对照最新只有 1 个样本；prefix cache 仍与 minivllm 不同；不是算法归因。
5. FP32 对照用 `blocks=64`，不是 512；它是数值控制，不是性能配置。
6. 诊断 hook 会保存逐层 hidden，属于侵入式诊断；正式计时路径无 hook。
7. 外部测试/调用方若继续持有 Runner、张量或 generator，GPU 显存不会严格归 0；
   本关的判据是 owner 引用断开、重复关闭安全、三次循环不持续增长，不是“显存归零”。
8. 历史 `step82` 结果文件保留旧 `tpot_s` 语义；不要把 82 与 83 的 `tpot_s` 混用。
9. `benchmark_runner` 的超时/OOM 是明确失败，但不会自动重试；OOM 后的 CUDA context
   是否可继续使用不在本关保证范围内。

## 9. 一句话

**83 把“同进程连跑多配置”改成“一配置一子进程”，把 shutdown 从空操作改成按所有权释放，
把旧 TPOT 改名并新增按首 burst 后 token 计算的 TPOT，再用共同历史 + 逐层 hidden + FP32
把 64-token 第一分歧定位为 target BF16 near-tie；它不优化模型，也不跑完整性能矩阵，但让
84–88 的结论第一次有了可复现、可归因的测量基础。**
