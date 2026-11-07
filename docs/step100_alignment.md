# 100 TP 采样与请求镜像：实施前冻结源码映射

| 冻结 vLLM 0.28.0 | 职责/顺序 | 本项目落点 |
|---|---|---|
| `v1/worker/gpu_worker.py::init_device/compile_or_warm_up_model` | 分布式初始化后按相同 model seed 设置 Python、NumPy、Torch/CUDA 随机状态；模型初始化、profiling、warmup 后再次设置，不加 rank | `worker/worker.py`，模型 seed 在 `ModelConfig` |
| `v1/worker/gpu/sample/states.py::SamplingStates.add_request` | 显式 seed 原样按 request slot 保存；None 从进程 NumPy RNG 产生，每 rank 必须请求顺序一致 | `worker/gpu/sample/states.py` 保持冻结公式和 slot owner |
| `v1/worker/gpu/sample/sampler.py::Sampler.sample` / `gumbel.py` | 请求映射、seed、绝对位置与 logits 共同决定采样；处理顺序按冻结实现 | `worker/gpu/sample/`，不改概率和采样顺序 |
| `v1/worker/gpu/model_runner.py` | 每个 TP rank 接相同调度快照、计算完整 logits、各自采样与 `post_update` 更新镜像 | `worker/gpu/model_runner.py`；测试专用 Worker 导出诊断 |
| `v1/executor/multiproc_executor.py::_get_output_rank` | 各 rank 执行，单一 output rank 返回结果，其余回执；Scheduler 只提交一次 | `executor/multiproc_executor.py` 与 EngineCore/Scheduler |

固定例：TP2、PP1、A seed17/B seed23、K=0；调度快照与真实 token 历史同源，两个 rank 的完整 logits、逐请求 seed/position、采样输出、执行侧状态均应一致。只做输出结果选择，不把 token 强制广播来遮盖分歧。Scheduler 独占 Request；测试探针只观测 Runner 自己的镜像，生产热路径不新增每轮 D2H。

既有 `SamplingParams` 在请求初始化拒绝 logit_bias；本关先记录并实测该负例，不宣称已支持 logit_bias，避免不具备 V1/V2 与投机 bonus 行契约的半套实现。完整真实模型多卡仍属101；多 rank Graph 属104/105；不解除 async_scheduling 端到端拒绝。

## 本关实现与测试状态

`ModelConfig.seed` 沿冻结模型配置默认为 0；多进程 Worker 完成组初始化后设置 Python/NumPy/Torch/CUDA 随机状态，`compile_or_warm_up_model` 返回前再次设置。单进程旧生命周期保持原有行为。Worker 仍只传 Scheduler 数据快照，全部 TP rank 执行 `sample_tokens`，只有 output rank 的结果向 Scheduler 返回；没有增加 rank0-only 采样或 token 广播。

测试专用 `Step100Worker` 在各 rank 模拟不同数量的 NumPy/Torch 初始化随机消耗，并在测试中导出真正 Runner 生成的原始 logits、请求 slot/position/seed、采样 token/数量/logprobs，以及本地 `all_token_ids`、`last_sampled`、总长与计算进度。探针不进入生产路径。TP2/TP4 均分别验证 greedy、带 top-k/top-p/min-p/惩罚的随机采样，以及真实 KV 容量压力导致的抢占恢复；普通场景让 A/B 在下一轮换 batch 行，取消 A 后让 seed=None 的 C 复用 slot；抢占场景让 seed=None 的 B 在 A 结束后恢复。每轮逐 rank 比较数据，且 Scheduler 历史每次只增加一次。

冻结 Gumbel 文件哈希与安装文件一致；固定 8192 个 seed 和位置的 CUDA 结果与冻结实现逐 token 相同。经验分布 `[0.18798828, 0.30895996, 0.50305176]`，softmax 目标 `[0.18632373, 0.30719587, 0.50648040]`，最大绝对差约 `0.00343`，预设验收阈值 `0.03`。logits rank 间预设 `atol=1e-5, rtol=1e-5`；离散 token、长度、seed、采样计数和 logprobs 按测试断言精确相同。

明确未开放：`logit_bias` 仍在 `SamplingParams` 初始化拒绝（未实现 bonus 行语义）；`logprobs=-1` 不扩大现有支持结论；真实 Qwen3-8B/14B 多 rank、投机跨 rank、多 rank Graph 和端到端异步调度未由本关验收。

最终 `benchmarks/results/step100/acceptance_release_traces/` 保存 6 份逐轮真实 rank 快照。每场景逐轮各 rank 的原始 logits 最大绝对差与相对 L2 都为 0；例如 random4 第 0 轮、rank 1、logits `[0,0]` 与 rank 0 同为 `-0.055132728070020676`。greedy4 原始 logits 的最小 top1/top2 margin 约 `0.00152491`（token 0: `0.22948602`，token 4: `0.22796111`）；random4 的最小原始 margin 约 `0.01001336`。两组均仅代表该 tiny FP32 测试，不代表 BF16 真实模型无近似平局。
