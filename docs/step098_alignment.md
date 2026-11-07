# 098 本地 Attention 与 KV 容量协议

## 实施前源码映射与状态所有权

| 职责 | 冻结 vLLM 0.28.0 | 本项目落点与顺序 |
|---|---|---|
| Q/K/V 头与本地 Attention | `model_executor/models/qwen3.py::Qwen3Attention.__init__` | Q/K/V 投影、norm/RoPE、Attention 的 Q/KV 头为本地；positions 不分片，输出由 Row 层归约 |
| 每 rank KV spec / 分配 | `v1/worker/gpu/model_runner.py`、`v1/kv_cache_interface.py` | 由本地 Attention 的 KV 头和 head_size 生成 spec；每个 Worker/Runner 拥有本地物理 tensor、metadata 和工作区 |
| 容量规划 | `v1/core/kv_cache_utils.py::get_kv_cache_configs` | 汇总每 Worker spec 与可用预算，先计算每 rank 块数再取共同最小，所有本地配置同步缩小 |
| Scheduler 规格 | `generate_scheduler_kv_cache_config` | 公共容量确认后创建唯一 Scheduler/BlockPool；相同逻辑 id 指向各卡各自的物理块 |
| Worker 预算来源 | `v1/worker/gpu_worker.py::determine_available_memory` | 冻结上游执行模型峰值 profiling；本项目当前 Runner 尚无 profile_run，必须显式记录预算口径，不把瞬时空闲内存冒称已完成上游 profiling |
| padding | `v1/worker/gpu/model_runner.py`、各 Attention 后端 | slot=-1 是无效行；Triton 写入跳过；Torch Graph 依旧依赖 null block，不能改变既有约束 |

固定例：8B、TP4、BF16、block=16、每卡 KV=2、D=128；每层每块 `2×16×2×128×2=16384` 字节。Scheduler 独占 Request/逻辑映射；各 rank 只持有执行快照与本地 GPU cache。KV 复制时按实际本地头计算，不按全局字节简单除 TP。

## 协议与实现

1. 097 已使 Qwen3 投影、RoPE、Q/K norm 与 Attention 使用本地头数；V2 `get_kv_cache_spec()` 从本地 Attention 创建 `FullAttentionSpec`，每层 cache 为 `[2, blocks, block_size, local_kv_heads, head_dim]`。Triton 工作区由原有 metadata builder 管理，按本地 Attention 的头数取桶。
2. `MultiprocExecutor.get_cache_config()` 分别收集 Worker 的本地规格和可用字节，再调用 `get_kv_cache_configs`。单 full-attention 组每块的消耗是 `层数 × page_size_bytes`；各卡各自计算容量，统一取最小值，Scheduler 使用同一逻辑块数，Worker 校验规格后分配各自的物理 tensor。
3. `num_gpu_blocks` 是项目原有的显式**上限**：实际块数为 `min(该上限, floor(当前空闲字节 / 每块字节))`，再跨 rank 取最小值。与冻结上游的 `num_gpu_blocks_override` 不同，不能说是同一个配置。容量不足一个最大长度请求时拒绝，FULL Graph 的 null block 另计。
4. Worker 返回的是 `torch.cuda.mem_get_info` 的空闲显存快照，不包含未来模型前向、图工作区或其他进程竞争的峰值。冻结 vLLM 使用 profiling；本项目尚无等价 profiling。该估计不能保证后续不会 OOM，也不能声称显存 profiling 已对齐。真实多卡模型加载仍在 Worker 入口拒绝；本关测试通过专用测试 Worker 注入 tiny 模型。
5. 只支持各 rank 同样的单 full-attention 组；PP 各 stage 异构层、多 KV group、混合类型仍拒绝。Scheduler 独占请求与块池；RPC 传快照，Worker 不回写 Request。

| 测试时刻 | 调度权威状态 | rank 本地状态 |
|---|---|---|
| prefill 17 行 | 一个请求占两个块，位置 0–16 | 每层写对应本地 KV 头 |
| decode 1 行 | 原块表不变，位置 17 | 每层只更新该位置 |
| 取消并新请求 | 旧请求释放块，新请求复用两块 | 新 KV 覆盖新请求的有效行 |

## 验证与范围

源码差分通过文件 SHA256 确认安装版 `v1/core/kv_cache_utils.py` 与冻结文件逐字节相同，再调用其中的实际规划函数；按相同全注意力规格和异构预算，两边均得到 `[6,6,6,6]`。上游 tensor 分配字节数亦核对。配置上限和 Graph null block 的差异在上节明确。

真实 4090 多进程使用两层随机 tiny Qwen3：TP1/2/4，FP32，block 16，17 行 prefill、1 行 decode，再取消并复用；按逻辑 block ID 和有效位置逐层比较每个本地 KV 头，TP4 对应的两个副本还逐值相等。Triton 的 `slot=-1` 在每个 rank 不更改实际 KV。测试 Worker 各自读取显存，并用测试专用上限模拟 8/7/6/8 块；公共容量为 6；另一个 rank 只有 3 块而最长请求需 4 块时初始化失败。tiny 权重仅用于机制验证，不是真实 Qwen3-8B/14B 多卡验收。

具体命令、原始日志、设备与版本见 `docs/results.json` 的 `step098`。多 rank Graph、真实模型多卡、完整显存 profiling 均不在上述通过结论内。

回归记录中保留一次预期之外的失败：首次 58—88 回归为 1288 passed / 1 failed / 23 skipped，失败是 `test_current_package_hash_matches_results`，因为本次改动后尚未更新 `current_package.sha256`。已按既定算法重算当前包摘要，不改历史记录；最终回归以重跑日志为准。

固定 tiny 历史、两层所有有效 KV 元素按 `atol=1e-5, rtol=1e-5` 比较；TP2 最大绝对误差 `5.96046448e-7`，相对 L2 `9.14845799e-8`，出现在取消后新请求、layer 1、rank 1、位置 2、局部 KV 展平元素 9（本项目分片 -1.1342331171035767，单卡 -1.1342337131500244）；TP4 最大绝对误差 `7.15255737e-7`，相对 L2 `1.07566045e-7`，出现在同轮 layer 1、rank 2、位置 1、局部元素 3（-1.1423944234848022 vs -1.142393708229065）。误差比较是项目 TP 与项目 TP1，不是 vLLM 完整模型对照。
