# 102 多卡 KV 生命周期与请求调度闭环

## 实施前冻结源码映射与状态约束

| 项目位置 | 冻结 vLLM 0.28.0 对应职责 | 顺序与布局 |
|---|---|---|
| `core/sched/scheduler.py::schedule` | `v1/core/sched/scheduler.py::Scheduler.schedule` | 唯一 scheduler 分配逻辑块、决定抢占和恢复；V2 新请求及恢复请求使用 NewRequestData，常驻请求使用 CachedRequestData |
| `core/kv_cache_manager.py` / `block_pool.py` | `v1/core/kv_cache_manager.py` / `kv_cache_utils.py` | 仅完整且已计算的块可发布为 prefix；引用和空闲队列属于控制面，块号是逻辑地址 |
| `worker/gpu/model_runner.py::execute_model` | `v1/worker/gpu/model_runner.py::GPUModelRunner.execute_model` | 顺序为完成/抢占清理 → free → 新请求 → 增量更新 → block table staging → input/slot_mapping → forward；各 rank 存本地 KV 头而共享相同逻辑 block IDs |
| `worker/gpu/states.py` / `block_table.py` | `v1/worker/gpu/states.py` / runner 更新 | scheduler 快照校正镜像 computed 与块表，重用 request slot 必须覆盖旧状态；物理 KV 不按跨 rank 逐值相等断言 |
| `engine/core.py::step` / Executor | `Scheduler.update_from_output` / `v1/executor/multiproc_executor.py` | 只有完整正常结果才能提交；取消的请求不因迟到结果复活；任一 rank 失败使副本停摆 |

固定示例：TP2、block size 4、A 的 6 个已计算 token 在示意逻辑块 [1,2] 中；第 2 块仅前两项有效。B 抵达导致指定场景的 Scheduler 抢占，A 按新调度协议恢复；取消 A 后 C 复用 slot 时不继承 A 的块表或随机状态。逻辑块 ID 以测试池为准，不作为固定生产编号。实现先核实 prefix on/off、每 rank 输入和有效 KV、取消窗口、容量最小 rank 与缺结果，测试期望采用冻结轨迹或独立重放 oracle。

## 实现与上游对照

实际发现 `SchedulerOutput` 缺少 `preempted_req_ids`：Scheduler 抢占后释放逻辑 KV 块，但 V2 Runner 虽已有 `finish_requests` 的抢占清理分支，却收不到通知。被抢占的请求继续占据 resident slot 和旧块表，可能妨碍其它请求进入。冻结 `Scheduler._preempt_request` 将请求加入 `reset_preempted_req_ids`，`schedule` 把集合装入输出，`_update_after_schedule` 重新绑定空集合；本关按同样顺序接通，Worker 复用已有清理实现。对冻结方法做实际调用及源码 SHA 对照，并在真实 TP2/4 上检查抢占当轮每个 rank 不再持有 victim。

移除 Qwen3 TP 的 prefix caching 加载拒绝；沿用既有 Scheduler / KV Cache manager，不在 Worker 独立决策块号。完整块仅在既有 `update_from_output` 的已计算边界后发布，这个既有实现较冻结上游 `allocate_slots` 发布更晚，可能少一些命中机会，不能称为时序逐字一致。101 的 prefix 负例随支持范围变化改为正向验证；其他非法组合断言保留。

测试通过真实生产 Worker，测试专用 RPC 在每次准备 attention 时记录请求、resident slot、computed、query 范围、token、position、逻辑 block table 和物理 slot_mapping；每轮采后快照同时记录各 rank 的局部 KV、种子与请求状态。独立算术 oracle 根据 position、block size 和块表计算物理 slot，并以活请求块表重建引用计数；对照 TP1 的对应历史时按 KV head 重组，TP4 的相邻 rank 按 GQA 复制规则比较。被抢占请求当轮须从每个 rank 的常驻表消失，恢复时由 NewRequestData 重建。

prefix 开关、TP2/4 和真实中途抢占交叉验证；相同 prompt 前缀命中与不命中输出单独比较。取消发生在下一次执行之前，或已拿到 Worker 结果、但 Scheduler 尚未提交的测试注入边界；检查迟到结果不能复活请求、新请求复用旧 slot 时块表、token 与 seed 已替换。rank 1 在采样时注入真实失败，核对 Scheduler 没有提交输出且下一轮调度前拒绝；不同 rank 显存上限取最小值，容量不足在运行前明确拒绝。测试 hook 与诊断 D2H 只存在测试路径。

执行记录：首次 58—88 回归为 1288 passed / 1 failed / 23 skipped，失败项检查当前包 hash；原因是回归前尚未把生产改动后的 hash 更新到 `docs/results.json`。已更新 hash 并单独复验该检查通过，完整回归以修正后的重新运行结果为准；首次失败保留 `single_regression_initial_hash_failure.log`。

在 102 改动后重跑 101 验收矩阵：tiny、真实 8B 的 TP1/2/4 短前向和真实 14B TP2 加载均通过；这不是 103 的 32 token 真实端到端验收。原始结果位于 `benchmarks/results/step102/step101_real_regression.json`。

最终结果：本关 pytest **17 passed / 0 failed / 0 skipped**，点名验收六类全部通过；58—88 重跑 **1289 passed / 0 failed / 23 skipped**（历史真实权重缺失），089—101 **169 passed / 0 failed / 0 skipped**。每 rank 原始轨迹位于 `benchmarks/results/step102/acceptance_final_traces/`。TP 多 rank Graph、EAGLE3 多卡、跨 DP 前缀共享、KV transfer 和在线迁移仍不属于本关；102 不证明真实模型 32 token 端到端。命令和源码 hash 见 `docs/results.json → step102`。
