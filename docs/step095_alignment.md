# 095 多进程失败传播与幂等清理

## 实施前源码映射

| 项目职责 | 冻结 vLLM 0.28.0 对照 | 调用顺序与所有权 |
|---|---|---|
| Worker 存活监控 | `v1/executor/multiproc_executor.py::start_worker_monitor` | ready 后监测各进程 sentinel；意外退出标记失败、关闭执行组、通知控制端；进程由 Executor 拥有 |
| 退出升级 | 同文件 `_ensure_worker_termination`、`shutdown` | 先关闭 death pipe 请求退出；有限等待，再 terminate、kill；进程退出后释放通信资源 |
| Worker 父进程死亡 | 同文件 `WorkerProc.monitor_death_pipe` | Worker 监控 pipe；父进程消失时唤醒队列等待，Worker 负责自己的 Runner 和通信组 |
| 控制侧停摆 | `v1/engine/core.py` | 一旦执行端致命失败，后续 schedule 前拒绝；Scheduler 唯一拥有 Request |
| 组清理 | `distributed/parallel_state.py` | 每个 Worker 清理自己拥有的组；失败清理不增加缺 rank barrier |

本关处理进程/控制状态，不改变模型 tensor 布局、采样或 KV 数据布局；命令传输仍使用 SchedulerOutput 快照。真实多卡模型、Graph 和投机尚未开放，使用测试 Worker 注入故障不能冒充真实模型验收。实施证据待补。

## 实现及边界

- Executor 的 monitor 等待各 Worker 的进程 sentinel 和独立故障管道；异常管道承载 rank、PID、方法和 traceback，避免父进程等较早 rank 时丢失故障 rank 的原始异常。意外退出仅能报告进程退出信息。故障管道是本专题为保留原始异常所作的明确扩展，并非声称上游使用了同一协议。
- failure 首次置位后通过 callback 通知 EngineCore；EngineCore 在 schedule 前再次检查执行端状态。RPC 的 Future 读取响应时检查失败并拒绝成功，propose 输出回收异常也使 EngineCore 停摆。错误优先保留首次观察到的异常，不能保证进程被杀后仍有 Python traceback。
- 对照上游 Worker death pipe monitor，Worker 单独线程监控父进程管道并唤醒本地队列；Executor 请求退出后限时 join、terminate、kill，关闭队列与本 Executor 持有的共享内存；关闭互斥并协调在途读取和写入。共享内存队列增加关闭检查，使满队列的写入等待可中断。这是本地队列生命周期扩展，不改变消息布局。
- 取消请求与 Worker 致命失败分开；不实现自动重启、请求迁移和半轮重试。真实多卡模型、真实多 rank Graph、真实多卡投机仍未开放；本关通过测试 Worker 在相应 Worker 接口注入故障，不将其等同于真实模型验收。

## 复现与验证

修改前两卡复现：rank 0 在 RPC 中等待、rank 1 直接退出，`collective_rpc(..., timeout=6)` 最终只报 RPC timeout；修改后无 RPC timeout 的同类调用可检测 rank 1 死亡并失败退出。完整命令与最终执行结果见 `docs/results.json → step095.results` 和 `benchmarks/results/step095/`。

验收期间检查原始日志发现了测试断言未覆盖的并发问题：Worker 的 death-pipe 监控线程可能在 Worker 关闭输入队列之后仍向已关闭的 ZeroMQ socket 发送唤醒，产生 `Exception in thread DeathPipeMonitor`。本实现对队列关闭与监控唤醒使用同一生命周期锁，并让验收脚本检查后台线程未处理异常；此前 13 个逻辑状态为 passed 的结果不能代替此修复后的验收。

最终源码包 SHA256：`ca720066d43737f6e12a2b9241aeee47ffd4bb7f64f95789f1c8b581a79c436d`。冻结 `multiproc_executor.py` 与安装版均为 `d5ca86635611758c2bd543ffd1fe9ad200db62880ce1f47eda8e4c63847595ea`；测试直接调用该上游 monitor，验证真实子进程退出触发关闭和 callback。

- 095 独立 pytest：最终生产源码下 15 passed；后加的 `kill_escalation` 独立执行 1 passed。它令 rank 0 忽略 SIGTERM，实际断言退出码为 `-9`。
- 独立验收：13 个原有场景通过，后加的 `kill_escalation` 单独通过。所有案例都检查没有未处理的后台线程异常。故障注入的应用父进程按预期非零退出；EngineCore 拦截与新实例成功路径以零退出。
- 最终源码下 093/094：24 passed；58—88：1289 passed、23 skipped。089—094 在最后的队列关闭竞态修复之前为 112 passed；受影响的 093/094 已在修复后重跑。旧真实权重 skipped 不算通过。
- 测试后 `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader` 输出为空。进程、共享内存与 IPC 路径检查限定在各测试自己创建的资源，不清理其他项目。
