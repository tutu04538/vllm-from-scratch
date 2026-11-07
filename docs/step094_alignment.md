# 094 Executor 广播协议与异步 RPC 结果

## 实现前的冻结源码对照

| 上游冻结入口 | 职责、时序与布局 | 项目落点 |
|---|---|---|
| `distributed/device_communicators/shm_broadcast.py::ShmRingBuffer` | 单写多读环形共享内存；槽数据区、写标志、逐读者消费标志；所有读者确认后槽可复用 | `minivllm/distributed/shm_broadcast.py`；仅单机本地读者 |
| `MessageQueue.export_handle/create_from_handle/wait_until_ready` | 句柄握手、订阅确认后才能发送；overflow 使用 ZMQ multipart | 同模块，保留单写多读和消息格式 |
| `MessageQueue.enqueue/dequeue` | pickle protocol 5 和 CPU tensor out-of-band buffer，按槽容量选择内联/overflow；读后确认 | 同模块；不把 GPU tensor 带入 CPU 队列 |
| `MultiprocExecutor.collective_rpc` / `WorkerProc.worker_busy_loop` | 任务元组 `(method,args,kwargs,output_rank)` 广播，各 Worker 按顺序执行；响应状态来自独立响应队列 | `minivllm/executor/multiproc_executor.py`；execute 与 sample 独立 |
| `FutureWrapper.result` | Future 按提交顺序消费响应，non_block 时立即交回尚未结算的 Future | 同模块；不把同步调用包装成已完成 Future |
| `MultiprocExecutor._get_output_rank` | PCP=1 时最后 PP stage 的 TP rank 0，即 `world_size-TP` | 同模块；结果对外一次 |
| `v1/serial_utils.py` / `v1/outputs.py` | 引擎消息序列化和结果类型；Worker RPC 由上述队列 pickle/OOB，不能把 Request 活对象交给 Worker | 保留现有 `SchedulerOutput` 数据快照和项目结果类型 |

源码差异：冻结 `WorkerProc.worker_busy_loop` 在只选 output rank 回复时，对非 output rank 方法异常仅记录日志，不入响应队列；本关要求非输出 rank 故障不能忽略。已在 `vllm_bugs/UPSTREAM_SUSPECTED_BUGS.md` 登记 SHA 匹配后直接调用上游方法的最小复现。实现需使各 rank 的状态可检查，但对外仍只返回 output rank 的结果，记录这一有意差异。正式权重分片、多卡模型推理和端到端 `async_scheduling=True` 仍不开放。

## 本关实现和明确差异

- 单机队列保留冻结上游协议 5 的共享内存环形槽格式（写标志、逐读者消费确认、长度与 out-of-band buffer）及 ZMQ overflow。CPU tensor 在上游对应阈值内内联或经 OOB buffer；当前拒绝 CUDA tensor 和未支持的特殊 tensor 布局，不把 GPU 数据通过 Python pickle 跨进程搬运。多机远端读者及 spinloop 扩展不在本专题范围。
- Worker 初始化和加载探针后创建输入/响应队列并通过 ready 消息传递响应句柄；主进程按冻结顺序先完成广播队列握手，再完成各响应队列握手。Socket 文件放于 Executor 独占临时目录，强制退出的子进程对应共享内存由其预分配名字回收；不处理其他 Executor 的资源。
- 上游 `unique_reply_rank` 下非输出 rank 失败可能只记录日志；本关规定不能遗漏，故每个 rank 使用同一 SUCCESS/FAILURE 响应格式回执，非输出 rank 成功时只传空回执，对外仍只交 output rank 的结果。上游最小差分见本地私有 `vllm_bugs/UPSTREAM_SUSPECTED_BUGS.md` 和 `tests/step094/test_step094_queue.py`。本地对该错误传播的行为不是逐字节对齐上游。
- `FutureWrapper` 按冻结实现延迟读取响应并按提交顺序结算；Future pending 不代表后台有线程自动设置结果。EngineCore 的执行/采样调用依然分开。使用 FakeRunner 的测试 Worker 能走通 EngineCore；生产 Worker 的多卡权重分片仍拒绝。

## 实测结果

- `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/step094 -q -ra`：**11 passed / 0 failed / 0 skipped**。单槽慢读者不被覆盖、内联 OOB tensor 在槽复用后仍保持数据、CUDA tensor 拒绝、ZMQ overflow、异常和资源清理均有测试。
- `python benchmarks/check_step094_acceptance.py --case all --timeout 180 --output benchmarks/results/step094/acceptance.json`：**4 场景通过**。TP=2/PP=2 四个真实 Worker 逐字段收到相同 SchedulerOutput（新请求、恢复请求、完成 ID）；慢 Worker 时 execute/sample 的 Future 都 pending，反序等待仍返回正确轮次，采样结果来自 rank 2。连续 12 条消息和 **25,169,920 字节** overflow 消息正确往返。
- 非 output rank 1 的方法抛异常、进程突然退出均在真实四 Worker 上检测；后续提交失败并清理残留 SHM/IPC。EngineCore 的 FakeRunner 探针生成 `[7,8]`，并验证 rank 1 执行失败后下一轮在 Scheduler 前拒绝；这不是模型前向或数值验证。
- 上游五份源码 SHA256 与安装版匹配；本项目写入→上游读取及上游写入→本项目读取双向实际跨进程互操作通过。直接调用上游 `worker_busy_loop` 复现非 output rank 异常仅日志、不发送回复。
- 58—88 单卡回归：**1289 passed / 23 skipped**；089—093 回归：**101 passed**。缺旧模型权重的 skip 不算通过。需求/冻结源码静态审计 8003 项、0 错误。
- 原始 JSON、逐 rank PID/GPU UUID、子进程退出码及日志在 `benchmarks/results/step094/`；命令、版本和包摘要见 `docs/results.json → step094`。无真实模型加载；多卡参数分片、Graph collective、端到端 async_scheduling 和多机不在本关完成范围。

包摘要说明：完整既有回归和验收之后仅移除 `shm_broadcast.py` 末尾空白行，没有改变执行语义；094 的 11 项测试在最终格式和最终包摘要下重跑通过。格式修正前摘要保存在 `docs/results.json → step094.results.pre_final_format_package_sha256`。
