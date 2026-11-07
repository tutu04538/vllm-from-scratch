# 093 多进程 Worker 启动与设备所有权

## 实现前：冻结源码对照

| 冻结 vLLM 0.28.0 | 职责和顺序 | 本项目落点 |
|---|---|---|
| `MultiprocExecutor._init_executor` | 全部 Worker 先启动，之后等待全部 ready；建组可能同步，逐个启动逐个等待会死锁 | `minivllm/executor/multiproc_executor.py` |
| `WorkerProc.make_worker_process` / `worker_main` / `wait_for_ready` | 进程句柄、ready pipe、父进程死亡管道；子进程执行 Worker 初始化、加载后报告 READY | 父进程 `spawn`；明确阶段/rank 错误；就绪后保持子进程，管道 EOF 时退出 |
| `WorkerProc.__init__`、`WorkerWrapperBase.init_worker` | worker wrapper 创建本地 Worker；依次 `init_device()`、`load_model()` 后才发送最终 READY | 直接用项目 Worker 完成职责；093 无模型探针 ready 只表示探针就绪 |
| `gpu_worker.Worker.init_device` / `init_worker_distributed_environment` | 设备绑定 → 分布式和模型组 → 创建 Runner；GPU 权重在各子进程拥有 | `minivllm/worker/worker.py`；本关真实多卡模型权重明确拒绝，后续分片关卡解锁 |
| `utils/system_utils.get_mp_context` | 选择进程创建上下文，不能让已初始化 CUDA 的父进程 fork | 本关固定 spawn；传可序列化配置而不传 GPU 对象 |

真实 vLLM 的 READY 是加载模型和队列初始化之后；需求 093 示例“ready 后创建 Runner”不能替代这个事实。093 测试使用无模型 `minivllm/testing` 探针覆盖部署，不创建模型 Runner，不进行分片、KV 绑定或模型前向；普通多卡模型入口要显式拒绝，不能复制整份权重或在探针就绪后开始调度。

## 本项目的启动边界

- `VllmConfig` 对 V2、CUDA、DP=1 的多进程布局允许进入 Worker 启动；V1 多卡、DP 多副本、external launcher 和端到端异步仍拒绝。工厂按 `mp` 选择真实 `MultiprocExecutor`。
- `MultiprocExecutor` 使用 `spawn` 启动所有子进程后，等待每个 rank 的 readiness 管道。每个 Worker 先绑定设备并初始化 092 通信组，再调用自己的模型加载阶段；只有成功后才发送 READY。父进程不构造 GPU 模型或跨进程传 CUDA 张量；配置递归检查拒绝 CUDA 张量。
- 因本关没有 TP 权重分片，生产 Worker 的多卡 `load_model()` 明确报错，不能完整复制权重。独立测试使用 `minivllm/testing/step093_probe.py` 的无模型 Worker：它只验证真正的进程、GPU、组和启动/关闭，READY 的 `model_loaded=false`；探针不能创建可调度引擎。
- `get_cache_config` 和执行 RPC 明确拒绝；`EngineCore` 因此不能先于容量和后续 RPC 建立 Scheduler。094 再接入共享内存 RPC，096 等关实现真实分片和模型执行。
- 进程编号 `local_rank` 和可见 GPU `device_index` 独立保存：`device_ids=(1,0)` 时进程 local_rank 仍是 0、1，而设备索引为 1、0。092 进程组的 local_rank 和 device_index 因此也分别记录；默认映射不变。
- 失败包含 rank、阶段与 traceback；初始化失败或 shutdown 会关闭管道、等待子进程，有界超时后终止/kill，清理所属 FileStore 临时目录。与上游的 shm 消息队列握手和执行期健康监测分别留给 094、095。

## 实测结果

- `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/step093 -q -ra`：**13 passed / 0 failed / 0 skipped**。覆盖不同 PID/GPU UUID、显式 `device_ids=(1,0)`、`CUDA_VISIBLE_DEVICES=2,0`、设备/加载阶段 rank 1 异常、未全部 ready 时禁止调度、CUDA 张量配置拒绝与冻结源码哈希核对。
- `python benchmarks/check_step093_acceptance.py --case all --timeout 120 --output benchmarks/results/step093/acceptance.json`：**6 场景通过**。2/4 Worker 在同一父进程各自连续启动、关闭两次；父进程前后 `torch.cuda.is_initialized()` 均为 `False`；每次就绪记录 PID、local_rank、设备序号、UUID 和 TP/PP 组。设备/加载失败以及生产模型加载拒绝都有 rank、阶段和清理证据。
- `CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest $(printf 'tests/step%s ' $(seq 58 88)) -q -ra`：**1289 passed / 23 skipped**。`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/step089 tests/step090 tests/step091 tests/step092 -q -ra`：**88 passed**。23 个旧权重缺失案例仍为 skipped，不能当作通过。
- 需求和冻结源码静态审计 **8003 checks / 0 errors**。逐 Worker 数据、故障日志和原始 JSON 在 `benchmarks/results/step093/`，此目录不纳入 Git；命令、包摘要和版本在 `docs/results.json → step093`。
- 本关没有加载真实模型、没有 TP 参数切分或模型前向，也未验证 Graph、PP/DP 推理或正式 RPC。后续关卡完成模型分片与协议后，才能放开多卡推理。
