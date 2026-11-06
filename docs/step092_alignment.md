# 092 进程组与通信原语

## 实现前对照：冻结 vLLM 0.28.0

| 冻结源码 | 职责、顺序和数据布局 | 项目映射 |
|---|---|---|
| `distributed/parallel_state.py::init_distributed_environment` | 初始化默认通信环境，确定全局 rank / local_rank 和设备；已存在时核对配置 | `minivllm.distributed.parallel_state`；记录默认组所有权 |
| `initialize_model_parallel` | rank 布局按 `[外层DP, DP, PP, PCP, TP]`；TP 末维，转置 PP 后取得各 PP 组；所有 rank 按相同顺序创建所有组 | PCP=1、一个稠密副本的 TP/PP 布局；DP 独立副本留 110 |
| `GroupCoordinator.__init__/destroy` | 每个组定义依次建 device group 和 Gloo CPU group，保存本 rank 所属组；按所有权释放；不能摧毁借用者所有的默认组 | `GroupCoordinator` 管理本组和局部失败清理 |
| `GroupCoordinator.all_reduce` / `CudaCommunicator.all_reduce` | 单 rank 直接返回输入；多 rank CUDA 的 PyTorch NCCL 路径 clone 后归约，返回新 tensor | eager NCCL，不实现 custom all-reduce 或 Graph communicator |
| `GroupCoordinator.all_gather/reduce_scatter` / `DeviceCommunicatorBase` | 单 rank 返回原 tensor；gather 先按 rank 堆在第 0 维，移动 rank 维后按 dim 拼接；reduce_scatter 将待分片维移到第 0 维再归约分片 | 同维度顺序、非连续 tensor 行为与返回约定 |
| `GroupCoordinator.broadcast/send/recv` | broadcast 就地写输入；src/dst 是组内 rank；send/recv 用对应 device group | 区分 CPU 元数据组与 CUDA 张量组 |
| `communication_op.py` | TP 集合原语委托给当前 TP coordinator | 同名薄包装 |

本关只验通信，不加载模型；固定 TP=2、PP=2 时 TP 组 `[0,1],[2,3]`，PP 组 `[0,2],[1,3]`。torchrun 仅为测试启动器；093 前正式多卡 executor 未开放。

## 生命周期与限制

本专题 device group 使用 PyTorch NCCL eager 路径，CPU 元数据使用 Gloo；不实现 custom all-reduce 或 PyNccl 图捕获。默认组可由本模块创建或借用：只销毁自行创建的默认组，借用组保持可用。所有 rank 均依次创建各 TP 组的 device/Gloo 子组，再依次创建各 PP 组；通信张量不得经 CPU 中转。非连续 all-gather 输入先在 GPU 上转连续，返回值按 rank 顺序拼接。

单 rank 的 all-reduce/all-gather/reduce-scatter/broadcast 返回原 tensor；单 rank 没有 device communicator，send/recv 明确拒绝。多 rank NCCL all-reduce 返回新 tensor，broadcast 原地修改。src/dst 是组内 rank，通过组成员表转换到全局 rank。超时、rank 异常由测试启动器限时回收整个进程组，不在失败退出路径插入 barrier。

**冻结上游数值边界：** `reduce_scatter(dim=2)` 对三维输入 `(2,4,6)`、TP=2 时，冻结的 upstream DeviceCommunicatorBase 方法返回 `(6,2,2)`，按该维切分预期 `(2,4,3)`。本项目实现与该冻结方法逐值一致，未静默修改上游语义；该维度组合不能用于正确的三维分片。复现记录在本地私有 `vllm_bugs/UPSTREAM_SUSPECTED_BUGS.md`。二维 `dim=0/-1` 已按实际张量值验证。

连续销毁并重建默认进程组的测试使用不同 FileStore，并在销毁后以测试进程的文件标记做带外同步。初次尝试在同一 torchrun rendezvous 上复用默认组，四 rank 的第二轮出现 Gloo 连接旧地址失败；独立 rendezvous 和带外同步后可重复通过。这些测试协调不进入生产通信热路径。

本关使用 PyTorch NCCL eager 路径：all-reduce 对照冻结 `CudaCommunicator` 的 PyTorch 分支，all-gather/reduce-scatter 对照 `DeviceCommunicatorBase`；当前 GPU 张量操作直接由本项目 `GroupCoordinator` 持有并调用 PyTorch 进程组，没有宣称已移植上游独立的 `CudaCommunicator` / `PyNcclCommunicator` 或多后端选择。正式 Worker 的通信初始化与所有权接线属于 093：091 仍禁止创建多卡引擎，本关不在单卡 Worker 中假建分布式环境。未来增加后端时需按冻结上游拆出设备通信器并补齐调用与生命周期验收。

## 实测结果

- `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/step092 -q -ra`：**9 passed / 0 failed / 0 skipped**，含真实 1、2、4 rank、TP=2/PP=2、进程部分初始化失败、两轮创建销毁、借用默认组和 `CUDA_VISIBLE_DEVICES=2,0` 的 GPU UUID 映射。
- `python benchmarks/check_step092_acceptance.py --case all --timeout 120 --output benchmarks/results/step092/acceptance.json`：**6 个场景通过**；TP=2/PP=2 的 TP 归约为 `[3,3,7,7]`、PP 归约为 `[4,6,4,6]`。每 rank 记录 UUID、全局和组内 rank、组成员、两轮结果。部分初始化失败场景子进程退出码 1，父进程约 4.5 秒返回；没有把超时视为通过。
- 冻结源文件与安装文件 hash 核验后，在同一 NCCL 组中调用上游 `CudaCommunicator.all_reduce` 与 `DeviceCommunicatorBase.all_gather/reduce_scatter`，与本实现逐值比较；非连续 all-gather、二维 `dim=0/-1` 等均验证。三维维度问题见上方限制。
- 58—88 单卡回归：**1289 passed / 23 skipped**；089—091 回归：**79 passed**。旧模型权重缺失导致的 skip 不算通过。需求和冻结源码静态审计 8003 项、0 错误。
- 原始 JSON、逐 rank 数据与 torchrun 日志在 `benchmarks/results/step092/`，路径不纳入 Git；详细命令、设备与源码 hash 见 `docs/results.json → step092`。本关没有加载或运行真实模型，不能代表多卡模型推理通过。
