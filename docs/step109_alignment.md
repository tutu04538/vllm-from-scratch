# step109：真实 EAGLE3 TP 与 CUDA Graph 对齐

## 冻结源码映射与实施前状态

本关以前置的 105 和 108 为基础，目标是让每个 TP rank 分别捕获并重放 target 与 EAGLE3 draft 的计算，并对真实权重和冻结 vLLM 进行逐轮比较。下述是实施前的源码对照，不代表本关已通过。

| 冻结 vLLM 0.28.0 入口 | 职责与执行顺序 | 本项目对应 |
|---|---|---|
| `v1/worker/gpu/spec_decode/autoregressive/cudagraph_utils.py:SpeculatorCudaGraphManager.capture` | 每次热身和捕获按描述符重新准备假输入和注意力元数据；构造的函数交由基类捕获 | `minivllm/worker/gpu/spec_decode/autoregressive/cudagraph_utils.py` 同名方法 |
| `v1/worker/gpu/spec_decode/autoregressive/speculator.py:AutoRegressiveSpeculator.capture` | 先清零索引，使用 target 常驻输入和元数据捕获 draft prefill；K>1 时再用 draft 输入捕获 decode 步或融合循环 | `minivllm/worker/gpu/spec_decode/autoregressive/speculator.py` 同名方法 |
| 同文件 `AutoRegressiveSpeculator._multi_step_decode` | 每个 draft 步按位置生成 slot mapping 和元数据，FULL 时重放 decode 图，其他模式执行对应函数 | 本项目同名方法；融合路径由 `_fused_multi_step_decode` 组织 |
| `v1/worker/gpu/model_runner.py` | Runner 组织模型、缓存及图初始化，并在执行时分派图模式 | `minivllm/worker/gpu/model_runner.py` |

Draft prefill 的 token、位置和辅助 hidden 行来自本轮 target 输入，图捕获使用 target 的常驻输入缓冲及注意力布局；decode 每请求一行，使用 draft 自己的输入缓冲，随后按步更新位置和槽位。TP 各 rank 保持同一批次描述符及 collective 顺序，权重和 KV 各自持有本地分片；Scheduler 仍独占请求状态。项目不实现冻结源码中的 DP 同步或编译分段图分支。

状态：本关限定的真实模型 TP 与 FULL decode Graph 实现及验收完成；历史关闭超时风险仍见 105 报告。

## 排查中的证据与修正

- tiny TP2/K2 与 TP4/K4 的初步运行已在每 rank 实际捕获并重放 target、draft prefill 和 draft decode 图，并比较对应相同历史的 target 有效 KV。最初 TP4/K4 测试档位只包含其中一类图所需尺寸，分别导致 target 或 draft decode 捕获数为零；测试明确失败，随后同时提供两种布局的档位，日志保留在 `benchmarks/results/step109/tiny_tp4_k4*.log`。
- 参考引擎异步接收 B4 请求时可能在所有请求送达前开始调度，先后参考的首轮批次因此不同，不能按轮直接比较。验收脚本令参考 EngineCore 同进程同步接收请求，TP Worker 仍为多进程；首次未固定调度的两次失败记录保留。参考 Graph 显式禁用 torch.compile，最初误启编译的诊断不计验收。
- 对照脚本首次遗漏显式 `async_scheduling=False`：冻结 vLLM 的默认推断可能启用异步调度，14B TP2 B4 的最后一轮出现参考多执行一个请求的现象。定位后两侧均显式关闭异步、参考端断言解析结果；之前默认异步的整套矩阵只能视作诊断，不能作为正式验收。同步调度重跑的 14B TP2 B4 eager 已通过，完整同步矩阵进行中。
- 真实 8B TP2 B4 首次比较在 target 第 0 层发现 BF16 fused add RMSNorm 的少量舍入差。真实输入重放证明 255 行使用 1024 线程、256/464 行使用 256 线程时可对齐冻结二进制算子；每个向量先局部求平方和再累加也必须保留。普通 RMSNorm 在大批次下同样需要调整归约线程数。项目重构 `layers/ops/rms_norm_cuda.py` 后，真实首层输入三种行数的输出和 residual 与冻结算子逐值一致。上述线程配置和累加分组通过冻结安装二进制差分验证；GitHub 当前源码仅用于辅助定位，不能声称它是专题冻结源码的一部分。
- 修复后，128/120/112/104 长度的真实 8B TP2 B4 Graph 诊断中，冻结参考先重复、登记阈值，再运行 mini；620 项 target/draft 有效 KV、hidden、logits 等比较通过，输出一致。正式矩阵改为每请求 128 输入和 32 输出，独立入口先重复参考，再登记合同并运行项目。正在执行的矩阵和专门测试未完成前，不把诊断结果标为 109 验收。
- 当前矩阵给参考固定的 KV 字节预算按 target 层数计算，参考还需为一层 draft 分配 KV，因此日志中的参考块数为 8B 186、14B 187；mini 配置 192 块。所有输入都在两侧容量内，比较按逻辑位置读取有效 KV，块数差异不影响本次数值状态对照；不同 KV 容量下的占用不能直接当作公平容量或服务吞吐比较。

## 当前实测

- 同步调度、禁用 compile、BF16、TRITON_ATTN、prefix off，128 输入/32 输出：`CUDA_VISIBLE_DEVICES=2,3,0,1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmarks/check_step109_eagle3_tp.py --gpu-memory-utilization 0.50 --timeout 1800 --output benchmarks/results/step109/sync_real_matrix.json`，24/24 配置通过（包 hash `b4262d4272f8e6d4aaaec3798bd2ad451b95245c31a372d1335a9148da9a8373`，在 stream 修复前），含 20 个 K2 主配置与 4 个 K1/K4 补充。每配置先运行两次同 TP 冻结参考并登记门槛，再运行项目；最终比较器重检日志 `benchmarks/results/step109/final_comparator_recheck.log` 为 24/24 通过、共 11880 项连续张量比较。包括每 rank 的采样前 hidden/logits、输入及接受状态、第 0 层 target/draft 有效 KV；有限性和 `-inf` 掩码逐位核对，不能把第 0 层 KV 覆盖写成全部层。
- `CUDA_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmarks/check_step109_forced.py --matrix benchmarks/results/step109/sync_real_matrix.json --output benchmarks/results/step109/forced_matrix.json`：8B/14B TP2 B1 K2 的 eager/Graph 四组固定历史通过；强制历史来自同模式的自然参考，冻结重复先登记门槛，并检查拒绝后的后续轮。各轮 top1/top2、margin 与 first_diff 位于 comparison JSON。
- 真实概率草稿：8B/14B TP2 B1 K2 Graph，temperature=0.8，固定参考历史并比较每 rank 映射 logits；各 300 项连续张量比较通过，非交集保持 `-inf`、计算 q 后非交集概率为零且归一化。概率和检验在 CPU 用 FP64 累加，最初 FP32 累加 151936 项发生约 1.38e-5 的数值误差，失败日志保留；这只是验证计算精度，不修改模型或采样算法。产物为 `benchmarks/results/step109/probability_{8b,14b}_graph_*`。
- 独立测试 `CUDA_VISIBLE_DEVICES=2,3,0,1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/step109 -q -ra`：16 passed、0 failed、0 skipped（包 hash `b4262d4272f8e6d4aaaec3798bd2ad451b95245c31a372d1335a9148da9a8373`，在 stream 修复前）。包括 TP2/4 K1/2/4 捕获与 replay、prefill eager、请求身份/长度变化及不重捕、引擎重建、draft capture 与 propose rank 故障，以及 255/256/464 行 BF16 RMSNorm 冻结算子差分。
- 在修复 stream 竞争和更正 RMSNorm 注释后的包 hash `324aa1fa47ca5240f6ee75181c05807aeeb91eef38b15ef0cc6e725d534a5786`，重新运行上述完整真实矩阵：`CUDA_VISIBLE_DEVICES=2,3,0,1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmarks/check_step109_eagle3_tp.py --gpu-memory-utilization 0.50 --timeout 1800 --output benchmarks/results/step109/final_matrix_hash324.json`，24/24 配置、120/120 子步骤及 11880 项连续张量比较通过。固定历史入口使用该矩阵重跑，20/20 子步骤通过，见 `benchmarks/results/step109/final_forced_hash324.json`。概率草稿按之前冻结参考登记门槛重跑最终包，8B/14B 各 300 项比较及 q 检查通过，见 `probability_{8b,14b}_graph_*_hash324.json`。
- 最终包 58—88 为 1289 passed、0 failed、23 skipped（`benchmarks/results/step109/legacy_hash324.log`），089—107 为 242 passed（`parallel_089_107_hash324.log`），108 为 63 passed（`step108_hash324.log`），109 为 16 passed（`pytest_hash324.log`）；后三项均 0 failed、0 skipped。跳过项不计验收通过。
- `benchmarks/step109_cost_probe.py` 在最终包、TP2 B1 K2、128 输入/32 输出下以 CUDA event 记录 proposal GPU stream 区间；8B eager/Graph 均 drafted=58、accepted=2（3.45%），rank0 每轮均值分别 2.744/1.581 ms；14B 均 drafted=56、accepted=4（7.14%），均值分别 2.570/1.845 ms。分别只有一次带探针运行，不代表稳定加速或服务吞吐。按唯一 storage 去重的参数加 KV，rank0 8B 为 8,873,699,328 bytes、14B 为 15,714,850,816 bytes（已扣共享 embedding）；allocator peak 分别 9,530,187,264 和 16,376,055,296 bytes，含其他工作区及图，不能拆成每模型各自峰值。完整每 rank 数值见 `benchmarks/results/step109/cost_{8b,14b}_{eager,graph}_hash324.json`；先前包的测量保留在无 hash 后缀的文件中。
