# 105 TP 整模型 CUDA Graph 接入

## 实施前冻结源码映射

| 项目位置 | 冻结 vLLM 0.28.0 | 职责、布局与时序 |
|---|---|---|
| `worker/gpu/cudagraph_utils.py::CudaGraphManager.capture/dispatch` | `v1/worker/gpu/cudagraph_utils.py::CudaGraphManager.capture/dispatch` | 按降序 bucket 依次热身和捕获，统一执行描述；运行期选择兼容的 FULL decode 图或可见的 eager 路径，每个 rank 拥有自己的图与常驻地址 |
| `worker/gpu/cudagraph_utils.py::ModelCudaGraphManager` | `v1/worker/gpu/cudagraph_utils.py::ModelCudaGraphManager` | capture 前准备本 rank 的静态 input_ids、positions、attention metadata/slot；图输出写入本地常驻 hidden，replay 后供真实采样 |
| `worker/gpu/model_runner.py::execute_model` | `v1/worker/gpu/model_runner.py::GPUModelRunner.execute_model` | 各 rank 根据相同调度快照选择 batch 描述，更新本地常驻输入、seq_len、block table 与 slot mapping；padding 行 `slot=-1`，仅有效行参加采样/输出 |
| `compilation/cuda_graph.py::graph_capture`、`distributed/parallel_state.py::GroupCoordinator.graph_capture` | `compilation/cuda_graph.py`、`distributed/parallel_state.py::graph_capture` | 捕获期间复用 104 的 TP 通信 stream/上下文；图及工作区先于本地 communicator 释放 |
| `worker/worker.py::_validate_tp_model` | 冻结 V2 支持检查 | 将本关普通 Qwen3 target、PP=DP=1、FULL decode 范围内解锁；draft、PP Graph 和其他未验组合仍明确拒绝 |

同一模型副本只有 Scheduler 拥有 Request、逻辑 KV 与提交位置；各 Worker/Runner 分别拥有本卡权重、设备 KV、常驻缓冲与 graph。Executor 拥有进程及 RPC；任意 rank 失败使副本停止，禁止单卡改 eager 继续。prefill 使用 eager；本关不引入 torch.compile/PIECEWISE 或 draft Graph。

## 设计与当前验证口径

多 rank 仅开放普通稠密 Qwen3、V2、TRITON_ATTN、PP=DP=1、无投机、prefix caching 关闭的 `FULL_DECODE_ONLY`；不支持的 Graph 组合在加载前明确拒绝。`CudaGraphManager.capture()` 沿用既有 Graph 缓存和 pool，TP>1 时通过 104 的 group capture context 取得同一捕获 stream，并显式传给 `torch.cuda.graph`；单卡维持既有路径。捕获前 KV 已绑定，各 rank 根据相同的 scheduler snapshot 独立分派一致 bucket；prefill 和混合批按既有描述符走 eager，decode 命中已捕获 FULL 图。图重放期间的输入、positions、seq_lens、block table、slot mapping 使用各 rank 的常驻设备缓冲。

固定例子 I1：两个 rank 的 mode/bucket/collective 顺序一致；I2：同一个 rank 中常驻缓冲地址和图个数稳定，内容随上下文更新；I3：3 个有效请求使用 bucket 4 时多余 slot 为 -1，不参与真实 KV 写入或采样；I4：Scheduler 拥有请求提交和逻辑 KV，Worker 只持有镜像/本地设备状态。动态 tiny 运行依次让有效 decode 请求从 3 到 4 再到 3，随后取消旧请求并新增请求；逐 rank 验证 slot 复用、位置跨越 block 边界、所有 4 层有效 KV 与 eager 一致。刻意把 padding token/position 设成非零，正确结果仍不变。

第一次动态验收中两个引擎输出一致，但测试把结束后的 scheduler-only 清理 tick 也算作 `Runner.sample` 调用，导致测试断言失败；日志保留在 `benchmarks/results/step105/dynamic_first*`。核对源码和 trace 后，改为比较两边实际采样轮，同时限定清理 tick 的数量；未改生产输出或数值阈值。改正后动态例子对 756 项 tiny hidden/logits/逐层 KV 做逐值比较，最大误差 0，且验证请求数变化、换页和 resident slot 复用。

真实 BF16 Qwen3 验证使用 103 在 mini 测量前登记的合同，不依据本关结果放宽：8B TP1/2/4、14B TP2/4，各 B1/4。正常生成比对冻结 vLLM 32 token 及 prefill/前 3 次 decode hidden、raw logits、第 0 层有效 KV；同一份固定 token 历史另做独立 teacher-forced Graph replay，逐 rank 核对 token/position、本地 block table→slot 算术和同一批张量。另将 Graph 与 104 重新执行的 eager 结果直接比对。正常生成与固定历史均是 10 个配置、各 780 项张量比较；真实 KV 的覆盖仅第 0 层，不称全部层已验。103 合同及源文件 hash 保持原值。

除 greedy 外，tiny TP2 Graph 使用 temperature=0.8、top_k=5、top_p=0.95、min_p=0.05 运行随机采样；测试用独立 CPU 计算允许的 token 集合，并检查同 TP 组采样状态一致（36 次 rank/token 检查），不要求跨实现仅以同 seed 获得相同随机序列。真实模型随机分布未在本关单独比较。

多引擎测试同时建立两套真实 `LLMEngine`、各 2 个生产 Qwen3 tiny Worker，分别占用 GPU 0/1 与 2/3；结束第一个后，第二个继续 Graph decode 并与先完成的引擎输出一致。关闭路径在 Worker 内以 weakref 检查 manager、捕获图、workspace pool、communicator 均无活引用，再检查进程已回收。另在实际 tiny 模型捕获期间让 rank 1 抛错，验证 Engine 初始化失败传播、整个 Executor 标记失败并回收全部 Worker；此测试不是用测试 Worker 假装真实模型捕获。

### 关闭超时异常（未确定根因）

最终代码的第一轮真实模型复验中，14B TP2 B1 已生成全部 32 token，但 `engine.shutdown()` 报 `Worker 终止超时，未退出 ranks=[1]`；该进程命令退出码为 1，不能当作该轮通过。原始结果与日志保留在 `benchmarks/results/step105/real_models_shutdown_timeout.json` 和 `graph_models_shutdown_timeout/14b_tp2_b1/`。系统当时 I/O pressure 的 `some/full avg10` 约 52%/52%，但**没有当次 Worker 的 `/proc` 等待点，不能据此断定原因**。

为定位，给测试驱动增加独立 Worker 退出状态采样和必要时的 Python 堆栈信号，不改变生产代码；以同一真实 14B TP2 B1 配置独立创建/关闭引擎 50 次，50 次退出码均为 0，监测到的退出时间约 1.11–1.31 秒。个别成功运行曾采样到 Linux `D`（不可中断等待）状态，等待点包括 `nvidia_close_callback`、`os_acquire_mutex` / `os_acquire_rwlock_write`；`D` 不等于一定是磁盘 I/O。这些观测说明复现期间驱动关闭曾短暂等待，但不能倒推首次超时的确切根因。未通过简单增加退出时限或屏蔽错误的方式处理。

诊断记录为 `benchmarks/results/step105/shutdown_diagnostic_{1..50}/shutdown_states.json` 和对应日志；完整矩阵后续重新执行并单独保存。首次异常仍是未解释的可靠性风险，后续通过的运行不能抹去它。


## 验收状态与边界

最终代码的 105 pytest 9 passed，七项独立验收通过；同一 Python 进程内连续两次运行完整 `tests/step105/test_step105_graph.py`，各 9 passed。58—88 回归 1289 passed、23 skipped；089—104 回归 197 passed。真实模型正常生成重新运行 10/10 通过、固定历史 10/10 通过、Graph 与 eager 的直接对照 10/10 通过，每组比较共 780 项。正常生成中每个请求的 32 token 与同配置冻结参考一致。

虽然上述复验和 50 次专门诊断均通过，第一次最终模型矩阵存在一次 Worker 关闭超时，且根因未知。**因此本关代码已实现、主要功能与数值已验证，但暂不将 105 标为完全关闭。** 后续若该现象再次发生，应使用同一测试驱动保留 `/proc` 状态和堆栈，在有证据时按冻结上游职责修复；不把延长超时作为已证实的修复。

本关不支持多 rank Graph 配合 prefix caching、draft/EAGLE3 或 PP；真实 8B/14B 只比较了第 0 层有效 KV，没有实测真实模型随机采样分布，也没有进行性能 benchmark。旧单卡回归的 23 个 skip 不计为通过。完整命令、设备、版本、每 rank capture/replay 和原始路径见 `docs/results.json → step105.results/models`；结果日志在本地 `benchmarks/results/step105/`，不加入 Git。
