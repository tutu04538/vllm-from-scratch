# step125：AWQ TP 真实端到端

实施中；下表为编码前的需求映射，不代表验收通过。

| 需求 | 拟改造位置与验证 |
| --- | --- |
| R125.01 | `VllmConfig` 与 `Worker` 支持边界；真实 8B/14B TP1/2/4 V2 eager |
| R125.02 | `MultiprocExecutor`、V2 Runner 的所有 rank 执行与 output rank，核对每 rank 快照 |
| R125.03 | AWQ 各本地线性层 Parameter 加载及后处理、逐 rank shape 与复制参数 |
| R125.04 | 同组采样结果和提交一致，跨 TP 单列数值差异 |
| R125.05 | Scheduler 生命周期及 TP 实测，覆盖混合、prefix、抢占与取消 |
| R125.06 | Executor/Engine 失败停摆与清理，注入单 rank 加载、执行、通信异常 |
| R125.07 | 独立进程与冻结 vLLM 按相同权重、输入、attention backend 差分 |
| R125.08 | 记录各 rank 显存与通信路径；性能留待后续集中测试 |

## 当前实现与证据（未收口）

2026-11-20 时配置和 Worker 开放 Qwen3 AWQ、V2、CUDA BF16、eager、无投机 TP1/TP2，并临时拒绝 TP4；2026-11-21 的四卡验收后解除 TP4 限制，见下文。Graph、投机、非 AWQ、PP/DP 仍明确拒绝。配置期解析 checkpoint 量化方法、校验格式与 dtype，避免不支持的格式启动多个 Worker 后才发现。

8B/14B 各在 TP1 和 TP2 分别用一个相同 token prompt 生成 128 token，隔离进程与冻结 vLLM 对照完全一致；TP2 每个 rank 记录的前四次最终 hidden、logits 和本地 KV 与上游最大绝对差为 0，两个 rank 的这四次采样 token 相同。不将单 prompt 的结果扩大成普遍数值一致；初次四请求固定历史因上游 EngineCore 先启动导致批次不同而有 logits 差异；受控匹配批次后两个 rank 的 hidden、logits 和首层 KV 均零差，具体局限见 [123](step123_alignment.md)。

真实 8B/14B AWQ 独立进程使用生产 Worker，分别在 TP1/TP2 运行 prefix、preempt、cancel_before、cancel_after：同一模式每轮 scheduler 状态和输出一致；TP2 所有 rank 请求镜像、执行快照一致，最后都释放请求和 block；prefix 有命中，preempt 模式确实发生一次抢占。TP2 注入 rank1 的采样异常后，已有 token 未改变、下一轮调度前拒绝并清理。命令及断言见 `tests/step125/step125_lifecycle_probe.py` 与 `step125_lifecycle_check.py`，日志和 trace 保存在本地 `benchmarks/results/step125/`。该证据不等于所有生命周期组合、NCCL 故障、加载故障或超时场景通过。

逐 rank 方法审计：8B 144 层、14B 160 层，各层都 ready 且三类量化参数进入 kernel 布局。8B 量化参数 TP1 为 3,608,543,232 字节、TP2 每 rank 为 1,804,271,616 字节；14B 分别为 6,864,076,800 和 3,432,038,400 字节。该计数不含未量化参数、KV 和内存分配器开销，具体显存原始数据见本地 summary。

测试：`CUDA_VISIBLE_DEVICES=1,2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest $(printf 'tests/step%s ' $(seq 113 125)) tests/step110 tests/step81/test_spec_metrics_pipeline.py::test_current_package_hash_matches_results -q -ra`：212 passed。`CUDA_VISIBLE_DEVICES=3 ... python -m pytest $(printf 'tests/step%s ' $(seq 58 88)) -q -ra`：1289 passed，23 skipped；首次回归因包 hash 尚未更新出现 1 failed，更新后该测试及全组重跑通过。该次 089—112 的四卡完整回归因 GPU0 被其他任务占用而未跑；2026-11-21 补跑结果见下文。性能测试按用户要求延后。

截至 2026-11-20，TP4、通信/加载故障等仍待补验；2026-11-21 的结果见下文。性能与命名 benchmark 按用户约定延后，123 的同源质量结论仍受公开权重来源限制。

14B 补验命令与 8B 相同，向 `step125_lifecycle_probe.py` 增加 `--model 14b`，分别运行 TP1/TP2 四种模式及 TP2 fail；`step125_lifecycle_check.py --directory <目录>` 全部通过。14B 日志及 trace 位于本地 `benchmarks/results/step125/lifecycle_14b/`。089/090 单卡回归：20 passed。

## 2026-11-21 TP4 与故障补验

解除 TP4 的临时配置门槛，开放 Qwen3 AWQ V2 CUDA BF16 eager、无投机的 TP1/2/4；Graph、投机和非 AWQ 仍拒绝。8B/14B 真实 checkpoint 均在四个实际 Worker 上运行，分别覆盖单请求 128 token 和四请求固定历史。每个 case 先用冻结 vLLM 采集两次参考，登记文件 hash 和既定数值门槛，再运行 mini；参考合同 SHA-256：`117239e265b928abe122c07ee0f26fe8bb7775be17fe7ad87ba7d8654a268b88`。上游两次重复所有记录的 hidden、logits、首层 KV 最大差为 0；mini 与上游 192 项张量比较最大差为 0，40 个 rank/请求生成序列全部精确一致。受控参考采用进程内 EngineCore 以固定批次到达时机，但 TP4 仍为四个独立 GPU Worker。

两个模型各做 TP4 的 prefix、抢占、执行前取消、执行后取消，以及 rank1 采样故障。每一轮四 rank 的请求镜像和执行快照一致，调度状态及输出与相同输入的 TP1 对照一致；实际观察到 prefix 命中与抢占，最终请求和 KV 引用释放。采样故障没有提交半轮结果，失败后拒绝下一轮调度。测试代码为 `tests/step125/step125_lifecycle_probe.py`、`step125_lifecycle_tp4_check.py`。

真实 8B AWQ TP4 另注入 checkpoint 读取故障，异常指出 rank1 且子进程回收；在其他 rank 进入 NCCL collective 时令 rank1 退出，1.20 秒内检测到失败，Executor 拒绝后续 RPC，Worker、rendezvous 和共享队列均清理；对 rank1 注入 30 秒 RPC 阻塞并设 2 秒超时，2.00 秒触发失败，同样完成有界清理。这些测试分别验证加载、通信掉 rank、RPC 超时；通信测试本身不是请求调度中的 NCCL 异常。测试代码 `tests/step125/step125_fault_probe.py`，原始数据和日志位于本地 `benchmarks/results/step125/tp4_20261121/`。

TP4 下每 rank 的实际模型参数总量：8B 为 1,525,082,112 字节（其中量化参数 902,135,808 字节），14B 为 2,494,781,440 字节（其中量化参数 1,716,019,200 字节）；每 rank 的 KV tensor 分别为 113,246,208 和 125,829,120 字节。统计发生在 `--batch 1`、192 KV 块、eager 配置下，并非全部运行时显存。

当前生产包 SHA-256 `05c7312a47ae9e793b6f80df0617db7ac591a81899cdadcff44f260f95b2c468`。真实命令和结果：`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest $(printf 'tests/step%03d ' $(seq 89 112)) -q -ra` 为 **418 passed、0 failed、0 skipped**；`CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest $(printf 'tests/step%s ' $(seq 58 88)) -q -ra` 为 **1289 passed、0 failed、23 skipped**；`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest $(printf 'tests/step%s ' $(seq 113 125)) -q -ra` 为 **163 passed、0 failed、0 skipped**。23 个 skip 的具体原因见原始回归日志。文档静态检查通过，不等同额外 GPU 测试。

仍未完成性能对比与本关点名 benchmark；按用户要求后续统一测量。公开 AWQ 与既定 BF16 的源 revision 尚未证明相同，因此不声称同源量化质量测试已通过。
