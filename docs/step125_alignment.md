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

配置和 Worker 开放 Qwen3 AWQ、V2、CUDA BF16、eager、无投机 TP1/TP2。TP4 在真正四 GPU 端到端验证前配置期明确拒绝；Graph、投机、非 AWQ、PP/DP 仍明确拒绝。配置期解析 checkpoint 量化方法、校验格式与 dtype，避免不支持的格式启动多个 Worker 后才发现。

8B/14B 各在 TP1 和 TP2 分别用一个相同 token prompt 生成 128 token，隔离进程与冻结 vLLM 对照完全一致；TP2 每个 rank 记录的前四次最终 hidden、logits 和本地 KV 与上游最大绝对差为 0，两个 rank 的这四次采样 token 相同。不将单 prompt 的结果扩大成普遍数值一致；四请求固定历史仍有 logits 差异，见 [123](step123_alignment.md)。

真实 8B AWQ 独立进程使用生产 Worker，分别在 TP1/TP2 运行 prefix、preempt、cancel_before、cancel_after：同一模式每轮 scheduler 状态和输出一致；TP2 所有 rank 请求镜像、执行快照一致，最后都释放请求和 block；prefix 有命中，preempt 模式确实发生一次抢占。TP2 注入 rank1 的采样异常后，已有 token 未改变、下一轮调度前拒绝并清理。命令及断言见 `tests/step125/step125_lifecycle_probe.py` 与 `step125_lifecycle_check.py`，日志和 trace 保存在本地 `benchmarks/results/step125/`。该证据不覆盖 14B 全部生命周期，也不等于 NCCL 故障、加载故障或超时场景通过。

逐 rank 方法审计：8B 144 层、14B 160 层，各层都 ready 且三类量化参数进入 kernel 布局。8B 量化参数 TP1 为 3,608,543,232 字节、TP2 每 rank 为 1,804,271,616 字节；14B 分别为 6,864,076,800 和 3,432,038,400 字节。该计数不含未量化参数、KV 和内存分配器开销，具体显存原始数据见本地 summary。

测试：`CUDA_VISIBLE_DEVICES=1,2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest $(printf 'tests/step%s ' $(seq 113 125)) tests/step110 tests/step81/test_spec_metrics_pipeline.py::test_current_package_hash_matches_results -q -ra`：212 passed。`CUDA_VISIBLE_DEVICES=3 ... python -m pytest $(printf 'tests/step%s ' $(seq 58 88)) -q -ra`：1289 passed，23 skipped；首次回归因包 hash 尚未更新出现 1 failed，更新后该测试及全组重跑通过。089—112 的四卡完整回归未跑：GPU0 被其他任务占用，不抢占；其中 step110 已在上述 212 项中覆盖。性能测试按用户要求延后。

尚待：TP4 四进程真实 8B/14B、TP2/TP4 多 prompt 严格数值与完整 rank 状态、14B 生命周期、通信/加载故障与超时清理、性能及命名 benchmark；四请求 logits 差异须先完成定位或按需求程序取得预登记参考噪声证据。故 123、124、125 均不能声称全部验收结束。
