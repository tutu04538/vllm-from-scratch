# 091 并行配置与支持矩阵

## 实现前的冻结源码对照

| 上游冻结入口 | 职责与顺序 | 本项目落点 |
|---|---|---|
| config/parallel.py `ParallelConfig.__post_init__` | 先得到模型执行组 world_size=PP×TP×PCP；普通 DP 分开计入跨副本资源。默认按规模选 uni/mp；external_launcher 是上游特殊例外。 | `config.ParallelConfig` 和执行器工厂；本专题 PCP=1、external_launcher 不开放。 |
| config/vllm.py `_get_v2_model_runner_unsupported_features` | V2 检查 external_launcher+PP、EAGLE3+PP 等组合。 | `VllmConfig` 配置期拒绝；V1 单卡保留。 |
| config/speculative.py `_verify_and_get_draft_tp` | 普通模型默认 draft TP=target TP，上游也接受 1；mlp_speculator 特例为 1。 | 项目仅接受 draft TP=target TP；不称为上游通用限制。 |
| v1/executor/abstract.py `Executor.get_class` | 按最终后端选择真实执行器类，后续再初始化 worker。 | 工厂返回真实 UniProcExecutor；mp 直至 093 不开放。 |

本关不修改模型 tensor 布局、Scheduler 或执行快照；Q/KV 头切分与 PP 层数仅做启动前合法性验证，不代表分片实现。资源预算是 TP×PP×DP，DP 不混进普通单副本 world_size。

## 当前行为与边界

| TP | PP | DP | world_size | GPU 预算 | 当前结果 |
|---:|---:|---:|---:|---:|---|
| 1 | 1 | 1 | 1 | 1 | V1/V2 单卡，工厂创建真实 UniProcExecutor |
| 2 | 1 | 2 | 2 | 4 | 预算可表达；多卡未开放 |
| 2 | 2 | 1 | 4 | 4 | 预算可表达；多卡未开放 |
| 4 | 1 | 2 | 4 | 8 | 四卡环境设备数不足 |

`external_launcher` 是上游 `world_size` 包含 DP 的特例；本项目不开放该后端。显式 `mp` 不伪造执行器，配置期拒绝。模型头数、层数只检查能否切分；本关没有建立通信组、GPU shard、跨进程执行或 DP 路由。PP+EAGLE3 在读取模型权重和创建 worker 前报错。`draft_tensor_parallel_size=None` 跟随 target；不接受本专题以外的 draft TP=1/target TP>1 组合，上游则接受。旧单卡自动调度解析及调用方注入 executor 的接口保留。

## 实测与产物

- `python benchmarks/check_step091_acceptance.py --case all --timeout 120 --output benchmarks/results/step091/acceptance.json`：退出码 0；内部 091 pytest **59 passed / 0 failed / 0 skipped**。包括实际 GPU tiny Qwen3 的 V1/V2 工厂构建和两 token 生成，四个冻结源文件 hash 核对，以及实际调用上游配置方法的差分。
- `CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest $(printf 'tests/step%s ' $(seq 58 88)) -q -ra`：**1289 passed / 0 failed / 23 skipped**；缺旧模型的 skip 不算通过。`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/step089 tests/step090 -q -ra`：**20 passed**。
- `CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmarks/check_step090_models.py --mode forward --model 8b --timeout 1800 --output benchmarks/results/step091/qwen3_8b_forward.json`：Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`，V2 单卡真实权重前向生成 token `104198`，GPU peak allocated 16,745,053,696 bytes。该脚本沿用旧的显式 executor 入口；新工厂的真实 GPU 路径由上述 tiny V1/V2 用例验证。不是逐 token 差分或吞吐测试。
- 专题文档/冻结源码静态校验 **8003 checks / 0 errors**。所有明细和命令、模型来源、版本与 hash 在 `docs/results.json → step091`；原始日志/JSON 在被忽略的 `benchmarks/results/step091/`。
- 未运行真实 14B 或 EAGLE3、多卡模型并行、Graph collective，留到各后续关卡。当前设备限制检查的多卡反例未创建 worker，不代表多 rank 运算。
