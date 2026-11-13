# 110：冻结支持边界与反例验收

状态：限定范围实现与验收通过。目标是在最早有足够信息时拒绝超出本专题范围的配置，避免错误配置启动 Worker 或 NCCL；已支持项必须指向真实证据。

| 项目入口 | 冻结 vLLM 0.28.0 | 职责、顺序与布局 |
|---|---|---|
| `config.py::VllmConfig._validate_parallel_config` | `config/vllm.py::_get_v2_model_runner_unsupported_features`、`_validate_v2_model_runner` | 在构造配置时检查并行组合，先于 executor、权重和通信初始化；PP/DP 的全面拒绝是项目范围，EAGLE3+PP 另有冻结 V2 限制。 |
| `config.py::_validate_parallel_config` | `config/speculative.py::_verify_and_get_draft_tp` | 本专题要求 draft TP 等于 target TP；冻结上游还接受 draft TP=1，不把项目范围扩大成上游限制。 |
| `executor/factory.py::get_executor_class`、`worker/worker.py::_validate_tp_model` | `config/parallel.py`、`v1/worker/gpu/spec_decode/eagle/utils.py` | 工厂检查设备与执行器，worker 在加载权重前检查架构和后端；各 rank 持有本地权重和 KV，Scheduler 独占请求状态。 |

生产 `MultiprocExecutor` 在创建 Worker 前对普通 Worker 的模型和后端元数据执行同一 `_validate_tp_model` 检查；显式注入的测试 Worker 可以验证无模型 RPC 生命周期。093 的生产非法模型测试因此从 worker `load_model` 阶段失败改为父进程拒绝。094 的真实四 rank RPC 测试由 TP2×PP2 改为 TP4，output rank 按现行协议从 2 改为 0；仍验证四进程、广播快照、指定 rank 回复、逆序 Future、非输出 rank 失败与清理，不删断言。

095 的四进程突然退出故障注入同样从 TP2×PP2 调整为 TP4，仍杀死 rank 1，验证 Future 的错误传播、失败后拒绝 RPC 与有界清理。旧配置在配置期按本关新边界拒绝，不再能抵达这项四 rank 故障测试。

不变量：PP/DP 越界时无 Worker、权重加载或 NCCL；不静默关闭 Graph、投机或降低 TP；单卡现有能力仍按其原有条件可用。109 的真实 8B/14B 同 TP EAGLE3 及 Graph 证据见 [109](step109_alignment.md)，105 的历史关闭超时仍未定位。

## 支持矩阵与证据范围

| 模型 | TP | 普通 eager | 普通 FULL decode Graph | 同 TP EAGLE3 eager / Graph |
|---|---|---|---|---|
| Qwen3-8B | 1、2、4 | [103](step103_alignment.md) | [105](step105_alignment.md) | [109](step109_alignment.md) |
| Qwen3-14B | 2、4 | [103](step103_alignment.md) | [105](step105_alignment.md) | [109](step109_alignment.md) |

这些链接指向既有真实模型的限定矩阵，并非所有工作负载的保证。103/105 各覆盖 B1/4、32 输出 token；109 包括 B1/4、K2，另有 TP2 B1 的 K1/K4 补充，细节以各报告为准。14B TP1 不作为本机 BF16 容量验收点。真实 KV 比较限于相应报告列明的层，23 个历史 skip 不算模型通过。

| 配置 | 分类及行为 |
|---|---|
| PP>1、DP>1 | 本专题范围限制：配置期拒绝，包含同时设置时；不代表上游普通 PP/DP 均不支持。 |
| V2 EAGLE3 + PP>1 | 冻结上游 V2 也明确拒绝。 |
| draft TP 不等于 target TP | 本专题范围限制；冻结上游还允许 draft TP=1。 |
| TP 不是 1/2/4、跨机、CP/EP、Ray | 非法值或本专题未实现，明确拒绝。 |
| 多卡非 Qwen3 稠密模型、量化、非 Triton 后端、Graph + prefix | 生产 executor 启动 worker 前检查并拒绝。 |
| 异步调度、动态 K、多卡非 EAGLE3 投机 | 项目尚未开放，明确拒绝；单卡既有能力按原条件保留。 |
| GPU 不足 | 设备限制，factory 明确拒绝；不降低 TP。 |
| 多卡 EAGLE3 + 结构化输出 | 请求提交时拒绝；普通多卡结构化输出未纳入真实模型支持矩阵。 |
| torch.compile | 配置期明确拒绝，图模式不等于启用 compile。 |

每个 TP rank 仍执行和采样，只有 output rank 返回；这不意味着单 rank 独占采样。矩阵不包括多机、PD、MoE、MTP/Gemma 多卡和多模态；缺少模型或硬件的项不能由 tiny 测试替代。

## 本关验证记录

- `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmarks/check_step110_acceptance.py --case boundary --output benchmarks/results/step110/boundary_reformatted.json`：49 passed；覆盖 PP/DP 的 2/4 和混合值、eager/Graph、投机开启与关闭，以及明确禁止调用 worker 启动、权重加载和初始化进程组。
- 同一入口 `--case frozen`：2 passed，核对冻结 draft TP 规则和 EAGLE3 Graph 源码；`CUDA_VISIBLE_DEVICES=2,3,0,1 ... --case all`：113 passed、0 failed、0 skipped，包含原 091 配置及 108 投机边界检查。
- `CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest $(printf 'tests/step%s ' $(seq 58 88)) -q -ra`：1289 passed、0 failed、23 skipped，日志 `benchmarks/results/step110/legacy_final.log`。skip 不计入真实模型验收。
- 初轮已完成并行回归 320 passed、1 failed（`parallel_final2.log`），唯一失败是 095 的四 rank 用例仍选 TP2×PP2；改成 TP4 后该用例独立重跑 1 passed（`abrupt4_revised.log`）。初轮本身仍是失败，不能写成全绿。
- 修正后 `CUDA_VISIBLE_DEVICES=2,3,0,1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest $(printf 'tests/step%03d ' $(seq 89 109)) -q -ra`：321 passed、0 failed、0 skipped，日志 `benchmarks/results/step110/parallel_final3.log`。
- 在本关新包上按 109 的冻结参考先重复、登记阈值再比较：8B 与 14B 均为 TP2、B1、K2 Graph，两组各 5 个子步骤全部通过（`benchmarks/results/step110/real_{8b,14b}_graph.json`）。这是针对初始化检查变更的真实模型复测，不替代 109 原有完整矩阵。

实际 Python 3.12.13、Torch 2.13.0+cu129、Triton 3.7.1、vLLM 0.28.0+cu129、Transformers 5.14.1；服务器为 4×RTX 4090。生产 Python 包按仓库规则计算的 SHA256 为 `dfedce8d6e73039072766c3c0b364de106dc75a01b11616d76b3be353bdab4a9`。冻结源码资料校验及模型身份见 `docs/results.json → step110` 和 [090](step090_alignment.md)。105 的历史关闭超时仍未定位，不因本关通过而关闭。
