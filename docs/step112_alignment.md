# step112：并行专题收口与证据审计

状态：限定 TP 范围验收通过；历史 105 Worker 关闭超时根因仍未知。

## 冻结源码映射与边界

冻结 `config/vllm.py::VllmConfig._get_v2_model_runner_unsupported_features` 汇总 V2 支持限制；项目 `minivllm/config.py` 在配置期和生产 Worker 模型检查中实施本专题更窄的单副本 TP 范围。冻结 `v1/executor/multiproc_executor.py::MultiprocExecutor.shutdown` 先通知并终止 Worker，再关闭响应和广播队列；项目 executor 负责 Worker、通信对象和失败传播的生命周期，现有 095 回归验证异常与清理。Scheduler 拥有请求及逻辑 KV，Worker 的请求状态是根据快照校正的镜像；模型权重、设备 KV 与图由对应 rank 本地拥有。TP rank 本地权重形状、QKV/KV 布局与资源释放仍须对应各关实测证据，不能用静态扫描替代。

本关将分开检查配置/源码、tiny 多进程、真实普通模型、真实 EAGLE3 和性能 trace 五类证据。105 的历史 Worker 关闭超时尚无根因，不因其他用例通过而关闭。PP/DP、跨机和其他未授权并行轴不属于本次收口。

## 审计进行中的发现

生产导入的 AST 静态扫描暂未发现直接引用安装的 vLLM；它不能排除所有动态导入形式，需要结合源码审查。权重读取使用 CPU safetensors 流，TP 线性层创建时即分配本地分片；真实 14B 加载峰值应以 101/103 的已有测量及本关复核为证，不能仅凭源码推出运行时峰值。

V1 `GPUModelRunner.cudagraph_selections` 与 V2 图管理器 `selections` 在每次分派时追加记录，生命周期内没有容量上限；这是目前发现的长期运行内存增长风险。冻结上游没有这套无限历史账本。V1 还无条件创建 `CUDAGraphLogging` 并累计原始统计，冻结上游仅在启用对应可观测性时创建并由 metrics logger 周期性输出清空；项目当前没有对应的周期管理。112 需要将仅用于测试/分析的观察移出生产热路径，并按上游的统计职责处理累积记录，同时保持原有图模式、padding 和 replay 测试的有效断言；已移到测试侧观察器，生产不再逐轮累积，最终回归已通过。V2 图调度对未匹配的形状执行 eager 是冻结设计的逐批分派，不等于把用户配置的 Graph 功能静默关闭。

## 新发现：RoPE 表的设备与转换语义

111 对 8B TP4 混合负载的固定历史逐层诊断表明，项目和冻结 vLLM 在位置 850 的 QKV 与 Q/K norm 一致，实际 RoPE 表有一个 BF16 项相差 0.0009765625，旋转后的 key 最大差 0.00390625，与首层有效 KV 的差值一致。定位到 `minivllm/layers/rotary_embedding.py::_canonical_device(None)` 忽略模型构造的默认设备；其后 `_apply` 在退出 CUDA 默认设备上下文后重新生成表，导致最终使用 CPU FP32 三角函数的结果。冻结 `model_executor/model_loader/base_loader.py::load_model` 在目标 device 上下文构造模型，`rotary_embedding/base.py::_compute_cos_sin_cache` 在该上下文生成表，模块转换则转换已有值。111 的原始超限、trace 和两次钩子实验均保留。

修复已合入工作树：原始表使用构造默认设备生成，设备/精度转换按原表转换并按来源共享，避免将 CPU 和 GPU 生成的不同数值误合并。定向 CUDA 差分和既有共享测试在隔离工作树 33 项通过；隔离工作树与当前生产包的 Python 文件哈希均为 `4f251f57ecf9a6fd740646abf8e1cc69f11fed53b06ffc7af569f4ddff8e8835`。其余两组原失败同样在修复后严格通过；完整矩阵及最终回归均已通过。

## 支持矩阵与证据索引

| 模型与 TP | 普通 eager | 普通 Graph | 同 TP EAGLE3 | 证据边界 |
|---|---|---|---|---|
| Qwen3-8B TP1/2/4 | 已有真实 GPU 证据 | 已有真实 GPU 证据 | 已有真实 GPU 证据 | [103](step103_alignment.md)、[105](step105_alignment.md)、[106](step106_alignment.md)、[109](step109_alignment.md)；111 的三组失败和修复后的结果另计。 |
| Qwen3-14B TP2/4 | 已有真实 GPU 证据 | 已有真实 GPU 证据 | 已有真实 GPU 证据 | [103](step103_alignment.md)、[105](step105_alignment.md)、[109](step109_alignment.md)；具体 B/K 和容量限制以各报告为准。 |
| Qwen3-14B TP1 | 未验 | 未验 | 未验 | 本服务器单卡 BF16 容量未做验收，不将未测写成配置期拒绝。 |

PP/DP>1、异步端到端、跨机与多卡 Graph+prefix 依配置明确拒绝；真实 MTP/Gemma 多卡及其他并行轴不属于本专题。每项历史证据对应当次源码 hash，不能替代 RoPE 修复后的回归。

## 当前验证与待运行项

工作树 `tests/step112` 在 GPU 不可见时为 8 passed / 1 skipped；跳过项是 CUDA 表对照，已在同包哈希的隔离工作树 GPU 定向测试中通过，不将本轮 skip 计为通过。`tests/step111` CPU 重检 25 passed；`check_step112_acceptance.py --case static` 通过，生产直接导入/常量动态导入静态扫描未发现上游 vLLM，包哈希匹配，需求检查 7911 项通过。静态扫描不替代运行路径审计。

历史资源受占用时，独占 GPU 门禁拒绝启动；随后四卡空闲，已依次重跑三组 111 失败配置、完整 50 组矩阵及 trace，最终回归已通过。105 历史 Worker 关闭超时仍无根因。

`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmarks/check_step112_acceptance.py --case models --timeout 1800 --output benchmarks/results/step112/models.json`：退出码 0，4/4 模型身份通过，耗时约 535.5 秒。检查了 8B/14B target 的权重和 tokenizer 哈希、两份草稿的来源权重、转换权重、配置和 manifest；这是文件身份审计，不是 GPU 模型前向。

2026-11-14 复查：四张 GPU 仍存在其他用户的计算进程；验收入口新增 GPU 占用门禁后实测，三组回归均按 `unrun`、退出码 125 记录，没有启动 GPU 工作。新增门禁测试后，CPU `tests/step112` 为 10 passed / 1 skipped；`benchmarks/results/step112/busy_gate.json` 保存当次检查。

`step112_rope_validation.py --case failed-cases` 的资源预检同样将三组原失败负载标为 `unrun`（各退出码 125）；完整矩阵支持 `--resume` 并检查原始配置和源码，避免覆盖已有执行记录。

新增三组性能复测的占用防护测试后，CPU `tests/step112` 为 12 passed / 1 skipped；跳过项仍为 CUDA 构造设备对照，此前同包哈希定向测试已通过。

续跑前已有摘要也会单独保留；相关 CPU 测试更新为 13 passed / 1 skipped（CUDA 对照此前同包哈希定向通过）。

补充验证 trace 门禁来源、历史矩阵保留和回归来源关联后，CPU `tests/step112` 为 17 passed / 1 skipped。修复后的矩阵登记只能引用相同包哈希的 trace 和回归；本次尚无修复后 GPU 结果。

2026-11-15 继续验证时，四卡仍有其他用户计算进程；三组失败负载与三组回归均由资源门禁标为 `unrun`（内部退出码 125）。`tests/step111 tests/step112` 在 GPU 不可见环境下 42 passed / 1 skipped，跳过项仍为 CUDA RoPE 对照；资源检查结果分别保存在 `benchmarks/results/step112/busy_20261115.json` 与 `benchmarks/results/step112/regressions_busy_20261115.json`。

新增回归设备选择检查：即使外部环境是 `CUDA_VISIBLE_DEVICES=0`，单卡旧回归显式使用 GPU 0，多卡回归显式使用 `2,3,0,1`；本次运行实际因进程占用均为 `unrun`。更新后 `tests/step111 tests/step112` 在 GPU 不可见环境下 43 passed / 1 skipped。

## 2026-11-15 修复后真实模型复测

四卡空闲后，用未调整的 111 跨 TP 参考合同复测原先三组失败的 16 请求混合负载。8B TP4、14B TP2、14B TP4 均为 `passed_performance`、`strict_equivalent=true`：逐 rank 状态门禁通过、完整 128 token 贪心输出一致，每组完成一次预热和三次正式计时。分别耗时约 422.0、418.3、175.3 秒；结果见 `benchmarks/results/step112/failed_retest_20261115_v2.json`。这支持 RoPE 表设备差异是原失败的原因；后续 50 组完整矩阵和最终回归也已通过。

首次复测入口误用 `--input-length 1024`（B 组为长短混合），导致 3 个 case 均在选择阶段报错、未运行模型；修正后使用 B 组选择器重跑，失败日志保留于 `benchmarks/results/step112/failed_retest_20261115.json`。

本日稍后四卡空闲，修复后完整矩阵入口为 `benchmarks/step112_rope_validation.py --case matrix --output benchmarks/results/step112/final_matrix_20261115.json`；50 组严格通过，最终回归与登记均已通过。此前的占用记录与三组选择器错误记录均保留。

## 2026-11-15 最终回归与证据登记

在 `source scripts/activate-4090.sh` 后执行 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmarks/check_step112_acceptance.py --case regressions --timeout 7200 --output benchmarks/results/step112/regressions.json`。该入口依次设置 GPU 可见设备并启动以下三组，结束前后生产包哈希一致，均为 `4f251f57ecf9a6fd740646abf8e1cc69f11fed53b06ffc7af569f4ddff8e8835`：

| 组 | 设备 | pytest 结果 |
|---|---|---|
| tests/step112 | 2,3,0,1 | 19 passed，0 failed，0 skipped |
| tests/step58—88 | 0 | 1289 passed，0 failed，23 skipped |
| tests/step089—111 | 2,3,0,1 | 395 passed，0 failed，0 skipped |

23 个 skip 不是通过，主要是本机未配置真实 Gemma4 等模型条件；具体原因见 `benchmarks/results/step112/legacy_final.log`。冻结需求静态检查 7911 项通过；这不是 GPU 测试。111 的 50 组最终矩阵均严格通过，六个代表点及两种 NUMA 的门禁和 trace 任务均成功，细节见 [111](step111_alignment.md)。105 的历史一次 Worker 关闭超时根因仍未知。

最终总验收命令：`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmarks/check_step112_acceptance.py --case all --timeout 1800 --output benchmarks/results/step112/final_acceptance_20261115.json`，退出码 0，状态 `passed`。四份模型及转换 manifest 的身份和哈希均通过；静态检查和本次所有回归证据的包哈希一致。静态 import 扫描不能排除所有动态构造的导入。原始产物不纳入 Git。
