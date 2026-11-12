# 103 真实 Qwen3 TP 端到端验收

## 实施前冻结源码映射

| 项目位置 | 冻结 vLLM 0.28.0 入口 | 职责与数据布局 |
|---|---|---|
| `worker/gpu/model_runner.py::execute_model` | `v1/worker/gpu/model_runner.py::GPUModelRunner.execute_model` | 消费 Scheduler 快照，准备 batch token、position、block table / slot mapping，按本地 KV heads 执行模型，保存采样所需 hidden |
| `worker/gpu/model_runner.py::sample_tokens` | `GPUModelRunner.sample_tokens` | 根据 logits_indices 选出需采样的 hidden，通过 LMHead 和 logits 汇总生成词表分数，由每 rank 采样并更新本地镜像 |
| `models/qwen3.py::Qwen3ForCausalLM` | `model_executor/models/qwen3.py` | checkpoint 相同配置与 shard，各层 QKV 和 KV 本地 heads，o_proj/MLP 归约后为完整 hidden；logits 为 `[采样行, 原词表大小]` |
| `executor/multiproc_executor.py` | `v1/executor/multiproc_executor.py::_get_output_rank` | 每 GPU 一个 worker，每 rank 同序执行；PP=1 的 output rank = world_size - TP = 0，但各 rank 均参与采样及状态更新 |

固定矩阵：8B TP1/2/4 × B1/4、14B TP2/4 × B1/4，BF16、V2、eager、TRITON_ATTN、无 prefix/投机/PP/DP。模型和冻结 vLLM 分进程串行运行，以本关在 mini 测量前登记的数值合同为阈值；普通 greedy 连续 32 token 与固定历史差分分开记录，不能用一个替代另一个。prefill 最后一行和前 3 次 decode 比较 hidden、raw logits 与有效 KV，B4 使用不同长度和完成时间并保持请求身份。权重、KV、工作区和 allocated/reserved 峰值按 rank 记录；本关不打开 Graph。

## 数值定位和生产修复

十组冻结参考各独立运行两次；在运行 mini 之前，按模型、batch、prefill/前三轮 decode 及 hidden / raw logits / 有效 KV 分别登记 48 项阈值，方法为冻结参考自身重复与合法 TP 配置之间最大误差的两倍、下限 1e-5。`benchmarks/results/step103/numerics_contract.json` SHA256 为 `863ca136b18661b4f3d966d69bd1c3f64a482cd29c2314ac3c1e644107ec0558`；包含原始参考文件的 SHA。090 合同 SHA 保持 `29bcbf64f1212cf100845387ff46ddcd43981b3d9fd4487c044a455ec28d0e7c`，其旧阈值只适用于原固定 fixture，不外推到 103 新输入。

首次参考 KV 采集误把冻结 Triton cache 每个 head 内打包的 K/V 解释为先按 K/V 分组，产生虚假的大误差；该参考、合同与日志保留在 `benchmarks/results/step103/invalid_reference_layout/`，在任何 mini 结果产生之前作废，修正布局后完整重新采集冻结参考并登记上述合同。该修正不是放宽阈值。

首次 mini 矩阵仅 3/10 通过。逐层诊断发现：

1. `minivllm/layers/ops/layernorm.py` 先前对 BF16 普通 RMSNorm 在乘权重前多做一次舍入；融合残差分支则在残差和舍入前归一化。对实际 8B token 198 和对应真实权重，项目与冻结安装包 `_C.rms_norm` 有 1087 个元素不同，修正舍入顺序后逐值一致；融合算子也由 1389 处不同变为逐值一致。
2. `layers/ops/activation.py` 的 BF16 SiLU 曾把 FP32 SiLU 与 up 直接相乘后才舍入；冻结安装包 `_C.silu_and_mul` 对同一真实 gate/up 输入在 SiLU 后先舍入。修正前两者有 1632 个元素不同，修正后逐值一致。
3. 即使修正舍入次序，Triton 的 FP32 方差归约顺序仍与冻结 CUDA 算子不同：真实 8B prefill 第 1 层输入归一化的 524288 个元素中有 2 个 BF16 输出不同，并在后续层放大。项目新增自身的 `layers/ops/rms_norm_cuda.py`，用 CUDA/CUB 实现 BF16 对齐向量归约、残差处理和权重计算；其余 dtype、Gemma 的 `add_one` 及不满足向量布局的情况继续使用既有路径。GPU 原始输入直接用于计算，没有调用安装 vLLM 的算子或通过 CPU 重建生产输入。扩展在匹配 CUDA 架构时延迟编译并跨 Worker 加锁复用，缺少 nvcc 时明确报错，不静默降级。源码随 Python 文件进入包 hash。

专题冻结清单包含 Python 源码，未随附该 CUDA 扩展的 C++ 源码；这里的 CUDA 数值行为以本机已验证版本的 `_C` 实际算子为对照，不声称逐字复刻了某个未冻结的 C++ 源文件。真实输入、多个 BF16 维度、带 stride 输入及 CUDA Graph replay 与实际算子对照；测试引用 vLLM，生产不引用。旧 85 CUDA 测试原以 native IR 为参考，现改为对应的 CUDA 算子，BF16 逐值断言没有放宽；CPU / Gemma 的 native IR 断言保留。

14B 修复前在第 20 个输出 token 出现 near-tie：冻结 TP2 top1/top2 分别为 71304/31935，BF16 logits 30.625/30.5；mini 两者均为 30.625。对各自同轮 hidden 和对应词表权重做 FP32 输出头重算，上游约为 30.6562/30.4648，mini 约为 30.5625/30.5806。这个诊断记录的是修复前事实，不能把其写成严格逐 token 一致；后续生产矩阵重新运行后，十组各 32 token 全部与冻结 vLLM 一致。历史失败、诊断、重测分开保留。

## 主矩阵与内存口径

8B 的 TP1/2/4 × B1/4 与 14B 的 TP2/4 × B1/4 共十组；每组以独立子进程串行运行两套引擎，真实输入 IDs 固定，B4 长度分别为 128/120/112/104，每请求输出 32 token，完成轮次为 32/33/34/35。所有请求的 32 token 与同配置冻结 vLLM 精确一致，prefill 最后一行与前三次 decode 的 hidden、raw logits 和**第 0 层全部有效 KV**均通过已登记阈值；所有 rank 单独检查，KV 使用各 rank 本地 head。第 0 层的覆盖不冒称逐层 KV 全覆盖。生产稳态不做测试 hook 的 D2H 采集。

每 rank 模型参数、KV tensor、已识别的 attention/worker workspace 以及 torch allocator 当前/峰值 allocated/reserved 来自独立进程的 `aux_matrix.json`，capture 未启用。本地已识别工作区不包含 CUDA 库内部或 torch 其他缓存，不能把 allocator 差额全称为 workspace；测试 hook 的临时分配也包含在测量峰值中，不能用于公平吞吐比较。比如 mini 14B TP2 B4 每 rank 参数 14,768,732,160 bytes、KV 251,658,240 bytes、已识别 attention workspace 5,324,800 bytes、峰值 allocated 15,212,865,024 bytes；完整 checkpoint 参数约 29,536,614,400 bytes，并结合 101 的加载审计说明没有完整权重瞬时驻单一张 GPU。

除 greedy 外，8B TP2 B4 和 14B TP4 B4 的两套引擎各自执行 temperature=0.8、top_k=5、top_p=0.95、min_p=0.05 的真实模型补测；每个 TP 组内前四轮采样状态一致，192 次 rank/token 检查均在 top-k 候选中。跨引擎不以相同 seed 强求随机输出逐 token 相等。

## 状态与验收口径

| 阶段 | 唯一 owner | 执行侧验证 |
|---|---|---|
| 请求输入、生成历史及提交 | Scheduler | Worker 只接收本轮快照；固定历史重放通过测试 hook 在采样返回时强制既定 token，后续仍由原生产流程提交和准备输入 |
| 参数、KV、block table 和工作区 | 各 rank Worker/Runner | 逐 rank 记录本地张量与显存；slot 检查使用该 rank 自己的 block table，不要求不同引擎的物理块编号相同 |
| 调度、进程与 RPC | Scheduler / Executor | 真实模型两引擎独立子进程串行执行；不同时驻留污染显存测量 |

各对照的判定分别为：32-token greedy 与重放的 token/position/slot 算术要求 exact；hidden、raw logits 和有效 KV 按预登记合同 tolerance；修复前 14B top 候选为 near-tie；最初 3/10 与中间两轮 6/10 的矩阵为 failed（原始日志保留）；全部层 KV、多 rank Graph、TP 投机、PP/DP、异步调度与性能对比为 unrun / 本关范围外。没有据此推断任何吞吐加速。

固定历史重放以冻结参考 run 0 的 token 序列为共同输入，20 个独立进程覆盖两引擎 × 十组；记录并核对每个 rank 的 token、position 与本地 block table → slot 映射，再比较采样前的 hidden、logits、有效 KV，共 780 项全部通过。自由生成矩阵同样为 780 项，其中九组张量逐值相同、8B TP2 B4 有差异但位于事先登记的阈值内。

109 的并行全量回归中，历史独立真实测试在上游启动前因可用 19.35 GiB 低于 0.82 预留所需 19.39 GiB 而失败（`benchmarks/results/step109/parallel_after_stream_fix.log`）。测试显式调整参考预留为 0.78，双方仍断言 192 个 KV 块，输入和数值门槛未改；保留 pytest 父进程 CUDA context 的单项复测通过，见 `benchmarks/results/step109/step103_reservation_078.log`。

实跑命令及结果（先 `source scripts/activate-4090.sh`，日志与 JSON 位于本地 `benchmarks/results/step103/`）：

```bash
python benchmarks/step103_collect_reference.py
python benchmarks/step103_register_contract.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmarks/check_step103_tp_e2e.py --case all --timeout 1800 --output benchmarks/results/step103/matrix_production.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmarks/step103_aux_matrix.py --mode all --timeout 1800 --output benchmarks/results/step103/aux_matrix.json
python benchmarks/step103_validate_aux.py --input benchmarks/results/step103/aux_matrix.json --output benchmarks/results/step103/aux_validation.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmarks/step103_aux_matrix.py --mode forced --timeout 1800 --output benchmarks/results/step103/forced_matrix.json
python benchmarks/step103_compare_forced.py --input benchmarks/results/step103/forced_matrix.json --output benchmarks/results/step103/forced_compare.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/step103 -q -ra
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest $(printf 'tests/step%s ' $(seq 58 88)) -q -ra
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest $(printf 'tests/step%03d ' $(seq 89 102)) -q -ra
python learning_notes/14_vllm_from_scratch/分布式并行完整需求/参考资料/check_requirements.py
```

上述命令退出码均为 0：主矩阵和固定历史均为 10/10，辅助检查 24/24（其中随机约束 192 次 rank/token 检查）；103 pytest 5 passed；58—88 回归 1289 passed、23 skipped；089—102 回归 186 passed。23 个 skip 为历史测试缺少真实模型等条件，不计为通过。静态需求校验 8003 项通过，仅证明文档和冻结源码资料完整。初次回归曾因生产源码改变后未先更新包 hash 导致历史 hash 检查失败，更新 hash 后完整重跑；该失败没有被当成通过。初次比较器路径导入失败也已修正并重跑。当前 `minivllm/` 的 188 个 Python 文件包 hash 为 `8fcccbe86ad2e7db8db0fcef208f3fe835af8a1ed6b0a9672f544933e8e4f253`。

环境：Python 3.12.13、Torch 2.13.0+cu129、Triton 3.7.1、vLLM 模块 0.28.0、Transformers 5.14.1、4×RTX 4090；详细结果、revision、文件 hash 和每 rank 显存快照见 `docs/results.json → step103.results/models`。结果、trace、日志和权重均不纳入 Git。
