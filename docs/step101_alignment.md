# 101 Qwen3 TP 模型接线和加载峰值

## 实施前冻结源码映射

| 冻结 vLLM 0.28.0 | 职责 / tensor 布局 | 项目落点 |
|---|---|---|
| `models/qwen3.py::Qwen3Attention.forward` | 本地 QKV → Q/K norm → RoPE → 本地 paged Attention → o_proj TP 归约 | `models/qwen3.py::Qwen3Attention`；本地 Q/KV 头数来自 097/098 |
| `models/qwen2.py::Qwen2Model.forward` | embedding → 每层 input norm / attention / residual / post-attention norm / MLP → 最终 norm；residual 为完整 hidden | `models/qwen3.py::Qwen3Model` / decoder layer |
| `models/qwen3.py::Qwen3ForCausalLM.load_weights`、`models/utils.py` | checkpoint 名字映射到参数；参数自己的 loader 切片并做覆盖检查 | `models/qwen3.py` / `model_loader/auto_weights_loader.py` / `layers/linear.py` |
| `model_loader/default_loader.py` | CPU 按张量读取 checkpoint，GPU 模型参数在创建时就是本地形状 | `model_loader/base_loader.py` 与 `weight_utils.iter_weights`；测试记录每 rank 参数、源区间与峰值 |
| `v1/worker/gpu/model_runner.py` | Worker 持有 GPU 本地模型和 KV；Scheduler 管请求和逻辑块 | `worker/worker.py` / `worker/gpu/model_runner.py` / `executor/multiproc_executor.py` |

固定 tiny：4 层、H=32、Q=4、KV=2、D=8、I=64、V=129、TP2。本地 Q=2/KV=1，attention 前后本地 Q 宽度16；embedding、o_proj、MLP 归约后和 residual 是完整 H=32；LMHead 汇总为全局 V=129。物理 KV 各 rank 独立、逻辑块号由唯一 Scheduler 决定；请求对象绝不传给执行侧修改。

090 的固定 128-token BF16 参考是先于本关登记的契约，仅可用于相同 fixture、对象和配置。tiny 逐层与冻结参考及 3 次 decode、8B TP1/2/4 数值、14B TP2 真加载峰值均须独立取证；不可把 098/100 测试 Worker 推导成生产模型已开放。生产仍须明确拒绝 PP、投机多卡、Graph 和不在本关支持清单的模型组合；加载峰值与 098 空闲显存快照不是同一度量。

## 实现边界与测试方法

生产 Worker 不再对所有多卡模型统一拒绝，而是在本地权重创建前验证：V2、TP>1、单副本、PP=DP=1、同步、eager、TRITON_ATTN、普通 dense `Qwen3ForCausalLM`、无投机/量化/滑窗/prefix caching/非默认 RoPE；head 布局由既有 `ParallelConfig.validate_model` 校验，MLP 列数可被 TP 整除。超出范围明确拒绝，不关图、不降 TP、不复制完整权重。旧 093 的 `production` 负例使用没有架构类型的虚构配置，现在检查“非稠密 Qwen3 被拒绝”，而不是要求所有生产多卡均被拒绝。

真实权重路径通过默认生产 Worker、CPU safetensors 惰性迭代和参数自带 loader 运行；测试函数通过 Executor RPC 读取模型实际 named_parameters 及 CUDA 峰值，并依据各模块的分片属性推导每参数 source 区间；它不参与生产计算，也不是对每次 loader 读取的逐调用追踪。每 rank 清单对账 checkpoint 的全局索引和每个张量真实形状：8B 的 291 本地参数映射 399 个 source tensor；14B 的 323 本地参数映射 443 个 source tensor，均无重复或缺失。数字、模型 revision、逐参数清单及实际源区间在结果目录中。

固定需求 tiny D=8 的 4 层 FP32、TP1 与 TP2 已覆盖 prefill/3 decode、各层 norm、QKV、本地 attention、归约后 attention、MLP、residual、hidden/logits；两 rank 分别对对应 Q/K/V 与 attention 头切片比对。测试在 layer 2 前插入短暂停顿，逐张量确认暂停前后结果相同；暂停只存在测试 hook。

冻结 vLLM 的 TRITON_ATTN 明确拒绝 D=8（至少 32），其 FLEX_ATTENTION 在该 fixture 上初始化超过 180 秒被超时终止；这两次尝试保留日志，不冒称固定 D=8 的冻结 GPU 参考通过。另加不替代固定例的 D=32 4 层 tiny，并使用真实冻结 vLLM V2 / TRITON_ATTN、TP2 验证 4 层及 prefill/3 decode：冻结默认 `tl.dot` 为 TF32 而项目 FP32 路径显式 IEEE，默认设置的第 0 层 Attention 相差约 1.59e-4。使用同一组实际 Q/K/V 与 CPU FP64 因果 Attention 对照，项目误差约 2.98e-8、上游默认约 1.59e-4；该结果支持精度模式是差异来源，但不将其记为上游 bug。额外以 `TRITON_F32_DEFAULT=ieee` 对齐参考计算精度，在公共协议原有 FP32 `atol=1e-5, rtol=1e-5` 下检查所有 4 层、两个 rank 和四个时刻；不按 mini 的误差放宽阈值。这个非默认精度条件和 D=32 范围单列，不能外推到默认设置下的逐值一致。

## 真实模型与结果

真实 8B 使用 090 的 128 token 固定输入和冻结参考，BF16、eager、一次 prefill、输出 1 token；以下列出各 TP 情况中各 rank 的最大误差（并非逐值相等）。阈值是 090 已预注册的固定合同，不按本关输出重新确定。`raw_logits` 的 top-1 都为 198，top-2 都为 271；对应 margin 为 TP1/2 13.125、TP4 13.5。

| TP | QKV max abs / 相对 L2 | attention max abs / 相对 L2 | hidden max abs / 相对 L2 | logits max abs / 相对 L2 |
|---|---|---|---|---|
| 1 | 0.0078125 / 0.002195 | 0.015625 / 0.003397 | 10.375 / 0.035213 | 0.625 / 0.035931 |
| 2 | 0.00390625 / 0.002321 | 0.015625 / 0.003955 | 6.9609375 / 0.031979 | 0.625 / 0.036093 |
| 4 | 0.00390625 / 0.002516 | 0.015625 / 0.004253 | 13.62109375 / 0.043265 | 0.509765625 / 0.029111 |
| 090 阈值 | 0.0078125 / 0.005171 | 0.03125 / 0.008007 | 18.8125 / 0.067431 | 1.1875 / 0.085890 |

14B checkpoint 全部 source tensor 共 29,536,614,400 bytes；TP2 两个 rank 各自本地参数 14,768,732,160 bytes，加载期间 PyTorch `max_memory_allocated` 各 14,905,982,464 bytes、`max_memory_reserved` 各 14,948,499,456 bytes。峰值包含被 PyTorch allocator 统计到的模型和加载过程分配，并非整卡总显存峰值；它低于完整 checkpoint 的参数字节数，结合本地 shape / 区间对账支持未在 GPU 构造完整 14B 参数副本。本次 14B 只测加载，未声称真实 14B 前向。

权重清单对账检查 source 名称、metadata shape、本地参数形状和对应区间以及遗漏/重复，不声称逐元素比较过完整 checkpoint 的权重字节。原始 trace、各 rank 清单、固定合同差分、峰值、两次 D=8 后端失败日志及汇总均在 `benchmarks/results/step101/`，该目录不提交 Git。tiny 固定 D=8 使用 TP1 作为单卡参考满足本关“冻结 reference/单卡”验收选项；额外 D=32 冻结参考的非默认 IEEE 条件及局限见上一节。

## 命令与边界

激活环境后，`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/step101 -q -ra` 为 3 passed；`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmarks/check_step101_acceptance.py --case all --timeout 1800 --output benchmarks/results/step101/acceptance_release.json` 的 tiny、8B TP1/2/4、14B TP2 五项均通过。逐项命令、回归数量、版本与模型 revision 记录在 `docs/results.json → step101`。不以本关加载探针声称 PP、EAGLE3 多卡、多 rank Graph、端到端异步调度或真实性能已经验收；本关不测吞吐。

历史回归：58—88 为 **1289 passed / 0 failed / 23 skipped**（历史真实权重不可用，不计作通过）；089—100 为 **166 passed / 0 failed / 0 skipped**。两者分别在单卡和多卡环境串行运行。
