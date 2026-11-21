# step124：AWQ 量化权重 TP 分片

实施中；下表为编码前的需求映射，不代表验收通过。

| 需求 | 拟检查的位置与验证 |
| --- | --- |
| R124.01 | `parameter.py`、`linear.py` 的 rank 逻辑切片；制作四类参数切分表 |
| R124.02 | `PackedvLLMParameter`、`GroupQuantScaleParameter` 的列/行分片，分别核对 K/N、打包轴、group |
| R124.03 | `QKVParallelLinear.weight_loader` 与 Parameter 的 head 复制，对 qweight/zero/scale 分别验算 |
| R124.04 | `MergedColumnParallelLinear.weight_loader`，各逻辑 shard 独立切片 |
| R124.05 | `AutoAWQMarlinLinearMethod.create_weights` 与本地后处理，核对本地 shape |
| R124.06 | AWQ 创建时检查 group、packed 对齐，Marlin 本地 tile padding |
| R124.07 | `RowParallelLinear.forward` 保留结果 all-reduce，不额外归约 scale |
| R124.08 | 仅本地抽层验证；完整模型 TP 由 125 开放 |

## 当前静态核对及 CPU 精确分片

119/117 已经实现按本 rank shape 建立 AWQ checkpoint 参数并调用对应 Parameter 加载器；本次先覆盖这些既有机制，不重复增加另一套分片代码。以下取逻辑 `W[K,N]`，AWQ checkpoint 在输出维以 8 个 4-bit 值打包，group=128：

| 类型 | qweight checkpoint | zeros checkpoint | scales | rank 范围 |
| --- | --- | --- | --- | --- |
| Column | `[K,N/8]` | `[K/128,N/8]` | `[K/128,N]` | 分别在 N/8、N/8、N 上按 rank 切 |
| Row | `[K,N/8]` | `[K/128,N/8]` | `[K/128,N]` | qweight 按 K 切，zeros/scales 按 K/128 group 切 |
| QKV | 分别加载 Q/K/V 各自的上列形状 | 同左 | 同左 | Q 独立切；KV head 不足 TP 时，以冻结 head 规则复制三个参数 |
| Merged | 分别加载 gate/up 各自的上列形状 | 同左 | 同左 | gate/up 先分别按 rank 切，再拼本地输出轴 |

CPU 测试覆盖 TP1/2/4 所有逻辑 rank、GQA 单 KV head 复制、非对称 merged、group 边界拒绝；输入图案不同，逐项比较三种参数的物理分片。`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/step124 tests/step117 tests/step119 -q -ra`：30 passed。上述 CPU 测试不是多 GPU 端到端证据；后续真实权重本地分片补验见下文。逐 rank 常驻峰值与点名 benchmark 尚未完成，性能延后。

## 真实 checkpoint 本地 rank 补验

8B 与 14B 各取首、中、末三个 Transformer 层，分别检查 QKV、o_proj、gate/up、down_proj；TP2 和 TP4 的每个逻辑 rank 在一张空闲卡上依次创建本地参数。共 144 个本地分片：checkpoint 中 qweight、qzeros、scales 的原始切片与独立索引算出的结果逐元素相同；经本地后处理后，使用相同 packed 参数和输入的本地 Marlin GEMM 与冻结 vLLM CUDA GEMM 最大绝对差为 0。脚本通过 `verify_oracle` 校验安装版与冻结上游的身份。

命令为 `CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python tests/step124/step124_real_tp_layers.py --model <固定 AWQ snapshot> --tp <2 或 4> --output <结果>`，8B/14B 与 TP2/TP4 组合分别运行。原始结果和日志保存在本地 `benchmarks/results/step124/`。第一次调用因测试脚本未将默认 dtype 切为 BF16 而失败，修正脚本后四组均通过。

这里的 TP4 是**同一张 GPU 上串行模拟四个本地 rank**，不涉及四个 Worker、collective，也不能作为 125 的 TP4 端到端证据；同 packed GEMM 对照也不是独立的上游重排差分。性能、逐 rank 显存计账及本关 benchmark 仍待完成。

## 2026-11-21 独立上游 repack 补验

在真实 8B/14B checkpoint 的首、中、末层，分别对 QKV、o_proj、gate/up、down_proj 的 TP2/TP4 所有本地 rank 进行完整 repack 比较，共 144 个分片。对每个分片，以本地准备的相同标准 packed 输入分别调用本项目 CUDA repack 与冻结 vLLM CUDA repack，整个整数输出逐元素一致；同 packed 的后续 GEMM 最大绝对差仍为 0。参考算子 `_custom_ops.py` 已对照冻结文件 SHA-256。原始 JSON/log 在本地 `benchmarks/results/step124/repack_*_20261121.*`。

该测试在单张 GPU 上串行执行所有逻辑 rank；真实四进程模型及通信验证另见 [125](step125_alignment.md)。性能及点名 benchmark 按用户约定留待后续集中测试，逐 rank 显存见 125 报告。
