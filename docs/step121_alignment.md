# Step 121：W4 Marlin 重排与工作区（实施中）

编码前映射：R121.01/02/06/07 为 `minivllm/layers/quantization/awq.py` 和 `marlin.py` 的标准布局到 kernel 布局流程；R121.03/08 为 `marlin.py` 的 workspace 与层内生命周期；R121.04 为 `marlin.py` 的 padding、scale/zero 置换与逻辑尺寸；R121.05 为 `tests/step121/` 独立快照差分。点名工具为 `benchmarks/benchmark_step121_marlin_repack.py`。仅支持无 act-order 的 AWQ W4A16 G128，GEMM 和模型推理仍属后续关。

真实 GPU 差分、Graph 与显存验收暂未进行；四张 GPU 均有其他进程使用。后续结果按实际运行记录，不能将 CPU 测试当作 GPU 验收。

## 实现及验证状态

| 子项 | 当前实现和证据 | 未完成 |
|---|---|---|
| R121.01 | `marlin.prepare_marlin_weights` 按 G128、无 act-order 处理 full-K、padding、workspace、zero、qweight 和 scale；通过 `AutoAWQMarlinLinearMethod` 接入 CUDA 后处理 | CUDA 执行及逐阶段冻结上游差分未运行 |
| R121.02 | AWQ 先执行 `_convert_awq_to_standard_format` 再调用 `gptq_marlin_repack`；真实执行仍拒绝 | 真实 CUDA 对照未运行 |
| R121.03 | 每个 LinearMethod 持有 workspace；按 SM 数分配 int32 清零，可检验规格后原地复用 | GPU 与 Graph 生命周期未运行；完整 reload 路径未实现 |
| R121.04 | `marlin_padded_nk`、padding、scale 和 zero 置换以及逻辑/填充尺寸元数据；group 截断拒绝 | 最终输出裁剪由 122 GEMM 路径接入，当前没有 GEMM 输出 |
| R121.05 | CPU 针对非均匀 scale/zero 使用冻结源码 AST 和独立 numpy pack 逐值对账；`step121_gpu_probe.py` 为独立进程 tiny repack 对照入口 | qweight 真实 GPU 精确差分及各阶段真实权重快照未运行 |
| R121.06 | 无 act-order 的 g_idx 与 sort_indices 使用设备 int32 空 tensor | 设备结果未运行 |
| R121.07 | 转换完成后三个参数标 kernel，既有 Parameter 的 checkpoint 写入门禁仍生效 | 真实显存峰值、临时释放尚未测量 |
| R121.08 | 每层独立 workspace，显式 reuse 检查；记录逻辑/填充尺寸 | Graph 中参数与 workspace 地址稳定、跨模型和跨 Worker 未验证 |

CPU 表示审计仍保留标准布局；CUDA 表示审计进入 kernel 准备路径，但未经过 GPU 验证；正常量化推理依旧明确拒绝。CUDA 处理严格限制 SM89、BF16/FP16、W4/G128、无 g_idx、zero point；其他情况拒绝，不做降级。现有模型加载流程只允许执行一次后处理，因此尚不能宣称模型 reload 后 packed 地址可复用。

## 本次执行

- `source scripts/activate-4090.sh` 后 `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/step115 tests/step116 tests/step117 tests/step118 tests/step119 tests/step120 tests/step121 -q -ra`：86 passed、0 failed、0 skipped；全部为 CPU、构建或 meta，不能代表真实 GPU。
- `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmarks/benchmark_step121_marlin_repack.py --check --output benchmarks/results/step121/check.json`：CPU 前置检查通过。`--compare` 实现 tiny 隔离 CUDA 对照入口但本次未执行；真实抽层计时、峰值、trace 尚未实现，不标记性能通过。
- `python learning_notes/14_vllm_from_scratch/量化完整需求/参考资料/check_requirements.py`：27 文档、24 关、849 源文件静态校验通过，不是模型验证。
- 四张 GPU 有其他用户进程，未运行 GPU、真实量化模型、TP、Graph、完整回归或性能。T121.01、T121.03 的 CUDA 部分，以及 T121.04/05 的实际生命周期和内存验证均未通过，121 保持实施中。
