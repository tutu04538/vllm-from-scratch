# Step 117：量化 Parameter 与加载生命周期（实现中）

本关将逻辑分片、物理打包轴和 scale 的加载职责放入项目内 Parameter，并在后处理前验证加载完整性；不开放量化矩阵计算。

编码前的 R 映射：R117.01/02 对应 `minivllm/layers/parameter.py`；R117.03/04 对应 `minivllm/layers/linear.py` 和 `minivllm/model_loader/auto_weights_loader.py` 的分片委派与覆盖检查；R117.05/06 对应 Parameter 布局状态与 `minivllm/model_loader/` 后处理；R117.07/08 对应加载器的量化方法、设备上下文、延迟 attention 和模型 hook 顺序；R117.09 对应小 fixture，不开放真实量化 TP。测试在 `tests/step117/`，点名入口为 `benchmarks/benchmark_step117_quant_loading.py`。

## 实现与已知边界

`parameter.py` 提供 Base、Column、Row、ModelWeight、Packed、GroupQuantScale 和 PerTensorScale 参数；参数显式记录输入/输出/打包轴和倍数、TP 状态与逻辑分片覆盖。行列与融合层将量化参数委派给各自的参数加载方法，QKV 的 KV 复制沿原 TP 头数规则取分片；checkpoint 内已融合的输出按逻辑边界拆分。读取 shape/dtype 不符、缺分片、重复分片、未对齐打包边界和未知 tensor 均明确拒绝。`permute_param_layout_` 只变更轴顺序，不执行 repack。

后处理由 `post_load.py` 在加载完成后依次处理量化方法、参数 TP 状态、延迟 Attention hook、模型 hook，并用状态阻止重复执行。方法须在真实转换后显式推进 checkpoint → standard → kernel，不能在尚未转换时声称 kernel ready。失败时模型不可返回或重试；权重替换时保留仍需的 loader 和加载状态。设备加载上下文按冻结顺序恢复 CPU 参数，并对被替换的 UVA 参数重新映射；UVA、特殊 Parameter 的转换资源清理与 GPU 失败传播均未实测。真实量化方法尚未开放，当前主要调用链经测试 fixture 进入；没有将 CPU 反量化作为生产热路径。

## 基础检查

- `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest tests/step117 -q -ra`：8 passed，0 failed，0 skipped，CPU 小张量与生命周期检查。初次执行有 3 个 Parameter 构造参数错误，修正后重跑通过。
- 在最后一次输入输出轴校验小改动之前，同环境执行 `tests/step117 tests/step116 tests/step115 tests/step090/test_step090_models.py tests/step096/test_step096_linear.py::test_single_gpu_free_compatibility_and_bias tests/step097/test_step097_packed.py tests/step097/test_step097_qwen_loader.py -k 'not test_true_gpu_and_frozen_or_real'`：72 passed、7 deselected，后者是未选择的 GPU 用例。随后只重跑了 step117 的 8 项。
- `benchmarks/benchmark_step117_quant_loading.py --check --output benchmarks/results/step117/check.json`：CPU 小样例加载与字节清单；`--compare`/`--trace` 为 not_run，不是 GPU 正确性或性能结果。

当前包内相对路径 SHA256 `d22c6eb37917618b05c3f3857b8fa630eba48a6bd2080ab704e87911f2b6f90d`。GPU/tiny 实机、真实权重、多 rank、Graph、失败资源清理及完整回归均未运行；UVA 量化后处理已接入代码但尚未验证。117 因此不能标记为完全验收，后续实现量化方法时需要继续完成上述边界。

追加检查：在最终 UVA 与失败状态接入之后，`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest tests/step117 tests/step116 tests/step115 -q -ra`：49 passed，0 failed，0 skipped；仅 CPU。

## 2026-11-20：UVA 后处理 GPU 测试及边界修复

真实 CUDA 测试发现 `device_loading_context` 重映射 UVA 参数时绕过默认 cpp 统一入口，并且同进程切换 GPU 时可能把目标 cuda:1 的参数恢复为 cuda:0 视图。原实现 `tests/step117/test_step117_uva_gpu.py` 实测 2 failed、1 passed，日志 `benchmarks/results/step122/step117_uva_before_fix_20261120.log`。

修复改用 `uva_ext.get_cuda_view_from_cpu_tensor`，默认继续 cpp 且不自动回退；参数持有 pinned CPU tensor 以保证借用存储的生命周期，并在目标设备上下文进行 pin/view。由于 pinned allocator 可能复用之前设备关联的内存，仅切换上下文仍可能得到错误设备：现在检查视图设备，错配立即报错而非静默返回错误设备；当前设备在退出上下文时恢复。新测试 3 passed，其中跨设备用例验证正确返回或明确报错，**不代表跨设备缓存错配已支持**。单独构造先在 cuda:0 缓存 pinned 内存，再在 cuda:1 重新映射，实际得到明确拒绝 `UVA 视图设备 cuda:0 与目标 cuda:1 不一致`。

尝试独立实验使用 stable `from_blob` 显式标记 cuda:1 时子进程退出码 139，未将该实验用于生产。原 pybind CUDA 指定设备原型因服务器高 I/O 编译长时间阻塞而中断；尚需安全定位并实现跨设备 pinned 缓存的完整支持。正常单 Worker 当前卡的 UVA 重映射、CPU owner 释放和异常恢复经过 GPU 检查，但不能推广为所有设备组合通过。
