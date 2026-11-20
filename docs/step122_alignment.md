# Step 122：W4A16 GPU 矩阵计算（实施中）

编码前映射：R122.01/04/07 为 `minivllm/quant_ops/ops.py` 的 GEMM ABI、参数校验和归约策略；R122.02/08 为 `minivllm/layers/quantization/marlin.py` 与 `awq.py` 的输入展平、pad、输出裁剪及方法边界；R122.03/05/07 为 `tests/step122/` 与独立 CUDA 对照入口；R122.06 为点名 trace；点名 benchmark 为 `benchmarks/benchmark_step122_w4a16_gemm.py`。无 GPU 正确性证据前，量化局部 forward 和 Engine 仍拒绝执行。

121 的真实 CUDA 重排、工作区、Graph 仍未验；四张 GPU 均为其他进程占用。本关先完成不依赖 GPU 的实现和测试，性能按用户要求留到大部分需求完成后。

## 本次实现与证据

| 需求 | 实现及本次证据 | 未完成 |
|---|---|---|
| R122.01 | `marlin_gemm_w4a16` 绑定本地完整扩展的 uint4 W4A16 schema；shape、dtype、stride、device、workspace、g_idx 等有检查；ScalarType ID 对照冻结源码，meta 检查通过 | 真实 CUDA 执行未验 |
| R122.02 | `apply_marlin_w4a16` 处理多维展平、K padding、N 裁剪和 batch shape 恢复；逻辑 bias 与 kernel bias 分开，保持 skip_bias_add 的逻辑顺序 | 非零 bias 真实 GPU 检查未运行 |
| R122.03 | CPU/meta 覆盖 M=1/2/7/16/32/128 和多维输入；tiny GPU 对照脚本覆盖这些 M | 实际 GPU 及真实 Qwen3 投影尺寸未运行或完成对照脚本 |
| R122.04 | atomic add 条件按冻结逻辑及其默认关闭环境策略；FP32 reduce 默认 true；CPU 条件测试通过 | GPU 可重复性未验证 |
| R122.05 | tiny GPU 脚本计划以同一 packed 输入隔离比较本地和安装版冻结 vLLM，并比较独立 FP32 oracle；子进程核对上游 Python 源 hash | 未执行；121 CUDA repack 前置未验，真实模型级对照未实现 |
| R122.06 | 热路径调用本地扩展、未包含全层反量化及每轮 repack；静态实现不是 profiler 证据 | GPU trace 未实现、未运行 |
| R122.07 | 本地接口检查形状、dtype、设备、stride、16 字节对齐和 workspace 容量；GPU 探针写有 100 次交错和单独 stream 检查 | 仅 CPU/meta 负例实测；GPU 100 次与 stream 未运行 |
| R122.08 | LinearMethod 已接入目标调用，`supports_execution=False` 保持拒绝；完整模型加载同样仍拒绝 | GPU 准确性通过之前不能放开局部 forward |

121 后处理此前首次运行时可能只加载 repack 扩展，使随后加载完整 GEMM 扩展发生重复 schema 冲突。当前在 CUDA 后处理前先加载完整扩展；已在单独隔离进程加载 repack-only 的场景，GEMM 要求干净进程加载完整扩展，不做自动混装。未运行 GPU 集成。

冻结上游 `MarlinLinearKernel` 修改 `layer.bias.data` 的 kernel 排列，而 `skip_bias_add` 的 Linear 直接返回 `self.bias`；静态组合及 CPU 探针显示可能返回重排后甚至包含 padding 的 bias。本项目将逻辑 bias 保持原样，为 GEMM 另存 kernel bias；此差异记录在本地 `vllm_bugs/UPSTREAM_SUSPECTED_BUGS.md` 和 `vllm_bugs/marlin_skip_bias_cpu_probe.py`。尚未做上游 GPU/端到端最小复现，不能扩大为所有模型已出错。

## 实测命令与边界

- `source scripts/activate-4090.sh` 后 `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=2 python -m pytest tests/step115 tests/step116 tests/step117 tests/step118 tests/step119 tests/step120 tests/step121 tests/step122 -q -ra`：98 passed、0 failed、0 skipped；仅 CPU、构建与 meta。随后修改冻结 scalar 测试的源码执行方式以避免写 bytecode，单独重跑 `tests/step122`：12 passed、0 failed、0 skipped。
- `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmarks/benchmark_step122_w4a16_gemm.py --check --output benchmarks/results/step122/check.json`：CPU 检查通过；同脚本 `--compare --case tiny --output benchmarks/results/step122/blocked.json` 在屏蔽 CUDA 的环境明确返回 blocked，未执行真实 GPU。
- `python learning_notes/14_vllm_from_scratch/量化完整需求/参考资料/check_requirements.py` 初次因测试读取冻结文件生成 `__pycache__/scalar_type.cpython-312.pyc` 而失败；修正测试不生成 bytecode、删除本次生成的文件后重跑通过：27 文档、24 关、849 源文件。
- 四卡当时均有其他任务，121 前置 CUDA 未验，未运行 GPU、真实权重、TP、Graph 或全量历史回归。性能按用户约定延后。trace、点名真实尺寸对照和实际独立 GPU oracle 仍未完成，122 不是完成状态。
