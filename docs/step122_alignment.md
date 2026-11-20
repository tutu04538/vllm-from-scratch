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

## 2026-11-20：GPU 正确性补验与局部前向

- `CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=2 python benchmarks/benchmark_step122_w4a16_gemm.py --compare --case tiny --output benchmarks/results/step122/gpu_compare_20261120.json`：BF16、K=256、N=128、M=1/2/7/16/32/128，同 packed 输入的独立 vLLM 进程与本地逐值相同，FP32 参考最大绝对误差 0.013907；100 次交错 workspace 复用和另一个 CUDA stream 检查通过。
- `tests/step122/step122_real_projection.py` 逐个运行两份公开 AWQ 首层的 qkv、o_proj、gate_up、down_proj，覆盖真实完整投影形状及上述六种 M（合计 48 次）。本项目与独立上游同 packed GEMM 输出最大绝对差均为 0；从原 checkpoint 独立构造 FP32 参考，检查每次输出前 16 列，最大绝对差不超过 0.002799。结果在 `benchmarks/results/step122/real_{8B,14B}_*_final_20261120.json`。qkv/gate_up 使用实际逻辑权重拼接，属于局部单卡数值检查，不代表模型 TP。
- GPU integration `tests/step122/test_step122_cuda_integration.py` 实测 N=64/72/128、非零 bias、skip_bias_add、M=1/2/7/16/32/128、两个层的 workspace 与 Graph 重放；其中 N=72 验证 padding 和输出裁剪。加入 profiler 用例后独立运行 8 passed；与 121 合跑 19 passed。真实 8B QKV 的 `--trace` 结果 `benchmarks/results/step122/real_8b_qkv_trace_20261120.json` 和同名 trace 核实本地 `minivllm_quant::marlin_gemm`、CUDA Marlin kernel，并检查前向不含 repack 和 `aten::linear`；不是性能测试。
- 因上述 GPU 测试通过，`AutoAWQMarlinLinearMethod.apply` 对 CUDA 上已完成 kernel 布局的层开放局部前向；CPU 标准布局仍拒绝。`supports_execution=False` 仍保留在模型加载生命周期上，正常量化 Engine 初始化继续拒绝，123 整模型未开放。上述 GPU 检查不能覆盖未测试的 TP/模型组合。
- `CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=2 python -m pytest tests/step113 tests/step114 tests/step115 tests/step116 tests/step117 tests/step118 tests/step119 tests/step120 tests/step121 tests/step122 -q -ra`：129 passed、0 failed、0 skipped，日志 `benchmarks/results/step122/regression_113_122_gpu1_20261120.log`。
- 首轮旧单卡 58—88 为 1 failed、1288 passed、23 skipped：唯一失败是当前包 hash 尚未写入 `docs/results.json`。更新 hash 后，`tests/step81/test_spec_metrics_pipeline.py::test_current_package_hash_matches_results` 单项 1 passed；首轮完整日志 `benchmarks/results/step122/regression_58_88_gpu0_20261120.log`，不能把首轮改写成全通过。
- 当前包 SHA256 `b9ca6877571f40bc95531375bf37cbf047d550756c6f180a83cc0dbc833607f7`。FP16、完整 FP32 所有输出列、真实 TP 和整模型 Graph 未覆盖；性能按用户安排延期。121 未完成全部原退出条件，因此 122 维持实施中，不宣称全关验收。

上述 tiny 算子差分在开放局部前向之前、包 hash `b02bb758ab14ebc6fc32bc2fc5fea92de25d702317c8f9f979d8978a90f16234` 下运行；八个真实投影最终结果和 GPU 集成重跑使用 `b9ca6877571f40bc95531375bf37cbf047d550756c6f180a83cc0dbc833607f7`。

冻结上游 `skip_bias_add` 疑点已补充真实 GPU tiny 层复现：输出 `[1,72]` 而返回 bias 为重排且补齐的 `[128]`。这是固定版本、特定组合的实测，不推断所有模型受影响；命令与差分见本地 `vllm_bugs/UPSTREAM_SUSPECTED_BUGS.md`。

八个真实首层投影后续补充了全部输出列的独立 FP32 oracle 和完整真实 qweight repack 对照：各 M 的本地/上游输出全部逐值相同，FP32 全列最大绝对误差不超过 0.007842；证据 `benchmarks/results/step122/real_*_full_oracle_20261120.json`，包 hash `3fd57c0121f098ac682fc671799d348ca609406769295eb7c09daeac923713a8`。同时 N=64/72/128 和 bias/skip_bias_add 的 BF16 与 FP16 合成 AWQ GPU 集成重测 14 passed。该记录覆盖全输出列，不覆盖 TP/整模型 forward。

最终包 `effea1cb00748c482a13ef4ebfd52214e6fb6cdb126e1e621792a7b8cf708908` 上重跑：八个真实投影的完整 repack、同输入上游 GEMM 和全部输出列 FP32 oracle 均通过（最大绝对误差 0.007842）；BF16/FP16 tiny 集成均包含在 113—122 联合回归的 138 passed 中。旧单卡 58—88 最终重跑 1289 passed、0 failed、23 skipped；跳过项见 `benchmarks/results/step122/regression_58_88_final_20261120.log`。091/108 环境隔离后包含 GPU Engine 的组合回归 62 passed。最终包上的 089—112 完整重跑在提交时仍进行中，不能将初次 417 passed、1 failed 改写为通过。
