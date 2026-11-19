# step113：冻结源码与量化能力基线（进行中）

状态：**未完成**。本关要求真实 4090 量化算子和 dispatch 探针；2026-11-19 其他用户训练进程占用四张卡，当前只实施不占 GPU 的部分。历史 step108 多卡回归有一次中断及一个待定位失败，见该关报告。

## 实施映射

| 子需求 | 冻结源码/规范 | 本项目位置 | 状态 |
|---|---|---|---|
| R113.01 | 工作树与包身份 | `benchmarks/benchmark_step113_quant_capability.py` 基线检查 | 已记录基线与当次 dirty 文件 hash；历史 108 多卡回归仍待结束 |
| R113.02 | 849 文件清单及安装包 | 同上 | 849 文件、346 安装版 Python、342 历史源码对账通过；GPU 硬件能力另验 |
| R113.03 | AWQ/GPTQ/FP8 参考分派 | `tests/step113/step113_quant_reference_probe.py` | GPU2 下记录别名、配置及 4096×4096 的候选/选择；尚缺完整形状与 override 对照 |
| R113.04 | W4、动态 FP8、KV 算子 | `tests/step113/step113_quant_reference_probe.py` | GPU2 的 BF16 W4、FP8 W8A8 和 FP8 cache 小例通过；非完整模型或 kernel 迁移验收 |
| R113.05 | Marlin/FP8 backend | 隔离参考子进程 | 默认 W4 Marlin、FP8 CUTLASS，显式 FP8 CUTLASS 一致；仅点名 shape |
| R113.06 | CMake、算子绑定、CUTLASS | `benchmarks/step113_cutlass_manifest.py` | 固定 checkout、头文件及许可 hash；本地 FP8 类型编译通过，量化 kernel 构建待 120 |
| R113.07 | 独立上游参考进程 | 本关测试与探针 | GPU2 隔离子进程、小张量通过；生产导入与动态链接检查通过 |
| R113.08 | 生命周期、能力表 | 本关测试与结果 | 两次 GPU 子进程成功退出；CPU 失败子进程回收已测，GPU 故障清理待验 |

`--check` 为环境和源文件完整性检查，不能替代 GPU 能力和数值验收；结果中必须区分未运行、失败与通过。性能测试按用户要求集中到功能基本完成之后。

## 2026-11-19 CPU 基线

旧变更的工作树已提交到 `2c3125c10695d49ff21d50c1f0bc42d1d3aa652e`，包内相对路径 hash 为 `2720f8bfb1f3f2c9e7311d9d4810a3f5cdeb21b883fcb448f6f683b3046517fc`（196 个 Python 文件）。探针另保存运行当时新建的未跟踪文件清单和逐文件 hash，不能把当时 dirty 工作树当成纯 HEAD。

`source scripts/activate-4090.sh` 后运行 `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmarks/benchmark_step113_quant_capability.py --check --output benchmarks/results/step113/cpu_identity_20261119.json`：849 份源码快照 hash、346 份安装版 Python、342 份并行专题快照及对应安装版一致。首次运行错误地要求两套冻结清单彼此包含，产生 342 个假失败；修正为各自与安装版逐项对账后通过。Python 3.12.13、Torch 2.13.0+cu129、Triton 3.7.1、vLLM 0.28.0+cu129、Transformers 5.14.1；nvcc 12.9.86。CUTLASS 远端 v4.4.2 tag 指向 `da5e086dab31d63815acafdac9a9c5893b1c69e2`，本次没有下载其头文件或构建算子。能力文件的 GPU 算子状态为 not_run，不能以 CPU 检查通过替代。

`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest tests/step113 tests/step115 tests/step090/test_step090_models.py -q -ra`：34 passed、0 failed、0 skipped；其中 113 的三项仅覆盖清单破坏检测、hash 口径及 GPU 未实现状态。量化 GPU、真实权重、TP/Graph 与完整回归均未运行。


### 隔离 GPU 探针准备

新增 `tests/step113/step113_quant_reference_probe.py`，覆盖 AWQ/GPTQ W4A16 Marlin（M=1、128）、参考 kernel 分派、CUTLASS FP8 W8A8、FP8 cache 转换；只在独立上游子进程引用 vLLM。探针尚未在 GPU 上运行，其实现及数值阈值均待实测验证，不计作对应 R/T 已通过。`--compare` 在启动前只读 `nvidia-smi` 检查选定卡上的计算进程；检测繁忙或查询失败返回 `not_run`，不启动子进程。注意启动前检测无法杜绝随后其他任务抢占设备。

当前 GPU0 有其他用户进程，`python benchmarks/benchmark_step113_quant_capability.py --compare --device 0 --output benchmarks/results/step113/gpu_busy_20261119.json` 返回码 2，记录 `not_run`；无 GPU 算子执行。`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest tests/step113 tests/step115 tests/step090/test_step090_models.py -q -ra`：35 passed、0 failed、0 skipped；`runpy.run_path(..., run_name='step113_import_only')` 的上游探针 CPU 导入通过，不代表算子可运行。性能 trace 依照用户要求延后。


## 2026-11-19：4090 GPU2 参考能力与 CUTLASS 源码

GPU0/1 正有其他用户任务；单卡 GPU2 空闲后执行 `python benchmarks/benchmark_step113_quant_capability.py --compare --device 2 --output benchmarks/results/step113/gpu2_reference_final_20261119.json`。冻结上游独立进程在 RTX 4090、SM89 下完成 11 项小张量检查，结果文件 `gpu2_reference_final_20261119_reference.json` SHA256 `fa25ddd22523211d20ac0dafe95f3ea42bbeb52499951b554d966ae90110f851`：

- AWQ 与 GPTQ W4A16 的 K=N=4096、group=128、BF16 输入默认选 `MarlinLinearKernel`；分别在 M=1/128 调用真实 Marlin GEMM，对其量化后权重的 BF16 参考，平均绝对误差与参考平均绝对值之比最大约 `1.2e-6`。
- FP8 动态 per-tensor/per-token 的默认及显式 CUTLASS 均选 `CutlassFP8ScaledMMLinearKernel`。CUTLASS GEMM 的 A/B 实际 dtype 均为 `torch.float8_e4m3fn`，M=1/128 的相对平均误差分别约 `0.00168`、`0.00166`；输出 BF16，因此此处确实是 W8A8，不是 W8A16。
- FP8 cache 以 scale=1 的小向量往返通过；不等于 FP8 attention 或校准 scale 已支持。首次参考进程中 FP8 分派未放在 vLLM 配置上下文而失败，其他算子通过；按冻结上游测试增加 `set_current_vllm_config(VllmConfig())` 后独立重跑通过，首轮失败记录 `gpu2_reference_20261119.json` 保留。两次成功进程退出后现场未发现其遗留 GPU 进程。

`nvidia-smi topo -m` 记录 GPU0/1 为 NUMA0、GPU2/3 为 NUMA1，`topo -p2p r` 所有跨卡项为 CNS；这是驱动拓扑报告，不声称当前再次运行了实际 CUDA peer 测试。当前环境身份重查见 `cpu_identity_after_115_20261119.json`：包 hash `30b5bae802894f03009f1e4e459af611f78979a0fed901fd4cacec9726ddb443`，当次工作树有尚未提交的探针文件，记录与 HEAD 分开。

CUTLASS v4.4.2 checkout commit `da5e086dab31d63815acafdac9a9c5893b1c69e2`，1159 个 `.h/.hpp/.cuh` 文件按相对路径及 sha256 聚合 `3cdee752e4b4f1c3ae45191362eaca9637b900bd6fe828bc91d2dcda691ee346`，许可证 SHA256 `7bc733829f16663593865344541c6dcc433beb6b6b555c8f86a43e9b399993cf`，详见本地 `cutlass_v442_manifest.json`。`nvcc -std=c++17 -arch=sm_89 -c tests/step113/step113_cuda_toolchain.cu -o benchmarks/results/step113/step113_cuda_toolchain.o` 编译通过，只证明 FP8 类型基础编译，**不是 Marlin/CUTLASS 量化 kernel 编译通过**。

`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest tests/step113 tests/step114 tests/step115 tests/step090/test_step090_models.py tests/step096/test_step096_linear.py::test_single_gpu_free_compatibility_and_bias tests/step097/test_step097_packed.py tests/step097/test_step097_qwen_loader.py -k 'not test_true_gpu_and_frozen_or_real' -q -ra`：72 passed、7 deselected。包含独立进程检查生产导入未加载 vLLM Python 模块或预编译 .so，以及 CPU 失败子进程退出。尚缺多 shape 拒绝矩阵、GPU 故障清理、完整历史回归和 120 的本地量化 kernel 构建；113 继续保持进行中。
