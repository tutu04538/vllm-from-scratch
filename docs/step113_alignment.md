# step113：冻结源码与量化能力基线（进行中）

状态：**未完成**。本关要求真实 4090 量化算子和 dispatch 探针；2026-11-19 其他用户训练进程占用四张卡，当前只实施不占 GPU 的部分。历史 step108 多卡回归有一次中断及一个待定位失败，见该关报告。

## 实施映射

| 子需求 | 冻结源码/规范 | 本项目位置 | 状态 |
|---|---|---|---|
| R113.01 | 工作树与包身份 | `benchmarks/benchmark_step113_quant_capability.py` 基线检查 | 已记录基线与当次 dirty 文件 hash；历史 108 多卡回归仍待结束 |
| R113.02 | 849 文件清单及安装包 | 同上 | 849 文件、346 安装版 Python、342 历史源码对账通过；GPU 硬件能力另验 |
| R113.03 | AWQ/GPTQ/FP8 参考分派 | `tests/step113/step113_quant_reference_probe.py` | 探针已编写并通过 CPU 导入检查；真实分派未运行 |
| R113.04 | W4、动态 FP8、KV 算子 | `tests/step113/step113_quant_reference_probe.py` | 探针已编写；等待 GPU 空闲后实跑与核对 |
| R113.05 | Marlin/FP8 backend | 隔离参考子进程 | 待 GPU 空闲 |
| R113.06 | CMake、算子绑定、CUTLASS | 基线与依赖记录 | 已解析远端 tag；头文件、构建和许可核验待做 |
| R113.07 | 独立上游参考进程 | 本关测试与探针 | 已采用隔离子进程，忙卡保护已测；真实 CUDA 隔离与失败清理待验 |
| R113.08 | 生命周期、能力表 | 本关测试与结果 | 待实施 |

`--check` 为环境和源文件完整性检查，不能替代 GPU 能力和数值验收；结果中必须区分未运行、失败与通过。性能测试按用户要求集中到功能基本完成之后。

## 2026-11-19 CPU 基线

旧变更的工作树已提交到 `2c3125c10695d49ff21d50c1f0bc42d1d3aa652e`，包内相对路径 hash 为 `2720f8bfb1f3f2c9e7311d9d4810a3f5cdeb21b883fcb448f6f683b3046517fc`（196 个 Python 文件）。探针另保存运行当时新建的未跟踪文件清单和逐文件 hash，不能把当时 dirty 工作树当成纯 HEAD。

`source scripts/activate-4090.sh` 后运行 `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmarks/benchmark_step113_quant_capability.py --check --output benchmarks/results/step113/cpu_identity_20261119.json`：849 份源码快照 hash、346 份安装版 Python、342 份并行专题快照及对应安装版一致。首次运行错误地要求两套冻结清单彼此包含，产生 342 个假失败；修正为各自与安装版逐项对账后通过。Python 3.12.13、Torch 2.13.0+cu129、Triton 3.7.1、vLLM 0.28.0+cu129、Transformers 5.14.1；nvcc 12.9.86。CUTLASS 远端 v4.4.2 tag 指向 `da5e086dab31d63815acafdac9a9c5893b1c69e2`，本次没有下载其头文件或构建算子。能力文件的 GPU 算子状态为 not_run，不能以 CPU 检查通过替代。

`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest tests/step113 tests/step115 tests/step090/test_step090_models.py -q -ra`：34 passed、0 failed、0 skipped；其中 113 的三项仅覆盖清单破坏检测、hash 口径及 GPU 未实现状态。量化 GPU、真实权重、TP/Graph 与完整回归均未运行。


### 隔离 GPU 探针准备

新增 `tests/step113/step113_quant_reference_probe.py`，覆盖 AWQ/GPTQ W4A16 Marlin（M=1、128）、参考 kernel 分派、CUTLASS FP8 W8A8、FP8 cache 转换；只在独立上游子进程引用 vLLM。探针尚未在 GPU 上运行，其实现及数值阈值均待实测验证，不计作对应 R/T 已通过。`--compare` 在启动前只读 `nvidia-smi` 检查选定卡上的计算进程；检测繁忙或查询失败返回 `not_run`，不启动子进程。注意启动前检测无法杜绝随后其他任务抢占设备。

当前 GPU0 有其他用户进程，`python benchmarks/benchmark_step113_quant_capability.py --compare --device 0 --output benchmarks/results/step113/gpu_busy_20261119.json` 返回码 2，记录 `not_run`；无 GPU 算子执行。`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest tests/step113 tests/step115 tests/step090/test_step090_models.py -q -ra`：35 passed、0 failed、0 skipped；`runpy.run_path(..., run_name='step113_import_only')` 的上游探针 CPU 导入通过，不代表算子可运行。性能 trace 依照用户要求延后。
