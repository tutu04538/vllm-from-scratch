# 104 Graph 可捕获通信与后端选择

## 实施前源码映射

| 项目位置 | 冻结 vLLM 0.28.0 | 职责、数据与时序 |
|---|---|---|
| `minivllm/distributed/parallel_state.py::GroupCoordinator` | `distributed/parallel_state.py::GroupCoordinator.graph_capture` | 设备组负责成员与组内 rank；捕获上下文选择专用 stream，并等待原 stream 上已提交初始化操作 |
| `minivllm/distributed/device_communicators/pynccl.py` | `distributed/device_communicators/pynccl.py::PyNcclCommunicator.all_reduce/all_gather` | 使用 CPU 组广播 128 字节 NCCL unique id，创建每 rank communicator；接收同设备连续 GPU tensor，以当前 CUDA stream 指针调用 NCCL；all-reduce 返回与输入同 shape 的输出，all-gather 输出按 rank 首维排列 |
| `minivllm/distributed/device_communicators/pynccl_wrapper.py` | `distributed/device_communicators/pynccl_wrapper.py::NCCLLibrary` | 按 NCCL C ABI 绑定数据类型、buffer 指针、stream、unique id、初始化和通信函数；只封装实际调用的接口 |
| `minivllm/distributed/device_communicators/cuda_communicator.py` | `distributed/device_communicators/cuda_communicator.py::CudaCommunicator.all_reduce` | 根据可用后端分派；当前专题只接 PyNccl 必要路径，不引入无调用方优化后端；不得强行启用依赖 P2P 的 custom all-reduce |

状态所有权：通信组拥有 communicator；调用方的 graph 管理器拥有捕获图和图输出；关闭时先释放图再销毁 communicator。所有 rank 使用一致的 collective 顺序；GPU 内存地址可跨 rank 不同，但每个 rank 的 replay 缓冲必须在图生命周期内有效。

## 后端与生命周期

`GroupCoordinator` 在创建 NCCL 设备组和 Gloo CPU 组后创建 `CudaCommunicator`；多 rank 设备通信的 all-reduce/all-gather 进入本项目的 `PyNcclCommunicator`。初始化时 rank 0 创建 NCCL unique id，由 CPU 组广播，随后每 rank 在本地 CUDA 设备上初始化并热身。运行时向 NCCL 传入真实 GPU tensor 地址与当前 CUDA stream；生产不导入安装的 vLLM。单 rank 保持原有直通行为；reduce-scatter 和点对点当前仍沿原有 PyTorch 路径，不声称它们已完成图内验收。

本实现只移植本关有调用方的 PyNccl 后端，没有实现 custom all-reduce；当前服务器实际 TP4 的任意两卡 peer access 均为 false，未强行启用依赖硬件条件的 custom 后端。该实现范围限制与这台机器的硬件观测分别记录，不将其他硬件上的专用后端标为支持。初始化失败明确抛错；NCCL abort 使用有界等待以处理 rank 意外退出。Graph 和其输出的所有者必须在销毁 communicator 前释放它们。

`GroupCoordinator.graph_capture()` 选择或复用捕获 stream，并令其等待当前 stream 已提交的初始化工作。测试在每 rank 分别持有图、输入和输出；两张图各重放 100 次，输入内容轮换，逐次校验 all-reduce 和 all-gather 的值及各自地址。两个进程组生命周期按顺序关闭、重新创建；独立 Executor 探针在 rank 1 的实际 CUDA Graph 捕获上下文中注入异常，检查错误上报、后续 RPC 被拒及 Worker 回收。

## 实测结果

- `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/step104 -q -ra`：6 passed。包括 TP2/4、捕获中 rank 故障、Executor 故障传播、两个顺序创建/关闭的 `LLMEngine`（测试 Worker）和冻结文件 hash 校验。
- `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmarks/check_step104_acceptance.py --case all --timeout 120 --output benchmarks/results/step104/acceptance_final.json`：5 个 case 全通过，其中直接注入失败的子进程按预期非零退出。每个 rank 两个进程组生命周期、每个生命周期两张顺序创建的图、每张图 100 次 replay；每次对 all-reduce 和 all-gather 做精确值与地址检查。另与实际冻结 vLLM PyNccl 执行 eager all-reduce 差分。
- 同一个 Python 进程中连续三次运行 `tests/step69/test_spec_cudagraph.py`：每次 19 passed、退出码 0；这是 089 生命周期复验，不冒充模型多 rank Graph 验收。
- 58—88 全量：1289 passed、23 skipped；089—103 全量：191 passed。旧 skip 仍是未验路径，不记为通过。
- 通信后端变化影响真实模型，故另外用 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmarks/check_step104_model_regression.py --timeout 1800 --output benchmarks/results/step104/model_regression.json`，按 103 的预登记合同重新运行 8B TP1/2/4 和 14B TP2/4、各 B1/4：10/10 passed、780 项张量对照通过，全部请求 32 token 与冻结参考一致。该验证为 eager；独立目录保存本次日志与结果，不覆盖 103 历史产物。
- `python learning_notes/14_vllm_from_scratch/分布式并行完整需求/参考资料/check_requirements.py`：8003 项静态检查通过。

环境为 Python 3.12.13、Torch 2.13.0+cu129、Triton 3.7.1、vLLM 0.28.0、NCCL 2.29.7、4×RTX 4090；包内 192 个 Python 文件 SHA256 为 `34543201a3374d524412bec60ded8849418012954c7df21c69379c619eb93886`。详细命令、退出码、rank 设备及结果在 `docs/results.json → step104.results/models`；原始日志和结果位于本地 `benchmarks/results/step104/`，不加入 Git。

本关未验多 rank 整模型 Graph、reduce-scatter/broadcast/send/recv 图捕获、custom all-reduce 或性能；105 负责普通 TP 整模型 Graph。两引擎生命周期的模型执行部分使用测试 Worker，并非真实模型推理。
