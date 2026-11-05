# 089｜服务器基线与通信成本

2026-11-05；基线 HEAD `03791f511d67799c8b9b8d13978f41fe29172c90`。本关建立环境证据和测量工具，不实现 TP/PP/DP 或模型推理。真实权重未下载；Graph 已知问题仍未解决，104/105 验收前必须关闭相关阻塞。

## 职责与源码对照

| 项目文件/行为 | 冻结 vLLM 0.28.0 文件/方法 | 边界与证据 |
|---|---|---|
| `scripts/activate-4090.sh`、`benchmarks/check_parallel_environment.py`：设备、C++ UVA、子进程探针 | `distributed/device_communicators/cuda_communicator.py::CudaCommunicator.__init__` | 上游为多 rank 创建 PyNccl，并按 TP 等条件初始化其他通信器；这里是测试环境，不创建生产通信器。 |
| `benchmarks/step089_communication.py`：PyTorch NCCL all-reduce 等 | `cuda_communicator.py::CudaCommunicator.all_reduce`、`pynccl.py::PyNcclCommunicator.all_reduce` | 上游有多个候选后端和选择次序；本基准只测 PyTorch NCCL，不宣称逐调用或性能等价。输入为每 rank FP32 一维连续 buffer；all-gather 输出为 world 倍、reduce-scatter 输出为 1/world。 |
| 拓扑与 peer 矩阵、NCCL 日志 | `custom_all_reduce.py::CustomAllreduce.__init__` | 上游对于超过两张不全互联的卡及 P2P 检查失败有禁用条件；本次 NCCL 日志观察到 `SHM/direct/direct`。这不是实测上游 custom-all-reduce 的初始化结果。 |
| `tests/step089/` | 三份冻结源码及 manifest | 实际源码快照、安装包 hash 一致且指定方法存在；collective 数值直接按数学期望检查，不把自己的实现当上游 oracle。 |

测试不持有 Request、KV 或 Scheduler 对象，不改变其 owner。父进程通过 torchrun 启动每 GPU 一 worker；每 rank 各自检查输入/输出，父进程收集完整状态。输入 1/2 的 all-reduce 各得 3，输入 1/2/3/4 各得 10。通信是环境探针，不是模型并行实现。

## 本次实测

环境：Python 3.12.13，Torch 2.13.0+cu129，Triton 3.7.1，vLLM 0.28.0+cu129，NCCL 2.29.7，四张 RTX 4090。GPU0/1 同 NUMA 0，GPU2/3 同 NUMA 1；非对角 peer access 均为 false。四个单 GPU 独立进程各通过 BF16、Triton、Graph replay、C++ UVA 检查。每个组均启动独立进程、所有 rank 检查五种操作；20 warmup、100 samples、3 repeats，四组 × 五操作 × 四大小，共 80 个配置，每 rank 每配置 3 组记录。`0,99` 注入 local rank 1 的设备初始化失败，父进程非零；缺 rank 初始化和父进程超时均有限退出。完成后 `nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader` 无计算进程。

例如，按各 rank 和 3 组 CUDA event 均值再平均，all-reduce 在 4 KiB / 16 MiB 分别为：GPU 0/1 **0.0391 / 1.1648 ms**，GPU 2/3 **0.0375 / 1.1599 ms**，GPU 0/2 **0.0384 / 1.1764 ms**，四卡 **0.0422 / 1.4890 ms**。这些仅是本次微基准观察，不推导模型加速倍数。

大小表示每 rank 输入字节；all-gather 输出是输入的 world 倍，reduce-scatter 输出是输入的 1/world。`effective_bytes` 对 all-gather 取每 rank 输出大小，其他取每 rank 输入大小；其除以 event 时间为应用层有效吞吐，不是链路流量或 bus bandwidth。墙钟从设备同步后开始，包含 Python 提交与结束事件等待；event 记录当前流的 GPU 时间。每条操作先单独检查正确性，再用零值输入测稳态；不把编译、初始化、结果检查纳入计时。原始每 rank 三组平均值与墙钟保存在本地 `benchmarks/results/step089/communication/`，包含 NCCL 日志、进程 PID、设备 UUID 和失败时最后进度。

## Graph 问题：复现，不等于修复

本次完整旧回归 **1288 passed / 0 failed / 23 skipped**，23 个缺权重或 tokenizer 的真实模型测试仍未验。历史两次全量曾为 1275/13/23；本次一次全绿不能推翻历史失败或证明根因已解决。

在 GPU0、单独 Python 进程中对同一 pytest 文件调用两次 `pytest.main`：第一次 19 passed，第二次 17 passed / 2 failed；失败为 `CUDACachingAllocator.cpp:3332` 的 `use_count > 0` 断言。进一步只运行 `test_polluted_padding_buffer_does_not_change_valid_output`，一次调用就可失败：clean 引擎执行并调用 `shutdown()` 后，dirty 引擎的图捕获使用 `_get_global_graph_pool()` 返回的进程级共享池 `(0, 1)` 时失败。作为仅作用于诊断进程的对照，将 `_get_global_graph_pool` 临时替换为每次调用 `torch.cuda.graph_pool_handle`，同一测试 **1 passed**。这说明共享池生命周期值得进一步排查；两种运行存在捕获池等差异，不能证明唯一根因，也不能把对照写成生产修复。未改关闭顺序或断言，尚需受控检查池持有者和关闭顺序。

复现和图池对照可分别在**新的进程**中运行（均需先激活环境）：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python - <<'PYCODE'
import pytest
raise SystemExit(pytest.main(['tests/step69/test_spec_cudagraph.py::test_polluted_padding_buffer_does_not_change_valid_output', '-q', '-ra']))
PYCODE
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python - <<'PYCODE'
import pytest, torch
import minivllm.compilation.cuda_graph as cg
cg._get_global_graph_pool = torch.cuda.graph_pool_handle  # 仅本诊断进程
raise SystemExit(pytest.main(['tests/step69/test_spec_cudagraph.py::test_polluted_padding_buffer_does_not_change_valid_output', '-q', '-ra']))
PYCODE
```

## 命令及产物

```bash
source scripts/activate-4090.sh
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest $(printf 'tests/step%s ' $(seq 58 88)) -q -ra
# GPU=0、1、2、3 分别在独立进程执行一次
CUDA_VISIBLE_DEVICES=$gpu python benchmarks/check_parallel_environment.py --single --output benchmarks/results/step089/single_gpu${gpu}.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/step089 -q -ra
python benchmarks/check_parallel_environment.py --communication --output benchmarks/results/step089/communication
python learning_notes/14_vllm_from_scratch/分布式并行完整需求/参考资料/check_requirements.py
```

本关 pytest **7 passed**；完整矩阵四组均 exit 0；需求与源码静态校验 8003 项通过。Graph 原始对照和失败保存在 `benchmarks/results/step089/graph_{repeat,two_engine_repeat,fresh_pool_probe}.log`。这些本地结果及私有需求目录被 gitignore，不随提交同步。具体命令、hash、未验项与产物见 `docs/results.json → step089.results/models`。
