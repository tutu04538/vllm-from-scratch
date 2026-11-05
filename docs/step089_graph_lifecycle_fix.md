# 089 后续｜V1 Graph 热身的推理模式与池生命周期

本次修复的直接原因是 **V1 `_dummy_run` 缺少推理模式边界**，并非证明 PyTorch 或 vLLM 有 bug。旧版热身/捕获会在开启梯度的调用环境下执行可求导前向，将 `CopySlices` 反向图接到 KV 缓存；进程保留相关对象时，销毁图后其共享池中仍可能有存活分配。之后复用同一池捕获，PyTorch allocator 在 `use_count > 0` 检查处报错。

## 固定源码对照

- 项目 `minivllm/worker/gpu_model_runner.py::GPUModelRunner._dummy_run`：热身和捕获共用的假前向入口。修复前无 inference mode；修复后按上游在整个方法上使用 `@torch.inference_mode()`，使模型计算、KV 写入和输出都处于推理语义。真实执行本来就有同样边界。
- 冻结 vLLM `v1/worker/gpu_model_runner.py::GPUModelRunner._dummy_run`：装饰器相同。2026-10-03 归档及当前安装文件 SHA256 均为 `7f2890eefca1efe25565bf1c7e5906a87948ae922610a7aaac620b28b46f26aa`；读取归档核验，而非依赖 latest。V2 的 `CudaGraphManager.capture` 和 `SpeculatorCudaGraphManager.capture` 本来已有 `@torch.inference_mode()`。
- 冻结上游平台 `get_global_graph_pool` 仍然维护进程级池句柄，项目保留此布局；没有为每张图或每个引擎新建池，没有关闭 Graph 或改变调度、输入、采样及模型执行路径。
- 项目 `CUDAGraphWrapper` 与冻结上游仍存在既有差异：项目对图输出使用强引用，上游采用不拥有存储的 Tensor 别名。此次输出探针显示旧版 entry 输出 `requires_grad=False`、`grad_fn=None`，不能把该差异当成本次已证实的根因；本次未改输出协议。

## 因果证据及范围

1. 旧版两层 KV 在引擎 A 运行后、shutdown 前均为 `requires_grad=True`、`grad_fn=CopySlices`；entry 输出自身 `grad_fn=None`。因此之前将输出强引用列为直接原因的猜测不成立。
2. 原测试在关闭 A 前复制 KV 摘要并保留到 B 执行后；旧版摘要实际为 `requires_grad=True`、`grad_fn=CatBackward0`，修复后为 `False` / `None`。因此摘要会沿旧版 KV 的求导链延长图池中间张量的存活。旧版引擎 A 关闭后，对共享池调用 `torch.cuda.memory_snapshot(pool)`，记录到 22 个 `active_allocated` block；再构造引擎 B 触发 `CUDACachingAllocator.cpp:3332`。在独立诊断进程仅将 `_dummy_run` 包裹 `inference_mode` 后，同一测试关闭时活动 block 为 0，B 创建通过。这个对照定位的是该测试工作负载，不将 22 推广为其他模型的固定数量。
3. 脱离模型的单卡 CUDA 微型对照：图销毁后同时保留池内输出 Tensor，`empty_cache` 后以原池捕获可复现同一断言；同时删除图和输出再捕获通过。说明池内分配的残留是能够触发该错误的机制；并不声称模型实验里每个活动 block 都由同一条 Python 引用直接持有。
4. 新增 GPU 回归在 `torch.enable_grad()` 下连续三次创建/运行/关闭 V1 引擎，逐一检查 KV 无 `requires_grad` 和 `grad_fn`、图发生 replay 且生成结果一致。诊断进程移除装饰器时该回归因 KV 的 `CopySlices` **确定性失败**，不是仅因加了新代码而通过。

对照原始日志保存在 `benchmarks/results/graph_lifecycle/`，该目录不随 Git 提交。所有 Graph 测试均使用 tiny 权重；不把结果冒充真实模型或多 rank 图验收。对外保留全局池时，其他未来路径仍需各自验证生命周期和捕获顺序。

## 验证

- 首次完整回归：1295 passed / 1 failed / 23 skipped；唯一失败为旧测试将历史 step81 hash 当作当前包 hash。保留该历史 hash，新增 `docs/results.json.current_package`；测试改为对当前 hash 做同样严格的计算和比较。
- 修改后完整回归：`CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest $(printf 'tests/step%s ' $(seq 58 88)) tests/step089 -q -ra`：**1296 passed / 0 failed / 23 skipped**，134.32 s。该回归收集时已包含无反向图/三引擎测试；其后又强化了摘要跨关闭保留断言，见独立定点复测。
- `CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/step83/test_gpu_runner_lifecycle.py tests/step69/test_spec_cudagraph.py tests/step71 -q -ra`：**44 passed**，5.76 s。
- 同一 Python 进程内连续三次执行 step69 Graph 文件：每次 **19 passed**；之前第二次为 17 passed / 2 failed。
- 独立重跑 089 多卡测试（不继承单卡设备限制）：**7 passed**，23.56 s，见 `benchmarks/results/graph_lifecycle/step089_separate.log`。
- 原测探针：禁用新边界时 KV 摘要 `CatBackward0` 且引擎 B 捕获失败；启用后摘要无反向图且测试通过（`digest_without_boundary.log`、`digest_fixed.log`）。
- 包摘要：`0dfdbb66ba9a6d91ef986a23c82108ab94882f616fcadcd2aabff85bc588fd2c`，177 个 Python 文件。
