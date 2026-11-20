# Step 120：量化 kernel 选择与本地算子边界（实现中）

本关为 W4A16 的能力判定和本地 CUDA 扩展建立接口；121/122 验收前不开放量化执行。

编码前 R 映射：R120.01/02 对应 `minivllm/layers/quantization/kernel_dispatch.py`；R120.03/05/06 对应 `minivllm/quant_ops/` 的构建身份、加载器和独立进程工具；R120.04 对应 `minivllm/csrc/quantization/marlin/` 的许可证、同源源码和生成器；R120.07 对应本地 operator schema 与 Python fake/meta；R120.08 对应 `tests/step120/` 隔离探针；点名入口 `benchmarks/benchmark_step120_quant_kernel_dispatch.py`。本关不将已编译、未验收的核对外标为可执行。

## 已实现的边界

`MPLinearLayerConfig` 登记全局/本地 `[K,N]`、权重类型、激活和输出 dtype、group、zero point 与 g_idx。`can_plan_marlin` 和 `plan_mp_kernel` 明确只覆盖 SM89、W4A16、G128、无 g_idx 的项目子集；非法设备、dtype、group、权重/zero point 组合、跨 group 分片和打包不对齐均拒绝。SM89 上冻结上游 CUDA 候选顺序依次为 CutlassW4A8、Machete、AllSpark、Marlin、Conch、Exllama、TritonW4A16、Humming；前两者最低 SM90，AllSpark 在 SM89 不接受 G128，因此限定条件下计划 Marlin。非 SM89 拒绝，未声称实现上游全部 registry。该计划的 `execution_available` 始终为 false，`require_mp_kernel` 明确拒绝；AWQ Linear 只保存对应的 `kernel_config`，正常 forward 仍不开放。

本地 `minivllm/csrc/quantization/marlin/` 移植冻结 `2cf0a6915ce544dc493a0990f2ea38d81601128a` 的 Marlin 源码、生成器和必要稳定 ABI 头，保留许可证与源码声明。`gptq_marlin_repack.cu`、`marlin.cu` 仅将 operator 注册命名空间从 `_C` 改为 `minivllm_quant`；本项目另提供独立 schema binding。源码内部 `vllm::ScalarType` 是随源码移植的 C++ 类型声明，不代表链接安装版 vLLM。测试对照冻结源码确认除两处命名空间外字节一致。

`minivllm/quant_ops/build.py` 仅在显式调用时构建，使用 CUDA 12.9、SM89、Torch stable ABI `TORCH_TARGET_VERSION=0x020B000000000000ULL` 和独立缓存。构建 key 登记所有源码 hash、生成器 profile、builder hash、Torch/CUDA 版本、C++ 编译器版本、nvcc 版本与编译参数。生成器在独立缓存目录限定为 FP16/BF16、uint4/uint4b8、G128及单 group 的实例，不更改冻结源或用户其他项目；生成物与 selector 记录 hash。编译错误直接抛出，不导入 vLLM 或切换 CPU/F.linear。首次编译因缺 `USE_CUDA` 导致稳定 ABI shim 声明不可见失败；显式补齐与冻结构建一致的宏后成功，失败不是通过证据。

`minivllm/quant_ops/ops.py` 为本地 repack 和 GEMM schema 注册 fake/meta，提供受限 shape、dtype、device 检查；只完成 meta 合约基础，不宣称与 GPU 实算已经比对。扩展中确实注册了两项 CUDA implementation，但模型并未接入运行，121/122 前不可声称计算可用。完整编译使用独立的 `minivllm_quant` namespace；117/119 的表示审计及正常量化 forward 拒绝保持。

## 已运行的检查

- `source scripts/activate-4090.sh` 后执行 `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=2 python -m pytest tests/step120 -q -ra`：16 passed、0 failed、0 skipped；仅 CPU、源码和 meta 检查。包含 5 个冻结上游 Marlin CPU 平台模拟正反例、源文件对账、无工具链明确报错、独立进程禁用 vLLM import、动态库依赖和 fake shape/dtype。首次隔离测试因子进程未加入项目路径失败，补齐路径后重跑通过。
- 同环境运行 `tests/step115 tests/step116 tests/step117 tests/step118 tests/step119`：63 passed、0 failed、0 skipped；不是 GPU 回归。
- `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=2 python benchmarks/benchmark_step120_quant_kernel_dispatch.py --check --output benchmarks/results/step120/cpu_build_meta.json`：完整扩展在屏蔽 vLLM import 的独立进程中加载，两项 CUDA dispatch 已注册；repack meta 输出 `[8,128]` int32，GEMM meta 输出 `[4,64]` BF16，错误 packed shape 被拒绝。进程中无 vLLM Python module 或 vLLM 动态库映射；`readelf` 的 NEEDED 依赖仅为 Torch、CUDA runtime 与系统库。执行的是 meta，不是 GPU kernel。结果 SHA256 `fa26f977e091d47a0a2dfb40c914afed8c2af93eb6c0c1357a7966024a237653`。
- 完整构建身份 SHA256 `be69fade8e7dcd95df49f6705b6e67590b8b0fbc3dd5a5aa6dbd5605591b462e`；动态库路径 `/data/syh/.cache/torch_extensions/minivllm_quant_full_be69fade8e7dcd95/minivllm_quant_full_be69fade8e7dcd95.so`，SHA256 `9c541c6441d063f9f8c905434ada76b281f3cb802c386dd4fd9a57a740270e0d`。selector SHA256 `6915cb70c8e0638d7f77c7a21246409d77d385770d04fce4c805c5f84407f897`；各生成文件 hash、完整工具链身份见 `benchmarks/results/step120/full_build.log`。不提交 .so 或结果文件。
- 量化需求静态脚本：27 个文档、24 关和 849 份冻结文件检查通过；这是文档检查，不是 GPU 测试。

当前生产包 SHA256 `40de241ea0d7f4d438970d63c404db4e4e32de3171edc97e1ac59d97c73b2b2b`。四卡均被其他任务使用，按用户安排未启动 GPU 实算；独立扩展真实小算子、与上游数值对照、fake 与真实输出比对、Graph、完整回归和性能均未跑。113/119 的 GPU 前置仍未验收，因此 120 不能标记为完全完成；121/122 的 kernel 数学与重排也尚未验收。

## 2026-11-20：项目独立 CUDA 算子实测

121 在 GPU0 进行六组随机/规律 repack，与冻结 vLLM int32 完全一致；122 同一 packed 输入六种 M 的 GPU GEMM 输出与冻结上游逐值一致，并对 FP32 oracle 通过预设容差。当前本地扩展有实际 GPU 执行证据，不等于全格式或整模型能力。详细命令与局限见 121/122 的新增实测章节。
