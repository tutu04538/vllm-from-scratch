# Step 119：AWQ 真实权重加载与格式转换（实现中）

本关建立 W4/G128/zero-point 的 AWQ checkpoint → 标准 packed 表示加载链，不开放量化 forward 或 Marlin kernel。

编码前映射：R119.01/02 对应 `minivllm/layers/quantization/awq.py`、`config.py` 和 `minivllm/model_loader/post_load.py`；R119.03 对应 Qwen3 stacked 映射、`linear.py` 和 Parameter shard 加载；R119.04/06 对应 `BaseModelLoader` 的明确表示审计入口及覆盖检查；R119.05 对应 `benchmarks/benchmark_step119_awq_load.py` 的真实模型抽层；R119.07 对应量化方法 `apply` 明确拒绝；R119.08 对应参数创建前的 group/packed 边界检查。独立测试在 `tests/step119/`。用户选择的公开 checkpoint 精确源 revision 和校准集仍未知，不能冒称是 114 原始 BF16 同源质量基线。

## 实现与当前边界

`AutoAWQConfig.get_quant_method` 对受支持且未排除的 Linear 分派项目内 `AutoAWQMarlinLinearMethod`。方法在本 rank 创建 AWQ checkpoint 布局的 int32 `qweight/qzeros` 和激活 dtype 的 `scales`，按 QKV、gate/up 独立记录每个参数需要的逻辑 shard；K 必须与 group128 对齐，每个本地输出 shard 必须与 pack8 对齐，失败直接报错。合法 padding 的更广泛上游场景没有静默支持，当前不满足本地约束即拒绝。

加载完成后，生产实现按冻结 `auto_awq.py` 的 tensor 位运算恢复 AWQ 位序，将 qweight 从 `[K,N/8]` 转换为 `[K/8,N]`，qzeros 转为 `[N/8,G]`，保留 scale dtype 和数值。每层最终状态停在 `standard`，没有伪造 `kernel` 状态，也不调用后续 Marlin repack。新建的参数保留加载身份、逻辑 shard 记录和 TP 状态；不保存 BF16 全层反量化权重。

`BaseModelLoader.load_model_for_representation` 是显式表示审计入口，使用普通 checkpoint 读取、名称映射与严格覆盖检查，Qwen3 embedding/lm_head/norm 沿用原有 BF16 参数加载。普通 `load_model` 仍在业务调度前明确拒绝 AWQ；表示审计入口返回的 AWQ Linear 在 `apply` 中明确拒绝 forward。GPTQ 和 FP8 继续保持未实现状态。`post_load` 允许该方法完成标准布局后以 `representation_ready` 标识模型，而不是称为推理 ready。

适配旧的 step115 两项断言：AWQ 由 `PendingQuantizedLinearMethod` 变为实际创建和转换方法；正常推理入口统一报未开放量化执行，不再靠 meta 构造报第一个待实现层名。这是能力边界变化，不表示 AWQ 推理已开放。

## CPU 基础检查与真实权重抽样

`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest tests/step119 -q -ra`：8 passed、0 failed、0 skipped。含生产转换与冻结 vLLM 独立 CPU 差分、QKV/gate-up 每个量化参数的 shard 路由、缺 scale/重复拒绝、group/pack 不对齐拒绝、CPU 模拟 TP2 的行分片，以及 forward 和重复后处理拒绝。模拟 TP 不启动分布式进程，不能称真实 TP 验证。

同环境执行 `tests/step115 tests/step116 tests/step117 tests/step118 tests/step119 tests/step090/test_step090_models.py tests/step096/test_step096_linear.py::test_single_gpu_free_compatibility_and_bias tests/step097/test_step097_packed.py tests/step097/test_step097_qwen_loader.py -k 'not test_true_gpu_and_frozen_or_real' -q -ra`：86 passed、7 deselected；被排除的 GPU 用例不是通过。

`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python benchmarks/benchmark_step119_awq_load.py --check --output benchmarks/results/step119/public_cpu_slices.json`：公开 8B/14B 首、中、末层各七个逻辑投影，共 42 个小块，从 checkpoint 加载 K=256、N=16（两组 scale），经项目 QKV/Merged/Row 参数加载及标准转换后，与隔离进程冻结 vLLM 逐字节对照，qweight、qzeros、scale 原 dtype 和数值均一致。每次读取前核对 config/index 和涉及的源分片 LFS hash。结果 SHA256 `2812ccb3f68f0fd134c57c181021ac2dfe65c65d35e266c3410a251e19974e46`。不是每层完整矩阵差分，不是 GPU kernel 对照。

`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python tests/step119/step119_full_cpu_audit.py --model <固定 AWQ snapshot> --output <结果路径>` 在 CPU 完整载入并转换后检查所有参数的 shard 覆盖和布局状态，再主动验证 AWQ forward 被拒绝。8B：144 个物理 Linear / 432 个量化参数完成标准转换，参数字节 6098479104，进程 peak RSS 12524172 KiB；embedding/lm_head 为 BF16。`benchmarks/results/step119/awq_8b_full_cpu.json` SHA256 `c7d596c1d28f84d3be919eb7c38ecdafde1753e9de6c33ba03199596689bf9c7`。这不是 8B 完整数值、上游输出或 GPU 测试。

抽样和完整 CPU 审计运行时生产包 hash 为 `4eb61bda369dc6b845f0926632a7cce1c59363b367439a4e0e2392ee43f59a0d`；之后仅补充表示审计入口对“没有量化层”的明确拒绝，当前包 hash 为 `5046414a52d2a9b26da0d69bfc464aee084ebf260ac3795d2e4550a696e8120e`，上述 86 项已在当前包重跑。不可把旧审计 hash 误写成当前包结果。

按用户决定，不运行 GPU 正确性、真实 TP/Graph、GPU 峰值、模型质量和性能；`--trace` 返回 not_run。公开 checkpoint 源 BF16 精确 revision 及校准数据依然未知；未据此声称 114 同源基线。119 的真实 GPU 和完整历史回归仍未完成，不能标记为完全验收。

14B 在相同命令、固定 `Qwen/Qwen3-14B-AWQ@31c69efc29464b6bb0aee1398b5a7b50a99340c3` snapshot 下完成完整 CPU 表示审计：160 个物理 Linear / 480 个量化参数完成标准转换，参数字节 9976576000，peak RSS 20097896 KiB；embedding/lm_head 为 BF16，forward 明确拒绝。结果 `benchmarks/results/step119/awq_14b_full_cpu.json` SHA256 `f012138f8d3abf50a0c48d847d71c03c90fbfec0b77282b3e85a7d6bd9568b58`，同样属于前述 `4eb61b...` 包版本的 CPU 表示审计。额外对两个 checkpoint 的 embedding、lm_head、norm 名称读取 safetensors dtype 元数据：8B 147 个、14B 163 个均为 BF16；并非 GPU 输出或质量测试。

## 2026-11-20：真实模型 CUDA 权重表示

使用 `tests/step119/step119_full_cuda_audit.py --model <固定 AWQ snapshot> --output <结果>`，逐模型单卡加载并检查全部量化参数为 CUDA kernel 布局、shard 覆盖完整、空设备 g_idx 和层间独立 workspace。8B：144 层/432 参数，峰值 allocated 6,984,556,544 B；14B：160 层/480 参数，峰值 allocated 11,571,875,840 B。结果为 `benchmarks/results/step119/awq_{8b,14b}_cuda_representation_20261120.json`。本证据不执行整模型前向，也不代替 TP 或质量测试。

上述整模型 CUDA 审计运行时包 hash 为 `b02bb758ab14ebc6fc32bc2fc5fea92de25d702317c8f9f979d8978a90f16234`，在随后开放局部前向前执行；不把它记成新包 hash 的实测。
