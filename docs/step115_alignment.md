# step115：量化配置解析（进行中）

状态：**部分实现，不能视为 115 验收通过**。113、114 仍未完成；未运行量化 GPU 算子和真实量化模型，也未做性能测试。

## 基线

实现前 HEAD 为 `c7c4886af40c2b3e00213d5c2bfa826302bc5b4d`，按包内相对路径计算的生产包 Python hash 为 `5a738fc095c544e59baf458a04cb0cf2beb25e89e4fe6e83e01c8911573b4565`；此前记录的 `48ba37036a952982e667679d3a5c5a6b309c6e7d2acc8db998201f6130834b90` 使用仓库根目录相对路径，与项目验收工具口径不同。本地忽略的 `learning_notes/14_vllm_from_scratch/量化完整需求/参考资料/工作区基线_20261119_step115/` 保存了状态、改动补丁、未跟踪文件和逐文件 hash；不是干净 HEAD，也不纳入 Git。

当时工作树的包内相对路径 Python hash：`2720f8bfb1f3f2c9e7311d9d4810a3f5cdeb21b883fcb448f6f683b3046517fc`；此前记录的 `162c26384ccc799961a5137a94600032816c5b50c3db2f5f103080f37f366a8f` 使用仓库根目录相对路径。两者均包含当时既有 Qwen3 未提交修改。项目标准 hash 按包内相对路径排序、逐文件 SHA256 后以 `路径\0hash\n` 拼接计算。

## 实现及缺口

| 项 | 对照及本项目位置 | 当前状态 |
|---|---|---|
| R115.01 | 上游 base_config；`minivllm/layers/quantization/config.py`、`methods.py` | 配置与方法接口已建；量化参数和 apply 为明确拒绝的占位，完整字段及覆盖顺序待验 |
| R115.02 | 上游 auto_awq/auto_gptq/fp8；同上 | 解析计划子集、常用字段及三组入口别名；实测上游注册名一致，尚未完成逐字段差分 |
| R115.03 | 上游 `get_quant_config`、同上 | `ModelConfig.quantization` 与外置文件已接入加载器；checkpoint 优先、用户方法冲突拒绝；尚未验证所有上游覆盖顺序 |
| R115.04 | 上游 `is_layer_skipped`、`get_quant_method`；`minivllm/layers/quantization/` | Qwen3 linear 和 fused skip CPU 对照已实现；尚未验证真实产物、MoE/FP8 KV 方法 |
| R115.05 | 同上 | 实现部分配置子集拒绝；仍缺层类型、完整格式及 MoE 检查 |
| R115.06 | 三种配置的 dtype 与 minimum capability | 加载入口校验 dtype/SM；`kernel.can_implement` 和真实 GPU 能力待实现/验收 |
| R115.07 | Qwen3、Linear | 参数传入 Qwen3 Linear，meta 预检给出不支持层名；多 rank 量化仍在现有 Worker 边界拒绝 |
| R115.08 | `minivllm/model_loader/base_loader.py` | checkpoint 配置解析后，模型构造前明确拒绝量化执行；不是量化功能开放 |

不能把当前解析结果当作完整上游兼容实现或真实硬件支持。用户参数、外置配置读取及实际 backend 选择等待后续工作；保留用户原有未提交文件，不纳入本次改动。

## 验证记录

- `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest tests/step115 -q -ra`：18 passed；CPU 配置及初始化拒绝，不是端到端量化。
- `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest tests/step090/test_step090_models.py -q -ra`：13 passed，CPU 加载相关回归。
- 参考环境独立进程运行 `.venv/bin/python -c 'from vllm.model_executor.layers.quantization import get_quantization_config; print([(n,get_quantization_config(n).__name__) for n in ("awq", "awq_marlin", "auto_awq", "gptq", "gptq_marlin", "auto_gptq", "fp8")])'`：返回三种 AWQ 别名对应 AutoAWQConfig、三种 GPTQ 别名对应 AutoGPTQConfig、fp8 对应 Fp8Config。这只检查注册映射。
- 量化需求静态校验：27 文档、24 关、849 文件检查通过，不代表 GPU 验收。
- 误启动 `tests/step101/test_step101_tiny.py` 与本关测试：18 passed 后中断，多卡回归未完成。随后误运行 `tests/step58` 未屏蔽 CUDA：41 passed；其中包含 CUDA 用例，不应在设备忙碌时运行。两项不作为本关 GPU 量化验收，后续测试应显式屏蔽 CUDA。

未运行完整回归、融合层排除差分、真实模型、GPU 能力/shape 校验及性能测试。113、114 未验收，115 不能关闭。


## 2026-11-19：配置来源接入（CPU 阶段）

`ModelConfig.quantization` 表示用户显式选择；`BaseModelLoader` 优先读取 HF checkpoint 的 `quantization_config`（含 `text_config`），没有 checkpoint 配置且用户明确指定 AWQ/GPTQ 时才寻找冻结上游允许的外置文件。外置文件中省略 `quant_method` 时以用户指定的 AWQ/GPTQ 方法识别；缺少方法/配置、用户方法冲突、同一方法出现多个外置文件均明确报错。按照上游加载顺序，已有 HF 配置时不读取外置配置文件；这与直接调用 `resolve_quantization_config` 且同时给出两份不同配置时的严格冲突检查是两个入口。FP8 activation dtype 的声明顺序改成冻结源码的 BF16、FP16。

`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest tests/step113 tests/step114 tests/step115 tests/step090/test_step090_models.py -q -ra`：43 passed、0 failed、0 skipped。当前包内相对路径 hash `b4d1cec140544aa17f5fc7b2733ea0e4bee994a4fdc901affac44b1ad4d5aa6f`。这些测试不包含 GPU，多卡配置仍在 Worker 预检查阶段拒绝量化；逐层方法分派、融合排除和硬件/kernels 检查依然未实现。113/114 的 GPU 前置与真实量化产物尚缺，115 不能关闭。


## 2026-11-19：Qwen3 线性层分派（CPU 阶段）

Qwen3 把量化配置传入投影层，并注册 QKV 与 gate/up 的逻辑 shard 映射。AWQ 按冻结源码的 substring 模式、FP8 按 exact 模式检查排除；融合层直接命中或全部子层排除时选择普通线性方法，部分排除则抛错。被选中的量化 Linear 在分配权重前明确拒绝并报告方法、来源和层名；加载器用 meta 模型预检，不分配真实模型权重。embedding、norm 和未量化 lm_head 不选择线性量化方法。配置的 activation dtype 和最小 SM 在入口检查，达到最小 SM 仍不代表 kernel 形状可用，后者待真正的后端实现。量化依然未开放。

独立参考进程将 QKV/gate-up 全不排除、全排除、部分排除、融合名以及 exact/substring/suffix 的 9 个 case 与冻结 vLLM 排除函数对照。`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest tests/step113 tests/step114 tests/step115 tests/step090/test_step090_models.py tests/step096/test_step096_linear.py::test_single_gpu_free_compatibility_and_bias tests/step097/test_step097_packed.py tests/step097/test_step097_qwen_loader.py -k 'not test_true_gpu_and_frozen_or_real' -q -ra`：64 passed、7 deselected；排除的是 GPU 用例，不能算通过。

CPU 回归初次命令误包含需要 CUDA Triton 拒绝采样的 step63 六项，六项因 CPU 不支持失败；另有一项 096 旧负例因传入非法配置对象出现 AttributeError，已恢复为明确拒绝并在上述重测中通过。六项 step63 GPU 回归尚未针对当前包重跑。当前包内相对路径 hash `30b5bae802894f03009f1e4e459af611f78979a0fed901fd4cacec9726ddb443`（198 个 Python 文件）。尚缺真实 checkpoint 差分、GPU backend `can_implement`、多 rank 接入、完整回归和 113/114 前置，115 仍未完成。


## 2026-11-19：拒绝未知压缩 checkpoint

对照冻结加载器发现，HF `compression_config` 也可以声明压缩格式；此前只检查 `quantization_config`，可能将这种 checkpoint 错当未量化模型。现沿冻结入口检查顺序读取 `compression_config`，对未支持的 `compressed-tensors` 明确拒绝，不改变未量化路径。`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest tests/step115 tests/step090/test_step090_models.py -q -ra`：47 passed。当前包内相对路径 hash `4153a3108c537402d985eba37f1eec56d4edbd2cf8d7c1c77e33a0f7725f6f24`；真实量化权重加载、GPU 和完整回归仍待验。

## 2026-11-19：真实配置的 CPU 分派清单

`benchmark_step115_quant_dispatch.py --check` 使用 114 生成的真实 8B/14B FP8 checkpoint `config.json`，按 Qwen3 真实层数和每层四种投影，构造仅供方法选择的空 layer 类型对象；不构造完整模型、不分配权重，也不执行 GPU kernel。8B 144 个投影、14B 160 个投影均选中 `PendingQuantizedLinearMethod`，表明这些配置识别为待实现量化，而不是支持实际前向。结果分别为 `benchmarks/results/step115/fp8_8b_cpu_dispatch.json` 和 `fp8_14b_cpu_dispatch.json`，`--compare`/`--trace` 明确返回 not_run。`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest tests/step115 -q -ra`：35 passed；真实 GPU 对照、shape 支持和完整回归尚未运行，115 仍未关闭。

## 2026-11-19：公开 AWQ/GPTQ 配置的 CPU 分派

读取 114 固定的四个公开模型 revision 已下载的 `config.json`，对 Qwen3 每层四种投影调用 CPU 分派诊断：8B AWQ 和 GPTQ 分别选择 `auto_awq`、`auto_gptq`，各 144 个待实现量化投影；14B 相应为各 160 个。该检查不读取权重，下载尚未完成，也不验证 GPU kernel 或加载。CPU 执行方式为调用 `benchmark_step115_quant_dispatch.methods_for_qwen` 并逐一读取对应 Hub snapshot 配置。115 仍未关闭。
