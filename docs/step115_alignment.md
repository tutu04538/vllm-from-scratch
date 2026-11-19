# step115：量化配置解析（进行中）

状态：**部分实现，不能视为 115 验收通过**。113、114 仍未完成；未运行量化 GPU 算子和真实量化模型，也未做性能测试。

## 基线

实现前 HEAD 为 `c7c4886af40c2b3e00213d5c2bfa826302bc5b4d`，生产包 Python hash 为 `48ba37036a952982e667679d3a5c5a6b309c6e7d2acc8db998201f6130834b90`。本地忽略的 `learning_notes/14_vllm_from_scratch/量化完整需求/参考资料/工作区基线_20261119_step115/` 保存了状态、改动补丁、未跟踪文件和逐文件 hash；不是干净 HEAD，也不纳入 Git。

当前生产包 Python hash：`162c26384ccc799961a5137a94600032816c5b50c3db2f5f103080f37f366a8f`；包含既有 Qwen3 未提交修改。hash 按相对路径排序、逐文件 SHA256 后以 `路径\0hash\n` 拼接计算。

## 实现及缺口

| 项 | 对照及本项目位置 | 当前状态 |
|---|---|---|
| R115.01 | 上游 base_config；`minivllm/layers/quantization/config.py` | 只有配置抽象及注册入口；方法生命周期和逐层分派未实现 |
| R115.02 | 上游 auto_awq/auto_gptq/fp8；同上 | 解析计划子集、常用字段及三组入口别名；实测上游注册名一致，尚未完成逐字段差分 |
| R115.03 | 同上 | 解析函数区分 checkpoint/外置来源并拒绝冲突；用户参数和外置文件尚未接入模型入口 |
| R115.04 | 上游 `is_layer_skipped`、`get_quant_method` | 未实现 |
| R115.05 | 同上 | 实现部分配置子集拒绝；仍缺层类型、完整格式及 MoE 检查 |
| R115.06 | 三种配置的 dtype 与 minimum capability | 提供元信息；未接入实际设备检查，kernel `can_implement` 未实现 |
| R115.07 | Qwen3、Linear | 未接入 |
| R115.08 | `minivllm/model_loader/base_loader.py` | checkpoint 配置解析后，模型构造前明确拒绝量化执行；不是量化功能开放 |

不能把当前解析结果当作完整上游兼容实现或真实硬件支持。用户参数、外置配置读取及实际 backend 选择等待后续工作；保留用户原有未提交文件，不纳入本次改动。

## 验证记录

- `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest tests/step115 -q -ra`：18 passed；CPU 配置及初始化拒绝，不是端到端量化。
- `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest tests/step090/test_step090_models.py -q -ra`：13 passed，CPU 加载相关回归。
- 参考环境独立进程运行 `.venv/bin/python -c 'from vllm.model_executor.layers.quantization import get_quantization_config; print([(n,get_quantization_config(n).__name__) for n in ("awq", "awq_marlin", "auto_awq", "gptq", "gptq_marlin", "auto_gptq", "fp8")])'`：返回三种 AWQ 别名对应 AutoAWQConfig、三种 GPTQ 别名对应 AutoGPTQConfig、fp8 对应 Fp8Config。这只检查注册映射。
- 量化需求静态校验：27 文档、24 关、849 文件检查通过，不代表 GPU 验收。
- 误启动 `tests/step101/test_step101_tiny.py` 与本关测试：18 passed 后中断，多卡回归未完成。随后误运行 `tests/step58` 未屏蔽 CUDA：41 passed；其中包含 CUDA 用例，不应在设备忙碌时运行。两项不作为本关 GPU 量化验收，后续测试应显式屏蔽 CUDA。

未运行完整回归、融合层排除差分、真实模型、GPU 能力/shape 校验及性能测试。113、114 未验收，115 不能关闭。
