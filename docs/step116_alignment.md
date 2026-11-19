# Step 116：LinearMethod 接入（实现中）

本关将线性层的本地权重创建和计算交给方法对象；列分片的 gather、行分片的 all-reduce 和 bias 仍由线性层负责。当前不开放量化模型执行。

开始编码前的 R 映射：R116.01/02/05 对应 `minivllm/layers/linear.py` 的五类 Linear 与 `minivllm/layers/quantization/methods.py`；R116.03/04 对应既有构造接口与 Qwen3 调用；R116.06 对应 `minivllm/model_loader/base_loader.py` 权重加载和方法后处理；R116.07 对应真实 TP/Graph 回归，按用户决定留待后续 GPU 正确性测试；R116.08 对应触及文件的注释及本报告。测试和点名脚本位于 `tests/step116/`、`benchmarks/benchmark_step116_linear_method.py`。

## 代码与边界

- `LinearMethodBase` 继承量化方法生命周期；`UnquantizedLinearMethod.create_weights` 根据本 rank 的输入和输出尺寸创建普通 Parameter，并保留原来的 `weight_loader` 与分片属性；`apply` 沿用 `F.linear`。五类 Linear 保留原构造入口、state_dict 名称、bias 与通信职责。待实现量化方法在参数创建时继续拒绝，不会转为 BF16。
- `BaseModelLoader.load_model` 在读取 checkpoint 期间将线性层标为未就绪，完整 `load_weights` 返回之后依次调用方法的 `process_weights_after_loading`；全部成功后才将这些层标为就绪。任何加载或后处理异常均不会返回模型。独立构造 Linear 的既有 API 仍允许调用者自行填充参数和调用，因此本阶段无法检测调用者是否真的手动填充了每个参数；该边界不应声称满足所有未加载负例。
- 未改通信顺序：Row 的 bias 只由 rank 0 参与局部计算，随后执行原 all-reduce；Column 按原条件 all-gather。QKV/Merged 保留既有 shard 加载及覆盖检查。没有修改 Qwen3 的 attention、norm 或 EAGLE3 语义。

## 已执行的基础检查

- `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest tests/step116 tests/step115 -q -ra`：41 passed、0 failed、0 skipped（CPU）。包含五类方法构造及后处理失败拒绝执行的基础检查。
- 同环境执行 `tests/step115 tests/step090/test_step090_models.py tests/step096/test_step096_linear.py::test_single_gpu_free_compatibility_and_bias tests/step097/test_step097_packed.py tests/step097/test_step097_qwen_loader.py -k 'not test_true_gpu_and_frozen_or_real'`：58 passed、7 deselected（GPU 用例不计通过）。
- `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python benchmarks/benchmark_step116_linear_method.py --check --output benchmarks/results/step116/construction.json`：五类 Linear 的 CPU 构造检查通过；不是性能测试或模型数值对照。

当前包内相对路径 SHA256 `2d7f0bc5b8595ec1e093b01a5e3e6dd27a5c48d3d9966781ced201297ca34063`。按用户要求，真实模型 BF16、TP1/2/4、Graph、EAGLE3 和重构前后数值差分以及完整回归均留待后续，尚未验收；点名脚本的 `--compare`/`--trace` 返回 not_run。115/116 代码工作可继续，但不能将需求标为完全完成。
