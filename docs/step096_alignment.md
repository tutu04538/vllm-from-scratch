# 096 TP 行列线性层与本地参数

## 实施前冻结源码映射

| 项目职责 | 冻结 vLLM 0.28.0 对照 | 布局、顺序及所有权 |
|---|---|---|
| Column 参数和加载 | `model_executor/layers/linear.py::ColumnParallelLinear.__init__/weight_loader`、`model_executor/parameter.py` | PyTorch 权重 `[out,in]` 按输出维分片；各 rank 创建 `[out/TP,in]`，bias `[out/TP]`；从 checkpoint 取 rank 连续切片并校验 |
| Column 前向 | `ColumnParallelLinear.forward`、`distributed/communication_op.py` | 输入完整、本地 GEMM；`gather_output` 时按输出维 all-gather；skip_bias_add 和 return_bias 对应上游 |
| Row 参数和加载 | `RowParallelLinear.__init__/weight_loader` | 权重 `[out,in/TP]` 按输入维分片；bias `[out]` 复制；按 rank 切权重列 |
| Row 前向 | `RowParallelLinear.forward` | 输入未分片则按 rank 切；仅 rank 0 融合 bias；`reduce_results` 时 all-reduce 部分和；保留独立 bias 返回约定 |
| 参数加载 | `model_executor/parameter.py`、项目 `AutoWeightsLoader._load_param` | 本地 Parameter 由本 rank 持有，挂载 loader；不创建完整 GPU 权重，生产不可调用上游类 |

固定例子 `TP=2`、`x=[1,2,3,4]`、`W=[[1,1,1,1],[1,2,3,4]]`：Column rank0/rank1 输出 `[10]`/`[30]`；Row 输入前/后半对应部分和 `[3,5]`/`[7,25]`，归约结果 `[10,30]`。本关不开放 QKV/Merged、词表或完整多卡模型；量化参数不支持并显式拒绝。

## 实现边界与上游差异

- 普通行列层按已初始化的 TP 组决定 rank 和本地参数大小；无分布式环境的旧单卡调用沿用 TP=1。若多 rank 默认组已建立而 TP 组缺失，直接报错，不能静默降级。`disable_tp=True` 按冻结行为显式选择完整复制。
- 为保留项目原有调用方式，构造函数中 `bias=False`、`prefix` 和 Column `return_bias` 的已有位置与默认值保持不变；新增分支使用关键字参数。冻结上游默认 bias 值及构造函数位置与项目既有接口不同，此为兼容路径差异，不改变显式配置的数学行为。
- `AutoWeightsLoader` 已通过每个参数的 `weight_loader` 分派，`iter_weights()` 本来就从 CPU 逐 tensor 读取 safetensors，因此不需要构造全模型 GPU state dict，也不需要改写加载器。每 rank 的 Parameter 自创建起即为分片大小。
- 非空量化配置和特殊 Parameter 明确拒绝；不支持 QKV/Merged 分片、词表分片或真实多卡模型。Column `skip_bias_add=True` 且 gather 时返回的 bias 仍是本 rank 分片，遵循冻结上游，而不是额外拼接 bias。
- 真实 GPU 测试加载小型 safetensors checkpoint，经 `AutoWeightsLoader` 分派；按 rank 逐值校验本地权重和底层 storage 字节数。FP32 组合与 BF16 整数 fixture 对冻结上游类逐值比较，同时核对独立完整矩阵参考；不把这些 tiny 层测试称为真实 Qwen 模型验收。
- 冻结源码差分直接实例化安装版 vLLM 的 `ColumnParallelLinear` / `RowParallelLinear`，先逐文件核对 `linear.py` 与 `parameter.py` 的 SHA256；测试中将上游 rank/world-size 查询和 collective 调用接到本项目实际 NCCL 组，以便两套层使用同一组比较。该差分覆盖上游层的参数加载和前向分支，不宣称重新验证了上游 communicator；后者见 092。

## 最终验证

生产包 SHA256：`b038c61640da5f5d971d88d13beed1e0698b16d42cc3dc67cf440e248cc6bba5`；冻结上游 `linear.py` SHA256 为 `fc4e96ba5e36895513b801b99e746e2b07959d402e36d56aae7d8ad3f124cfe7`，`parameter.py` 为 `ff6054fbd19ec932c548d562c6f4cc4506383b71cbe411fcfd554c8b1d87b510`。

- `tests/step096`：4 passed、0 skipped；包括真实 1/2/4 GPU 独立进程。
- 独立验收 3/3 通过；7 个 rank 各自完成 44 个 FP32 分支组合和 2 个 BF16 fixture 对照，另检验 disable_tp、真实 CPU safetensors 流、本地 storage 大小和非法维度。TP=2 原样手算例两 rank 的 Column `[10]`/`[30]`、Row `[3,5]`/`[7,25]` 和通信结果 `[10,30]` 均精确通过。
- 089–095 并行回归 128 passed；58–88 单卡回归 1289 passed、23 skipped。跳过的旧真实权重用例没有计入通过。
- 不加载真实 Qwen 权重，未运行完整模型多卡推理、QKV/Merged 分片、真实多 rank CUDA Graph；这些仍是后续关卡的验收范围。

完整命令、逐 rank 设备 UUID、结果和原始产物路径见 `docs/results.json → step096.results/models` 及 `benchmarks/results/step096/`。
