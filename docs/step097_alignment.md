# 097 Merged / QKV 分片与 GQA 复制

## 实施前冻结源码映射

| 职责 | 冻结 vLLM 0.28.0 | 输入/本地布局与顺序 |
|---|---|---|
| Merged 构造和加载 | `model_executor/layers/linear.py::MergedColumnParallelLinear.__init__/weight_loader` | 各逻辑输出各自除 TP，目标 `[gate_local; up_local]`；独立 shard 或完整 packed 逐段切出本 rank，Parameter 从创建起本地大小 |
| QKV 参数构造 | `QKVParallelLinear.__init__` | Q 头数除 TP；KV≥TP 时均分，KV<TP 时每 rank 一头并按 TP/KV 复制；目标 `[Q_local;K_local;V_local]` |
| QKV 加载 | `QKVParallelLinear.weight_loader`、`model_executor/parameter.py` | 完整 packed 从全局 Q/K/V 边界拆开；独立 q/k/v 先按 TP rank（KV 用 rank//replicas）切源，再放入本地偏移；沿用缺漏、重复对账 |
| Qwen3 构造 | `model_executor/models/qwen3.py` | 显式 head_dim；保存全局和本地头数，split/view 和每 rank Attention 用本地维度；q/k norm 每 head 维度共享 |

本关只实现/测试权重、构造和本地投影；完整多 rank Attention/KV、真实多卡模型和 Graph 不在范围。Gemma 共享 KV 多卡仍不开放；TP1 路径必须保持。

## 设计与边界

- `QKVParallelLinear` 从本进程 TP 组取得 rank/size；Q 头均分，KV 在 `KV>=TP` 时均分、`KV<TP` 时按冻结规则计算复制因子 `TP/KV`，以 `rank//replicas` 选择 KV 源。全局 Q/KV 头数与本地头数、`q_size/kv_size` 分开保存。本地参数从创建起仅分配 Q/K/V 本地行数。
- `MergedColumnParallelLinear` 按每个逻辑输出独立切分，目标按源 shard 顺序拼接。两类层都支持独立输入和完整 packed 输入，先检查源张量全局形状，再按 rank 取源切片并写入本地槽；拒绝特殊/量化 Parameter。
- 保留并扩展加载对账：weight 和 bias 分别统计，避免 bias 写入被误算为重复 weight；启用 bias 时相应 shard 缺失或重复也报错。`Qwen3Model.load_weights` 加载前清零本轮记账，完成后检查每层 q/k/v 和 gate/up；Gemma 保留已有按共享层实际消费的 shard 集合对账。
- `Qwen3Attention` 保留全局头数并从 QKV 层取得本地头数、split 宽度和 Attention 头数；`head_dim` 仍按配置取，q/k norm 保持每头维度。`VllmConfig` 明确拒绝 Gemma TP>1；真实 Qwen 多卡模型加载、KV cache/Attention 端到端与 Graph 尚未开放。
- 测试直接调用 SHA 匹配的安装版冻结 `QKVParallelLinear` 和 `MergedColumnParallelLinear`；仅将其 TP rank/size 查询接入真实测试组，分别对独立、完整 packed、weight/bias 和多逻辑 shard 加载做比较；还独立计算 rank 源切片。TP=1/2/4 的 tiny 测试在真实 GPU 上执行 Qwen 局部 QKV/norm/RoPE 和 MLP，并通过 NCCL 归约比较完整矩阵参考。

## 真实权重身份

抽查的是各自固定 revision 的第 0 层 Q/K/V、gate/up 和 q/k norm；原始文件均从 Hub 缓存 CPU mmap 读取，只有本 rank 切片被写入 GPU 参数。实际文件重新计算 SHA256，并与 090 已冻结的身份记录逐项一致：

- `Qwen/Qwen3-8B`，revision `b968826d9c46dd6066d109eabc6255188de91218`，`model-00001-of-00005.safetensors`：`31d6a825ae35f11fb85b195b4c42c146c051e446433125a215336abdf95cbf5f`。
- `Qwen/Qwen3-14B`，revision `40c069824f4251a91eefaf281ebe4c544efd3e18`，`model-00001-of-00008.safetensors`：`e942bdbdf08857d16a8fef7d1dae9fceabeb4e84def6043485fe2f6f085dab0e`。

这些是参数抽层校验，不是完整模型加载或完整推理；首个目标文件哈希重算输出见 `benchmarks/results/step097/real_weights_sha256.log`。

测试开发过程：新增 `attention_pre` 探针首次因直接构造模块后未按 `BaseModelLoader` 执行 `.to(device)`，导致 RoPE cache 留在 CPU；测试已按真实加载顺序移动完整模块。新增 MLP 探针首次引用了不存在的 `Qwen3MLP`，实际 Qwen3 使用 `Qwen2MLP`；已修正测试名称。两者均保留为测试构造问题，不归因于生产实现。

## 最终验收

- 最终包 SHA256：`271b3b62cb83d796ae565de8e28bf9d2bda98db367183373e2244a7f667fb119`。本关 pytest 15 passed；随后新增的非法复制因子反例单独 1 passed。源码校验 8003 项通过。
- 独立验收 7/7 通过：tiny TP=1/2/4；真实 8B、14B 分别 TP=2/4。tiny 的 QKV 每 rank 覆盖 KV=1/2/8 × 独立/packed × bias 开关；Merged 用三个不同逻辑 shard。TP4、KV2 的 KV 源头依次为 `[0,0,1,1]`。
- Qwen tiny 使用 `hidden_size=32`、`Q=8`、显式 `head_dim=8`，实际运行本地 Q/K/V 的拆分、norm 和 RoPE；MLP 在真实 GPU 使用本地 packed gate/up 和 Row all-reduce，TP=1/2/4 与完整矩阵参考的最大绝对误差均约 `3.73e-9`。
- 真实权重抽层：8B 的 TP2/4 本地 QKV 形状分别为 `[3072,4096]` / `[1536,4096]`；14B 分别为 `[3584,5120]` / `[1792,5120]`。各 rank 按全局配置独立推导源偏移逐值核对；q/k norm 各完整加载 128 个元素。
- 089—096 并行回归 132 passed；58—88 回归 1289 passed、23 skipped。旧权重缺失项不算通过。测试结束 GPU 计算进程查询为空。

完整命令、冻结源码哈希、逐 rank 形状与 UUID、固定权重身份、日志和未验范围见 `docs/results.json → step097.results/models` 和 `benchmarks/results/step097/`。
