# 099 词表分片与完整 Logits：实施前源码映射

| 冻结 vLLM 0.28.0 | 职责与布局 | 项目落点 |
|---|---|---|
| `vocab_parallel_embedding.py::VocabParallelEmbedding.__init__/_get_indices/weight_loader` | 默认按 64 行补齐；先切原始词表与 padding，每 rank 创建本地行并按全局区间加载，padding 置零 | `minivllm/layers/linear.py` 词表模块与参数加载器 |
| `VocabParallelEmbedding.forward/get_masked_input_and_mask` | 非本地 token 映射安全索引并 mask，局部 lookup 后 TP all-reduce；输入行不分片 | 同文件 Embedding forward / TP group |
| `ParallelLMHead` 和 `LogitsProcessor._get_logits/_gather_logits/forward` | 本地词表 GEMM → CUDA all-gather → 去 padding → soft cap → scale；每 rank 得到完整词表 | `linear.py` 和 `layers/logits_processor.py`；调用方保持全局 logits |
| `platforms/interface.py::Platform.use_all_gather` | CUDA 使用 all-gather，每 rank 都有结果 | 使用项目既有 TP group all_gather |

固定教学语义：同一批 token/hidden 输入每个 rank 一份；按词表分片而不分序列。每 rank embedding 仅本地有效 token 贡献值，all-reduce 后返回完整 embedding；每 rank LMHead 计算本地词表列，all-gather 后恢复 `[num_logits_rows, vocab_size]`。padding 行不是 token，不能进入采样。真实布局按上游实例验证，不采用教学图的 8 行假设。

模型参数/KV/工作区归 Worker/Runner；请求进度只归 Scheduler；本关不开放生产多卡模型加载、不引入局部 top-k 采样、LoRA/量化或 EAGLE3 多卡词表映射。现有单卡模型的 logits、草稿宽度及共享权重语义必须保留。

## 执行顺序与固定例子

以 V=7、TP=2、维度=4、默认 padding=64 为例，冻结上游先将 7 补到 64，每 rank 的物理分片都是 32 行。rank 0 的真实范围为 `[0,7)`、物理范围 `[0,32)`；rank 1 的真实范围为 `[7,7)`、物理范围 `[32,64)`，因此 rank 1 全是补齐行。输入 token 5 时，rank 0 取本地第 5 行，rank 1 将其 mask 为 0，两 rank all-reduce 后均得到该 embedding；两个输出头分别产生 32 列，all-gather 后删去真实词表外的 57 列，再交现有采样器。

| 步骤 | 输入 | 产物 / owner |
|---|---|---|
| 创建/加载 | 全局 checkpoint、TP rank | 每 Worker 持有 `[padded_vocab/TP, hidden]` 参数；仅复制真实本地权重，尾部 padding 清零 |
| 输入 lookup | 所有 rank 的相同 token IDs | 本地 mask + lookup + TP all-reduce；所有 rank 拿到完整 embedding |
| 输出 GEMM | 每 rank 对应 hidden rows | 每 rank 的本地词表 logits |
| 输出汇总 | 本地 logits | 每 rank all-gather 得完整列，切到真实词表，再 soft cap、scale |
| 后续采样 | `[num_logits_rows, vocab_size]` | 复用现有采样；跨 rank 采样协议由 100 单独处理 |

`ParallelLMHead.forward` 输出物理本地列，`LogitsProcessor.forward` 管理聚合及去 padding。Qwen3、Qwen3 MTP、EAGLE3 和 Step3p5 原先直接调用词表头的地方改为通过该处理器，以保留单卡真实词表宽度；这不开放对应模型的多卡组合。已有 DFlash2 候选选择保留 TP1，先排除真实词表以外的 padding；TP>1 的局部 top-k 明确拒绝，属于本关排除范围。

对旧测试的调整：单卡参数现在遵循上游 64 行补齐。step65/66 原先直接把整个参数和未补齐 checkpoint 比较，改为逐值比较真实行并额外断言补齐行为零；step80 的直接复制改为调用权重加载器并检查补齐后的 shape；step77 的手算参考仅取真实 Markov 词表行；step65 MTP 的直接输出比较也排除物理 padding。原有模型输出、概率和 token 断言没有放宽。

## 明确边界

只支持普通原始词表，不实现 LoRA 扩展词表、量化、分布式 local top-k 或 EAGLE3 的多卡缩减词表映射。TP1 的输入是 padding ID 时，按冻结上游行为，若该 ID 落在物理 padding 内可查到零行；padding 绝不能进入输出词表或被采样。真实 Qwen3-8B/14B 多卡加载与完整推理仍由 101 验收；099 的计算差分是人工确定权重的层级测试，不是整模型端到端结论。

偏置按冻结 `LogitsProcessor._apply_head` 的调用约定由 `embedding_bias` 显式传入，词表头本身不自动重复加偏置；多 rank 测试还对同一批本地偏置直接比较冻结 `LogitsProcessor`。099 的浮点层级对照阈值预先固定 `atol=1e-5, rtol=1e-5`；离散区间、权重及 token ID 使用精确断言。

089—098 首次全量回归记录 153 passed / 1 failed（095 `fresh`：rank 1 在既有 shutdown 超时内未退出）；095 探针不调用本关词表层，本次没有修改 Executor 清理逻辑，单独重跑该场景已通过。首次失败和完整重跑分别留存日志，不将首次失败改写为通过。

## 最终数值与验证

真实 RTX 4090、TP1/2/4 各 rank 运行 vLLM 原始 `VocabParallelEmbedding` / `ParallelLMHead` / `LogitsProcessor`，安装文件与冻结 `vocab_parallel_embedding.py`、`logits_processor.py`、`platforms/interface.py`、`platforms/cuda.py` 逐字节 SHA256 一致。测试将冻结实例的 rank / 通信入口绑定到实际 NCCL 组，并非自己重写分片公式作唯一 oracle。

| V | 补齐后全局行数 | TP2 真实区间 | TP4 真实区间 |
|---:|---:|---|---|
| 7 | 64 | `[0,7), [7,7)` | `[0,7), [7,7), [7,7), [7,7)` |
| 129 | 192 | `[0,96), [96,129)` | `[0,48), [48,96), [96,129), [129,129)` |
| 151936 | 151936 | `[0,75968), [75968,151936)` | `[0,37984), [37984,75968), [75968,113952), [113952,151936)` |

两行固定 FP32 输入和人工构造权重覆盖每个真实 shard 边界、padding 强制高分、显式局部偏置、soft cap 与 scale；每 rank 的全词表 logits 与冻结上游及直接参考比较最大绝对误差、相对 L2 均为 0；两行 top1/top2 均为 `V-1/V-2`，margin 分别约 `0.0015/0.00075`。这些精确结果只针对该固定层级夹具，不外推到 BF16 真实模型。原始每 rank 区间、最差位置、分数、设备 UUID、源码哈希见 `benchmarks/results/step099/acceptance_release_tp{1,2,4}_golden.json`（gitignore）。

首次 58—88 回归为 1276 passed / 13 failed / 23 skipped：失败原因是旧测试直接比较未补齐权重或把补齐行传给 DSpark 的参考函数。按上文说明增加真实行和补齐行断言后重跑通过。后续最终源码状态再次回归；命令、版本与全部日志见 `docs/results.json → step099`。
