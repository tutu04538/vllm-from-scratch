# 090｜真实模型身份、转换与加载

状态：090 范围已完成；本关未实现项目 TP/PP/DP，14B 项目端完整 GPU 推理留 103，EAGLE3 真实前向、逐轮差分和映射后 logits 留 113。运行结果、tensor manifest 和参考张量位于本机 `benchmarks/results/step090/`，不进 Git；模型位置见下表。

## 1. 来源与模型身份

所有模型使用需求固定 revision，原始快照在 `/data/syh/.cache/huggingface/hub/`，转换产物在 `/data/syh/models/converted/AngelSlim/`。2026-11-05 的下载预检：四份权重合计 47,937,135,500 bytes（约 44.65 GiB）；包括临时副本按至少 100 GiB 预留，`/data` 当时约有 1.7 TiB 可用。下载走本机配置的 Hugging Face endpoint，完成后 15 个权重文件 SHA256 与该 revision 的 Hub LFS 元数据全部一致；详见 `source_lfs_verification_full.json`。

| 模型 | revision | 实测 |
|---|---|---|
| Qwen/Qwen3-8B | `b968826d9c46dd6066d109eabc6255188de91218` | 5 分片，399 tensor，16,381,470,720 tensor bytes；本项目 V2 BF16 单卡短前向通过 |
| Qwen/Qwen3-14B | `40c069824f4251a91eefaf281ebe4c544efd3e18` | 8 分片，443 tensor，29,536,614,400 tensor bytes；CPU 流式审计通过；项目端 14B GPU 未运行 |
| AngelSlim/Qwen3-8B_eagle3 | `9629dfce7a4a10564dd48d3e5485c3976095653c` | 15 tensor 转换和重复对账通过；原始浮点 tensor 为 BF16 |
| AngelSlim/Qwen3-14B_eagle3 | `3d13517724e81cb409ddf1d4650772ec52f1e18e` | 15 tensor 转换和重复对账通过；**原始浮点 tensor 为 FP16**，虽然 config 声明 BF16；转换未改 dtype |

两个 target 的 tokenizer.json SHA256 均为 `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`，以各自快照记录为准。8B 为 36 层、H=4096、Q/KV=32/8；14B 为 40 层、H=5120、Q/KV=40/8；head dim=128、词表=151936，均不 tie word embeddings。权重张量字节不是 GPU 加载峰值。

## 2. 源码和执行顺序

| 项目路径 / 方法 | 冻结上游 | 差异与证据 |
|---|---|---|
| `benchmarks/check_step090_models.py` 固定 revision、空间检查和文件 hash | `model_executor/model_loader/weight_utils.py`、`default_loader.py::DefaultModelLoader.load_weights` | 准备工具与生产加载分开；生产仍只读 safetensors，不启用任意 pickle 加载。源文件和配置 hash 在结果 manifest 中。 |
| `convert_draft` / `audit_eagle_mapping` | `models/llama_eagle3.py::Eagle3LlamaForCausalLM.load_weights`、`config/speculative.py` | CPU `torch.load(weights_only=True)`，保留名称/shape/dtype/原始字节；完整逐 tensor 对账、重复转换不覆盖原件。Llama 架构按配置识别，不凭 Qwen 名字添加 Q/K norm。 |
| `minivllm/model_loader/weight_utils.py::iter_weights` | `DefaultModelLoader.get_all_weights` / safetensors 迭代 | 项目 `AutoWeightsLoader` 使用递归 groupby，同模块名字必须连续。真实 checkpoint 存在跨 shard 的层，原按文件输出依次在 8B `model.layers.7`、14B `model.layers.3.mlp` 触发分组错误；现在校验所有文件后对**名字**全局排序，逐 tensor 从原 shard 惰性读取，不重建 tensor 或改变模型权重布局。冻结上游不依赖本项目这个分组约束。 |
| `benchmarks/check_step090_reference.py`、`step090_reference_hooks.py` | 冻结 V2 Runner、Qwen3、LogitsProcessor、TRITON_ATTN | 仅测试进程调用安装的上游；先逐文件检查专题快照 342 项与历史执行归档 21 项的安装源码 hash。通过测试 RPC 逐 rank 取固定历史、有效 KV、首层算子及 logits；不进入生产包。 |

请求及逻辑 KV 仍由 Scheduler 管理；这关没有改 Scheduler 或执行侧镜像。格式转换只处理权重文件；第 14B target 没有在项目端每卡复制完整模型后再切。

## 3. 转换、映射和加载证据

两份 draft 的原始与转换 state dict 各 **15 tensor**，逐个名称、shape、dtype 和字节 SHA256 相同，原始 `.bin` 未覆盖；重复转换与第一次 tensor manifest 一致。两份 `d2t` 均为 int64 `[32000]`，`t2d` 均为 bool `[151936]`。按冻结规则 `target_id = arange(32000) + d2t`，两者均是 32000 个不重复且不越界的 target id，`t2d` 的 true 集合恰与这些 id 一致。

两份真实转换产物经项目 loader 在 CPU 加载检查，模型均为 Llama EAGLE3 路径，Q/K norm 为 None。checkpoint 均未提供 target embedding：已确认模型标记需要共享，**未在该检查中绑定 target，也未拿这个未绑定模型做前向**；完整共享关系和草稿数值闭环留 113。14B draft 以 BF16 构造 CPU 模型是显式加载转换，原始和转换文件仍保持 FP16。

8B target 的项目 V2 eager、TRITON_ATTN、BF16 单卡短前向完成，聊天模板 `enable_thinking=False`，实际 18 个 token IDs 和生成 token 保存在 `forward_8b_manifest.json`。实际模型含 291 个参数项、36 个 KV tensor，KV 物理合计 301,989,888 bytes；GPU peak allocated 16,745,053,696 bytes、reserved 16,833,839,104 bytes，进程 CPU peak RSS 17,098,136 KiB。14B CPU 权重审计 peak RSS 为 29,406,720 KiB；两份草稿的 CPU 加载峰值见 `draft_load_final_{8b,14b}.json`。这些是本次测量，不是将来并行加载容量保证。

## 4. 冻结参考数值登记

`numerics_contract.json` 只登记**本次固定 128 token target fixture**：由固定 tokenizer 输出的 18 个 token ID 循环截取，位置 0—127、block size 16、BF16、eager、TRITON_ATTN、每配置重复三次。8B 参考 TP1/2/4，14B 参考 TP2/4；这是运行冻结 vLLM 的参考测量，不是项目 TP 已实现。逐 rank 保存输入 ID、position、有效 slot、有效 block table、seq_lens、首层 qkv/attention/KV、最终 hidden、raw logits；组内相同数据及 rank/UUID 进行了检查，局部 qkv 与相同输入/权重的 FP32 GEMM 对照。

登记规则在看到任何 mini 数值差分之前固定为：按对象分别取参考重复及合法 TP 间的最大观测误差的 2 倍，absolute/relative L2 各设置最小值 1e-5；local qkv 使用 BF16 与 FP32 局部重放的观察值。结果包含最大误差索引、两侧数值、relative L2 和 top1/top2/margin。它们只用于这个固定 fixture 的对应对象，不是其它层、输入或实现的通用精度担保；超限须定位，不能根据 mini 结果放宽。

同配置三次重复本次均一致。跨 TP 的参考自身存在 BF16 累积差异：8B 相对 TP1 的 raw logits 最大绝对差为 0.59375，14B TP4 相对 TP2 为 0.90625；本次 top1 均为 198，top1/top2 margin 分别约 13.25—13.625。这不是项目与 vLLM 的逐 token 等价证明。EAGLE3 **映射后 logits 未登记**，实际 draft 前向在 113 先建立独立参考契约，不能套用 target 门槛。

契约 SHA256：`29bcbf64f1212cf100845387ff46ddcd43981b3d9fd4487c044a455ec28d0e7c`；冻结专题源码清单 SHA256 `b37356d29313680953f67fc1ceacf10c83e726931efaed87fe9c06f7dc67bea7`；历史执行归档 SHA256 `c2917a2106bb7de0bdbadc023e9c75b6ba6fee372f0aee406135635431e807ea`。完整阈值、max error 位置和值以及原始产物路径见 `docs/results.json → step090.results.numerics_contract` 与本地结果。

## 5. 测试与已知边界

- `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/step090 -q -ra`：13 passed；覆盖跨 shard 小 Qwen 实际参数加载、真实配置/索引、转换的 alias/int/bool、重复转换、错版本/错配/词表及重复/缺失/错 shard 负例、mapping 和数值指标。
- `CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest $(printf 'tests/step%s ' $(seq 58 88)) tests/step090 -q -ra`：1302 passed / 0 failed / 23 skipped；23 个旧真实模型用例因资料不足未验。
- `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/step089 tests/step090 -q -ra`：20 passed；多卡用例没有继承单卡限制。
- `check_step090_models.py --mode {convert,load-draft,forward}` 的真实模型运行、`check_step090_sources.py`、五个独立参考进程和 `compare_step090_reference.py` 实跑；逐命令和退出码见 `docs/results.json`。需求静态校验 8003 项通过。
- 保留失败记录：第一次 8B 前向测试入口把 Transformers 5 返回的 BatchEncoding 键当 token，报 `too many dimensions 'str'`；已修正并补测试。上游参考第一次 RPC 被默认序列化限制拒绝、TP2 第一次 hook 经 `__main__` 无法跨进程 pickle，分别改为仅在测试进程启用函数序列化及可导入 hook 模块后通过；参考汇总第一次因 TP1 缺 UUID 被拒，补采通过。日志均保留在本地结果目录。

没有训练 draft、没有完成真实 EAGLE3 逐轮闭环，没有本项目 14B GPU 推理，也没有量化/offload/多卡功能声明。快照、模型权重及生成结果不入 Git，私人需求资料和转换目录需按实际服务器路径另行管理。
