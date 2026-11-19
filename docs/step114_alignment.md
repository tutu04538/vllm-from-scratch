# step114：真实量化模型身份与评测基线（进行中）

状态：**未完成**。113 的 GPU 能力和上游 dispatch 尚未验收，本关暂时只准备独立的数据身份与校准/评测材料；六份真实量化权重尚未生成，不能标 ready。

| 需求 | 参考/本地职责 | 状态 |
|---|---|---|
| R114.01 | 090 模型 revision 与源身份 | 继承历史身份，待本轮重新对账 |
| R114.02—05 | AWQ/GPTQ 离线工具与 `benchmarks/step114_convert_fp8.py` | 8B/14B FP8 已完成 CPU 审计；8B FP8 上游短前向通过；8B AWQ 正在制备，GPTQ 真实产物未制备 |
| R114.06 | 固定校准与 held-out 数据、精确 token | 已固定 train/test 与 GSM8K 样本、token IDs、模型 tokenizer、hash；尚未应用于离线量化 |
| R114.07 | 产物路径、hash、manifest | 数据产物已登记；量化权重尚未生成 |
| R114.08—09 | 冻结 vLLM 真实加载、参考值和容差 | 待 GPU 空闲及产物具备 |

评测数据的准备不是量化模型质量测试，数据脚本或 CPU 测试通过不代表真实模型或 GPU 验收通过。


## 2026-11-19 数据准备与验证

固定 WikiText-2 raw 的 Hub 仓库 `Salesforce/wikitext` revision `b08601e04326c79dfdd32d625aee71d232d685c3`，train 用作校准、test 用作 held-out；数据卡声明 CC-BY-SA-3.0 和 GFDL。固定 GSM8K `openai/gsm8k` revision `740312add88f781978c0658806c59bc2815b9866`，数据卡声明 MIT，按原始 test 顺序选前 256 题。数据原文件在 `/data/syh/.cache/huggingface/hub/`，所有文件 sha256、字节数和 tokenizer 目录文件摘要记录在产物 manifest。

对 8B/14B 各使用固定源 revision 的本地 tokenizer，按原文非空行分别编码、每行追加 EOS，再顺序截取不重叠的 2048-token 窗口：train 前 128 个作为校准，test 前 128 个作为 held-out。窗口不补齐，评测使用相邻 token 预测，首 token 无预测目标，剩余有效目标的 loss mask 为 1；精确 token IDs、窗口 ID、起点及 hash 均已保存。两模型本次生成的数据文件 hash 相同，并不替代模型身份独立登记。GSM8K 使用零样本用户消息、关闭 thinking，追加 `Give the final answer after ####.`，生成预算 512，答案解析规则登记于 manifest；这只是预登记，不代表已运行质量评测。

数据准备命令：`CUDA_VISIBLE_DEVICES='' TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmarks/prepare_step114_quant_data.py --arrow-python /data/syh/models/converted/_tools/step114-data-pyarrow/bin/python --output /data/syh/models/converted/_quant_data/step114_wikitext_gsm8k_20261119`。Parquet 读取器使用独立 Python 环境、pyarrow 24.0.0，不修改项目 `.venv`。第一次执行因 Transformers 5 的聊天模板返回字典而失败，临时文件已清理；随后显式设置 `return_dict=False` 重新生成。

产物在 `/data/syh/models/converted/_quant_data/step114_wikitext_gsm8k_20261119/`，manifest SHA256 `3b2799bf8bc959e70616793fe4b038467ff3c81f77b338d6593f99c1ba6d1d0e`；训练与测试的完整 2048-token 窗口 hash 交集为 0，held-out 窗口各不重复。`python benchmarks/check_step114_quant_data.py --data /data/syh/models/converted/_quant_data/step114_wikitext_gsm8k_20261119` 独立检查数据文件与源文件 hash、token IDs、窗口长度、样本数量和交集，通过；它不是语言模型输出对照。`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest tests/step113 tests/step114 tests/step115 -q -ra`：26 passed，0 failed，0 skipped，均为 CPU 测试。

尚未固定 AWQ/GPTQ 量化工具和量化模型 recipe、未制备六份权重，也未测 vLLM 加载、参考值、量化质量或性能。113、114 均不能关闭。


## 2026-11-19：8B FP8 CPU 转换（上游未验）

固定源 8B revision 的 5 个 BF16 权重分片重新逐文件与 090 的 LFS 清单对账；逐张量转换七种投影，每个原始 `[N,K]` 线性权重独立计算 `s=max(abs(w.float()))/448`，全零取 `s=1`，量化值为 `cast_e4m3fn(clamp(w/s,-448,448))`，保存 FP32 反量化 scale。embedding、lm_head、norm 及其他非投影保持源 dtype。转换脚本的 hash、每个源与输出文件 hash、每层 scale 和 recipe hash 保存在产物 manifest；不会覆盖已有目录。

命令：`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmarks/step114_convert_fp8.py --model 8B`。产物 `/data/syh/models/converted/Qwen/Qwen3-8B/b968826d9c46dd6066d109eabc6255188de91218/fp8-36023869cf7aff20`，recipe SHA256 `36023869cf7aff2019bc524329ca694ab0d96c03888d1c97b61b8fea300380e8`，tensor 总字节 9435704304，252 个逻辑投影 scale。产物 manifest SHA256 `a30018b825396073d1dd79c7ac2381aa43847e719cf84fb615114a94d2230bdd`。

`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmarks/check_step114_fp8_artifact.py --artifact <上述目录> --output benchmarks/results/step114/fp8_8b_cpu_audit.json`：CPU 审计通过；核对全部输出文件 hash、651 个 tensor 的索引关系、252 个 scale 的 dtype/数值，并对首层 Q 和末层 down 投影重算 FP8 bytes，抽查 embedding/lm_head 原样。小张量转换测试 3 passed，含零值、异常值拒绝和两份独立输入的重复编码。上述两层是生成公式的 CPU 对账，不是独立质量或 GPU 数值验证。

14B CPU 转换仍在运行，AWQ/GPTQ 工具环境正在准备。**8B 产物尚未由冻结 vLLM 加载或前向，状态是 converted_not_validated_by_vllm，不是 ready。**


## 2026-11-19：14B FP8 与 8B 上游短前向

14B 使用同一 FP8 转换脚本和 recipe，8 个源分片逐一与 090 LFS hash 对账，结果 `/data/syh/models/converted/Qwen/Qwen3-14B/40c069824f4251a91eefaf281ebe4c544efd3e18/fp8-36023869cf7aff20`，tensor 字节 16324557920，280 个投影 scale；manifest SHA256 `04f321e5d74269e714b2665cc0225dbba0c7a28ebf34d04c33771b3cf301b4a5`。同一 `check_step114_fp8_artifact.py` CPU 审计通过：723 个 tensor 的索引、全部 scale 和输出文件 hash，首层 Q、末层 down 的 FP8 bytes，以及 embedding/lm_head 抽样。14B 上游短前向启动前 GPU3 被其他用户占用，忙卡保护返回 not_run；本次没有真实加载 14B。

8B 在 GPU3 空闲期间以固定 tokenizer 的 held-out 窗口前 32 token、greedy 生成 4 token、BF16 激活、TP1、eager、TRITON_ATTN、固定 256 MiB KV，使用冻结 vLLM V2 Runner 实际加载并前向通过。首层 QKV 与 gate/up 均为 `Fp8LinearMethod` + `CutlassFP8ScaledMMLinearKernel`，权重 dtype `float8_e4m3fn`；输出 token `[279, 18096, 3776, 19584]`，每个 prompt token 的 logprobs 及生成 logprobs 保存在 `benchmarks/results/step114/fp8_8b_vllm_gpu3_retry_reference.json`。这只证明该 checkpoint 在该上游配置上可短前向，不是完整 logits 差分、质量评估或 minivllm 实现验证。第一次已加载模型但在 worker 函数序列化阶段失败；隔离参考子进程设置 `VLLM_ALLOW_INSECURE_SERIALIZATION=1` 后重跑通过，首轮日志保留。后续上游探针新增了 AWQ/GPTQ 格式的选择，初次 FP8 结果使用的是新增之前的探针版本。

## AWQ/GPTQ 工具准备（真实模型待验）

AutoAWQ 0.2.9 安装在 `/data/syh/models/converted/_tools/step114-awq/`，其包自述已停止维护，最后测试组合是 Torch 2.6/Transformers 4.51.3。初次选择 Transformers 4.57.6 时因 `PytorchGELUTanh` 无法导入失败；只在隔离工具环境降到 4.51.3 后，CPU 上真实 tiny Qwen3 的 AWQ 量化/保存通过。当前 Torch 为 2.13.0+cu129，与其最后测试版本不同，真实量化和上游加载仍需检验。工具 `_apply_quant` 会将线性权重转 FP16 后打包；source 和校准 model 使用 BF16、目标推理激活仍要求 BF16，recipe 单独登记此 FP16 中间步骤，不把它隐藏为 BF16 全程计算。

GPTQModel 5.6.0 安装在 `/data/syh/models/converted/_tools/step114-gptq/`，Transformers 4.57.6、Torch 2.13.0+cu129；首次隔离构建因找不到 torch 版本失败，指定 `TORCH_VERSION=2.13.0`、`BUILD_CUDA_EXT=0` 重试成功。CPU tiny Qwen3 的 GPTQ 对称 W4/G128、desc_act=false 量化和保存通过；这不证明真实权重或 GPU kernel 可用。两环境分别运行 `tests/step114/step114_tiny_quantizer_probe.py --method awq/gptq`，日志在 `benchmarks/results/step114/`；环境列表 SHA256 分别为 `9f60f95837cc1de6b38a388dda550d99a4fb67a8937163a78c403406189929b3`、`2223532fca852aff6d1824ec707829822397663b07b021619ec57fcc7aab6ad6`。

GPU2 上已启动 8B AWQ：128×2048 固定校准、group 128、zero point true、BF16 model/calibration、串行校准样本；量化任务尚未完成，日志与进度见 `benchmarks/results/step114/awq_8b_gpu2.log`。启动器仅在目标卡计算进程为空时启动，不保证运行期间其他用户不会占卡。GPU0/1/3 目前有其他用户任务，完整 4 卡回归和 14B FP8 GPU 验证待空闲。
