# step114：真实量化模型身份与评测基线（进行中）

状态：**未完成**。113 的 GPU 能力和上游 dispatch 尚未验收，本关暂时只准备独立的数据身份与校准/评测材料；六份真实量化权重尚未生成，不能标 ready。

| 需求 | 参考/本地职责 | 状态 |
|---|---|---|
| R114.01 | 090 模型 revision 与源身份 | 继承历史身份，待本轮重新对账 |
| R114.02—05 | AWQ/GPTQ/FP8 真实量化及独立工具环境 | 待实施，GPU 后端身份待 113 |
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
