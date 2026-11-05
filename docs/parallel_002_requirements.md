# 并行专题需求与编号整理交付记录

日期：2026-11-05。编写前HEAD：`4115a39a390439096bea2d74534087ea24cbc2a0`。

本次交付详细需求，供Terra逐关实现Qwen3-8B、Qwen3-14B及各自真实EAGLE3的单机四卡并行。没有修改生产实现、测试或运行环境；本记录不代表TP/PP/DP已经实现。

## 1. 本地交付入口

- [并行专题总纲](../learning_notes/14_vllm_from_scratch/分布式并行完整需求/000_总纲与实施顺序.md)：范围、实施依赖与最终支持矩阵。
- [冻结源码与职责映射](../learning_notes/14_vllm_from_scratch/分布式并行完整需求/001_冻结源码与职责映射.md)：项目落点、源码职责和禁止替代的机制。
- [测试与交付协议](../learning_notes/14_vllm_from_scratch/分布式并行完整需求/002_测试与交付协议.md)：数值门槛预登记、逐rank证据、失败反例、真实模型和回归规则。
- [参考资料说明](../learning_notes/14_vllm_from_scratch/分布式并行完整需求/参考资料/README.md)：冻结源码、模型元数据、编号迁移及校验入口。
- [当前AGENTS.md](../AGENTS.md)：四卡环境、最新证据、现存边界、实现与交付约束。

这些文件在本仓库本地存在，但`learning_notes/`和`AGENTS.md`有意保持gitignore。本次Git仅提交本记录和docs索引，不强制提交私有资料；新克隆不会包含上述资料，需要另行同步给异机上的Terra。

## 2. 拆分与约束

共35篇需求：000—002三篇公共契约，以及089—120共32个实施关卡。各关都有痛点、固定小例子、冻结源码入口、修改范围、状态所有权、不变量、实现顺序、正负例验收和交付物。

| 关卡 | 主题 |
|---|---|
| 089—090 | 服务器基线、通信成本、模型身份与EAGLE3无损权重转换 |
| 091—095 | 并行配置、进程组、Worker启动、共享内存RPC、失败传播与清理 |
| 096—103 | TP线性层、QKV/GQA、KV容量、词表与采样、Qwen3加载、调度闭环、真实8B/14B验收 |
| 104—105 | 可捕获通信与TP整模型CUDA Graph |
| 106—109 | PP切层、中间张量与反馈、容量闭环、多在途批次 |
| 110—112 | 稠密DP副本、负载路由、四卡混合拓扑 |
| 113—117 | 真实EAGLE3单卡差分、分片、拒绝后状态、TP/Graph及DP混合 |
| 118—120 | 支持边界反例、公平性能矩阵、证据收口 |

冻结参考为vLLM 0.28.0，随附342份Python文件及95处具体方法入口。与三个既有源码归档的313个Python条目逐字节一致，0不同、0缺失；条目数包含归档间重复。新增并行文件只声称2026-11-05采集，未虚构更早归档证据。

新多卡主线固定V2，保留已有V1单卡行为。同步PP的在途队列容量遵循`max_concurrent_batches=PP`；稠密DP遵循独立副本重配置；TP采样遵循冻结CUDA完整logits/all-gather路径。禁止导入安装vLLM的生产模块充数。

冻结V2明确拒绝EAGLE3+PP>1，因此本专题不实现该组合。PP限定eager；TP/DP的Graph分阶段验收。draft TP等于target TP是本专题范围限制，不宣称上游所有路径都仅支持这一种。多机、CP/SP/EP/MoE、PD、量化、compile及端到端异步调度等留待后续。

## 3. 模型身份

| 模型 | 固定revision |
|---|---|
| Qwen/Qwen3-8B | `b968826d9c46dd6066d109eabc6255188de91218` |
| Qwen/Qwen3-14B | `40c069824f4251a91eefaf281ebe4c544efd3e18` |
| AngelSlim/Qwen3-8B_eagle3 | `9629dfce7a4a10564dd48d3e5485c3976095653c` |
| AngelSlim/Qwen3-14B_eagle3 | `3d13517724e81cb409ddf1d4650772ec52f1e18e` |

仅取得固定版本的配置、发布元数据及target权重索引，未下载或运行四份权重。两个target的BF16权重张量分别约15.26/27.51 GiB；后者用于验证超单卡容量，前者用于单卡参考和DP。两个草稿发布为`.bin`，090要求无损转safetensors及逐tensor对账，不能随机补参数。

## 4. 旧文档与AGENTS整理

`learning_notes/14_vllm_from_scratch/`内103个Markdown文件名的短数字前缀补齐三位，精确引用同步更新：77个旧资料文件和当时的AGENTS，共78个文件。正文关卡数字、旧tests/stepNN及历史docs/stepNN文件名保持原样。

迁移清单保存原名、新名和内容hash。与迁移前完整备份比较283个原资料文件，除精确文件名引用替换外，0处非预期内容变化。历史已有533条失效链接主要指向旧本机路径或缺失资料，本次未改写历史内容；新增失效链接为0。

AGENTS压缩重复流水账，保留代码不变量与测试/提交约束，纠正三类旧状态：

- 5090最终历史回归为1311 passed，attention异步加载修复及性能边界以最终报告为准。
- 4090最后重复回归仍为1275 passed / 13 failed / 23 skipped，Graph长进程问题未关闭；环境通过不等于模型并行通过。
- Gemma真实文本路径、Step3p5真实MTP子集与完整模型未验边界分别记录，不再笼统写成全部缺权重。

整理前AGENTS保留在专题参考资料中，仅供历史追溯。

## 5. 本次实际验证

```bash
source scripts/activate-4090.sh
python learning_notes/14_vllm_from_scratch/分布式并行完整需求/参考资料/check_requirements.py \
  --output learning_notes/14_vllm_from_scratch/分布式并行完整需求/参考资料/需求校验结果.json
git diff --check
```

静态校验：**8003项检查通过，0失败**；32关依赖、342份源码hash/AST、95处方法入口、4个模型身份/元数据、103个重命名及新链接均通过。另做上述283文件备份内容核对。`git diff --check`通过。

没有运行pytest或GPU模型实验；passed/failed/skipped不适用于本次文档任务，不能将静态检查数当成运行测试数。旧服务器失败与skip未被本次工作关闭。没有修改`docs/results.json`历史运行记录。

生产包177个Python文件未变，包摘要仍为`d5a0104ef1f6d54bcf54862dda6fe2979c10718462630bb8ee7cc9775ea46c9e`。

| 本地交付身份 | SHA256 |
|---|---|
| 源码清单.json | `b37356d29313680953f67fc1ceacf10c83e726931efaed87fe9c06f7dc67bea7` |
| 需求索引.json | `0137aae0b9b720c8bcf66bdeb30170bc8738e371ae9f07122219c3e599d0e10e` |
| 模型基线.json | `294cb0972e430d9b1d98f27a083dcdf5a5598904e3dc3bc55183e08e66938760` |
| AGENTS.md | `8f911de43341f4aaa9beba308f63940bc064b7fe9e2e230ad953cff6006d252e` |
