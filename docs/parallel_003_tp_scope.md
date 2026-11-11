# 并行需求收缩为 TP 专题

日期：2026-11-11。修订前HEAD：`4b2012990b22ebd4990a9cfe5b3890f5614a423c`。

用户决定“只搞TP”。本次修改需求与协作入口，撤销PP、DP及混合并行的实施义务，保留Qwen3-8B/14B、对应真实EAGLE3、TP Graph和性能对照。没有修改生产代码、测试、模型权重或历史运行结果。

## 当前入口与有效关卡

[专题总纲](../learning_notes/14_vllm_from_scratch/分布式并行完整需求/000_总纲与实施顺序.md)现在有24个有效实施关卡：089—105、113—116、118—120，以及000—002三篇公共契约。

- 106—109（PP）、110—111（DP）、112（混合并行）、117（DP/TP与EAGLE3）共8关撤销。原文件名保留撤销说明，原文完整归档，编号不复用。
- 有效关卡不依赖撤销项；118前置改为103/105/116。后续主线为105→113→114→115→116→118→119→120。
- 091规定PP>1或DP>1在加载权重、初始化NCCL和启动Worker前拒绝，包含普通生成、Graph、EAGLE3及混合配置；这是用户范围限制，不冒称上游普通PP/DP不支持。
- 092的组划分例子改为单TP组；094的输出rank按冻结公式在当前范围取0，所有TP rank仍执行和采样。原有通用层测试不因范围收缩删除。
- 119改为单副本TP的延迟、吞吐、Graph、EAGLE3对照及通信归因；TP2只占两张卡，不能用DP副本补满四卡。增加同一TP2配置下同/跨NUMA GPU组合的通信观察。
- 002/120按有效且已实现的关卡生成回归目录；不连续展开089—120，不为撤销项创建空实现、空测试或skip占位，也不计为最终缺口。

[AGENTS.md](../AGENTS.md)、源码职责矩阵、需求索引和参考资料说明同步收缩范围。冻结源码、模型revision/元数据及其hash保持原值。总纲的模型/环境进展引用090、103、105已有报告，保留105一次Worker关闭超时根因未知的边界；本次没有重新验收这些能力。

## 本地归档与校验

修订前需求正文、索引、校验器、参考说明及AGENTS完整保存在[范围调整前归档](../learning_notes/14_vllm_from_scratch/分布式并行完整需求/参考资料/20261111_TP范围调整前.tar.gz)。当前索引明确区分`scope_status=active/withdrawn`；撤销项没有前置、源码阅读或测试执行义务。归档内的旧PP/DP命令仅供追溯，不是当前待办。

实际执行：

```bash
source scripts/activate-4090.sh
python learning_notes/14_vllm_from_scratch/分布式并行完整需求/参考资料/check_requirements.py \
  --output learning_notes/14_vllm_from_scratch/分布式并行完整需求/参考资料/需求校验结果.json
git diff --check
```

结果：**7877项静态检查通过，0失败**。检查包括24个有效关卡、8个撤销项、总纲/正文/索引依赖一致、无撤销项依赖、TP性能矩阵、342份冻结源码、69处有效方法入口、4个模型元数据身份及本地链接。新增失效链接为0，冻结资料hash未变；`git diff --check`通过。

未运行pytest或GPU实验；静态检查不是运行测试passed/failed/skipped，历史验收数量不在本次重新声明。`minivllm/`、`tests/`、`benchmarks/`、`scripts/`及`docs/results.json`均未修改。

需求索引SHA256：`74a79571fd16ba2df46f7d10de3b1e7c14f5e49d3c37bbbe1431a5cadfdaef49`。

需求正文、归档和AGENTS继续保持gitignore；本次Git仅提交本记录及docs索引。异机Terra需要另行同步这些本地资料。此前[parallel_002](parallel_002_requirements.md)是2026-11-05原交付的历史记录，当前范围以本记录及专题000为准。
