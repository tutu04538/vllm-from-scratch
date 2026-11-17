# 量化专题需求交付记录

日期：2026-11-17。范围：需求编写与本地协作约定更新，**未实现量化代码，未运行模型/GPU验收，未生成或下载量化权重**。

## 交付内容

本地入口：`learning_notes/14_vllm_from_scratch/量化完整需求/000_总纲与实施顺序.md`。目录含27份主文档：000—002共同约定，以及113—136共24关；共198个R子需求、121组点名验收用例。每关包含固定例子、状态所有权、冻结方法导航、按顺序实施项、不变量、负例、benchmark及退出条件。

路线：配置/Linear/Parameter → W4表示与真实AWQ → 本地Marlin构建/repack/GEMM → TP/Graph → GPTQ → FP8权重与W8A8 → FP8 KV → 量化target+BF16 EAGLE3 → 质量性能矩阵与收口。

模型限定Qwen3-8B、Qwen3-14B及各自AngelSlim EAGLE3。四份源revision沿用TP专题；六份计划量化产物明确标为未制备/未验收，114负责真实产物、工具配方、校准/评测数据与文件hash。PP/DP继续排除，草稿本期保持BF16。

冻结849份上游源码、构建和测试文件，总8,866,020字节，来源为本机0.28.0+cu129版本记录对应commit `2cf0a6915ce544dc493a0990f2ea38d81601128a`。其中346份Python与安装版逐字节一致，旧TP342条清单也全部一致。源码清单含每文件sha256与Python AST符号定位；CUDA源码本次未编译，外部CUTLASS实际依赖与构建证据仍属于113/120。

性能清单固定70配置、3种负载，共210个未来性能case：46个普通权重配置、12个FP8 KV配置、12个EAGLE3配置。实现差分、相对BF16质量、性能/容量独立验收，不预设加速倍数。

FP8 KV按冻结加载链路限定：校准scale主线采用FP8 checkpoint对应方法；BF16+FP8 KV的scale=1兼容结果单列。不会要求实现者凭空给AWQ/BF16扩展默认侧车scale入口。

本地AGENTS更新当前量化入口和实施边界，保留TP历史结论与105关闭超时风险；原版备份于专题参考资料。需求目录和AGENTS均gitignore，**本提交只含本交付记录，普通push不会同步需求正文与冻结资料**。

## 静态验证

仓库根目录执行：

```bash
source scripts/activate-4090.sh
python learning_notes/14_vllm_from_scratch/量化完整需求/参考资料/check_requirements.py
python learning_notes/14_vllm_from_scratch/分布式并行完整需求/参考资料/check_requirements.py
```

- 量化需求检查通过：27文档、24关、198个R项、121组T项、849份源码hash/AST、295个相对链接、依赖DAG、模型身份及70配置/210case边界，0错误。
- 旧TP需求检查最终通过：7,911项检查，0失败；旧历史未解析链接保持原有533条，没有本次新增坏链接。初次检查发现备份AGENTS的根目录相对链接被当作新需求链接，已保留原内容为.txt档案后复查通过。
- 在临时副本故意破坏源码hash、相对链接和矩阵唯一性，校验器均正确返回失败；未修改原冻结文件。
- 任务开始前已有的11份未提交/未跟踪文件逐份sha256复核均未改变，包含Qwen3、step108 oracle及既有报告；没有混入本提交。

未运行pytest或GPU benchmark：本次没有生产代码修改，未来关卡的测试/benchmark脚本尚属需求。没有更新生产包hash或将文档静态检查写入实现通过记录。

## 使用方式

交给Terra时从本地量化目录000/001/002和113开始。先冻结真实工作区基线与上游能力，再逐关实现；新R项按AGENTS独立提交并立即push。历史HEAD、当前dirty Qwen3修改和未来量化结果不能混为同一数值证据。
