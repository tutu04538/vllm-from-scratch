# vllm-from-scratch

从零逐步实现一个 vLLM 式推理引擎：continuous batching、paged KV cache、prefix caching、sequence packing、Triton attention、CUDA Graph、multi-head / GQA、投机解码。

**当前状态：仓库只有一个实现包 `minivllm/`**（原 `stepNN/` 逐步演进的最后一关，含两轮验收修复）。
按"每关一个目录"的旧约定已停止：`step01`–`step56` 的代码目录已删除，最新代码直接住在 `minivllm/`，
后续改动只在它上面做。历史记录（每关做了什么、为什么）保留在 `docs/` 与 git 历史里。

包名不叫 `vllm` 是刻意的：`benchmarks/` 里的对照脚本要在**同一个进程**里 `import vllm`
（site-packages 的真 vLLM 0.28.0），目录同名会静默遮蔽它。

## 环境

实测环境：conda 环境 `vllm-omni-dev`，PyTorch 2.13.0+cu130，Triton 3.7.1，RTX 5090 Laptop，FP32。

- CPU 可以跑通全部回归脚本与 `--device cpu` 的短生成；
- `--device cuda` 与真实 vLLM 对照需要 CUDA 环境。

## 目录

```text
minivllm/         唯一的实现包（对齐 vLLM V1 的文本生成子集）；入口 python minivllm/demo.py
                  README 里有"层 ↔ vLLM 模块"对照表与用法
fixtures/         验收用的外部模型目录（随机初始化的 Qwen3 tiny 模型，非预训练权重）
models/           本机真实模型（Qwen3-1.7B 等），不进版本控制
benchmarks/       回归脚本与对照脚本（跟踪）+ results/ 实测产物（**不跟踪**，见 .gitignore）
docs/             每次改动的记录：需求、改动、设计要点、验证、遗留；索引见 docs/README.md
```

## 实现概览

`minivllm/` 的分层与 vLLM V1 一一对应（细节与差异账本见 `minivllm/README.md`）：

```text
LLMEngine → InprocClient → EngineCore（调度 → 执行 → 采样 → 更新）
                            ├─ Scheduler / KVCacheManager（控制面：块表、前缀缓存、抢占）
                            └─ UniProcExecutor → Worker → GPUModelRunner（执行面：输入打包、KV 写入、采样）
spec_decode/                投机：ngram 与 draft model、拒绝采样、草稿时序
```

历史推进脉络（每一步的设计与验证都有一篇文档）：

- **1–6**：调度与生命周期；**7–9**：一批请求合成一次模型调用；**10–17**：KV 缓存到块池与 token 预算；
- **18**：前缀缓存；**19–21**：打包、slot mapping、按块 attention；**22–24**：Triton、元数据缓冲、CUDA Graph；
- **25**：多头与 GQA；**26–30**：真实模型结构、RoPE、模型目录；**31–35**：真实权重、BF16、算子融合、采样策略；
- **36–42**：性能诊断与关键路径；**43–51**：增量输出、抢占恢复、优先级、空闲块队列、增量历史；
- **52–57**：投机解码（ngram → 批量 → 随机 → draft model → GPU 拒绝采样）与对齐 vLLM V1 的架构重构。

## 运行

```bash
python minivllm/demo.py                                  # 默认 models/Qwen3-1.7B + 默认问题
python minivllm/demo.py --device cpu --max-new-tokens 32 "问题"
python minivllm/demo.py --scheduler-trace "同样的问题" "同样的问题"   # 看调度决策与前缀命中
```

## 测试（回归脚本）

```bash
python benchmarks/check_step57_engine_protocol.py    # 15 个脚本、354 项，全部可 CPU 运行
python benchmarks/check_step57_draft_model.py
python benchmarks/trace_step57.py                    # 一条覆盖五种事件的状态轨迹
```

与真实 vLLM 的对照（需要 GPU 与本机 Qwen3）：

```bash
VLLM_WSL2_ENABLE_PIN_MEMORY=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    python benchmarks/compare_step57_vllm.py
VLLM_WSL2_ENABLE_PIN_MEMORY=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    python benchmarks/check_step57_vllm_boundaries.py
```

`benchmarks/results/` 与 `*.log` 是本地实测产物，已在 `.gitignore` 里，不再跟踪。

## 文档

`docs/` 下每次改动一篇记录（历史命名 `stepNN_<主题>.md`），固定五节：需求大概、改动内容、设计要点、
验证、接口变化与遗留。索引见 [docs/README.md](docs/README.md)。
