# 第 80 关模型能力矩阵

> 基线：2026-10-28，80A–80G 已自检；80H 做收口。
> 本表只描述**当前仓库代码与已跑命令能证明的范围**，不把 tiny/契约/注册当成真实模型支持。
> 图例：**对齐** = 有逐值差分；**部分** = 只覆盖声明的子集；**契约** = 只验证接口/复用协议；
> **未实现** = 没有生产代码；**待验** = 代码在，但缺本机无法完成的真实端到端。

## 1. 固定列

| 模型/配置 | 实际工厂分支 | 模块结构是否对齐 | 输入/KV协议 | tiny差分 | 真实checkpoint | 后端/group前置 | 下一任务 |
|---|---|---|---|---|---|---|---|
| Qwen3 MTP tiny（`tiny_gqa`，K=1/2/4） | V2：K=1 `MTPSpeculator`；K>=2 `MultiModuleMTPSpeculator`；V1：`MTPProposer` | 对齐：`Qwen3MTP` 多模块、`% num_mtp_layers`、共享 `lm_head` | 80A packed 输入 + 80B 拒绝后 KV 修复 + 80C 整段图；非 Q-only | 80A 43 项逐缓冲差分；80B 8 项独立重放；80C 9 项状态/采样；K=1/2/4 eager 与 plain 相同 | **未下载**真实 Qwen3 MTP；tiny 是现场生成材料 | Torch + CUDA；单 KV group；Triton 拒绝采样；V2 需 UVA | 接真实 Qwen3 MTP checkpoint；补真实草稿接受率 |
| Gemma4 assistant tiny（80E.E1） | V2 `Gemma4Speculator`；V1 `Gemma4Proposer` | 部分：assistant 模型层/Q-only/跨宽度 projection 已对齐；E1 不用 attention 替身 | 80D Q-only 源层映射 + 只读；复用 target cache，不分配第二份 | 80E.E1 10 项：形状/权重路由/masked embedder/Q 投影+q_norm+RoPE | assistant 77.7M 已下载；tiny 差分不依赖它 | Torch SDPA；sliding/full 两组；V1 多组图明确拒绝 | 保留 bf16 near-tie 记录；V2 centroid 小图仍待优化 |
| Gemma4 target + assistant 真实（80E.E2） | V1/V2 同上；target 文本栈 `Gemma4ForCausalLM` | 部分：文本 target/YOCO/Q-only 尾层/PLE/softcap 已接；vision/audio 未接 | 80D + 81 子集：2 组、异构 head_dim、packed shared slab、常量位置 | 80E.E2 12 项；tiny 跨 window/block 对 HF max diff 2.98e-7 | **已下载** target 4.63B/10.25GB + assistant 77.7M；float32 8-token target-only eager == 投机 graph；bf16 near-tie | Torch SDPA；35 层/2 组/20 Q-only 尾层；V1 多组图拒绝；V2 支持 | 81 的 Mamba/抢占矩阵/KV transfer/滑窗块回收；多模态 |
| Step3p5 MTP V1（80F） | V1 `Step3p5MTPProposer` | 部分：MTP 模型/权重路由/per-group metadata/分步 head；target 196B 未接 | V1 per-group block table/slot/metadata；每层自己的 KV，不共享 target | 80F 12 项中的 tiny hidden/logits 逐值、K=1/2/4 e2e、slot29/45、逐组不广播 | **已下载** 52 tensors/5.9GB MTP 子集并前向；196B target 未下载 | Torch；sliding/full 两组；真实 MTP 三层均 sliding；full 层 llama3 scaling 未实现 | 真实 196B target 端到端；fused MoE/expert 权重；full RoPE |
| Step3p5 MTP V2（80F §13.5） | V2：K=1 `MTPSpeculator`；K>=2 `MultiModuleMTPSpeculator` | 部分：通用 multi-module per-group 扩展、`spec_step_idx` 采样 head；无 Step3p5 专用类 | V2 逐组 AttentionGroup/metadata；每层映射到 shape 匹配的 target group | K=1/2/4 eager 与 plain 相同；K=2 `FULL_DECODE_ONLY` 与 plain 相同；spy 验证采样 step 0..K-1 | **未跑**真实 196B target 的 V2；真实 MTP 子集只做加载/前向 | Torch；多 group；V2 需 UVA；K=4 图未单独捕获 | 真实 target V2；K=4 图；UVA 大模型路径 |
| DeepSeek MTP 稀疏索引契约（80G） | `MTPSpeculator` 四 hook；内层只认 `model.model` | 契约：`set_skip_topk`/`compact_topk_indices` 顺序与压紧；`TopKIndexBuffer` 是教学对象 | 索引缓存 `[max_tokens, index_topk]`；prefill end 压紧，decode 复用；不涉及 KV group | 80G 6 项：四 hook、K=1 不压紧、缺接口、stale 复位、重排行与冻结 DeepSeek compact 差分 | **未接**真实 DeepSeek；无 MLA/indexer 权重 | 真实模型需要 MLA/sparse MLA/indexer kernel；教学对象纯 CPU | 独立接 DeepSeek V3.2/V4.1 的 MLA/indexer；80G 不等于模型支持 |
| DeepSeek V3.2/V4.1 真实 DSA/CSA2 | 无工厂分支（未实现） | 未实现：本地没有 DeepSeek 模型类/稀疏 MLA | 未接 | 无 | **未下载**；V4.1 需要 1M 上下文/FP4/CSA2 支持 | 需要 MLA + indexer + sparse MLA + 多 group/压缩 KV | 单独立项；不能由 80G 契约测试替代 |

## 2. 证据分层

```text
L1 接口/契约      80G 四 hook、TopKIndexBuffer
L2 输入布局       80A 43 项逐缓冲差分
L3 模块/KV        80B 独立重放、80D Q-only 映射、80F per-group
L4 图/生命周期    80C 整段图、80E.E2 真实 graph/eager
L5 真实权重       80E.E2 Gemma4、80F Step3p5 MTP 子集
L6 真实端到端     仅 Gemma4 E2；Step3p5 196B / DeepSeek 均未完成
```

**L1–L4 通过不等于 L6 通过。** 表中“下一任务”列就是缺口。

## 3. 不列入矩阵的材料

- `minivllm/testing/sparse_attention.py`：教学版 learned top-k 稀疏注意力，
  随机权重，不是生产模型，也不是 Step3p5/DeepSeek 的数值复刻；
- `minivllm/testing/mtp_sparse_index.py`：80G 教学 buffer，不是真实 indexer。

它们分别由 `test_sparse_attention_teaching.py`（5 项）和
`test_mtp_index_reuse_contract.py`（6 项）覆盖，不能计入真实模型能力。

## 4. 状态结论

```text
执行机制收口：80A/B/C + 80D + 80G + 80H 记录 -> 可进入 81
80 全部模型适配：仍需 Step3p5 196B、真实 Qwen3 MTP、DeepSeek DSA/CSA2
```

本表与 `docs/step80_alignment.md`、`docs/results.json` → `step80.subrequirements.80A…80H`
一起构成 80H 的收口证据。
