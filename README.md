# vllm-from-scratch

从零逐步实现一个 vLLM 式推理引擎：continuous batching、paged KV cache、prefix caching、sequence packing、Triton attention、CUDA Graph、multi-head / GQA。

每一关一个自包含的目录 `stepNN/`，只依赖 PyTorch（第 22 关起可选 Triton）。每关都在上一关的基础上改动，旧目录保留不动，方便对照。第 30 关起实现按模块拆分，不再是一个大文件。

## 环境

实测环境：conda 环境 `vllm-omni-dev`，PyTorch 2.13.0+cu130，Triton 3.7.1，RTX 5090 Laptop，FP32。

- 第 1–21 关：CPU 即可跑通。
- 第 22 关起：`attention_backend="triton"` 需要 CUDA；`"torch"` 后端仍可在 CPU 上做参考。
- 第 24 关起：`use_cuda_graph=True` 需要 CUDA + Triton，其他组合会明确报错。

## 目录

```text
stepNN/          每关一个子目录；第 1–29 关是单个 stepNN/stepNN.py，第 30 关起拆成多个模块，
                  入口仍是 stepNN/stepNN.py，直接 python stepNN/stepNN.py 可跑内置示例
                  （第 17 关的重构版是 step17/step17_refactor.py）
fixtures/        验收用的外部模型目录（第 30 关：随机初始化的 Qwen3 tiny 模型，非预训练权重）
                 第 31 关用本机真实 Qwen3-0.6B，目录不在本仓库内，只读使用
benchmarks/      每关的性能脚本；results/ 下是实测记录（JSON，按时间戳命名）
docs/            每次改动的记录：需求、改动、设计要点、验证、遗留
step36/          例外：这两关是诊断关，没有代码包，也不放实现；各只有一份 README
step42/          说明该关结论与数据位置（实测数据仍在 benchmarks/results/ 下）
```

## 关卡

| 关 | 文件 | 主题 |
|---:|---|---|
| 1 | `step01/step01.py` | 单请求生成与停止 |
| 2 | `step02/step02.py` | 一次提交多条请求 |
| 3 | `step03/step03.py` | 每条请求轮流生成一个 token |
| 4 | `step04/step04.py` | 请求完成就通知调用者 |
| 5 | `step05/step05.py` | 生成中途接收新请求，`Engine` 生命周期 |
| 6 | `step06/step06.py` | 限制运行数量与等待补位（容量调度） |
| 7 | `step07/step07.py` | 一轮请求合成一次模型调用 |
| 8 | `step08/step08.py` | 让模型真正读取上下文（`TinyCausalLM` + causal attention） |
| 9 | `step09/step09.py` | 真实模型接入，不同长度请求一起生成 |
| 10 | `step10/step10.py` | 单请求 KV 缓存与增量计算 |
| 11 | `step11/step11.py` | 把 KV 缓存交给请求管理 |
| 12 | `step12/step12.py` | 不同长度缓存一起做 decode |
| 13 | `step13/step13.py` | Engine 接入批量 KV 缓存 |
| 14 | `step14/step14.py` | 不同长度 prompt 一起 prefill 并返回缓存 |
| 15 | `step15/step15.py` | KV 缓存预分配与原位追加 |
| 16 | `step16/step16.py` | KV 缓存池的申请、归还与复用（分块缓存闭环） |
| 17 | `step17/step17.py` | token 预算与分块 prefill 闭环 |
| 18 | `step18/step18.py` | 跨请求前缀缓存与共享块管理 |
| 19 | `step19/step19.py` | 无 padding 打包，prefill/decode 一次 forward |
| 20 | `step20/step20.py` | slot mapping 与批量 KV 读写 |
| 21 | `step21/step21.py` | 不拼完整历史，按块读 KV + online softmax |
| 22 | `step22/step22.py` | 整批请求一次 Triton attention kernel |
| 23 | `step23/step23.py` | attention 元数据固定缓冲、一次 H2D 上传 |
| 24 | `step24/step24.py` | GPU forward 接入 CUDA Graph |
| 25 | `step25/step25.py` | 多头 attention 与 GQA / MQA |
| 26 | `step26/step26.py` | 多层 decoder：RMSNorm、SwiGLU、逐层 KV |
| 27 | `step27/step27.py` | RoPE 旋转 Q/K、缓存位置一致性 |
| 28 | `step28/step28.py` | 模型目录与权重加载 |
| 29 | `step29/step29.py` | 独立 head_dim 与 Q/K 归一化 |
| 30 | `step30/`（包，入口 `step30/step30.py`） | 读取外部 Qwen3 模型目录；代码按模块拆分 |
| 31 | `step31/`（包，入口 `step31/step31.py`） | 接入真实 Qwen3-0.6B：BF16 加载、tied 权重、EOS 配置、文本入口 |
| 32 | `step32/`（包，入口 `step32/step32.py`） | BF16 运行精度：显式 FP32 累计边界、半显存 |
| 33 | `step33/`（包，入口 `step33/step33.py`） | 融合 RMSNorm：一个 Triton kernel 顶掉一串小算子 |
| 34 | `step34/`（包，入口 `step34/step34.py`） | 只为需要采样的行算 logits：lm_head 只处理 M 行 |
| 35 | `step35/`（包，入口 `step35/step35.py`） | 采样策略（top-p / 三种惩罚）；beam search 与 Triton 采样 kernel 已移出主线 |
| 36 | `step36/`（只有 README） | 诊断关：与 vLLM 的单卡同条件基准、六测点差距与 profiler 定位 |
| 37 | `step37/`（包，入口 `step37/step37.py`） | query 分块的 prefill attention：一个 program 处理多行 query，QK/PV 走 `tl.dot` |
| 38 | `step38/`（包，入口 `step38/step38.py`） | 融合 RoPE：一次 forward 一个 kernel，每次调用 12 → 1 个 GPU kernel |
| 39 | `step39/`（包，入口 `step39/step39.py`） | 合并 QKV 与 gate/up 投影：每层投影 7 → 4 次 |
| 40 | `step40/`（包，入口 `step40/step40.py`） | KV 按需分配：块随进度增长，容量不可能满足时明确拒绝而非空转 |
| 41 | `step41/`（包，入口 `step41/step41.py`） | 缩短 decode 关键路径：slot mapping 在 CPU 侧按块区间展开，不再每请求每步 H2D |
| 42 | `step42/`（只有 README） | 归因关：decode 一步的带宽账（有效带宽口径），结论按证据强度收窄 |
| 43 | `step43/`（包，入口 `step43/step43.py`） | 增量输出观察点 `on_token`：不必等整条请求结束就能收到新 token |
| 44 | `step44/`（包，入口 `step44/step44.py`） | 重计算式抢占与恢复：超卖准入 + 尾部犠牲者，被抢占者重放历史后继续 |
| 45 | `step45/`（包，入口 `step45/step45.py`） | 阻塞者感知的恢复准入：不急着恢复刚被抢占的请求，反复抢占 17 → 4 次 |
| 46 | `step46/`（包，入口 `step46/step46.py`） | 抢占恢复与前缀缓存整合：恢复时先复用仍在缓存里的完整块 |
| 47 | `step47/`（包，入口 `step47/step47.py`） | 优先级调度与抢占：名额、KV 块、token budget 三种资源都体现优先级 |
| 48 | `step48/`（包，入口 `step48/step48.py`） | 行为不变重构：`request.py` 分离、KV 计划/提交、调度分阶段（为投机解码铺路） |
| 49 | `step49/`（包，入口 `step49/step49.py`） | 空闲 KV 块改队列增量维护：补块不再扫全池，`_plan_block_growth` 与池子大小无关 |
| 50 | `step50/`（包，入口 `step50/step50.py`） | 闲置缓存改 vLLM 式单链表 LRU 索引：淘汰不再扫全池，8192 块 491 μs → 0.66 μs |
| 51 | `step51/`（包，入口 `step51/step51.py`） | 增量维护完整 token 历史：只读视图 + 单一写入点；prefix 登记时机与 `preemption_mode` 解耦 |
| 52 | `step52/`（包，入口 `step52/step52.py`） | 单请求贪心 n-gram 投机解码：草稿只进本轮计划，目标模型一次 forward 验 K+1 行，拒绝的 KV 撤回；抢占改无条件，删掉 `preemption_mode` / `over_subscribe` |
| 53 | `step53/`（包，入口 `step53/step53.py`） | 批量投机验证与抢占恢复：不同 K 的投机请求 + 普通 decode + 中间 prefill 同批，先缩草稿再抢占 |
| 54 | `step54/`（包，入口 `step54/step54.py`） | 随机采样投机解码：拒绝采样 + 纠正分布 + 逐行惩罚历史；采样参数不再受限 |

大致的推进脉络：

- **1–6 关**：调度与生命周期——从单请求走到动态到达 + 容量限制。
- **7–9 关**：把一批请求合成一次模型调用，并接上真实的 attention。
- **10–17 关**：KV 缓存——增量计算 → 请求级缓存 → 批量 decode/prefill → 预分配原位写 → 块池 → token 预算。
- **18 关**：前缀缓存，共享块与 LRU。
- **19–21 关**：输入与 KV 的存取方式——打包成一维、slot mapping、按块 attention。
- **22–24 关**：GPU 执行——Triton kernel、元数据缓冲、CUDA Graph。
- **25 关**：多头，以及多个 query head 共享 KV head。
- **26–30 关**：真实模型结构——多层 decoder、RoPE、模型目录的存取、独立 head_dim 与 Q/K Norm，最后能直接吃下别人导出的 Qwen3 目录。
- **31 关**：接上真实 Qwen3-0.6B 与 tokenizer，从一句话生成到一句话。
- **32 关**：同一模型可选 FP32 / BF16 运行，量化统计与 attention 累计留在 FP32。
- **33 关**：算子融合——RMSNorm 从 6/9 个 kernel 收成 1 个，norm 后端与 attention 后端独立可选。
- **34 关**：按需算 logits——先挑出要采样的行，再对这几行做最终 norm 与 lm_head。
- **35 关**：生成策略——按请求配置 greedy / random、beam search 的多候选与 KV 分支、最终选 token 的 Triton kernel。

`step17/step17_refactor.py` 是第 17 关的一次重构版本，保留用于性能对照。

## 运行

```bash
python step25/step25.py     # 跑该关内置的示例
python benchmarks/bench_step25_engine.py    # 跑性能脚本
```

## 文档

`docs/` 下每次改动一篇记录（命名 `stepNN_<主题>.md`），固定五节：需求大概、改动内容、设计要点、验证、接口变化与遗留。索引见 [docs/README.md](docs/README.md)。

已记录：第 18–54 关。
