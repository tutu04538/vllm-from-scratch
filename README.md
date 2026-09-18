# vllm-from-scratch

从零逐步实现一个 vLLM 式推理引擎：请求调度、分页 KV 缓存、前缀缓存、打包输入、Triton attention、CUDA Graph、多头 / GQA。

每一关一个自包含文件 `stepNN.py`，只依赖 PyTorch（第 22 关起可选 Triton）。每关都在上一关的基础上改动，旧文件保留不动，方便对照。

## 环境

实测环境：conda 环境 `vllm-omni-dev`，PyTorch 2.13.0+cu130，Triton 3.7.1，RTX 5090 Laptop，FP32。

- 第 1–21 关：CPU 即可跑通。
- 第 22 关起：`attention_backend="triton"` 需要 CUDA；`"torch"` 后端仍可在 CPU 上做参考。
- 第 24 关起：`use_cuda_graph=True` 需要 CUDA + Triton，其他组合会明确报错。

## 目录

```text
stepNN.py        第 NN 关的实现，直接 python stepNN.py 可跑内置示例
benchmarks/      每关的性能脚本；results/ 下是实测记录（JSON，按时间戳命名）
docs/            每次改动的记录：需求、改动、设计要点、验证、遗留
```

## 关卡

| 关 | 文件 | 主题 |
|---:|---|---|
| 1 | `step01.py` | 单请求生成与停止 |
| 2 | `step02.py` | 一次提交多条请求 |
| 3 | `step03.py` | 每条请求轮流生成一个 token |
| 4 | `step04.py` | 请求完成就通知调用者 |
| 5 | `step05.py` | 生成中途接收新请求，`Engine` 生命周期 |
| 6 | `step06.py` | 限制运行数量与等待补位（容量调度） |
| 7 | `step07.py` | 一轮请求合成一次模型调用 |
| 8 | `step08.py` | 让模型真正读取上下文（`TinyCausalLM` + causal attention） |
| 9 | `step09.py` | 真实模型接入，不同长度请求一起生成 |
| 10 | `step10.py` | 单请求 KV 缓存与增量计算 |
| 11 | `step11.py` | 把 KV 缓存交给请求管理 |
| 12 | `step12.py` | 不同长度缓存一起做 decode |
| 13 | `step13.py` | Engine 接入批量 KV 缓存 |
| 14 | `step14.py` | 不同长度 prompt 一起 prefill 并返回缓存 |
| 15 | `step15.py` | KV 缓存预分配与原位追加 |
| 16 | `step16.py` | KV 缓存池的申请、归还与复用（分块缓存闭环） |
| 17 | `step17.py` | token 预算与分块 prefill 闭环 |
| 18 | `step18.py` | 跨请求前缀缓存与共享块管理 |
| 19 | `step19.py` | 无 padding 打包，prefill/decode 一次 forward |
| 20 | `step20.py` | slot mapping 与批量 KV 读写 |
| 21 | `step21.py` | 不拼完整历史，按块读 KV + online softmax |
| 22 | `step22.py` | 整批请求一次 Triton attention kernel |
| 23 | `step23.py` | attention 元数据固定缓冲、一次 H2D 上传 |
| 24 | `step24.py` | GPU forward 接入 CUDA Graph |
| 25 | `step25.py` | 多头 attention 与 GQA / MQA |

大致的推进脉络：

- **1–6 关**：调度与生命周期——从单请求走到动态到达 + 容量限制。
- **7–9 关**：把一批请求合成一次模型调用，并接上真实的 attention。
- **10–17 关**：KV 缓存——增量计算 → 请求级缓存 → 批量 decode/prefill → 预分配原位写 → 块池 → token 预算。
- **18 关**：前缀缓存，共享块与 LRU。
- **19–21 关**：输入与 KV 的存取方式——打包成一维、slot mapping、按块 attention。
- **22–24 关**：GPU 执行——Triton kernel、元数据缓冲、CUDA Graph。
- **25 关**：多头，以及多个 query head 共享 KV head。

`step17_refactor.py` 是第 17 关的一次重构版本，保留用于性能对照。

## 运行

```bash
python step25.py            # 跑该关内置的示例
python benchmarks/bench_step25_engine.py    # 跑性能脚本
```

## 文档

`docs/` 下每次改动一篇记录（命名 `stepNN_<主题>.md`），固定五节：需求大概、改动内容、设计要点、验证、接口变化与遗留。索引见 [docs/README.md](docs/README.md)。

已记录：第 18–25 关。
