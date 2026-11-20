# step123：AWQ 单卡闭环

实施中；以下为编码前的需求映射，不代表验收通过。

| 需求 | 拟改造位置与验证 |
| --- | --- |
| R123.01 | `VllmConfig` 支持边界、`BaseModelLoader`、Qwen3 V2 Runner；真实 8B/14B 前向 |
| R123.02 | `BaseModelLoader`、`process_weights_after_loading`；检查每层方法、参数和 READY |
| R123.03 | 独立进程采集 mini/vLLM 同输入逐层输出及 logits |
| R123.04 | 复用 Scheduler/Runner 状态协议，覆盖混合批、分块、prefix、取消和抢占 |
| R123.05 | 单卡 14B 记录加载峰值、参数、workspace、KV 和剩余显存 |
| R123.06 | 加载与执行故障注入，核对 failure 和关闭 |
| R123.07 | 同源 BF16 质量比较；公开 AWQ 的源 revision 和校准数据未知，不能冒称同源 |
| R123.08 | 配置期拒绝尚未开放的 Graph、TP、投机及其他量化方法 |

性能对照按用户要求留待大部分量化需求完成后集中进行。

## 当前实现与证据（未收口）

- 已接通 V2、CUDA、BF16、TP1、eager、无投机的 Qwen3 AWQ；权重全层后处理成功后才置 READY。表示审计仍不可执行，CPU、V1、图、TP 和投机明确拒绝。
- 单独进程在同一张 RTX 4090、TRITON_ATTN、相同 192 KV 块及相同 token 输入下，8B/14B 的 128 token 与冻结 vLLM 全部一致；前四次采样对应的最终 hidden、logits、首层 KV 最大绝对差均为 0。该结果是一条输入，不能扩大成全面数值验收。
- 14B 测得参数 9,976,576,000 字节、KV 503,316,480 字节；推理后统计的 CUDA 峰值 allocated 11,571,880,960 字节、reserved 12,626,952,192 字节。这个峰值覆盖加载和运行，尚未拆分每种 workspace 或记录剩余显存。
- 真实权重为公开 AWQ，与所指定 BF16 的源 revision 及校准数据未建立同源，故不声称同源质量结论。
- R123.03 批量 teacher forcing 未达默认误差门槛；R123.04/06 已有真实 8B 的生命周期与单 rank 执行故障证据，但覆盖尚不完整；R123.05 完整计账、R123.07 同源质量、点名 benchmark 与全部多卡回归尚未完成。性能测试按用户要求延后。

命令：`CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python tests/step123/step123_model_probe.py --model {8b,14b} --tp 1 --batch 1 --output <目录>`，及相同参数调用 `step123_frozen_probe.py --repeats 1`；四次分别运行。结果位于本地不跟踪的 `benchmarks/results/step123/`。`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/step123 tests/step115 tests/step119 tests/step122 -q -ra`：79 passed。

## 后续逐层、批量及生命周期证据

- 8B/14B 各一个 128 行 prefill 的 embedding、所有 Decoder 层输出和 residual、最终 norm，与冻结 vLLM 精确相等，分别比较 75/83 个 tensor。8B 的另一条 32 token 单请求固定历史也逐层及 logits 零差。代码与结果分别为 `tests/step125/step125_{model,frozen}_probe.py --layer-audit`、`tests/step123/step123_compare_layers.py` 和本地 `benchmarks/results/step123/layerwise/`。这是同 token 输入的逐层证据，不代替所有不同批次。
- 固定四条不同输入历史并并发：两模型的四个 greedy token 都与冻结上游相同，但完整 logits 不满足 `rtol=1e-2, atol=5e-2, mean_abs<=1e-2` 起始门槛。8B 四个请求的 mean_abs 为 0.10537、0.08424、0.04718、0.04926；14B 为 0.10515、0.07716、0.06611、0.05814。原始结果在本地 `benchmarks/results/step123/teacher_forcing/`，`tests/step123/step123_compare_teacher.py` 按门槛返回失败。
- 本批次中 mini 首轮处理 128 行，而 vLLM 首轮为 65 行；同一冻结 vLLM 重复这组请求，部分请求 logits 的 mean_abs 差达 0.07989、0.02900。8B 同输入独立运行时逐层零差。这些现象说明 batch 组成和上游重复噪声值得继续核查，**尚不足以证明本项目无误**，不能据此放宽门槛或标记验收通过。
- 真实 8B AWQ TP1/TP2 的 prefix、抢占、两种取消及 rank1 采样故障见 `docs/step125_alignment.md`；不能代替 14B 或所有故障场景。
- 审计发现 8B 共 144 个 AWQ 线性层，14B 共 160 个，TP1/TP2 各 rank 全部 READY、kernel 布局；14B TP1 量化参数 6,864,076,800 字节，workspace 合计 81,920 字节。权重参数总数与 KV、峰值见上文。
