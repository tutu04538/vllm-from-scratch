# 107：EAGLE3 分片模型与词表映射

状态：107 草稿模型分片及限定范围验收完成。106 单卡主路径已建立真实参考；106 历史 stream 竞争与 GPU0 显存不足分别保留在其报告中。

本关生产包 hash：`6c63b3ca0f81ebdbf8ec214c3e1fd150f80db570d692667a9ede805d94ee0fe3`（递归 Python 文件按相对路径排序后，对 `路径\0sha256\n` 求 SHA256；后续改动必须重算）。

## 冻结源码、布局与所有权

| 项目位置 | 冻结 vLLM 0.28.0 | 职责与布局 |
|---|---|---|
| `models/qwen3_eagle3.py::Eagle3Attention` | `models/llama_eagle3.py::LlamaDecoderLayer`、`models/llama.py::LlamaAttention` | Q/K/V 以草稿的总 head 数构造 QKV；每 rank 使用本地 head 数和本地 Q/KV 宽度，KV head 不足时按冻结规则复制 |
| `models/qwen3_eagle3.py::Eagle3Model`、`layers/linear.py` | `llama_eagle3.py::LlamaModel.__init__`、`linear.py::ReplicatedLinear` | 特征 FC 和 norm 各 rank 复制；QKV/Merged 为列分片，O/down 为行分片；层号和权重源不变 |
| `models/qwen3_eagle3.py::Eagle3ForCausalLM.compute_logits` | `Eagle3LlamaForCausalLM.compute_logits`、`layers/logits_processor.py` | 各 rank LM head 先按词表分片、all-gather、裁剪 padding，得到全局 draft 列后按 `arange+d2t` 映射到 target 词表 |
| `worker/gpu/spec_decode/eagle/utils.py::load_eagle_model` | 同名 `_should_share/load_eagle_model` | target/draft 使用同一执行 TP 组；embedding 仅共享同 rank 参数；独立缩减词表 head 不覆盖 |

Scheduler 仍拥有 Request 和逻辑 KV；每个 worker 仅拥有本地模型分片、设备 KV 和执行镜像。107 验证模型及词表协议，不开放多 rank 投机整条链（108），不把相同 draft 复制到 rank0 模拟 TP。

## 当前已做与尚缺证据

- 草稿 attention 按 QKV 层给出的本地 Q/KV head 数拆分，FC 使用本项目的非量化 `ReplicatedLinear`；EAGLE packed 权重加载检查缺失与重复 shard。旧 step63 测试的 FC 尺寸属性改用新层的 `input_size/output_size`，维度断言未放宽。
- `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/step107 -q -ra --maxfail=1`：3 passed；TP2/4 真实 NCCL 检查包括草稿 QKV 拆分、RoPE、输出投影和跨分片缩减词表映射。attention 主体为测试替身，不是实际 KV/attention 验收。
- `CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/step63 tests/step64 tests/step67 tests/step72 tests/step74 tests/step106 -q -ra --disable-warnings --maxfail=1`：170 passed、11 skipped；旧模型权重缺失的 skip 不计为实测通过。
- 固定真实 8B 和 14B 草稿权重，`benchmarks/step107_draft_probe.py` 分别在 TP2/4 执行特征 FC、全词表汇总及映射后的 logits；各 rank 与 106 冻结上游对应记录逐值相等。8B 使用 106 真实 128 行特征，14B 使用 106 的固定合成特征。TP2/4 的 QKV 权重及输出与相同 TP 配置、hash 校验过的冻结上游 QKV 层逐值相等。8B 输入来自 106 实际历史，14B QKV 输入为固定合成值；不代表全层前向通过。记录见忽略目录 `benchmarks/results/step107/`。
- 8B 若只把真实 128 行 FC 输入截为 1 行再与原 128 行输出比较，BF16 最大绝对差为 32；恢复原 128 行计算后逐值相等。QKV 不同 TP 布局与 TP1 直接比较也有最大 0.25 的差异，同 TP 布局与冻结上游逐值相等。这些现象不能代替同布局全链路误差分析。
- 初次诊断：8B 冻结参考记录中的 prefill Q/K/V 按 rank 分片后喂给实际 Triton attention，TP2/4 的输出和写入 KV 均与对应的冻结记录切片逐值相等。整层重放时，106 的 post-hook 输入已经被 residual 原地更新；误用该输入造成 TP1 也有显著偏差。改用同轮捕获的 FC 输出作为层起始 hidden 后，TP1 整层输出和 residual 与参考逐值相等；TP2/4 对 TP1 的输出相对 L2 分别约 0.00694/0.00744、最大绝对差 6/8，residual 相对 L2 约 0.00153/0.00165。当时还没有相同 TP 布局的冻结整层对照；这些跨 TP 差异本身不能判断正确性。
- 2026-11-12 已补同 TP 的冻结引擎对照：真实 8B/14B 分别 TP2/4，用同一 128 token prefill 连续执行两次参考，按每 rank 注册参考重复及局部 FP32 重放后再运行项目模型。FC、各 rank QKV、Triton attention、写入 KV、草稿层输出及 residual、最终 norm 后的映射 logits 与参考均逐值相等。`CUDA_VISIBLE_DEVICES=2,3,0,1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmarks/check_step107_acceptance.py --case all --timeout 1800 --output benchmarks/results/step107/acceptance_final.json`：独立 pytest 5 passed；17 个子步骤全部通过。8B/14B 各自 TP2 的 16 项、TP4 的 32 项比较全部通过；每项严格比较包含有限性掩码及 dtype。对照从参考记录取得本轮实际 target 特征/embedding，手动绑定本地 KV 用于模块验收；这不验证请求状态跨轮闭环。
- 同配置冻结参考各 rank 两次逐值一致；验收门槛固定为逐值相等。FP32 重放用于量化参考自身 BF16 误差，不当作放宽对照的阈值。8B TP2/4 的 FC 首行相对 L2 为 0.003486/0.003142，14B TP2/4 为 0.004161/0.004245；草稿 LM head 首行相对 L2 约 0.00166。各 case 的完整统计、top1/top2/margin 与源文件 SHA 见 `benchmarks/results/step107/acceptance_final_runs/` 的 contract 和 comparison。
- 并行回归阶段记录：`CUDA_VISIBLE_DEVICES=2,3,0,1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest $(printf 'tests/step%03d ' $(seq 89 106)) -q -ra` 为 235 passed、1 failed。103 的参考引擎因父 pytest 已保留 CUDA context，启动时可用 20.06 GiB 低于 0.86 比例要求的 20.34 GiB，mini 未运行。这次回归期间还修改过 107 模型加载器，故不能声称是单一源码版本的最终回归。
- 对 103 真实回归提供显式参考预留比例参数，在测试中指定 0.82；两侧仍按固定 192 KV 块执行，新增块数对账，其余 token/KV/logits 断言不变。刻意令父 pytest 保留 CUDA context 后重跑该测试，1 passed（`benchmarks/results/step107/step103_persistent_context_final.log`）。旧 0.86 失败记录保留；这是环境容量调整，不改变精度或模型输入。
- 最新包完整并行回归：`CUDA_VISIBLE_DEVICES=2,3,0,1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest $(printf 'tests/step%03d ' $(seq 89 106)) -q -ra`，236 passed、0 failed、0 skipped；日志 `benchmarks/results/step107/parallel_hash6c_20261112.log`。完整旧回归 1289 passed、0 failed、23 skipped；skip 不算验收通过。
- **边界**：这些检查覆盖真实草稿固定 prefill 模块；本项目完整多卡投机仍在初始化时拒绝，跨轮拒绝状态与 Graph 分别留给 108/109。106 的历史 stream 竞争与 105 曾有的关闭超时不因本次通过自动消除。

## 108 修改 RMSNorm 后的复验

108 将 CUDA 融合 RMSNorm 的 residual 原地行为与冻结实现对齐后，重新运行真实草稿模块比较：8B TP2 与 14B TP2 使用参考默认的 0.75 显存预留，8B/14B TP4 因物理 GPU1 受其他进程占用，显式使用 `--gpu-memory-utilization 0.60`；四组同 TP 对照均通过，记录在 `benchmarks/results/step108/step107*_after_inplace.json`。首次按 0.75 跑 TP4 时因可用显存低于参考引擎初始化门槛而失败，保留在 `step107_after_inplace.json`；调整只改变参考 KV 显存预留，不改变模型、输入和数值比较。
