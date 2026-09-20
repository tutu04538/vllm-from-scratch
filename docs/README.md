# docs —— 改动记录

每次改动一个文件：`docs/stepNN_<主题>.md`，与代码同名对应。

代码放在各自的子目录下：`stepNN/stepNN.py`（第 17 关的重构版是 `step17/step17_refactor.py`）。

每篇记录固定写这几节：

0. **需求大概**：这次要做什么、边界在哪，两三句话讲清。
1. **改动内容**：改了哪些方法、为什么。
2. **设计要点**：架构层面的事实，尤其是容易被后续改动破坏的约定。
3. **验证**：跑了什么、结论是什么；没跑的就写没跑。
4. **接口变化与遗留**：公开接口的变动、已知未做的部分。

## 索引

| 文档 | 对应代码 | 主题 |
|---|---|---|
| [step18_prefix_cache.md](step18_prefix_cache.md) | `step18.py` | 跨请求前缀缓存、共享块引用、LRU 淘汰 |
| [step19_packed_forward.md](step19_packed_forward.md) | `step19.py` | 无 padding 扁平打包、prefill/decode 一次 forward |
| [step20_slot_mapping.md](step20_slot_mapping.md) | `step20.py` | slot mapping、整批 KV 写入与向量化读取 |
| [step21_block_attention.md](step21_block_attention.md) | `step21.py` | 按块读 KV、online softmax 合并 attention |
| [step22_triton_attention.md](step22_triton_attention.md) | `step22.py` | 整批请求一次 Triton attention kernel |
| [step23_metadata_buffer.md](step23_metadata_buffer.md) | `step23.py` | attention 元数据固定缓冲、一次 H2D 上传 |
| [step24_cuda_graph.md](step24_cuda_graph.md) | `step24.py` | GPU forward 接入 CUDA Graph（按 N 缓存、replay） |
| [step25_multi_head_gqa.md](step25_multi_head_gqa.md) | `step25.py` | 多头 attention、GQA/MQA、KV 池按 KV head 存 |
| [step26_multi_layer_decoder.md](step26_multi_layer_decoder.md) | `step26.py` | 多层 decoder：RMSNorm、SwiGLU、逐层 KV |
| [step27_rope.md](step27_rope.md) | `step27.py` | RoPE 旋转 Q/K、缓存位置一致性 |
| [step28_model_dir.md](step28_model_dir.md) | `step28/step28.py` | 模型目录（config.json + safetensors）与 Engine 加载 |
| [step29_head_dim_qk_norm.md](step29_head_dim_qk_norm.md) | `step29/step29.py` | 独立 head_dim、Q/K Norm，对齐 Qwen3 结构 |
| [step30_external_qwen3_dir.md](step30_external_qwen3_dir.md) | `step30/`（包） | 外部 Qwen3 目录的配置/权重适配；代码按模块拆分 |
| [step31_real_qwen3_text.md](step31_real_qwen3_text.md) | `step31/`（包） | 接入真实 Qwen3-0.6B：BF16→FP32、tied 权重、可配置 EOS、文本入口 |
| [step32_bfloat16_inference.md](step32_bfloat16_inference.md) | `step32/`（包） | BF16 运行精度：精度边界、半显存、误差来源分析 |
| [step33_fused_rmsnorm.md](step33_fused_rmsnorm.md) | `step33/`（包） | 融合 RMSNorm：一个 Triton kernel、与非融合路径的 A/B |
| [step34_sample_rows_only.md](step34_sample_rows_only.md) | `step34/`（包） | 只为需要采样的行算 logits，M=0 与 Graph 按 (N,M) 分键 |
| [step35_sampling_beam_triton.md](step35_sampling_beam_triton.md) | `step35/`（包） | 采样策略与惩罚、beam search 与 KV 分叉、Triton 采样 kernel（**后两者已移除**，文档保留原文） |
| [step36_benchmark_gap.md](step36_benchmark_gap.md) | `benchmarks/`（无新增包） | 与 vLLM 同条件基准、六测点差距、profiler 归因与下一项改动 |
| [step36_remove_beam_triton_sampler.md](step36_remove_beam_triton_sampler.md) | `step35/`（就地删除） | 移除 beam search 与 Triton 采样 kernel：牵连面、验收影响、残留引用 |
