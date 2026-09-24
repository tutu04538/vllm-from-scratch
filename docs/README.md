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
| [step37_query_tiled_prefill_attention.md](step37_query_tiled_prefill_attention.md) | `step37/`（包） | query 分块的 prefill attention：`tl.dot`、tile 归属、图缓存键与路径回退 |
| [step38_fused_rope.md](step38_fused_rope.md) | `step38/`（包，新增 `rope.py`） | 融合 RoPE：一次 forward 一个 kernel、`fp_fusion` 的取舍与量化 |
| [step39_merged_proj.md](step39_merged_proj.md) | `step39/`（包） | 合并 QKV 与 gate/up 投影：融合参数、每层 7 → 4 次 launch、装载时兼容旧三键 |
| [step40_kv_on_demand.md](step40_kv_on_demand.md) | `step40/`（包） | KV 按需分配：准入承诺额度 + 生长时补块、不可能完成时明确拒绝 |
| [step41_short_critical_path.md](step41_short_critical_path.md) | `step41/`（包） | 缩短 decode 一步的关键路径：slot mapping 改 CPU 侧算，步墙钟 −18% |
| [step42_bandwidth_account.md](step42_bandwidth_account.md) | **无新增代码**（归因关） | decode 一步的带宽账：840 MiB 四种投影的有效带宽，结论按证据强度收窄 |
| [step43_incremental_output.md](step43_incremental_output.md) | `step43/`（包） | 增量输出观察点 `on_token`：每产生一个 token 通知一次，旧接口不变 |
| [step44_recompute_preemption.md](step44_recompute_preemption.md) | `step44/`（包） | 重计算式抢占：容量超卖、从 running 尾部选犧牲者、历史按 `all_token_ids` 重放 |
| [step45_blocker_aware_resume.md](step45_blocker_aware_resume.md) | `step45/`（包） | 阻塞者感知的恢复准入：记住是谁迫使让路，它结束前不急着恢复 |
| [step46_resume_prefix_cache.md](step46_resume_prefix_cache.md) | `step46/`（包） | 抢占恢复与前缀缓存整合：恢复时复用仍在缓存里的完整块，只重算剩下的历史 |
| [step47_priority_scheduling.md](step47_priority_scheduling.md) | `step47/`（包） | 优先级调度与抢占：名额/容量/token budget 三种资源都按 `(priority, arrival_order)` |
| [step48_refactor_kv_scheduler.md](step48_refactor_kv_scheduler.md) | `step48/`（包） | 行为不变重构：请求状态拆到 `request.py`，KV 准入/补块改「先计划后提交」，调度拆成阶段 |
| [step49_free_block_queue.md](step49_free_block_queue.md) | `step49/`（包） | 空闲 KV 块改队列增量维护：真正空闲路径不再随池子线性扫描 |
