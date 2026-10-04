# docs —— 改动记录

每次改动一个文件：`docs/stepNN_<主题>.md`（沿用历史命名；这里的 `stepNN` 只表示"第几批工作"，
目录不再是代码目录名）。

**代码位置（2026-10-03 起）**：唯一实现包是 `minivllm/`（原 `step57/`）。`step01`–`step56` 的代码目录
已删除，所以旧记录里出现的 `stepNN/...` 路径是**历史路径**，只在 git 历史里能检出；文档本身原样保留。

包名不叫 `vllm` 是刻意的：`benchmarks/` 的对照脚本要在同一进程里 `import vllm`（真 vLLM），同名会遮蔽。

顺带一条：**验收记录里按旧路径写的脚本**（`from step57 import ...`）需要改一行才能再跑，例如
`sed 's/from step57/from minivllm/g' 脚本.py > /tmp/probe.py`；仓库不再为旧路径保留 alias。
同理，`fixtures/` 也已删除：测试用的 tiny 模型改成现场生成，需要旧布局时用
`python -m minivllm.testing.tiny_models --out <目录>` 重建，再把脚本里的模型路径指过去。

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
| [step50_idle_lru_index.md](step50_idle_lru_index.md) | `step50/`（包） | 闲置缓存改 vLLM 式单链表 LRU 索引：淘汰不再扫全池，8192 块从 491 μs 降到 0.66 μs |
| [step51_incremental_history.md](step51_incremental_history.md) | `step51/`（包） | 增量维护完整 token 历史：`_plan_tokens` 13×、发布 32×，并把 prefix 登记时机与 preemption_mode 解耦 |
| [step52_ngram_speculative.md](step52_ngram_speculative.md) | `step52/`（包） | 单请求贪心 n-gram 投机解码：草稿 → 一次 forward 验证 K+1 行 → 只提交认可的 → 回滚 KV；抢占改无条件（对齐 vLLM V1） |
| [step53_batched_speculative.md](step53_batched_speculative.md) | `step53/`（包） | 批量投机验证与抢占恢复：行映射两个坐标系、真实 token 优先的预算、先缩草稿再抢占 |
| [step54_random_speculative.md](step54_random_speculative.md) | `step54/`（包） | 随机采样投机解码：拒绝采样 + 纠正分布 + 逐行惩罚历史 + 随机数归请求 |
| [step55_draft_model.md](step55_draft_model.md) | `step55/`（包） | draft model 双 KV：一般 p/q 拒绝采样 + 独立提议层 + 双池容量与对齐 + 分片加载 |
| [step56_gpu_rejection.md](step56_gpu_rejection.md) | `step56/`（包） | GPU 批量拒绝采样：counter-based RNG、两个内核、一张结果张量一次回传；与 CPU oracle 逐位对照 |
| [step56_triton_kernel_design.md](step56_triton_kernel_design.md) | `step56/`（附篇，无代码改动） | 把验证搬进 kernel 的设计思路：先讲清 counter RNG 与指数竞赛两个名词，再讲原做法差在哪、内核担什么、要备哪些输入、易错点与代价 |
| [step56_state_space.md](step56_state_space.md) | `step56/`（附篇，无代码改动） | 这一关的状态空间地图：五层轴、把不可能组合删掉的不变量、每个函数实际要装几个量、容易混的轴 |
| [step56_vs_vllm_rejection.md](step56_vs_vllm_rejection.md) | `step56/`（附篇，无代码改动） | 与 vLLM 0.28 拒绝采样的逐项对照：同样的密度、不同的取舍，以及哪些复杂度是需求逼出来的 |
| [step57a_skeleton.md](step57a_skeleton.md) | `step57/`（**新包**） | 对齐 vLLM V1 架构的竖直骨架：Engine/Executor/Worker/Runner 边界、协议数据包、统一预算调度；57A 只做这一段 |
| [step57b_real_model.md](step57b_real_model.md) | `step57/`（在 57A 骨架上继续长） | 把真实模型接进协议：Qwen3（GQA + q/k norm + RoPE）、三层权重加载（打包路由 + 覆盖检查）、Attention 边界与分页 KV、Runner 输入打包（198 §4 逐值）、full/chunk/decode 与 HF 逐位置对照；单卡 eager、只支持 TP=1 |
| [step57c_kv_and_prefix.md](step57c_kv_and_prefix.md) | `step57/`（KV 这条链重写） | 真实 KV 控制面：块池（引用计数 + O(1) 空闲队列 + 同 hash 多块）、前缀缓存（链式 hash、发布边界、共享不覆写）、priority 与重算式抢占（计划撤销/预算退回/恢复整表替换）、可打印 scheduler_trace |
| [step57d_sampling_and_stop.md](step57d_sampling_and_stop.md) | `step57/sample/`（新增） | 普通采样与停止：按行的 SamplingMetadata、min_tokens 的两处职责（采样侧屏蔽 vs 调度侧结束）、三种惩罚与 top-k/top-p 边界、指数竞赛抽样、用户增量输出与 stop_reason |
| [step57e_speculative.md](step57e_speculative.md) | `step57/spec_decode/`（新增） | 投机回接：SpecDecodeMetadata 的两个坐标系、min(1,p/q) 验证与 max(p−q,0) 恢复、「轮 t 提议 → 轮 t+1 采用」的时序、被拒草稿的进度回退、ngram 与真实 draft 模型（同 KV group、每层各自 tensor、规格不兼容明确报错） |
| [step57_architecture.md](step57_architecture.md) | `step57/`（对照关，无代码改动） | 回到真实源码：分层与状态归属、一条覆盖五种事件的状态轨迹、194 §8 的七问×五事件、与真实 vLLM 的数学对照（logits 2.4e-7 / 端到端 fp32 逐 token 一致） |
| [step57_alignment.md](step57_alignment.md) | `step57/`（对照关，无代码改动） | 57 关的差异账本汇总：按 vLLM 模块归类、每项九段（本机做法/本项目做法/为何简化/影响/对应测试/何时消除） |
| [step57_acceptance_fixes.md](step57_acceptance_fixes.md) | `step57/`（验收修复） | 独立探针抓到的 6 个问题：KV 上的 autograd 图（显存随步数涨）、提议者越界写 lookahead 槽位、CUDA 设备、提议与记账顺序、混批 ragged K 的展开、seed 被全局 RNG 污染；逐条按 vLLM 修 + 补回归用例 |
| [step57_draft_lockstep.md](step57_draft_lockstep.md) | `step57/`（验收修复·续） | 204 §6.2 的第二个选项换成第一个：drafter 每轮与 target 跑同一段位置（中间 prefill 块只同步 KV、不提草稿），于是**撤销**发布边界的 `min(target, draft)` 夹取，回到与 vLLM 相同的"只按 target 发布"；不变量改由用例盯着（发布边界 ≤ draft 进度，且发布位置上的 draft KV 非零） |
| [step57_lifecycle.md](step57_lifecycle.md) | `step57/`（验收修复·再续） | 205 的三类收尾：草稿写入同时过**逻辑上界 + 物理槽位**（块表容量向上取整不能当模型长度）；请求三态（未调度保留 / 恢复重置 / finished 删除）与"草稿只活一轮"的 q 契约；sample/propose 异常 → **失败态**，下一轮在调度前被拒绝；补 `model.eval()` 与七条回归用例 |
| [step58_alignment.md](step58_alignment.md) | `minivllm/` | 第五十八关：draft 第一遍按上游扩容分支组织输入（padded 物理行 + is_rejected 掩码 + slot 哨兵 + `extend_all_queries_by_N`），prefix 命中段不再重算；Scheduler 双预算（token + input）与抢占返还；固定输入工作区（`data_ptr` 稳定）；与上游 Triton kernel 逐值差分；`docs/step58_results.json` / `step58_models.json` 记录实测与模型 |
| [step59_alignment.md](step59_alignment.md) | `minivllm/sample/` | 第五十九关：拒绝采样搬到上游位置（`sample/rejection_sampler.py`）并换成 **Triton 批量内核**（greedy/random 两条内核 + recovered token 内核 + expand 内核），`SpecDecodeMetadata` 换成上游 7 字段（GPU int32、构造收进 Runner 的 `_calc_spec_decode_metadata`），`SpecDecodingStats` 只统计已验证候选；逐候选 D2H 385 → 0（B=32/K=5）；与上游同层函数注入同一随机数逐值差分，Torch 版参考实现移到 `minivllm/testing/`；`docs/step59_results.json` / `step59_models.json` 记录实测与模型 |
| [step60_alignment.md](step60_alignment.md) | `minivllm/spec_decode/` | 第六十关：CPU ngram 校准到上游逐值一致（最长后缀匹配、**同长度取最早那处**、`prompt_lookup_min/max` 窗口、`k` 的两个上限、跳过规则），新增 GPU ngram（`NgramGPUKernel` 一个 batch 一次匹配 + `NgramProposerGPU` 的**显存常驻历史 + 增量写入**，输出固定宽度 `[B,K]` + 每行有效个数）；调度侧交接处按有效数裁占位（`update_scheduler_for_invalid_drafts`），哨兵 `-1` 不出门；每步 D2H 与历史长度无关；`docs/step60_results.json` / `step60_models.json` 记录实测与模型 |
| [step61_alignment.md](step61_alignment.md) | `minivllm/spec_decode/` | 第六十一关：接入外部包 **arctic_inference** 的 Suffix Decoding（上游钉 0.1.1、本机装 0.3.0，suffix_decoding 子系统源码逐字节相同）：请求内 prompt 树 + **跨请求全局树**、候选长度按匹配长度与频次**动态**（`max_spec_factor` / `min_token_prob`）、全局缓存按请求条数 FIFO 淘汰（0 = 关全局树）、同 ID 重用先 evict 再建树；§3 六步调用顺序用 spy 断言；与上游 proposer 逐事件 trace 差分（候选 + 缓存状态，21 条轨迹 0 差异）；依赖清单 `docs/step61_dependencies.json`，实测 `docs/step61_results.json` / `step61_models.json` |
