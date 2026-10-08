> **交付物位置（有意偏离需求总纲 §4）**：总纲点名 `docs/stepNN_results.json` 与
> `docs/stepNN_models.json` 两份文件；为了让 `docs/` 根目录只留一个机器可读件，本项目把它们
> （以及 61 关的依赖锁定记录）**合并成一份 [`docs/results.json`](results.json)**，按关分键：
> `stepNN.results` / `stepNN.models` / `stepNN.dependencies`，字段口径与逐字内容不变。
> 这是格式与路径上的偏离，内容没有删减；需要拆回单文件时按这三个键拆即可。

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
| [step58_alignment.md](step58_alignment.md) | `minivllm/` | 第五十八关：draft 第一遍按上游扩容分支组织输入（padded 物理行 + is_rejected 掩码 + slot 哨兵 + `extend_all_queries_by_N`），prefix 命中段不再重算；Scheduler 双预算（token + input）与抢占返还；固定输入工作区（`data_ptr` 稳定）；与上游 Triton kernel 逐值差分；`docs/results.json` → `step58.results` / `step58.models` 记录实测与模型 |
| [step59_alignment.md](step59_alignment.md) | `minivllm/sample/` | 第五十九关：拒绝采样搬到上游位置（`sample/rejection_sampler.py`）并换成 **Triton 批量内核**（greedy/random 两条内核 + recovered token 内核 + expand 内核），`SpecDecodeMetadata` 换成上游 7 字段（GPU int32、构造收进 Runner 的 `_calc_spec_decode_metadata`），`SpecDecodingStats` 只统计已验证候选；逐候选 D2H 385 → 0（B=32/K=5）；与上游同层函数注入同一随机数逐值差分，Torch 版参考实现移到 `minivllm/testing/`；`docs/results.json` → `step59.results` / `step59.models` 记录实测与模型 |
| [step60_alignment.md](step60_alignment.md) | `minivllm/spec_decode/` | 第六十关：CPU ngram 校准到上游逐值一致（最长后缀匹配、**同长度取最早那处**、`prompt_lookup_min/max` 窗口、`k` 的两个上限、跳过规则），新增 GPU ngram（`NgramGPUKernel` 一个 batch 一次匹配 + `NgramProposerGPU` 的**显存常驻历史 + 增量写入**，输出固定宽度 `[B,K]` + 每行有效个数）；调度侧交接处按有效数裁占位（`update_scheduler_for_invalid_drafts`），哨兵 `-1` 不出门；每步 D2H 与历史长度无关；`docs/results.json` → `step60.results` / `step60.models` 记录实测与模型 |
| [step61_alignment.md](step61_alignment.md) | `minivllm/spec_decode/` | 第六十一关：接入外部包 **arctic_inference** 的 Suffix Decoding（上游钉 0.1.1、本机装 0.3.0，suffix_decoding 子系统源码逐字节相同）：请求内 prompt 树 + **跨请求全局树**、候选长度按匹配长度与频次**动态**（`max_spec_factor` / `min_token_prob`）、全局缓存按请求条数 FIFO 淘汰（0 = 关全局树）、同 ID 重用先 evict 再建树；§3 六步调用顺序用 spy 断言；与上游 proposer 逐事件 trace 差分（候选 + 缓存状态，21 条轨迹 0 差异）；依赖清单 `docs/results.json` → `step61.dependencies`，实测 `docs/results.json` → `step61.results` / `step61.models` |
| [step62_alignment.md](step62_alignment.md) | `minivllm/spec_decode/` | 第六十二关：自定义 Proposer 接入与**配置分派边界**——`method` 的推断只在 `SpeculativeConfig` 里判一次（点号路径 → `custom_class`、`ngram` → ngram、其余 → draft_model），Runner 只按 `method` 分派；`create_custom_proposer` 按上游逐条分类报错（无点号/模块不存在/类不存在/构造失败/`propose` 缺失或不可调用，异常链完整）并**返回实例本身不加套壳**；插件只拿到 `VllmConfig`（拿不到 Request/KVCacheManager）；空/全错/变长候选下 greedy 输出与非投机逐 token 相同；示例插件 `examples/custom_proposer.py` + demo 的 `--spec-model` 入口；`docs/results.json` → `step62.results` / `step62.models` 记录实测 |
| [step63_alignment.md](step63_alignment.md) | `minivllm/spec_decode/` | 第六十三关（**阶段 A/B/C 已完成并实跑**）：EAGLE/EAGLE3 的第一遍输入对齐——整体左移一格 + 按 `query_start_loc[1:] - 1` 打补丁（需求 §3 的 [a1,a2,b1,b2,b3]+[a3,b4] → [a2,a3,b2,b3,b4] 逐元素固定）、**特征与 positions 不动**、扩容行位置 = 该请求最后一行；与上游 `copy_and_expand_eagle_inputs_kernel(shift_input_ids=True)` 在 CUDA 上逐值差分（含被拒行），并记录两条通路的物理布局差异；真实 checkpoint manifest（`AngelSlim/Qwen3-1.7B_eagle3`：3 个辅助层、draft_vocab 32000 + d2t/t2d）；EAGLE3 draft 模型（`Eagle3Qwen3ForCausalLM`/`Eagle3LlamaForCausalLM`：多层特征 fc 融合 + layer0 拼接 + 返回 `(hidden, prenorm)`）已把**真实 checkpoint**（`AngelSlim/Qwen3-1.7B_eagle3`，3 辅助层、draft 词表 32000）12 个张量全部落位、`embed_tokens` 按"检查点缺这一份"与 target 共享；提议者/Runner 接线与 greedy 端到端（K=1/2/4）已通过，并完成与上游真实实现的数值对照（combine 逐位相同、logits max|Δ|=7.0e-4）；真实 checkpoint 完整生成见 §5；`docs/results.json` → `step63.results` 记录实测 |；**2026-10-08 复核修复**：EAGLE/MTP 自回归步的起点改为上游口径（位置 = 第一遍采样行 position + k、上下文 = (第一遍 seq_lens − 被拒) + k，见该文 §7）——实测接受长度不变（1.1211→1.1213），差距在 draft 第一遍/自回归的条件本身（同 prompt 上游 1.2819） || [step64_alignment.md](step64_alignment.md) | `minivllm/models/` + `minivllm/spec_decode/` | 第六十四关：HiddenStateExtraction 的 **cache-only 执行路径**——把 target 的辅助层特征 `[T, L, H]` 当作 KV 存进形状 `[num_blocks, block_size, L, H]` 的分页缓存（**L 当 head 数、H 当 head_size**，一个 token 一个 slot），于是与 target KV 共用同一份 `slot_mapping`、同一张逻辑块表、同一套 prefix 复用与释放；`propose()` 返回 `sampled_token_ids[:, :1]`（草稿就是 target 本轮自己采出的 token，K 固定 1，不计入加速算法）；验收含**读物理 slot** 验证层/请求/位置不互换（tiny + 真实 Qwen3-1.7B 双路径）、chunked prefill、prefix 命中、拒绝尾部、请求释放复用、宽度>1 只取第 0 列；`docs/results.json` → `step64.results` / `step64.models` 记录实测 |
| [step65_alignment.md](step65_alignment.md) | `minivllm/models/` + `minivllm/spec_decode/` | 第六十五关：**原生 MTP**（权重就在 target checkpoint 里）——24 个方法别名一律归一为 `method="mtp"`（与上游 `MTPModelTypes` 逐项一致）；draft 配置从 target 派生（`n_predict`/`architectures`，K 与 `n_predict` 的整除约束）；加载器认两派命名（`mtp.*` 与 `model.layers.{N+i}.*`）并让 **target 侧跳过 spec 权重**、MTP 侧只挑 spec 层 + 共享 `embed_tokens`/`lm_head`；提议循环**复用 `EagleProposer`**（MTP 吃 target 最后一层 hidden、单返回值 → 上一步的 hidden 回灌下一步）；与上游真实 `Qwen3NextMTP` 的胶水 forward / `compute_logits` **max|Δ|=0.0**；并修掉 63 关的一个真 bug（CUDA 上 `_upload()` 漏传 hidden → 特征从没进过 draft 模型）；`docs/results.json` → `step65.results` / `step65.models` 记录实测与别名表 |
| [step66_alignment.md](step66_alignment.md) | `minivllm/models/` + `minivllm/spec_decode/` | 第六十六关：**Medusa 多头提议**——N 个纯 MLP head 并行读同一份 target hidden、各自 argmax 成 `[B, num_heads]` 的**线性链**（不是论文的树；`max_paths`/`topk` 在 V1 无读取点），argmax = 点质量 q 所以 `draft_probs=None` 是正确语义；配置期把旧 FasterDecoding checkpoint 归一（`medusa_num_heads/_layers` 改名、缺省的 `model_type`/`architectures`、**K 就是 head 数**、`vocab_size`/`truncated_vocab_size` 对齐 target），加载期认三种权重命名（真实旧格式 `{h}.{l}.linear.weight`/`{h}.{l}.weight`、`medusa_heads.` 前缀、本模型名）并把"故意不加载"的名字记账（K 之外的 head、未开启的 bias）而认不出的名字当场报错；Runner 取"产出 bonus 的那一行"（块内第 `采样数-1` 行，stride 用调度快照，修掉上游在混合 prefill 批次下的错位）；与上游真实 `Medusa`/`MedusaProposer` 的每 head logits 与候选列顺序 **max\|Δ\|=0.0**；**`mlp_speculator` 按需求 §3 只交"版本缺口证明"**（本机 0.28.0 配置层认得、注册表那行是注释掉的、模型类没有 `forward`、Runner 无分派）→ 生产路径明确报错、不写自创实现；`docs/results.json` → `step66.results` / `step66.models` 记录实测 |
| [step67_alignment.md](step67_alignment.md) | `minivllm/spec_decode/` | 第六十七关：**异构词表 TLI**（token 级交集）——`VocabMapping` 按 **token 字符串**（空格标记 Ġ/▁ 归一化）建三张表（`draft_to_target_ids` / `target_to_draft_ids` / `intersection_mask_draft`）+ 两侧 unk 兜底（`unk→eos→报错`，**0 是合法 unk**），四条路径与上游逐条对应：第一遍的历史行/扩容行、自回归的上一枚草稿 → `map_target_to_draft_ids`；草稿 logits → `constrain_draft_logits`（非交集列 `-inf`，永远选不到）→ argmax → `map_draft_to_target_ids`（交出去的草稿必须是 **target 空间** id，q 是点质量）；**只换 id 不重新分词**（行数/位置/KV 槽位不变）；与上游真 `VocabMapping` 逐位差分（三表 + 两个 map + 约束输出）；tiny 异构词表对（同 KV 规格、vocab 11 vs 13、两套**真** tokenizer 文件）端到端 greedy 与非投机一致、草稿 id 全在交集像内、反「假接线」（第一遍输入真过映射）；真实规模记录 Qwen3-1.7B × gpt2 交集 **42257**（target 27.8% / draft 84.1%）；概率草稿的 TLI 按上游边界**配置期拒绝**（需求 §3.5）；`docs/results.json` → `step67.results` / `step67.models` 记录实测 |
| [step68_alignment.md](step68_alignment.md) | `minivllm/structured_output/` + `minivllm/sample/` + `minivllm/engine/` | 第六十八关：**采样约束、Logprobs 与结构化输出的投机语义**——logprobs 四种模式（`raw_logprobs`/`raw_logits`/`processed_logprobs`/`processed_logits`）与上游 `Sampler.forward` 逐值一致；投机下"第 j 个位置读第 j 行"（接受候选读候选行、恢复 token 读同一位置的行、bonus 读 bonus 行），被拒候选位多算但由 `parse_output` 的**同一张 valid_mask** 连同 token 一起滤掉（截断尾部不漏出）；结构化输出只接 **xgrammar**（`choice` 在请求期改写成 EBNF），掩码每请求 `1+K` 行、按候选**逐步试走**再`rollback`，**只有真正提交的 token 才推进 FSM**；`-1` 占位位不推进且其后不填掩码；`min_p`（argmax 不变处理器）与三种惩罚/min_tokens 的逐行"假设历史"不重复应用；提交**三态矩阵**（支持 / 上游不支持 / 本项目尚未接入）与**采样约束处理顺序表**（逐行标出 bonus 行与候选行的不同：`min_p` 与 `logit_bias` 在上游只作用于 bonus 行，实测 `min_p=1.0` 时投机交付 779 个位置里 85 个 rank>1）；六个未接入字段与三个未接入后端一律请求期拒绝；⚠️ 2026-10-06 **独立复核**把原先记的两条"上游疑似 bug"全部推翻（`log_softmax` 幂等 → `processed_logprobs` 的 bonus 位数字正确；`logprobs=-1` 在 `gpu_input_batch.py:435-440` 已被归一化成 `vocab_size`，用户请求到不了 `topk(k=-1)`）——真正待修的是**本项目**缺这条归一化（§7 第 4 条）；`docs/results.json` → `step68.results` / `step68.models` 记录实测 |
| [step69_alignment.md](step69_alignment.md) | `minivllm/forward_context.py` + `minivllm/cudagraph_dispatcher.py` + `minivllm/compilation/` + `minivllm/worker/gpu_model_runner.py` + `minivllm/attention/` | 第六十九关：**V1 投机输入 Padding 与编译 CUDA Graph**——把「每轮形状都在变」的投机批补齐成有限几种固定形状（档位 → `BatchDescriptor` 键），再把 target 前向录成 CUDA Graph 重放：`forward_context.BatchDescriptor` / `ForwardContext` 持有本轮运行模式与键，`CudagraphDispatcher` 负责键表 / 补齐映射 / `dispatch()`（没键就按规则回退 eager），`CUDAGraphWrapper` 只做直通 / 捕获 / 重放并**始终校验输入地址**；补齐出来的假行 / 假请求「物理存在但不留痕」（槽位 `PADDING_SLOT_ID(-1)`、`seq_lens=0`、块表行清零），代价是图内写 KV 用 `clamp_min(0)` → **0 号块必须留白**（`reserve_null_block`，兑现 64 关的待办）；补齐行是紧凑布局的尾巴，所以 68 关的掩码 / logprobs 行映射不用改；drafter 侧实现上游 `prepare_inputs_padded` 的 eager 等价版（与真 kernel 逐值一致）与 `num_rejected_tokens_gpu` 的上下文修正，但 **draft 自己的图未接线**（上游只在 PIECEWISE 下做）；⚠️ 缺口：无 torch.compile 一轴、无 PIECEWISE（混合批回退 eager）、图内注意力物化 K/V gather（长上下文显存吃紧）；实测 launch 121→4.2/step（非投机）、381.6→296.4/step（K=3），eager 与图 greedy 逐 token 相同、真实块 KV 逐位相同；`docs/results.json` → `step69.results` / `step69.models` 记录实测 |
| [step69b_piecewise_cudagraph.md](step69b_piecewise_cudagraph.md) | `minivllm/models/qwen3.py` + `minivllm/attention/backend.py` + `minivllm/compilation/stats.py` + `minivllm/config.py` + `minivllm/worker/gpu_model_runner.py` | 第六十九关补充（71 关前置）：**PIECEWISE 分段图**——把每个 decoder 层切成「注意力前段 / 注意力核心 / 后段」，只把前/后两段录进图、注意力留在图外，于是**混合 prefill+decode 批**也能用图；`CUDAGraphMode` 五值语义与上游一致（默认回到 `FULL_AND_PIECEWISE`），`AttentionCGSupport`（本后端 `UNIFORM_BATCH`）+ `VllmConfig.resolve_cudagraph_mode_and_sizes()` 实现能力协商（显式 `full` → 降级并 warning），PIECEWISE 的键**不带请求数**、只补 token 行（请求级缓冲按真实数切片），段间用显式静态缓冲（embedding 输出 + 注意力输出各拷一次，按 `max_num_batched_tokens` 开一次不重建），新增 `CUDAGraphStat`/`CUDAGraphLogging` 命中统计表（含「没走图」那一类）；⚠️ 机制差异：上游靠 torch.compile + `splitting_ops` 拆 fx 图，本仓库手工分段（段内是未融合的 eager 算子、无 compile cache）、段间显式拷贝（上游靠分配器确定性）、drafter 图仍未做（属 74 关）；实测真实 1.7B 捕获 9 张 / 2.70 s / 池 3758 MiB（分段图仅多占 252 MiB），`none`/`full_decode_only`/`full_and_piecewise` 与混合批场景**逐 token 相同**；`docs/results.json` → `step69_piecewise.results` 记录实测 |
| [step70_alignment.md](step70_alignment.md) | `minivllm/core/sched/async_scheduler.py` + `minivllm/engine/core.py` + `minivllm/worker/gpu_model_runner.py` | 第七十关：**异步调度占位符与 GPU 结果回传（骨架 + 明确边界）**——把「token 有/没有」的二元状态扩成三元（**预留 → 已确认 → 抢占时作废**）：`AsyncScheduler` 只覆写两个钩子（调度后按 `1+K` 记占位、结果回来按**实际交付长度**结账），占位参与三处上游原式（`num_new_tokens` 加占位、候选数减占位、发布上界减占位），抢占把在飞输出标成 stale（只交付、不再改计数）；`EngineCore.step_with_batch_queue()` 用**有界批队列**（=2，上游 1+pp）先把下一轮排出去、队列满了才回头取结果；`AsyncGPUModelRunnerOutput` 在**侧流**上非阻塞 D2H 到 pinned 缓冲 + blocking 事件，`get_output()` 幂等；⚠️ 本项目**默认关**且显式开启会被 `NotImplementedError` 拒绝（不静默降级）：实测与前缀缓存同开会 device-side assert、长跑下偶发与同步不一致——根因是「不等上一轮结果就组装下一轮输入」要求执行侧 GPU 驻留（上游的 `prev_sampled_token_ids` scatter + 进度校正），而本仓库输入组装与提议器都在 CPU 侧，属 74/75 关那条路；本关交付状态机骨架 + 14 项单测（纯 CPU 确定性）+ 17 项探针；`docs/results.json` → `step70.results` 记录实测与拒绝证据 |
| [step71_alignment.md](step71_alignment.md) | `minivllm/spec_decode/dynamic/utils.py` + `minivllm/config.py` + `minivllm/core/sched/{output,scheduler,async_scheduler}.py` + `minivllm/worker/gpu_model_runner.py` + `minivllm/spec_decode/{draft_model,ngram_proposer}.py` | 第七十一关：**动态投机长度与调度 Graph 兼容**——把「低并发多猜、高并发少猜」做成一张**用户给的闭区间表**（`[(1,2,4),(5,8,1)]` = 批 1~2 猜 4、批 5~8 猜 1；空隙沿用前段、尾部延续最后一段、每项按最大 K 裁剪、索引 0 不用），Scheduler 每轮用**实际被调度的请求数**查稠密表并把本轮 K 放进 `SchedulerOutput.num_spec_tokens_to_schedule`，提议者按轮 K 提草稿（**K=0 仍跑完 draft 第一遍**再返回空草稿，ngram 不写 KV 则直接空）；**两轮时序**：本轮 K 管「本轮提的草稿」、本轮验证的是上一轮提的候选，改 K 不重解释旧候选的长度/q；配置期两条改写——含 full graph 的模式一律降级 `PIECEWISE`（full 图冻结不了逐轮变的 `1+K`）、DP>1 清空表退回固定 K；方法边界上只有 `draft_model/eagle/eagle3/mtp/ngram` 支持，`ngram_gpu/suffix/medusa/extract_hidden_states/custom_class` 在配置期明确拒绝（保留上游固定 K 的断言）；实测：与上游 `dynamic/utils.py` 315 组逐值一致；tiny 场景 K 序列 `[4,4,0,2,2,2,0,4]`（第 2 轮 K=0 仍验证第 1 轮提的 4 枚、draft 第一遍照跑 1 次）；不开投机 / 固定 K=3 / 动态表三者贪心输出逐 token 相同；真实 1.7B + ngram 14 轮全部命中 PIECEWISE（8 张图、FULL 0 次）；`docs/results.json` → `step71.results` |
| [step72_alignment.md](step72_alignment.md) | `minivllm/spec_decode/utils.py` + `minivllm/spec_decode/{draft_model,eagle}.py` + `minivllm/config.py` + `minivllm/models/qwen3_eagle3.py` + `minivllm/worker/gpu_model_runner.py` | 第七十二关：**PARD 与 P-EAGLE 并行提议**——把「串行 K 次 forward 凑 K 枚草稿」换成「**一次** forward 出 K 个位置」：块内布局 = `[有效行][锚点][K−1 个 mask token][被拒尾部]`（P-EAGLE 左移、锚点复用 target 最后一行；PARD 不左移、锚点与 mask 全部新占），槽位净增 **PARD=K / P-EAGLE=K−1**（K=1 时 P-EAGLE 为 0）；`expand_parallel_draft_inputs()` 是上游 `copy_and_expand_eagle_inputs_kernel` 的逐行等价（两种 shift），差分测试直接调上游真 Triton kernel、六组用例的 input_ids/positions/两个 mask/采样行/hidden 映射**全字段相等**；mask token 与 `mask_hidden` 必须来自 checkpoint（缺了在配置期/加载期报错，串行权重开并行直接被拒）；实测 tiny K=3：并行每轮 1 次 draft 前向、串行 3 次，并行 greedy == 非投机；⚠️ 本机没有按并行草稿训练的权重 → 质量/加速比、drafter 图（74 关）、DFlash/DSpark（76/77）标待验；`docs/results.json` → `step72.results` |

