# step57 差异账本（57F）

200 §5 要求：本文件每项按**固定九段**写清楚——差异不是犯错，**没有意识到差异、却拿自创约束
代表 vLLM**才是问题。

```text
主题：
本机路径 / 类 / 方法：
本机做法：
本项目做法：
为何简化：
影响哪些功能/性能/可复现性：
对应测试：
以后何时消除差异：
```

另外两条维护约定：

- 每关（57A–57E）的**段落级**差异账本在各关文档里（`step57{a..e}_*.md`），本文件是**汇总**：
  按 vLLM 的模块归类，方便"读 vLLM 时逐条对账"。
- 新增/消除差异时**只改本文件 + 那一关的文档**，不去改别的关卡文档。

## 1. 部署与进程模型

```text
主题：Inproc + 同步直调，没有进程 / Future / RPC wrapper
本机路径 / 类 / 方法：v1/engine/core_client.py::InprocClient、v1/executor/uniproc_executor.py、
                       v1/engine/core.py::EngineCore.step / post_step
本机做法：客户端与 EngineCore 同进程，`execute_model(non_block=True)` 返回 Future；执行与采样
          可以在两个线程里重叠；WorkerWrapperBase 负责动态 worker 初始化
本项目做法：`InprocClient` 同步直调（`EngineCore.step()` 现场返回），`UniProcExecutor` 直接调
            Worker；`execute_model` 返回 None + `sample_tokens` 两步，但不造"假装异步的线程"
为何简化：本关学职责与协议，不混入部署复杂度；同步直调让每一步的顺序在代码里就是字面顺序
影响：吞吐（没有 CPU/GPU 重叠）、不能多进程；可复现性更好（没有跨线程竞态）
对应测试：check_step57_engine_protocol.py（执行/采样分离、空轮不执行模型）
以后何时消除：57 之后如果要做异步/多进程执行，先补这一层，再谈 CUDA Graph
```

```text
主题：单客户端输出，没有多 client-index 分桶
本机路径 / 类 / 方法：v1/engine/core.py::EngineCoreOutputs（client_index）、v1/outputs.py
本机做法：一次 step 的输出按 client_index 分桶，支持多客户端（数据并行 / 多 API 前端）
本项目做法：`EngineCoreOutputs.outputs` 一个列表，不做分桶
为何简化：本关单客户端，分桶只会多一层没人读的字段
影响：功能（不支持多客户端）；性能无
对应测试：check_step57_engine_protocol.py
以后何时消除：接入多 API 前端或数据并行时
```

## 2. KV 控制面与物理存储

```text
主题：手动 KV 容量、只有一个 Full Attention group
本机路径 / 类 / 方法：v1/core/kv_cache_manager.py::KVCacheManager、v1/core/kv_cache_coordinator.py、
                       v1/worker/gpu_worker.py::determine_available_memory
本机做法：启动时做显存 profiling 自动定容；KV group 由 `KVCacheConfig` 描述，支持全注意力 +
          滑窗 + Mamba 混搭，`HybridKVCacheCoordinator` 在组之间取共同边界
本项目做法：`CacheConfig.num_gpu_blocks` 手动配置；`UnitaryKVCacheCoordinator` 只有一组，
            没有滑窗/Mamba/混合路径
为何简化：暂不做显存 profiling 与混合模型；单组已经能覆盖"分页 KV + 前缀缓存"的全部机制
影响：功能（不能跑混合注意力模型；容量要手算）；性能（没有按显存自动定容）
对应测试：check_step57_block_pool.py、check_step57_prefix_cache.py、docs/step57c_kv_and_prefix.md
以后何时消除：需要支持滑窗/Mamba 时，先补 group 设计与共同命中边界，再扩 coordinator
```

```text
主题：prefix 在**结果处理之后**才发布
本机路径 / 类 / 方法：v1/core/kv_cache_manager.py::allocate_slots（内部调 coordinator.cache_blocks）
本机做法：分配时就把"可提交范围"截到 `request.num_tokens` 并顺手发布完整块
本项目做法：`Scheduler.update_from_output()` 里处理完结果再调
            `kv_cache_manager.cache_blocks(request, num_computed_tokens)`，上限
            `floor(min(num_computed, num_tokens)/block_size)`
为何简化：第一版不支持"尚在本轮执行中的块被复用"；发布与执行顺序严格配合才算安全
影响：命中机会略少（性能）；正确性不受影响（发布的一定是已写完 KV 的完整块）
对应测试：check_step57_prefix_cache.py（发布边界、开/关缓存输出一致）
以后何时消除：研究"提前登记的安全条件"之后，单独立项
```

```text
主题：不支持 partial-tail cache / COW / null block 等路径
本机路径 / 类 / 方法：v1/core/block_pool.py（null_block、partial hash、evict_blocks）、
                       v1/core/single_type_kv_cache_manager.py（CoW、fine-grained hash）
本机做法：null block 占一个物理块；局部块 hash 支持"半块也登记"；命中可以做局部 CoW
本项目做法：没有 null block（`num_gpu_blocks` 个块全部可用）；只登记**完整块**；没有 CoW
为何简化：只学完整块的基本机制；局部 hash 与 CoW 需要与后端 kernel 配合
影响：**容量对照要注意**（vLLM 实际可用块数是 n−1）；功能（不支持局部命中）
对应测试：check_step57_block_pool.py（"本关没有 null block"那条）
以后何时消除：需要局部命中/局部复用收益时
```

```text
主题：逐出顺序用"非缓存块进队头、带 hash 的块进队尾"，没有访问计数
本机路径 / 类 / 方法：v1/core/block_pool.py::free_blocks / FreeKVCacheBlockQueue
本机做法：同上（本关照抄），但它还有 metrics 钩子与 KV 事件
本项目做法：语义一致，去掉事件与 metrics；队头/队尾的取舍写在 `block_pool.py` 的注释里
为何简化：本关不做对外事件/监控
影响：可观测性（没有 kv_cache 事件流）
对应测试：check_step57_block_pool.py（队列不变量、同 hash 多块不误删）
以后何时消除：要接外部网关/监控时
```

## 3. 调度

```text
主题：抢占用的 victim 选择与"本轮不再接纳等待者"
本机路径 / 类 / 方法：v1/core/sched/scheduler.py::_try_schedule_running / _preempt_request:L1336
本机做法：FCFS 取 running 尾部；priority 取 `max(running, key=(priority, arrival_time))`；
          抢占后本轮不再接纳 waiting
本项目做法：同样两条规则；额外多了一个"连续两轮排不出 token 就报错"的教学保护
为何简化：教学保护是为了让"不可行配置"当场报错（196 §9.7 要求失败策略可查），vLLM 没有
影响：行为（本关在配置不可行时更快失败）；性能无
对应测试：check_step57_preemption.py、check_step57_scheduler_basic.py（空转保护）
以后何时消除：不需要消除；若要与 vLLM 完全一致可以去掉这条保护（但会失去可读的失败原因）
```

```text
主题：`num_lookahead_tokens` 已按 vLLM 实现；仍然没有 `input_budget` /
      `max_num_new_slots_for_drafting`
本机路径 / 类 / 方法：v1/config/vllm.py::VllmConfig.num_lookahead_tokens、
                       v1/core/kv_cache_manager.py::allocate_slots(num_lookahead_tokens)、
                       v1/core/sched/scheduler.py（input_budget）
本机做法：`num_lookahead_tokens` = K（EAGLE / draft model）或 0（ngram），
          scheduler 在 `allocate_slots` 时一律带上；token 预算与输入预算分开算
本项目做法：**同一条规则**（`Scheduler.num_lookahead_tokens`：draft_model → K，ngram → 0），
            `num_tokens_need_slot = min(computed + new + lookahead, max_model_len)`；
            提议者仍然问 `BlockTable.covers()`，没有槽位就少提几枚；
            仍然没有 input_budget——draft 前向的输入不预分配缓冲，只检查位置在
            `[0, max_model_len)` 内
为何简化：本关只有普通自回归 draft（不是 EAGLE/MTP 那种"提议者就是 target 自己"），
          输入预算不预分配也能保证不越界
影响：功能（EAGLE/MTP 接不进来）；性能（每轮可能重建输入张量）
对应测试：check_step57_draft_model.py §6（prompt 恰好占满整块 / 中间 prefill / CUDA 设备）
以后何时消除：接 EAGLE/MTP 之前必须补 input_budget
```

**注（2026-10-02 修正）**：这里原来写的是"草稿在本轮调度范围内，所以不需要预留 lookahead"——
**那是错的**。target 的 query 覆盖的是上一轮采用的草稿，而提议者这一轮要写的是**更后面** K 个
位置。独立验收探针用"prompt 恰好占满整块"复现了越界。详见
[`step57_acceptance_fixes.md`](step57_acceptance_fixes.md)。

```text
主题：没有流式会话 / 阻塞状态 / KV 连接器
本机路径 / 类 / 方法：v1/core/sched/scheduler.py（WAITING_FOR_REMOTE_KVS、skipped_waiting、
                       connector、`_handle_stopped_request`）
本机做法：为 P/D 分离、远程 KV、流式请求准备了一整套状态与队列
本项目做法：只有 WAITING/RUNNING/PREEMPTED + 结束态；没有连接器与流式
为何简化：本关单卡、无远程 KV
影响：功能（不能做 P/D 分离）
对应测试：check_step57_request_progress.py（状态与原因映射）
以后何时消除：接 KV 连接器时
```

## 4. 执行侧

```text
主题：未就绪的 prefill 先跳过采样
本机路径 / 类 / 方法：v1/worker/gpu_model_runner.py::_bookkeeping_sync（discard_request_mask）
本机做法：先对所有请求的末行算 logits 并采样，再按 mask 丢掉不该提交的，并处理 generator offset
本项目做法：在 `_prepare_inputs` 就算出 ready 行，只对 ready 行算 LM head 与采样；未 ready 返回 []
为何简化：不照搬绑定具体采样实现的 `generator.get_offset()` 回退约定
影响：少算一些用不上的 LM head（性能，略微有利）；RNG 消费轨迹与 vLLM 不同（199 §8 允许）
对应测试：check_step57_runner_inputs.py（只对 ready 行算 logits）
以后何时消除：不需要；若要 RNG 逐位对齐才需要改
```

```text
主题：推理边界（`torch.inference_mode()`）——**不是差异**，是必须对齐的一条
本机路径 / 类 / 方法：v1/worker/gpu_model_runner.py（`execute_model`/`sample_tokens`/`load_model`
                       上共 8 处 `@torch.inference_mode()`）
本机做法：每步入口都在推理模式下跑
本项目做法：`execute_model` / `sample_tokens` 加 `@torch.inference_mode()`；
            **`load_model` 不加**（那会把权重变成"推理张量"，而测试要直接用这些权重做前向）
为何简化：本关没有 warmup / dummy run，不需要在加载期划边界
影响：**没有这条就是显存泄漏**——KV 写入（`index_copy_`）会挂 `CopySlices` 反向图并逐步累积
对应测试：check_step57_runner_inputs.py §7（KV 缓存 `grad_fn is None`）、
          验收探针 review_inference_boundary.py
以后何时消除：如果以后加了 warmup/dummy run，把 `load_model` 也划进推理模式
```

```text
主题：Torch attention / Torch 采样参考实现，Graph 关闭
本机路径 / 类 / 方法：v1/attention/backends/*、v1/sample/ops/topk_topp_sampler.py（FlashInfer/Triton）
本机做法：分页 attention 与采样都走定制内核；支持 CUDA Graph 捕获
本项目做法：`TorchAttentionImpl` 逐请求 gather + Torch 数学；top-k/top-p 走排序 + 掩码；
            采样用指数竞赛（Torch）；不捕获 CUDA Graph
为何简化：先验证架构与数学；不声称性能接近 vLLM
影响：性能（明显更慢）；数值（与 HF/vLLM 在 fp32 下逐值一致，bf16 下有 logits 网格量级的差异）
对应测试：check_step57_model_logits.py、compare_step57_vllm.py（A/B/D 段）
以后何时消除：本关之后如果要做性能，先做这一层（并有性能矩阵验收）
```

```text
主题：简化模型 loader / TP=1 layers
本机路径 / 类 / 方法：vllm/model_executor/model_loader/*、layers/linear.py（ColumnParallelLinear 等）
本机做法：支持 TP/PP/EP、量化、HF hub 下载、多格式（bin/gguf/…）
本项目做法：`model_loader/{weight_utils,auto_weights_loader,base_loader,default_loader,loader}.py`
            只读本地 safetensors（单文件 / index 分片）；`layers/linear.py` 保留 Parallel 命名但
            **TP=1、没有通信**；额外做了**覆盖检查**（vLLM 没有）
为何简化：真实名字映射与分片写入保留（学习要点），分布式不实现
影响：功能（改 tp_size 不会跑起来）；好处：漏加载参数会当场报错（vLLM 是静默的）
对应测试：check_step57_weight_loading.py
以后何时消除：需要多卡时，先把通信加回 layers，再扩 loader
```

```text
主题：显式排空最后一条 finished 通知 / 不可行配置直接报错
本机路径 / 类 / 方法：v1/core/sched/scheduler.py（finished_req_ids 与代际编号）、
                       v1/engine/core.py（异常处理）
本机做法：用 request 代际编号允许 ID 立刻复用；不可行配置往往表现为空转或超时
本项目做法：结束消息**下一轮**送出；同一 ID 在清理消息送出去之前禁止复用；
            "连续两轮排不出 token"直接报错并列出 running/waiting/空闲块
为何简化：教学生命周期策略（195 §4 的"禁止复用"就是本关的选择）
影响：行为（ID 复用晚一轮；配置不可行时更快失败，且报错更好读）
对应测试：check_step57_engine_protocol.py（结束清理两段式、ID 唯一性与回滚）
以后何时消除：不需要；若要支持"同 ID 立刻复用"，需要引入代际编号
```

## 5. 采样与投机

```text
主题：**发布边界不夹 draft 进度**（与 vLLM 相同；上一版夹过，2026-10-03 撤销）
本机路径 / 类 / 方法：v1/core/kv_cache_manager.py::allocate_slots（`num_tokens_to_cache =
                       min(total_computed_tokens + num_new_tokens, request.num_tokens)`）、
                       v1/core/sched/scheduler.py::update_from_output / update_draft_token_ids、
                       v1/spec_decode/llm_base_proposer.py::set_inputs_first_pass
本机做法：drafter 与 target **每步跑同一段位置**——中间 prefill 块也跑（注释原话："The prefill
          forward pass above already ran to keep the drafter KV cache in sync"），产出的草稿由
          `update_draft_token_ids` 丢掉（"Ignore draft tokens for prefill chunks"）。发布只按
          target 的 `num_computed_tokens` 截断，不改上下限
本项目做法：**同一条规则**：执行端对每个被调度的请求都同步 draft 的 KV（中间 prefill 块也同步），
            发布处不夹边界。区别只在"草稿提不提"：本关在提议阶段就不给非 ready 行提，
            vLLM 提完再丢
为何上一版要夹（为什么那是错的）：本关原来把"prefill 块忽略草稿"实现成"干脆不跑提议者"，
            draft 的进度因此停在 0，而 target 这一轮就会发布完整块 → 会把只有 target 算过的块
            登记成可复用（199 §9 明确不允许：**"不能把只有 target 算过的块作为双模型命中"**）。
            当时的修法是在控制端夹 `min(target, draft)`；把执行端改成"同步"之后，按 199 §9 的
            第一个选项做，就不需要这层对账了
影响：命中机会与 vLLM 相同（中间 prefill 块的完整块也能发布，上一版会推迟到 draft 追平）；
      正确性由"两边同步"保证，而不是控制端兜底，所以必须有用例盯着
对应测试：check_step57_draft_model.py §6（中间 prefill 轮次 draft 进度 == target 进度、
          发布边界 ≤ draft 进度、**发布位置上的 draft KV 确实非零**、prefix 开/关对照、
          恢复后 draft 进度不落后）、验收探针 review_draft_boundaries.py（6/6，其中一条直接读
          `_draft_computed` 与 `request.num_computed_tokens`）
以后何时再动：只同步不提草稿，比 vLLM 少跑 K 次前向。vLLM 在 prefill 块会跑出一个
            "prefill lookahead token"，它的 KV 会污染该块（vLLM 靠 `num_reprefillable_tokens`
            排除 + EAGLE/MTP 命中时丢最后一块来兜）；本关不提就不写，所以没有这两处机制
```

```text
主题：普通 draft 限定兼容 KV 规格与同词表
本机路径 / 类 / 方法：v1/spec_decode/draft_model.py::DraftModelProposer（`_create_draft_vllm_config`）、
                       v1/core/kv_cache_coordinator.py（多 KV group）
本机做法：draft 与 target 可以有不同词表（有 vocab mapping 路径）与不同 KV 规格（各自 group）
本项目做法：加载时断言**同词表、同 dtype、同 KV head 数/head_dim**；不满足就明确报错，
            不用"隐藏双 pool"兜底；单 group 共用逻辑块表，每层绑自己的物理 tensor
为何简化：不把独立双池当作对齐完成；更广的分组留待以后
影响：功能（很多小模型配不起来，会明确报错）；这比"看起来支持"更诚实
对应测试：check_step57_draft_model.py（词表/KV 规格不兼容的两条报错）
以后何时消除：需要更广分组时，先补 group 设计评审
```

```text
主题：提议侧不施加惩罚（只做温度 + top-k/top-p）
本机路径 / 类 / 方法：v1/spec_decode/llm_base_proposer.py::compute_probs_and_sample_next_token
本机做法：注释写明"we ignore most of the sampling parameters in generating the draft tokens.
          We only use the temperature"（只做温度）
本项目做法：额外做了 top-k/top-p（与 target 的 p 更接近、接受率略高，代价是每次提议多一次排序）；
            **惩罚两边都不做**
为何简化：拒绝采样对任何 q 都成立——省掉约束只影响接受率，不影响输出分布
影响：接受率（我们的略高）；输出分布不受影响（`check_step57_rejection_sampler.py` 的
      recovered 分布与接受概率都对着解析值验过）
对应测试：check_step57_rejection_sampler.py、compare_step57_vllm.py（C 段统计对照）
以后何时消除：如果需要与 vLLM 逐位对齐提议分布，就把 top-k/top-p 也去掉
```

```text
主题：投机没有指标与事件
本机路径 / 类 / 方法：v1/core/sched/scheduler.py::make_spec_decoding_stats、
                       v1/spec_decode/*（acceptance rate 统计）
本机做法：逐请求统计草稿数/接受数/无效草稿数，进 metrics
本项目做法：只有 `num_drafts_proposed` 与 `Scheduler.trace` 里的 `scheduled_spec_decode_tokens`
为何简化：本关不做指标
影响：可观测性（看不到接受率）——57E 的差异账本里已记为遗留
对应测试：check_step57_spec_lifecycle.py（trace 里的草稿数）
以后何时消除：做指标那一层时
```

## 6. 采样（57D）与输出

```text
主题：采样元数据按行现扫；惩罚在算子内部拼张量
本机路径 / 类 / 方法：v1/worker/gpu_input_batch.py::_make_sampling_metadata、
                       v1/sample/ops/penalties.py
本机做法：InputBatch 里维护 CPU 张量与 `all_greedy`/`no_top_p`/`top_k_reqs` 这类增量集合，
          按需拷到 GPU；惩罚用 bins + 自定义算子
本项目做法：`SamplingMetadata.from_input_batch` 每轮扫一遍（批量小）；惩罚在算子内部把逐行 list
            拼成 padded 张量
为何简化：可读优先、便于逐值对照
影响：性能（O(批) 常数 + 每轮重建张量）；正确性无
对应测试：check_step57_sampler.py（惩罚逐值对照手写公式）
以后何时消除：做采样性能时
```

```text
主题：没有 logprobs / 白名单 / bad words / min_p / 结构化输出过滤
本机路径 / 类 / 方法：v1/sample/sampler.py（max_num_logprobs、gather_logprobs）、
                       v1/sample/logits_processor/*（框架与内置处理器）
本机做法：一整套 logits processor 插件 + logprobs 回传
本项目做法：只保留"会改变 argmax 的约束"里本关用得到的 min_tokens 屏蔽；其余明确不做
为何简化：本关不要求 logprobs 与插件框架
影响：功能（不能返回 logprobs；不能做 grammar/白名单）
对应测试：check_step57_sampler.py、check_step57_stop_and_outputs.py
以后何时消除：需要这些特性时，先补 logits processor 的框架（本关是函数而不是插件）
```

```text
主题：用户输出是"累计 token + 一次性交付"，没有流式队列
本机路径 / 类 / 方法：v1/engine/output_processor.py（RequestOutput、增量 delta、metrics）
本机做法：把增量交给用户（`CompletionOutput` 带 delta text），支持流式回调与 metrics
本项目做法：每轮交付**累计**快照；`stop_reason` 只在结束那一条上；没有 metrics
为何简化：本关不要求流式；累计快照更容易验证（相邻两轮互为前缀）
影响：功能（不能流式；没有首 token 延迟等指标）
对应测试：check_step57_stop_and_outputs.py（增量→累计、快照隔离、abort 与 ID 复用）
以后何时消除：做流式输出时
```

## 7. 本文件怎么用

读 vLLM 某个模块时，先在本文件里搜模块名，看**我们已经声明的差异**：

- 如果差异在列 → 按"本项目做法"那一栏读我们的代码，不会误以为自己理解错了 vLLM；
- 如果**不在列**而两边行为不同 → 那就是新发现的差异（或者我们写错了），补一条并写测试；
- 每条都要有"对应测试"与"以后何时消除"：**没有测试的差异等于没声明**。
