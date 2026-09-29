# step57A：Engine 分层的竖直骨架

- 对应代码：`step57/`（**新建**，不是从 step56 复制改名；只做 57A 这一段）
- 包摘要 SHA256：`db6ffabcb2ceaae5…`（23 个 .py / 1909 行；口径 = 包内 `*.py` 按相对路径排序，
  每个文件取自身 sha256，拼成 `名字\0哈希\n` 再取 sha256）
- 验收脚本：`benchmarks/check_step57_{engine_protocol,request_progress,scheduler_basic}.py`
  （对应需求里点名的 `test_engine_protocol.py` / `test_request_progress.py` /
  `test_scheduler_basic.py`，本仓库的统一命名是 `check_stepNN_*`）
- 参考实现：本机 `vllm 0.28.0`（`vllm/v1/{engine,core/sched,executor,worker,request,outputs}.py`）

## 0. 需求大概

这一关换方向：不再自己发明协议，而是**做一个能逐层映射到本机 vLLM 的、可运行的文本生成
子集**。57A 只做第一段——**竖直骨架**：

- `config` / `Request` / 输入输出协议 / `LLMEngine` — `InprocClient` — `EngineCore` —
  `UniProcExecutor` — `Worker` —（Runner）这条主线跑通；
- 普通统一预算调度（先 running 后 waiting），**无 prefix、无投机、无抢占、无真实模型**；
- 用测试注入的 Runner 做 CPU 假执行。

判定点（200 §57A）：三 token prompt、预算 2 的三轮例子；结果按 ID 映射；快照无共享可变列表；
结束清理；空轮不执行模型；活动 ID 唯一。**完成这一段就停**。

## 1. 改动内容（新建的层与它们各自回答什么）

| 层 / 文件 | 对应 vLLM | 它回答的问题 | 57A 的边界 |
|---|---|---|---|
| `config.py` | `vllm/config/*` | 配置怎么分类、在哪里校验 | 只留用到的字段；`enable_prefix_caching=True` **构造时拒绝**（属 57C）；不做显存 profiling 自动定容 |
| `sampling_params.py` | `vllm/sampling_params.py` | 采样参数属于谁 | 只留子集；**不持有 `torch.Generator`**（参数是可复制的配置，随机流属执行侧） |
| `request.py` | `v1/request.py` | 请求状态谁拥有、谁能改 | `num_computed_tokens` 由 Scheduler 写；不放 cache/generator/GPU 张量/抢占链/counter |
| `outputs.py` | `v1/engine/__init__.py` + `v1/outputs.py` | 三个边界上各传什么 | 同进程，用 dataclass 而非 msgspec；不做多 client-index 分桶 |
| `core/kv_cache_manager.py` | `v1/core/kv_cache_manager.py` | 块归谁 | **简单空闲表**：够就发、不够返回 None；无 hash、无 prefix 命中、无引用计数（57C） |
| `core/sched/output.py` | `v1/core/sched/output.py` | 调度 → 执行发什么 | 字段名保持一致；不复制 V2/投机专用字段；**包内不许有活对象** |
| `core/sched/request_queue.py` | `v1/core/sched/request_queue.py` | 等待队列的顺序 | FCFS + priority（懒惰删除）；不做阻塞状态队列 |
| `core/sched/utils.py` | `v1/core/sched/utils.py` | 什么时候停 | 只做 token 级；不做字符串 stop 匹配、不做事后重复检测 |
| `core/sched/scheduler.py` | `v1/core/sched/scheduler.py` | 谁这一轮算几个 token、进度谁维护 | 无抢占、无投机、无 encoder/connector；**多了一条空转保护**（见 §3） |
| `engine/core.py` | `v1/engine/core.py` | 一轮怎么编排 | 同步直调，**不造假装异步的线程**；执行/采样分离保留 |
| `engine/core_client.py` | `v1/engine/core_client.py` | 传输与执行解耦 | 只有同进程实现；Client 用 Protocol 而非 ABC |
| `engine/llm_engine.py` | `v1/engine/llm_engine.py` | 用户侧入口 | 只收 token IDs（字符串编码属 57B） |
| `engine/output_processor.py` | `v1/engine/output_processor.py` | 增量 → 用户可见结果 | 无流式队列、无 metrics；**结束结果交付之后才删状态** |
| `executor/uniproc_executor.py` | `v1/executor/uniproc_executor.py` | 交给哪种执行部署 | 省略 `WorkerWrapperBase`，因此**不能**说成完整 RPC |
| `worker/worker.py` | `v1/worker/gpu_worker.py` | 执行端怎么准备 | 57A 没有模型：没注入 runner 就**明确报错**，不静默退回假执行 |
| `testing/fake_runner.py` | （无对应，测试专用） | 执行侧只靠协议能不能干活 | 只被测试 import；脚本化输出，不给看不见的伪随机兜底 |

## 2. 设计要点（容易被后续改动破坏的约定）

1. **跨边界只传快照**：包里不许有 `request` / `seq` / `scheduler` / `kv_cache_pool` 活对象，
   也不许塞闭包。`FakeRunner` 每轮**主动检查**这条并记录（用例断言它是空的）——靠"看代码"
   守不住。
2. **快照必须复制可变容器**：新请求的 `prompt_token_ids`、`all_token_ids`、块表都是新 list/tuple。
   用例会往包里乱改（append 伪造字段、改块表、改旧进度），再断言 Scheduler 的 Request 与块表
   一字未变。
3. **进度是"已安排计算"**：`num_computed_tokens` 在**打包之后**才推进（`_update_after_schedule`），
   所以包里的值是旧快照；正常 decode 边界上 `num_tokens - num_computed_tokens == 1` 是常态
   （采样出 token ≠ 它的 KV 已经算好）。
4. **结果按 ID 取，不按行序**：`model_runner_output.req_id_to_index[req_id]`；执行侧允许重排
   紧凑 batch（用例故意反着返回行，仍要求各拿各的输出）。
5. **结束是两段式的**：请求在 `update_from_output` 里结束并释放块，但"可以清理了"的消息要等
   **下一轮**的 `SchedulerOutput.finished_req_ids` 才送到执行侧。因此：
   - `has_requests()` 在"只有清理消息没送"时仍然为真（引擎不能提前停）；
   - 同一个 ID 在清理消息送出去之前**禁止复用**——否则下一轮的包会同时含"清理 r1"与"新增 r1"，
     执行侧的状态必然错乱（vLLM 用代际编号解决，本关选择直接禁止）。
6. **`new_block_ids` 是增量**：普通续跑是**新增**块（执行侧追加），`resumed_req_ids` 里的请求是
   **整张替换**；没有新增时是 `None`，不是空列表。这条在 `FakeRunner` 的镜像里实现了一遍，
   也写进了 `KVCacheBlocks` 的 docstring。
7. **`max_num_seqs` 只数 `running`**：新接纳的请求当场 append 进 `running`，再和
   `scheduled_new_reqs` 相加会把同一条数两遍（这一版最初的写法就是这样，被用例抓住）。
8. **执行/采样分离**：`execute_model()` 存下计划并返回 `None`，`sample_tokens()` 才产出结果。
   空轮（0 token）不碰模型，直接回空结果——这条约束落在**执行侧**，Scheduler 不必知道。

## 3. 与 vLLM 的差异账本（本关允许的简化，都在代码里标出来了）

| 差异 | 原因 / 后续 |
|---|---|
| `allocate_slots()` 失败时**本轮停止排序**，不挑 victim | 抢占属 57C；现在只是"这轮不排后面的" |
| 多了"连续两轮排不出 token 且仍有未完成请求 → 报错" | 教学保护（196 §9.7 要求失败策略可查）；vLLM 没有这条 |
| `_handle_stopped_request` 内联成"必定结束" | 它只为 streaming 会话服务 |
| 没有 `skipped_waiting` / 阻塞状态 / PoC 字段（`preempted_req_ids` 等） | 都属于 V2、远程 KV、投机路径 |
| `KVCacheBlocks` 只表示"本次新增的块" | 与 vLLM 同名类型的语义一致，但本关没有 group/多池 |
| `Request` 没有 `events` / `num_output_placeholders` / `num_in_flight_tokens` | 异步调度与指标用的；本关同步 |
| `check_stop` 不做事后重复检测、不做字符串 stop | 前者属可选特性，后者属输出侧解码 |

**已知缺口**（会在后续段落补，不假装已完成）：真实模型与 GPU 执行（57B）、KV 物理存储与
prefix cache（57C）、抢占与恢复（57C）、采样与惩罚（57E）、投机（57E）、异步与多进程。

## 4. 验证

| 脚本 | 项数 | 覆盖 |
|---|---:|---|
| `check_step57_engine_protocol.py` | 23 | 三轮轨迹逐字段、空轮不执行模型、结果按 ID 映射（执行侧反序）、快照隔离、结束清理两段式、活动 ID 唯一 + 失败回滚、执行/采样分离、协议包无活对象 |
| `check_step57_request_progress.py` | 29 | prompt 复制、只读视图、唯一写入点同步两份列表、派生量、状态与原因映射、排序与两种队列（含懒惰删除）、`check_stop` 六条分支 |
| `check_step57_scheduler_basic.py` | 25 | 统一预算、先 running 后 waiting、`max_num_seqs` 与块容量两个上限、分配失败原子性、`all_token_ids` 发送规则、结果按 ID 更新、结束两段式、abort、空转保护 |

三个脚本全部通过（共 77 项）。**没有跑**真实模型（57B 才有）、没有性能测试（本关不涉及）。

## 5. 接口变化与遗留

- **新包 `step57/`**：与 step56 并存，互不 import——step56 是"自己发明的协议"，step57 是"对齐
  vLLM 的协议"，两者不是改名关系，这次是**新写**。
- 入口：`from step57 import LLMEngine, VllmConfig, ModelConfig, CacheConfig, SchedulerConfig,
  SamplingParams, UniProcExecutor, Worker`；测试另加 `step57.testing.FakeRunner`。
- 用法（57A 没有真实模型，只能注入假执行）：

  ```python
  config = VllmConfig(model_config=ModelConfig(model="dummy", max_model_len=64),
                      cache_config=CacheConfig(block_size=4, num_gpu_blocks=4),
                      scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=2))
  runner = FakeRunner(tokens={"r1": [11, 12]})
  engine = LLMEngine(config, UniProcExecutor(config, Worker(config, model_runner=runner)))
  engine.add_request("r1", [10, 11, 12], SamplingParams(max_tokens=2, eos_token_id=99))
  while engine.has_unfinished_requests():
      for out in engine.step():
          ...
  ```

- **遗留**：`step57` 还没有 `stepNN.py` 命令行入口（要等 57B 能加载真实模型才有意义）；
  旧的 `step56/` 保持不变，不删也不改。
