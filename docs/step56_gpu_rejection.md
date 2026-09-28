# step56：GPU 批量拒绝采样与单次结果回传

- 对应代码：`step56/`（从 `step55/` 复制，入口改名 `step56.py`；旧包不改）
- 包摘要 SHA256：`a6a7fc786c7d7590…`（22 个 .py / 5949 行；口径 = 包内 `*.py` 按相对路径
  排序，每个文件取自身 sha256，拼成 `名字\0哈希\n` 再取 sha256）
- 基线：`step55/` 的投手机制**行为逐字节不变**（`benchmarks/diff_step55_step56.py` 88 项，
  默认的 `rejection_backend="torch"` 就是第五十五关那条参考路径）
- **新增 3 个模块 + 改动 7 个代码文件**（另：包 README 重写、2 个检查脚本、`docs/README.md`
  索引加一行）。表里标 **本次** 的是「把 triton 后端调通」那一批改动（提交 `9c010e2`），
  没标的是这一关更早已经提交过的部分——一并列在这里，清单才是完整的：

| 文件 | 改动 |
|---|---|
| `rejection.py`（新增） | 批量执行层 `BatchedRejectionSampler`：`prepare_batch()`（逐行目标分布 + 提议分布）、`verify_batch()`（后端分派）、`materialize_results()`（唯一一次回传）；`_torch_backend()` 逐请求调既有验证函数（行为与第五十五关一字不改）；后端常量与错误码；模块级 `_row_layout()`（两个坐标系）与 `_greedy_draw()` |
| `rejection_rng.py`（新增） | counter RNG 的 CPU 参考：`philox4x32_10()`、`event_words()` / `event_word()` / `event_uniform()`、`exponential_race()`、`derive_rejection_seed()`；**本次新增 `make_rejection_seed()`**（`seed=None` 时从全局随机源取一个） |
| `rejection_triton.py`（新增） | 三个内核：`event_uniform_kernel()`、`verify_prefix_kernel()`、`sample_token_kernel()`。**192 号**：溢出判据改为按「本轮实际需要的事件」算（接受 EOS / 贪心 / 已报错都不需要那次 categorical）。**本次**：`_event_word()` 从 `_event_uniform()` 里拆出来（抽样要原始 32 位字，不能拿 `[0,1)` 的 float 再转回去）；`verify_prefix_kernel()` 多 `greedy_ptr` 与贪心分支（贪心不消费随机数）；`sample_token_kernel()` 的 `best_e` 显式声明 FP64、错误码改 2 |
| `rejection.py` 的 triton 段（**本次**） | **验收方 191 号第 3 条**：起始 counter 在主机侧校验（`rejection_rng.check_event_index`，与 CPU 参考共用同一个上界常量）、新增错误码 4；`_triton_backend()` 整体重写（两个坐标系 + 贪心 + 错误合并 + 不做数据依赖形状）、`_weight_rows()` 重写（收尾行、被拒位置夹住、按项的 `has_q_item`；`proposal_row_index` 保留 -1 哨兵、不再另存 `has_q`）、`_pack()`（列布局按 `max(kmax, 1)`、报错项不算 categorical）、`materialize_results()`（报错项返回空结论 + `num_rng_events` 统计）、`BACKENDS` 放开 triton、新增错误码常量 |
| `sample_runtime.py` | **验收方 191/192 号**：`is_speculative_item()` 只读计划项的 `speculative` 标志（判据含「请求已有输出」，覆盖抢占恢复的重算末块）、`_commit_verified()` 按 `num_real_inputs` 换算保留长度、CPU greedy 的 `argmax(...).tolist()` 只在 `uses_greedy_fast_path` 的后端上算（triton 那份白回传一次）；`__init__` 多 `rejection_backend` / `device` 两个参数并转给验证层；`run()` 的第 3/4 步重排（**先整批验证并检查错误，再按 `picked` 顺序提交**）；删 `_commit_drafts()` / `_commit_drafts_random()`（搬进 `rejection.py` 的参考后端），新增 `_needs_verification()` / `_commit_verified()`；`_is_greedy_without_penalty()` 移到 `rejection.py` 成模块级函数 |
| `scheduler.py` | `Scheduler.__init__` 多 `rejection_backend="torch"`；在 `add_request()` 里为 triton 后端派生 `seq.rejection_seed`（**只在这个后端下**，否则换后端会动到原有随机流）；计划项新增 `"speculative"` 标志（decode 分支打 True，prefill / 中间重算 chunk 打 False）——随机流路由只看它，不看本轮跑出几枚草稿（验收方 191 号第 1 条） |
| `engine.py` | `Engine.__init__` / `from_model_dir()` 多 `rejection_backend="torch"`；`_init_runtime()` 调 `check_rejection_backend(rejection_backend, device, speculative_mode)`，并把 `rejection_backend` 与 `device` 传给 `SampleRuntime` / `Scheduler` |
| `validation.py` | 新增 `REJECTION_BACKENDS` 与 `check_rejection_backend(rejection_backend, device, speculative_mode=None)`（三条拒绝规则见 §5；`speculative_mode` 这条是**本次**加的） |
| `request.py` | `SequenceConfig.__init__` 新增 `self.rejection_seed = None` 与 `self.rejection_rng_counter = 0` |
| `step56.py`（入口） | `main()` 加 `--rejection-backend {torch,triton}` 并透传给 `from_model_dir()` |
| `__init__.py` | 包说明改成第 56 关（模块清单补三行 `rejection*`）；导出 `BatchedRejectionSampler` / `RejectionItem` / `ItemResult` / `PackedResult` / `REJECTION_BACKENDS` / `derive_rejection_seed` / `make_rejection_seed` |
| `README.md` | 整体重写：用法、这一关的三条新约定、第五十五关就定下不能破的三条、验证表、遗留 |
| `benchmarks/check_step56_gpu_rejection.py` | 从「SKIP 工装」改成常驻用例：oracle 对齐贪心/错误语义、打桩采样器 `ScriptedSampler`、ragged 批、抽样内核逐事件对照、定向观测改口径、新增后端组合段 |
| `benchmarks/check_step56_real_qwen3.py` | 新增第 3 段：triton 后端的 greedy 等价、真跑过批、同 seed 可复现、随机采样跑完 |

`attention.py`、`cache.py`、`draft.py`、`loading.py`、`model.py`、`norm.py`、`rope.py`、
`sampling.py`、`speculative.py`、`formats/` 的**代码**未改（`loading.py` 只改了注释里的包名）。

## 0. 需求大概

第五十二到五十五关把投机的**算法**做完整了（n-gram 与 draft model 两条提议路径、
`min(1, p/q)` 接受、`max(p-q, 0)` 纠正、逐行惩罚历史、双 KV 回滚对齐），但**验证的执行
方式**一直没变：一条请求取一批标量回 CPU → Python 里决定接受/拒绝 → 再取下一批。这些回读
是**带数据依赖的决策**（这一枚接不接受，决定还要不要看下一枚、要不要抽纠正），所以既不能
批量也不能预取。量级：K=2 时每条请求 5 次，16 条请求 **80 次** `aten::_local_scalar_dense`，
每次 25–40µs 的固定往返，全部落在这一步的关键路径上、且随请求数线性增长。

**「等 GPU」的量级要说清楚**（免得把需求的口号当实测结论）：本引擎一次 target forward 的
**入队**就要 0.3–1 秒（Python 逐层建元数据），而 GPU 真正跑完只要 1–7ms。所以在**当前这个
引擎**里，走到验证代码时 forward 早就跑完了——实测在验证前插一次 `torch.cuda.synchronize()`
只阻塞 0.06ms，回读并没有真的在等。这里省下的是**往返与串行化**（CPU 在做这些往返时 GPU
空转，且下一步的 forward 要等它做完才能发起），不是等待时间。需求里「反复让 CPU 等 GPU」
描述的是 GPU 成为瓶颈的工况（批量大、CPU 侧被图模式和更少的 Python 削薄），那时第一次回读
确实必须等 forward 跑完。两种工况都成立，只是本引擎当前落在前一种。详见附篇 §一 第 5 条。

这一关把**决策搬进 GPU**，只把结论拿回来：

1. 一轮的验证做成**一次批量计算**：判定接受前缀、算纠正/bonus 分布、抽 token 全在设备上；
2. 结果打包成**一张**固定容量张量，**一次**回传，不逐请求、不逐字段；
3. 随机数改成 **counter-based**（随机数 = f(请求 seed, 事件编号, token 下标)；比需求 §4 的
   式子少一维「事件类型」，理由见 §2.1），
   因为批量路径要「为整段 K 提前算好候选随机数」，只有与事件编号绑定的随机数才能保证
   「**没发生的事件不推进计数器**」——这条是这一关随机流语义的地基；
4. 判据不是「跑得快」，而是**与同一套算法的 CPU oracle 逐位一致** + 分布/随机流检验 +
   定向观测（`verify_batch()` 内没有任何回传）。

参考路径（`"torch"`）原样保留：CPU 能跑、行为与第五十五关一字不改，两个后端在引擎创建时
**二选一固定**，运行中不切换。

另有两篇附篇，都不贴代码：

- [step56_triton_kernel_design.md](step56_triton_kernel_design.md)：设计思路（原来差在哪、
  内核担什么责任、要备哪些输入、易错点与代价），开头先把 counter RNG 与指数竞赛两个名词讲清；
- [step56_state_space.md](step56_state_space.md)：**状态空间地图**——这一关有哪些轴、哪些
  组合不可能出现（不变量）、每个函数实际要同时装几个量、哪些轴容易混。读代码前先看这页；
- [step56_vs_vllm_rejection.md](step56_vs_vllm_rejection.md)：与 vLLM 0.28 的逐项对照——
  同一件事的两种取舍，以及「我们为什么看起来更绕」（哪些轴是需求逼出来的）。

## 1. 数据流：三个入口与一轮的顺序

`rejection.py` 的 `BatchedRejectionSampler` 只负责**产生验证结论**，三件事分开：

| 入口 | 在哪算 | 做什么 |
|---|---|---|
| `prepare_batch(logits, items, greedy)` | CPU 元数据 + 设备上的张量 | 每项的 K+1 行目标分布 `p`（逐行惩罚历史）、每枚草稿的提议分布 `q`、还有哪些项走贪心 |
| `verify_batch(batch)` | 后端分派 | `"torch"`：逐请求调 `verify_drafts()` / `verify_drafts_random()`；`"triton"`：两个内核 + 向量化打包，产物仍留在 GPU 上 |
| `materialize_results(outcome)` | CPU | 把后端产物变成 `ItemResult` 列表；triton 后端在这里做**唯一一次** `.cpu().tolist()` |

它**不认识** `Scheduler`、不释放请求、不提交 token。回滚、对齐、提交、收尾仍然是
`SampleRuntime.run()` 的事，而且严格按 `picked` 顺序做（`sample_runtime.py`）：

```text
1) 行映射（纯）：scheduled_items -> rows（原始输入行号）+ picked
2) 没有草稿的项   -> 采样后端整批 select_batch（一次 .tolist()），**先不提交**
3) 贪心快路径项   -> 整批一次 argmax（一次回传）
4) 投机项         -> prepare_batch -> verify_batch -> materialize_results
                     **整批先检查错误**：任何一项非法就抛，绝不「提交了半批再报错」
5) 按 picked 顺序提交：普通项走 _commit_tokens，投机项走 _commit_verified
                     （truncate 两套 KV -> 记账 -> 经唯一提交入口落 token）
6) 收尾：把「预留了却没算到」的整块还给池子（对已验证的项是空操作）
```

步骤 2/3 的「先算好、后提交」不是风格问题：提交要在第 5 步按 `picked` 顺序统一做，
否则事件顺序会变成「先普通后投机」。第 4 步那个「整批先检查再提交」也是硬约束——
triton 后端把非法输入编码成结果张量里的**错误码**（而不是像参考路径那样当场抛），
所以必须整批拿到结论之后才决定提交谁。

## 2. counter-based 随机流

### 2.1 事件与四字计数器

一个逻辑事件是 `(事件编号 event_index, token 下标 token_index)`，喂给
**Philox-4x32-10**（Random123 的标准 counter-based 生成器）：

```text
c0 = 事件编号（int64 截到 32 位；每请求独立计数）
c1 = 0（保留）
c2 = 词表内下标（接受事件恒为 0；抽样事件按 token ID 区分）
c3 = 0（保留）
k0, k1 = 请求 seed 的低/高 32 位
```

**比需求 §4 的式子少一维「事件类型」**，是有意的：事件编号本身已经保证「每个被消费的事件
各不相同」，接受事件与抽样事件不可能撞上同一个编号，多一维类型并不带来额外隔离。代价只有
一个——同一个 seed 下派生出来的具体数字与带类型时不同；这无所谓，因为对照测试两边用的是
同一套公式（CPU 参考与 GPU 内核逐位一致，两万个事件零差异）。判据里原来那条「换事件类型
给出不同随机数」随之删掉，其余隔离性质照旧。

`event_index` 就是这条请求**已消费的事件数**（`SequenceConfig.rejection_rng_counter`）
加本轮第几个候选事件。每请求独立计数，所以请求重排、别的请求插进来、抢占重算都不会
改变同一条请求的随机流。

### 2.2 为什么不是「预抽 K 个 uniform」

最省事的做法是每条请求进验证时预抽 `K` 个均匀随机数。**这样随机流会错位**：

- 第一枚草稿就被拒（或第一枚就是终止 token）时，后面的位置根本没发生接受判断，预抽的
  随机数却已经被消耗掉了，下一轮的事件编号就对不上了；
- 反过来，「必接受 / 必拒绝」的位置按约定**不抽**随机数（抽了也是白抽），预抽也没法表达
  这件事。

counter-based 的规则是「事件的随机数只由它的编号决定」：没发生的事件编号**根本不用**，
自然不会推进计数器。GPU 侧仍然可以**为整段 K 提前算**候选随机数（内核就是这么写的），
只是用不到的那些不体现在计数器的增量里——这正是 counter-based 相对「顺序抽取」的意义。

### 2.3 消费规则（测试逐条数）

| 情况 | 消费的随机事件 |
|---|---|
| `0 < min(1, p[d]/q[d]) < 1` | 一个接受事件 |
| 必接受（比值 ≥ 1）/ 必拒绝（比值 ≤ 0） | 不消费 |
| 首次拒绝后的纠正抽样、全部接受后的 bonus 抽样 | 一个抽样事件（词表内部由 token ID 区分） |
| 贪心请求的纠正 / bonus | 不消费（逐行 argmax） |
| 贪心请求的接受判断 | 不消费（纯比较，见 §5） |
| 接受的草稿是终止 token | 当场结束，其后的位置与 bonus **一个都不消费** |
| 报错的项 | 只算它真的抽过的那些接受事件，不算 categorical |

「接受终止 token 之后一个随机数都不能再抽」是第五十五关就定下的约定（否则这条请求的
随机流与「没有投机时」错位），本关只是把它搬到 GPU 上并**逐条数**。

### 2.4 一次 categorical 抽样：指数竞赛

纠正/bonus 要从一行权重里抽一个 token，而词表是 151936——不可能一个 program token 一个
线程地做前缀和。这里用**指数竞赛**：对每个 token 取 `e_i = -log(u_i) / w_i`（`u_i` 是这个
事件在 token `i` 上的随机数），取最小的那个。独立指数分布竞速恰好等价于按 `w` 抽样。

三个附带好处，都是实现里用得上的：

- **不必归一化**：全体权重乘一个正常数不改变 argmin，所以纠正分布 `max(p-q, 0)` 不用在
  内核里归一化（CPU 参考也同样不归一化，两边才对得上）；
- `w_i = 0` 给 `+inf`，永不入选——`max(p-q, 0)` 的零质量就是这样排除的；
- 归约可以**分块**做（每个 program 一行、词表切块，逐块取更小者），不需要把整行读进来。

`u = (x + 0.5)·2^-32` 在 FP64 里算：落在 (0,1) **开区间**内，不会出现 `log(0)` / `log(1)`
那种边界；`x` 必须是 Philox 的**原始 32 位字**，不能拿 `[0,1)` 的 float 再转回来
（那样只剩高 8 位，`u` 恒等于 0，`-log(u)` 是垃圾——见 §6 的反证）。并列取**较小下标**，
与 CPU 侧的 `argmin` 对齐。

### 2.5 seed 派生

- `derive_rejection_seed(seed)`：由请求的 `seed` 线性同余混合而来，常数与 draft 侧**不同**
  （同一个请求的 draft 流与拒绝验证流不能是同一条）；
- `make_rejection_seed(params)`：`seed=None`（调用方没要求复现）时从**全局随机源**取一个。
  **不能退化成 0**：那样所有未播种的请求会共用同一条随机流——同一个事件编号拿到同一个
  随机数，它们的接受/拒绝完全相关，比「不可复现」坏得多。
- 种子在 `Scheduler.add_request()` 里**只对 triton 后端**派生：torch 路径不看这个字段，
  留成 `None` 才能保证「换后端不改变原有随机流」（否则白抽一次全局随机源，未播种请求的
  draft 流会跟着变）。

一个副产品：**同一个 seed 的 triton 运行可以逐 token 复现**（两条随机流都由请求 seed
唯一决定，counter RNG 又无内部状态）。这一条在真实模型上验过。

### 2.6 事件编号的 32 位契约（越界必须报错，不能静默回绕）

事件编号就是计数器的 c0，**只有 32 位**。越界必须明确报错——静默截断意味着「第 2³² 个事件」
复用事件 0 的随机数，输出看着完全正常，事后查不出来。两处各管一半：

| 情况 | 谁查 | 结果 |
|---|---|---|
| **起始** counter ≥ 2³² 或为负 | 主机侧（它是 CPU 已知的整数，无需读 GPU） | `ValueError`，整批不启动 |
| 起始合法，**本轮抽着抽着跨过** 2³² | 内核自己数——按**本轮实际需要用到哪一个编号**，不是按计数器指向的下一位 | 错误码 4，该项整项作废 |

内核那条判据的细节（验收方 192 号第 2 条：上一版把「用完最后一个合法事件」误判成溢出）：

```text
接受事件用掉的是        base .. base + consumed - 1     ← 最后一个是 base + consumed - 1
纠正/bonus 用掉的是     base + consumed                 ← 只有真要抽时才用
```

所以「还要不要用 `base + consumed`」取决于三件事：**在接受 EOS 上停下的项不抽**（没有 bonus）、
**贪心不抽**、**已报错的项不抽**。三者都不成立时才把上界算到 `base + consumed`。四类边界都
有用例钉着：最后合法事件用于接受 EOS（通过）、用于 K=0 的 categorical（通过）、用于接受判断但
之后还要抽 bonus（报错）、接受事件本身就用到了越界编号（报错）。

上界定义在 `rejection_rng.EVENT_INDEX_LIMIT`（CPU 参考与 GPU 路径**共用**这一个常量，不许
各写一份）。上游的 `SequenceConfig.rejection_rng_counter` 是 Python 整数（无上限），所以这道
检查不是形式——没有它，2³² 会安静地绕回 0。

## 3. 两个坐标系（以及踩过的坑）

`_row_layout()` 把一批项摊平成两个坐标系，全部是 **CPU 侧整数**，不读任何 GPU 标量：

```text
行坐标系：每项 K+1 行（K 枚草稿行 + 收尾行）首尾相接，row_offsets[i] 是第 i 项的起始行
位置坐标系：只覆盖草稿位置（P = ΣK），pos_offsets[i] 是第 i 项的起始位置
           ratio / eos / invalid 与接受前缀内核都按它排
```

分两套是因为**它们要回答的问题不同**：接受判断只关心草稿位置（K 个），而纠正/bonus 要从
「拒绝点那一行」或「收尾行」取整行的权重——后者在位置坐标系里根本没有对应项。

四个必须记住的约定（每个都对应一次真实的错）：

1. **收尾行必须在行坐标系里。** `kind == 0`（全部接受）时 bonus 就是从第 K 行抽的，
   只收前 K 行的话 `rows_of_item + K` 就指到别人的行上去了（高级索引不查边界，读到的
   是别的请求的分布，不会报错）。
2. **被拒草稿的位置必须夹住。** `kind != 1` 的项算出来的位置会越界（全接受的项
   `num_accepted == K`，正好越过本项），这个下标会喂给 `scatter_`——**越界下标在
   `scatter_` 里直接触发 CUDA device-side assert**。夹住之后读到的是一个合法 token id，
   而这条分支马上被 `kind == 1` 的 where 丢掉。
3. **`q` 只按位置收「真有 q」的项。** ngram 的提议是确定性的（`q` 是 d 上的 one-hot），
   物化整行是纯浪费：`has_q[j]` 为假的位置 `q[d]` 按 1.0 算。注意这是**按位置**的属性，
   而「用不用 `max(p-q,0)`」是**按项**的属性（`has_q_item`），两者不能混用。
4. **不做数据依赖的形状。** 不能按 `kind` 去筛子集（`mask[bool_mask]` 会把形状从设备拷回
   主机，是一次隐式同步），所以接受终止 token 的项也照常参与抽样，结论随后丢掉。

两个退化形状也要走通（都有常驻用例）：单批里 K=0 的项（直接抽它唯一的目标行）、
整批 K 全为 0（输出列只剩哨兵列 + 五个字段列，所以列布局按 `max(kmax, 1)` 算）。

## 4. 打包与结果契约

结果张量的每一行：前 `max(kmax,1)+1` 列是 `output_ids`（未使用的位置是哨兵 `-1`），
之后依次是长度、接受数、保留输入数、消费的随机事件数、错误码。

`kind` 是「接受前缀为什么停下来」，三个取值（`rejection.py` 顶部有同名常量，与错误码是
**两个不同的命名空间**：`KIND_FIRST_REJECT == 1` 说的是首拒绝，`ERR_INVALID_PROPOSAL == 1`
说的是非法提议）：

| `kind` | 含义 | `accepted` | 抽不抽 | 权重取哪一行 |
|---|---|---|---|---|
| `KIND_ALL_ACCEPTED`(0) | K 枚全接受（K=0 也在这支） | `= K` | 抽 bonus | 收尾行（第 K 行） |
| `KIND_FIRST_REJECT`(1) | 第 `accepted` 枚被拒 | `< K` | 抽纠正 | 第 `accepted` 行，分布换 `max(p-q,0)` / 挖掉 d |
| `KIND_ACCEPTED_EOS`(2) | 刚接受的那枚是终止 token | `>= 1` | 不抽 | — |


```text
lengths = accepted + (kind != 2)                 接受终止 token 时不算 bonus
kept    = 1 + accepted - (kind == 2)             本轮输入保留几个位置
rng     = consumed + (kind != 2) & ~greedy & (errors == 0)
```

错误码四种：`1` = 非法提议（`q[d] = 0`，从 q 里抽不出一个 q 质量为零的 token）、
`2` = 无剩余质量（纠正/bonus 的权重整行为零）、`3` = 贪心落进了「要抽随机数才决定接受」
那一支（= 「贪心的目标分布是 one-hot」这个前提被破坏了）、`4` = **本轮消费的事件编号跨过了
2³² 上界**（起始 counter 越界在主机侧就报错；「抽着抽着跨过」只有内核数得清——见 §2.6）。报错的项在 `materialize_results()` 里返回
**空结论**（`committed_ids = []`、`num_accepted = 0`、`kept_inputs = 0`），配合
`SampleRuntime` 的「整批先检查再提交」，非法输入不会在报错前已经提交了半批。

`torch` 后端不走这张张量：它逐请求调用既有函数，产出的就是 `ItemResult`，
`rng_consumed` 恒为 0（随机流由生成器自己推进）。

## 5. 后端组合与退化形状

后端在引擎创建时固定，`validation.check_rejection_backend()` 在构造时明确拒绝三种组合：

| 组合 | 结果 |
|---|---|
| 未知后端名 | `ValueError`，列出可选值 |
| `"triton"` + 非 CUDA 设备 | `ValueError`（内核与 counter RNG 都在 GPU 上） |
| `"triton"` + 不开投机 | `ValueError`——那时没有任何一项需要验证，它会**空转**，而普通采样的随机流也不归它管。留一个「看起来开了、实际什么都不做」的组合，就是「悄悄改用参考循环还报告成功」的另一种形态 |

### 5.1 「走哪条随机流」怎么路由：**只看配置与本轮是否采样**

这是验收方 191 号第 1 条与 192 号第 1 条两次打回的地方，判据定死成**三个条件合取**
（调度器在计划项上打 `"speculative"`，`SampleRuntime` 只读它）：

| 条件 | 为什么 |
|---|---|
| 引擎开了投机 | 不开投机时 triton 后端本来就被拒 |
| **本轮真的要采样**（`can_sample`：算到了历史末尾） | 中间的重算 chunk 不采样，不进验证批 |
| **这条请求已经生成过 token**（`len(output_ids) > 0`） | 首次 prefill 的采样按既定策略走普通后端（旧 generator）；从那以后这条请求的每一次采样都留在 counter 上 |

**判据里没有「本轮走的是 decode 还是重算」，也没有「本轮跑出几枚草稿」**——这两件事分别是
「KV 怎么算」和「运行结果」，都不该决定下一枚 token 用哪条随机流。两个后端各自的规则：

| 后端 | 规则 |
|---|---|
| `torch` | 有草稿才走验证（第五十五关的行为一字不改）。这个后端里两条路抽随机数用的**都是** `sampling_state.generator`，所以 K=0 退回普通采样没有副作用 |
| `triton` | 三个条件同时成立就走验证——**包括**「按投机计划、实际一枚草稿都没有」（预算缩零、draft 池不够、ngram 无匹配），也**包括**「抢占后重算到末尾、直接产出下一枚 token」的末块 |

191 号打回的是前半段（按「有没有草稿」判，K=0 的投机行会切回旧 generator）；192 号打回的是
后半段，例子最清楚：

```text
A 生成 3 枚后被优先级抢占 → 恢复时按预算重算历史（10 枚）→ **末块算完直接采样**
（这一轮：已有输出 3 枚、本轮输入 2 枚、走的是重算分支 → 旧判据给 speculative=False）
```

插不插入那个抢占者，A 的第四枚输出就变了——而 A 全程 K=0、每轮都从同一个均匀分布抽一枚，
本该用同一串随机事件。所以路由必须看「这条请求是否已经在生成」，而不是「这一轮 KV 怎么算」。

**首次 prefill 走普通后端**是明确写下来的策略（不是遗漏）：它只在请求生命周期的开头发生一次，
切换点由 prompt 长度决定，不随调度漂移，所以同 seed 复现仍然成立。**这样「只在首次
prefill → decode 之间切一次随机流」这句话才真正成立**（192 号指出：在上一版里它不成立，
因为抢占恢复的重算末块还会切回去）。

**重算末块的长度要在回滚时算对**：验证层数出来的 `kept_inputs` 里含一个隐含前提——「本轮只算
了 **1 枚真实 pending token**」（`1 + accepted − EOS` 的那个 1）。decode 行成立；重算末块可能
一次算好几枚，照搬会把刚算好的 KV 裁短（10 枚裁成 9 枚）。所以计划项另带
`num_real_inputs`，`_commit_verified()` 里按 `kept_inputs + (num_real_inputs − 1)` 换算。

**两份名单同源**：`run()` 里「走普通采样」与「走验证」都读同一份 `verify = {id(item): ...}`
（同一个 `_needs_verification` 判据），互斥且完备。各写一个表达式的话，K=0 的投机行会同时落进
两边——普通采样那份算完被丢掉，但请求自己的 `torch.Generator` 白白前进了一格，下次它真走普通
路径时抽到的随机数就跟着漂了。

**另一处少算的回传**：`run()` 开头那份 CPU 侧 `argmax(...).tolist()` 只在后端真的会走
`greedy_fast` 快路径时才算（`uses_greedy_fast_path`）。triton 后端一律走一般路径，那份结果没人
用——验收方在真实 tiny 双模型上数到过「K=2 与 K=1 的轮次各有 2 次结果回传」（一次 argmax 的
`.tolist()`、一次验证结论的 `.cpu()`），现在只剩后一次。

## 6. 验证

### 6.1 脚本清单（全部通过）

| 脚本 | 项数 | 覆盖 |
|---|---|---|
| `check_step56_gpu_rejection.py` | 39 | 本关主脚本：与 CPU oracle 逐位对照、ragged 批、抽样内核逐事件对照、分布/流隔离、定向观测、后端组合 |
| `check_step56_rejection_rng.py` | 9 | counter RNG：CPU/GPU 逐位一致（2 万个事件）、均匀性、流隔离、32 位边界、指数竞赛分布 |
| `check_step56_real_qwen3.py` | 15 | 真实 Qwen3-1.7B + 0.6B 端到端：greedy 逐 token 相同、triton 后端真跑过批、同 seed 可复现 |
| `check_step56_{speculative,batch,rejection,random,engine,combinations,draft_kv,loading}.py` | 316 | 从第五十五关移植的回归套件（默认 torch 后端） |
| `diff_step55_step56.py` | 88 | 投机关闭时与第五十五关逐步逐字节对照 |

### 6.2 有判别力的几条

- **与 CPU oracle 逐位一致**：`check_step56_gpu_rejection.py` 自己写了一份 CPU oracle
  （`oracle_verify()`），用的是**同一个 counter RNG 与同一个指数竞赛**——不是「拿不同随机
  算法的同 seed 输出强行比较」。18 条用例 × 两个事件起点，比的是输出 token、接受数、
  保留 KV、消费的随机事件数四元组。
- **抽样内核逐事件对照**：同一条权重行、512 个事件编号，GPU 内核与 CPU `exponential_race()`
  给出**完全相同的 token 序列**，且权重故意不给成归一化的。
- **ragged 批**：K = 2 / 0 / 2 / 1 混在一张批里逐项对照。§3 里那些「按项的行区间」的坑
  全是这条用例抓出来的。
- **反证（去掉修复就红）**：
  1. 去掉被拒位置的 `clamp` → `CUDA error: device-side assert triggered`（§3 第 2 条）；
  2. 抽样内核把原始 32 位字换回 `_event_uniform()` 的 float → 6 项对照失败，接受判断
     仍然一致、**抽出来的 token 全错**（`u ≡ 0`）。
- **定向观测**（`verify_batch()` 内）：
  - 同步点数与批大小无关（16 条请求 16 次 vs 1 条请求 16 次；都是每步固定的元数据上传）；
  - `.item()` / `.cpu()` / `.tolist()` **一次都没有**（用 `torch.cuda.set_sync_debug_mode
    ("warn")` 与逐个打桩两个独立口径）；
  - `materialize_results()` 里整批**恰好一次**回传。
- **真实模型**：triton 后端在 Qwen3-1.7B + 0.6B 上 greedy 输出与普通 greedy **逐 token
  相同**（贪心两边都不抽随机数，这条是最硬的端到端等价判据）；同 seed 两次运行逐 token
  相同；两套 KV 池结束后引用归零。

### 6.3 不承诺的事

不测 tok/s、不把接受率当质量指标（接受率低不代表实现错）。本关的收益是「去掉每条请求的
CPU↔GPU 往返」，不是「端到端更快」——draft 提议阶段与概率构造的逐行 Python 循环原样没动。

## 7. 接口变化与遗留

### 7.1 接口变化

| 位置 | 变化 |
|---|---|
| `rejection.py` | 新增 `BatchedRejectionSampler(backend, eos_token_ids, sampler, device)`：`prepare_batch` / `verify_batch` / `materialize_results`；`BACKENDS = ("torch", "triton")`；`PackedResult` / `RejectionItem` / `ItemResult` |
| `rejection_rng.py` | 新增：Philox-4x32-10 CPU 参考、`event_word` / `event_uniform` / `exponential_race`、`derive_rejection_seed` / `make_rejection_seed` |
| `rejection_triton.py` | 新增：`event_uniform_kernel`、`verify_prefix_kernel`、`sample_token_kernel` |
| `SampleRuntime.__init__` | 多两个参数：`rejection_backend="torch"`、`device=None`（设备**显式**传，不从采样器猜） |
| `Scheduler.__init__` | 多一个参数：`rejection_backend="torch"`；`add_request()` 在 triton 后端下派生 `seq.rejection_seed` |
| `Engine` | 公开参数多一个 `rejection_backend="torch"`（`__init__` 与 `from_model_dir`） |
| `SequenceConfig` | 新增 `rejection_seed` / `rejection_rng_counter`（默认 `None` / `0`，torch 后端下不用） |
| `Scheduler` 的计划项 | 新增 `"speculative"`（bool）：这一项要不要用 counter RNG 采样。由「投机开启 + 本轮要采样 + 请求已有输出」三个条件合成，随机流路由的唯一判据（191 号第 1 条 + 192 号第 1 条）；另新增 `"num_real_inputs"`：本轮算了几个**真实** token，供重算末块换算 `kept_inputs` |
| `ItemResult.kept_inputs` | 口径不变（「本轮输入保留几个位置」），但**调用方要换算**：它含一个隐含的「1 枚真实 pending token」，重算末块按 `num_real_inputs` 补差额 |
| `validation.py` | `check_rejection_backend(rejection_backend, device, speculative_mode=None)` |

默认值全是 `"torch"`：不传 `rejection_backend` 的调用方行为与第五十五关完全一致。

### 7.2 遗留

- **每步 O(1) 的元数据上传没有消除**：`verify_batch()` 里仍有约 16 次小的
  `torch.tensor(..., device="cuda")`（位置、种子、计数器这些），每次都是一次 H2D 拷贝。
  它们与请求数无关（16 条与 1 条请求的同步点数一样），所以不是「每请求一次」，但也确实
  是每步的固定开销。要再往下压，得把元数据拼成一张张量一次上传、或用 pinned 暂存 +
  `non_blocking`。
- **抽样内核是「按次」而不是「按行」计费的**：随机数按 token 下标取，所以抽一枚 token 要对
  15 万个候选各算一次 Philox；而内核里**一个 program 负责一行、串行走 38 个词表分块**，
  于是它的墙钟由**单行的串行循环**决定——实测 1/2/4/8/16/32 行都是 ~1.45ms，到 64 行才
  略涨（1.61ms）。所以：
  - 参考路径 5.57ms、triton 3.80ms（16 条请求，反超约 1.8ms），多出来的那 ~1.45ms 是
    **每次启动的固定代价**，不是「每行多少钱」；
  - 1 条请求时参考路径 0.37ms、triton 3.66ms（**反而慢十倍**）——小批亏在固定代价上；
    交叉点大约在 N≈4–8。
  要压的话方向是把这个串行分块归约做掉（例如多个 program 分块 + 二级归约，或按块取随机数
  + 块内用更便宜的抽样），而不是改回顺序抽——那会违反「随机数按 token 下标取」这条语义。
- **本关只覆盖「验证」**：普通（非投机）采样的随机源仍然是 `sampling_state.generator`，
  没有换成 counter RNG；于是「同一个请求的普通采样与投机验证」用的是两套不同的随机机制。
- **draft 提议阶段的同步没动**：逐行 `distribution()`（Python 循环）、提议层的
  `int(...)` 取草稿 id 都还在。
- `_pack()` 里仍有 `torch.cat` 的一次分配；整批一张结果张量，因此每步的显存占用随
  `批大小 × max(kmax,1)` 走，暂不打算做成复用缓冲。
