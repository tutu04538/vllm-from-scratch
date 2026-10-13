# 第七十五关：V2 块验证、Synthetic 模式与分块 Logits

> 需求：`learning_notes/14_vllm_from_scratch/投机解码完整需求/075_V2块验证Synthetic模式与分块Logits.md`
> 基线：本机 `vllm==0.28.0` 文件快照（2026-10-03）。执行路径：**V2 为主**（`minivllm/worker/gpu/`），
> synthetic **同时回归 V1**（`minivllm/sample/rejection_sampler.py`）。

> 📖 **零基础读者**：先看文末的 **[附录 A](#附录-a用大白话讲块验证和-synthetic零基础版)** ——
> 它用"老师（大模型）/ 学生（小模型）"的比喻 + 两个能手算的例子，把块验证与 synthetic 讲一遍，
> 再看正文的实现与差异账本。

## 0. 这次做了什么（两三句话）

73 关把 V2 的验证采样器搬过来时，块验证（`block`）与合成接受率（`synthetic`）两条分支**只搬了内核
没接配置**：配置期与采样器构造期都明确报错。这一关把两者接上，并给"分块 logits"补上 79 关要用的
保护与验收：

    standard   逐位置概率比检验（59 关已做），本关**保留为对照**。
    block      Sun et al. 2024：整块联合验证。新增的递推在
               `_compute_cumulative_log_p_kernel`（累计对数比，**上限加在累计值上**）与
               `_compute_local_residual_mass_kernel`（下一位置的残差质量）里；阈值
               `h_i = M / (M + 1 - p_i)`，接受长度取**最后**通过测试的位置；恢复分布是
               `max(p_tau · target - draft, 0)` 的归一化形式。
    synthetic  不做 p/q 检验，按配置给的**条件**接受率 `u < c_i` 接受草稿。
               `c_i` 由配置里的**无条件**概率换算（`unconditional_to_conditional_rates`）。
    分块       仍按**整条请求**切（绝不切开请求），chunk 内 `cu_num_logits` 重置为局部偏移，
               `draft_logits` 保持全局 slot 寻址；`logits.shape[0] <= max_chunk_logits` 这条
               单块分支就是 79 关"压紧后不能用陈旧 CPU offsets 挑边界"的保护。

## 1. 路径映射（本项目 ↔ 上游）

| 本项目 | 上游 | 状态 |
|---|---|---|
| `minivllm/config.py::SpeculativeConfig` | `config/speculative.py` | 新增 `synthetic_acceptance_rates` / `synthetic_acceptance_length` 两个字段；`_acceptance_length_to_rates` / `_resolve_synthetic_acceptance_rates` 逐行同义（连错误原文都同）；`rejection_sample_method` 的取值校验从"只接 standard"改成 `{standard, synthetic, block}` |
| `minivllm/spec_decode/utils.py::unconditional_to_conditional_rates` | `v1/spec_decode/utils.py:598-601` | 逐行同义（无条件的"前 i+1 枚全接受"→ 条件率 `c_i = p_i/p_{i-1}`） |
| `minivllm/worker/gpu/spec_decode/rejection_sampler.py::RejectionSampler.__init__` | `v1/worker/gpu/spec_decode/rejection_sampler.py:75-96` | 模式分派：`synthetic` 建条件率表（device 上）、`block` 置 `use_block_verification`；这两个布尔就是全部开关 |
| 同上 `_verify_in_chunks` | 同上 L152-226 | 补上单块分支的注释（79 关保护）；切块算术未改（73 关已按上游移植） |
| `minivllm/worker/gpu/spec_decode/rejection_sampler_utils.py` | `v1/worker/gpu/spec_decode/rejection_sampler_utils.py` | 73 关整文件移植时已含 `SYNTHETIC_MODE` / `USE_BLOCK_VERIFICATION` 两条分支与五个块验证内核，**本关未改一行**（只补了调用方） |
| `minivllm/sample/rejection_sampler.py` | `v1/sample/rejection_sampler.py:70-90, 723-830` | V1 的 synthetic：两条内核各加 `uniform_probs_ptr` / `synthetic_conditional_rates_ptr` / `SYNTHETIC_MODE`，`rejection_sample()` 加两个参数；⚠️ **block 在 V1 明确报错**（见 §5） |
| `minivllm/worker/gpu_model_runner.py` | `v1/worker/gpu_model_runner.py` | 构造 `RejectionSampler` 时把 `device` 传进去（条件率表要在 GPU 上读） |
| `minivllm/testing/torch_block_verification.py` | 无（测试参考实现） | 块验证的 **Torch 数学参考**：按源码递推写的 `p_i` / 阈值 `h_i` / 残差分布 / τ 分布闭式；只给测试 |
| `tests/step75/spec75_helpers.py` | 无（测试夹具） | V2 扁平布局构造器、`BlockCase` → 扁平布局的摊平器、V1 metadata 夹具、tiny 引擎夹具 |

## 2. 块验证的数学（照源码递推，不是照论文公式）

需求 075 §3.2 的原话是"`p_tau` 按源码递推，不能凭变量名自行替换成简单联合概率乘积"。这里的每一步
都能在 `rejection_sampler_utils.py` 里找到对应行：

1. **前缀联合比**（`_compute_cumulative_log_p_kernel`）：
   `log_p_i = min(log_p_{i-1} + (log p(d_i) - log q(d_i)), 0)`，`p_i = exp(log_p_i)`。
   上限加在**累计值**上——某一步比值 > 1 能补前一步的亏空；这不是"每步 `min(1, p/q)` 再累乘"。
   实测差别：K=3、σ=1.0 的合成分布下，朴素模型给 E[τ]=1.03，内核给 1.75（差 70%）。
   `-1` 占位草稿**跳过归约并沿用上一个累计值**（内核注释同款）。
2. **阈值**（`_compute_local_residual_mass_kernel` + `_compute_global_residual_mass`）：

       M_i = Σ_x max(p_i · M_b(x | 第 i+1 行) - M_s(x | 第 i+1 行), 0)      # 残差质量取自**下一行**
       h_i = M_i / (M_i + 1 - p_i)        # denom ≤ 0 时取 1.0（M=0 且 p_i=1 的兜底）
       h_i = p_i                          # 最后一位（没有"下一行"）或下一位是 -1 占位

   点质量草稿（`draft_probs=None`）走闭式 `M_i = p_i · (1 - M_b(d_{i+1}))`（`HAS_DRAFT_LOGITS=False`）。
3. **接受长度**（`_rejection_kernel` 的 `USE_BLOCK_VERIFICATION` 分支）：**不早停**——每一位都掷硬币，
   `accepted_length = max{i+1 : u_i ≤ h_i}`（都不满足则 0）。非贪心下 `-1` 占位会终止整块验证。
   贪心请求根本不进这条分支（`is_greedy` 先判）。
4. **恢复分布**（`_resample_kernel` 的块分支）：`max(p_tau · target - draft, 0)` 归一化，
   `p_tau = exp(cumulative_log_p[rejected_idx - 1])`（τ=0 时 1，所以首拒时块残差 == standard 残差）；
   点质量草稿下 `p_tau` 在归一化里约掉 → 直接"target 去掉被拒草稿 token"；τ=K（全接受）→ bonus 行。

## 3. kernel 输入输出表（需求 075 §5 点名要的）

`RejectionSampler._verify()` → `rejection_sample()` 的六个内核，按调用顺序：

| 内核 | 输入（形状/含义） | 输出 | 块验证用到的输出 |
|---|---|---|---|
| `_compute_local_logits_stats_kernel` | `target_logits [num_logits, V]`（**已按温度缩放**）；`draft_logits [max_num_reqs, K, V]`（**未除温度**，内核现除）；`expanded_idx_mapping [num_logits]`；`expanded_local_pos [num_logits]`；`temperature [max_num_reqs]` | `target_local_argmax/max/sumexp`、`draft_local_max/sumexp`，各 `[num_logits, cdiv(V, 8192)]` | 贪心行的 argmax、全局 logsumexp（残差质量与 logprob 都要） |
| `_compute_cumulative_log_p_kernel`（**仅块验证**） | 上面的 local max/sumexp；`draft_sampled [num_logits]`；`cu_num_logits [num_reqs+1]`；`idx_mapping [num_reqs]`；温度 | `cumulative_log_p [num_logits]`：`[start+i] = log p_{i+1}` | `p_i`（阈值）与 `p_tau`（残差） |
| `_compute_local_residual_mass_kernel`（**仅块验证 + 有草稿分布**） | 上面的 + `cumulative_log_p`；按 `(num_logits, cdiv(V, 8192))` 网格 | `local_residual_mass [num_logits, num_blocks]`：每块的部分和 | 第 i 步读第 `i+1` 行求和得 `M_i` |
| `_rejection_kernel` | 上面的 + `seeds [max_num_reqs]`、`pos [num_logits]`、`synthetic_conditional_rates [K]` 或 None | `sampled [num_reqs, K+1]`、`num_sampled [num_reqs]`（= 接受数+1）、`target/draft_rejected_logsumexp [num_reqs]` | 接受长度；被拒位的两个 logsumexp 给恢复分布用 |
| `_resample_kernel` | 上面的 + 两个 logsumexp；网格 `(num_reqs, cdiv(V, 1024))` | `resampled_local_argmax/max [num_reqs, num_blocks]`（每块一个赢家） | 块残差 `max(p_tau·M_b − M_s, 0)`；`APPLY_TEMPERATURE=False`（logits 已经缩放过） |
| `_insert_resampled_kernel` | `resampled_local_*`；网格 `(num_reqs,)` | 原地写 `sampled[req, num_sampled]` 并把 `num_sampled` **无条件 +1** | 交付序列 = `sampled[0 : num_sampled]` |

`num_rejected = num_logits - num_sampled` 由 Runner 侧另算（`get_num_sampled_and_rejected`）。

## 4. synthetic 的语义（与"无损"的边界）

* 配置里写的是**无条件**概率（"前 i+1 枚全被接受的概率"），长度必须 == K、每项 ∈ [0,1]、单调不增；
  也可以只给 `synthetic_acceptance_length ∈ [1, K+1]`，配置期换算成"最小方差档位表"后清空该字段。
  例：K=4、length=2.5 → `[1.0, 0.5, 0, 0]`（Σ = 1.5 = 期望接受枚数）。
* 内核里用的是**条件率**：`[0.8, 0.4, 0.2] → [0.8, 0.5, 0.5]`；直接拿原数组逐位置抽签会让
  `P(τ≥2)` 从 0.4 掉到 0.32（实测差 0.08 = 容差的 3 倍以上）。
* 贪心行**也要**掷这一次硬币（接受判定完全不看 target argmax），所以 `uniform_probs` 的生成条件是
  `synthetic_mode or not all_greedy`（上游同款）。`-1` 占位仍然必拒（`draft_token_id >= 0`）。
* **synthetic 不证明模型分布无损**（075 §4）：它绕过了 p/q 检验，"接受长度"是配置出来的。
  实测反证（`check_step75` F1）：target 把概率几乎全给 token 0（logit 8.0）、草稿给 token 1，
  rates 全 1 时内核**照收** token 1 —— 它在 target 下的概率只有 **3.343e-4**；把它的 logit 改成
  `-inf`（概率恰好 0）也照样被接受。所以这个模式只能做"接受长度 → 吞吐/调度/显存"的可复现实验。

## 5. 差异账本（临时差异逐条）

| 编号 | 差异 | 理由与影响 |
|---|---|---|
| D-75-1 | **V1 遇到 `rejection_sample_method="block"` 明确 `NotImplementedError`** | 上游 V1 的 `RejectionSampler.__init__` 里根本没有 block 分支 → 会**静默按 standard 跑**，用户以为拿到的是块验证的接受长度。需求 075 §4 点名要求"V1 不得静默把 block 当 standard"，所以本仓库报错并指路 V2。配置期不拦（`block` 对 V2 合法，配置不知道 Runner 选哪个） |
| D-75-2 | `block` 与 `synthetic` 互斥的 assert 在 `rejection_sample()` 里（上游同款），**配置期不额外拦** | 上游就是"配置允许两者同时给、内核 assert 挡住"；本关照抄。实测：`use_block_verification=True` + 非 None 的 rates → `AssertionError: Block verification is incompatible with synthetic acceptance rates.` |
| D-75-3 | V2 的 `rejection_sample_method` 校验 | 上游 V2 对未知取值没有显式校验（配置的 Literal 挡住了）；本仓库同样只在配置期校验取值集合，`validate_v2_model_runner()` 里删掉"只接 standard"的限制（75 关实现后它就成了假限制） |
| D-75-4 | 分块上限仍是 `MAX_CHUNK_BYTES = 2**30`（1 GiB） | 上游同值。**没有**按设备显存自适应（上游的 TODO 也写着"分块是 workaround"）；一行 fp32 = `V × 4` 字节，151936 词表 → 1766 行/块 |
| D-75-5 | 79 关（adaptive verification）未实现 | 本关只保留保护：`logits.shape[0] <= max_chunk_logits → 单块`。**没有**实现压紧本身，也没有"压紧后 cu_num_logits_np 更新"的新逻辑。测试用"陈旧 CPU offsets + 行数不超上限"的构造把保护钉住 |
| D-75-6 | `block` 的性能主张不作为验收条件 | 论文（arXiv:2403.10444）说块验证的每轮期望产出"永不差于标准"；实测只在**草稿接近 target** 的区间成立（§6.3），对抗性合成分布下更差。需求 075 §5 要求"以源码递推为实现基准"，所以本关只断言"与源码递推一致"，性能数字如实记录 |
| D-75-7 | `synthetic` 在 V1 的 `uniform_probs` 语义 | 上游 V1 与本仓库一致：`synthetic_mode or not all_greedy`。差别只在**随机源**（每请求 generator vs V2 的 `(seed,pos)` 哈希），所以 V1/V2 之间比分布不比逐值 |

## 6. 实测（2026-10-10，RTX 5090 Laptop / WSL2）

### 6.1 测试与探针

| 命令 | 结果 |
|---|---|
| `pytest tests/step75 -q` | **52 passed**（block 18 + synthetic 19 + chunks 15），连跑多次稳定 |
| `python benchmarks/check_step75_verify_modes.py` | **全部通过（20 项）** |
| `pytest tests/step58..tests/step75 -q` | **772 passed**（基线 720 + 本关 52） |

### 6.2 块验证的三方对照（check_step75 B/C 段）

* **逐 kernel 差分**：五个内核（stats / cumulative_log_p / residual_mass / rejection / resample+insert）
  × 11 个中间缓冲，点质量草稿与概率草稿两种输入下与冻结实现**逐值相等**（`torch.equal`）。
  （逐 kernel 时缓冲用 `zeros` 分配，"内核没写的格子"也能比较；整体通路另用真分配差分。）
* **τ 分布**（残差例 K=2、N=8000）：经验 `[0.6633, 0.3055, 0.0312]` vs 参考闭式
  `[0.6651, 0.3036, 0.0313]` → **TV = 0.0019**。
* **τ=1 的交付 token 分布**（n=2444）：vs **块残差** `max(p_tau·M_b − M_s, 0)` → **TV = 0.0074、
  卡方 = 1.90**（7 自由度）；vs standard 残差 `max(M_b − M_s, 0)` → TV = 0.1901。
  两条参考之间 TV = 0.1917 → 匹配块残差是有判别力的（写成 standard 会被卡方抓住）。
* **附录 A 的手算例（B）跑真内核**（K=3、V=3、2000 次）：standard 的 τ 分布
  `[0.653, 0, 0, 0.347]`（解析值 `[2/3, 0, 0, 1/3]`，平均 1.041），block 的 τ 分布 `[0, 0, 0, 1]`
  （平均 **3.000**）——这个例子同时钉住了"上限加在累计值上"：若按每步 `min(1,p/q)` 累乘，
  block 的平均接受长度会退化成 1.0（与标准验证相同），与实测不符。

### 6.3 E[τ] 随草稿质量（check_step75 D 段，K=3、V=8、200 组设计取平均）

| mix（草稿与 target 的接近程度） | 0.00 | 0.25 | 0.50 | 0.75 | 0.90 | 1.00 |
|---|---|---|---|---|---|---|
| E[τ] block | 0.303 | 0.531 | 1.294 | 2.540 | 2.906 | 3.000 |
| E[τ] standard | 0.333 | 0.557 | 1.212 | 2.381 | 2.840 | 3.000 |

即：论文的"永不差于标准"在**它的实验区间**（草稿接近 target，mix ≥ 0.5）成立；对抗性合成分布下
（草稿与 target 差得远）本实现实测更差。`vllm_bugs/LEAD_20260910_block_verification.md`（本地，
不提交）记着这条线索，本关不改实现。

### 6.4 synthetic 的经验接受率（check_step75 E/F 段）

* 配置 `[0.8, 0.4, 0.2]`（K=3、N=8000）实测前缀接受率 `[0.8011, 0.4057, 0.2046]`，
  平均接受枚数 `1.4115`（配置 Σ = 1.4）。
* 错误口径（原数组当条件率）会得到 `[0.8, 0.32, 0.064]`，与配置差 `[0, 0.08, 0.136]`。
* tiny 引擎端到端（V2、eagle3、synthetic `[0.8,0.4,0.2]`、60 枚 token）：25 轮里接受 34 枚
  → **1.360 枚/轮**；V1 同配置 27 轮 32 枚 → **1.185 枚/轮**；V1 `standard` 对照（夹具里
  draft == target）15 轮 44 枚 → **2.933 枚/轮**——标准模式跟着**模型**走，synthetic 跟着
  **配置**走，这正是它的用途。
* **真实权重端到端**（Qwen3-1.7B + `Qwen3-1.7B-eagle3`、bf16、K=3、2 条 prompt × 48 token、
  累加 Scheduler 的 `spec_decoding_stats`）：

  | 运行 | 轮数 | 接受枚数 | 接受/轮 | token/轮 |
  |---|---|---|---|---|
  | V1 standard（贪心） | 82 | 12 | 0.146 | 1.17 |
  | V1 synthetic（贪心） | 43 | 54 | **1.256** | 2.26 |
  | V1 synthetic（temp=0.7） | 39 | 56 | **1.436** | 2.44 |
  | V2 standard（贪心） | 86 | 8 | 0.093 | 1.09 |
  | V2 synthetic（贪心 / temp=0.7） | 38 | 57 | **1.500** | 2.50 |
  | V2 block（贪心） | 86 | 8 | 0.093 | 1.09 |

  两个结论：① synthetic 的接受长度在 V1/V2 上都收敛到配置的 1.4 附近，**与草稿质量无关**
  （这几条 prompt 上真实 EAGLE3 的 standard 只有 0.09~0.15 枚/轮，受 63 关已知差距影响）；
  ② `block` 在**草稿很差**时退化成与 standard 同量级（§6.3 的 D 段：对抗性分布下 block 更差）。
  CLI 入口：`python minivllm/demo.py --spec-method eagle3 --spec-model <draft 目录> --spec-k 3
  --rejection-sample-method synthetic --synthetic-acceptance-rates 0.8,0.4,0.2 ...`。

### 6.5 分块（check_step75 G 段）

* 上限口径：151936 → **1766** 行/块、32000 → **8388**、128256 → **2092**。
* 三条请求各 4 行（K=3）：上限 4 → 3 块、8 → 2 块、11 → 2 块、12 → 1 块（不切开请求）。
* 单块 vs 多块：`sampled`/`num_sampled`/`logprob_token_ids`/`selected_token_ranks`/`logprobs`
  逐值一致（多块走 `LogprobsTensors.cat`，`cu_num_generated_tokens` 是**全局**请求偏移）。
* 陈旧 CPU offsets（前缀和比 GPU 那份大）：行数 ≤ 上限 → **单块**（切块器一次都没被调用）；
  新鲜 offsets 超上限 → 真的切块 `[(0,1),(1,2),(2,3)]`；陈旧 offsets 若被用来切块会切出
  `[(0,1),...]` 而新鲜的是 `[(0,2),(2,3)]`（这就是保护要防的错位）。
* 引擎层：把 `get_max_chunk_logits` 换成 3，tiny V2 引擎跑出的 token 与每个位置的 logprobs
  与不切块时**完全相同**（rank 相同、logprob 差 < 1e-6）。

## 7. 已知边界

* 块验证**只对非贪心请求生效**：贪心请求走 `is_greedy` 分支（argmax 顶替），块阈值不参与——
  `tests/step75/test_block_verification.py::test_block_greedy_requests_take_the_argmax_path` 钉住。
* `block` 的**质量/加速比**未在真实权重上测（本机没有按块验证调优的权重，且它不改变草稿分布）：
  本关只验数学与协议。真实权重上的接受长度仍受 63 关已知差距影响（`docs/step63_alignment.md` §8）。
* synthetic 的 rates 长度 == K（配置 K，即最大 K）；动态投机长度（71 关）下每轮 K_i ≤ K，
  所以 `rates[i]` 的索引安全。**没有**为"每轮不同 K 配不同 rates"做扩展（上游也没有）。
* V1 的 synthetic 与 V2 只在**分布**上可比（随机源不同：每请求 `torch.Generator` vs
  `tl_rand32(seed,pos)`），逐值差分只在 V1↔上游 V1（钉死随机源）之间做。
* 分块的 `adaptive verification` 保护只有"单块分支"这一层；真正的压紧与预算上限属 79 关。
* `benchmarks/check_step57_vllm_boundaries.py` 的 **C1 在基线上就失败**（`git worktree` 上的
  2b2338d 复现同样症状："跑了 1 轮，违例 无"——上游引擎的 trace 只看到 1 轮），与本关无关；
  本关改动的探针只有 `check_step59_rejection.py`（E5 期望值）与新增的 `check_step75_verify_modes.py`。
* 顺手补的 CLI（`minivllm/demo.py`）：`--rejection-sample-method {standard,synthetic,block}`、
  `--synthetic-acceptance-rates 0.8,0.4,0.2`，以及"`--spec-model` 是 EAGLE/EAGLE3 的 draft 目录时
  自动构造 `draft_model_config`"（此前 eagle3 在 CLI 上根本建不起来）。

---

## 附录 A：用大白话讲"块验证"和"synthetic"（零基础版）

> 这一节假设读者**完全没接触过**投机解码。文中每个数字都能按计算器自己验算；标"实测"的数字来自
> `tests/step75/` 与 `benchmarks/check_step75_verify_modes.py`（命令与完整结果见 §6）。
> 想看代码：规则写在 `minivllm/worker/gpu/spec_decode/rejection_sampler_utils.py`（V2）与
> `minivllm/sample/rejection_sampler.py`（V1）；本文用到的"数学参考"在
> `minivllm/testing/torch_block_verification.py`。

### A.0 这件事到底在干什么

大模型（下面叫**老师**）写得准，但一个字一个字往外挤，慢；小模型（**学生**）快得多，但水平差。
投机解码（speculative decoding）的套路是：

1. 学生先"抢答" K 个字（K = 配置里的 `num_speculative_tokens`，本附录的例子取 3）；
2. 老师**一次**把这些位置全看一遍——注意：老师看一遍，就能算出"每个位置上、每个字有多大把握"，
   代价和老师自己写 K 个字差不多；
3. 按一套**验证规则**决定：学生抢答的字里，有几个可以算数（这个数字叫**接受长度**，记作 τ）；
4. 被拒的位置，老师补一个自己的字；如果全收下，老师再额外送一个字（叫 **bonus**）。

整件事有一条硬要求，叫**无损（lossless）**：把"学生抢答 + 老师批改 + 补字"合起来看，
**最终每个位置上的字，分布必须和老师自己写一模一样**。否则投机解码就成了"拿学生的水平换速度"。

所以本关的两个主角，回答的是两个不同的问题：

| 名字 | 一句话 | 它在回答什么问题 |
|---|---|---|
| **块验证**（block） | 换一套"批改规则" | 同一批抢答，能不能少浪费、多算数几个字？ |
| **synthetic** | 不批改了，直接照一张"通过率表"掷骰子 | 我想做实验：把接受长度固定成 1.4，系统会怎样？ |

### A.1 三个前置概念

**① 概率表**：老师对"下一个字"会给每个字一个分数，归一化之后是一张概率表，加起来等于 1。
本附录只有 3 个字（猫 / 狗 / 鸟），所以表就是三个数，比如：

    老师的表 p：猫 0.5   狗 0.3   鸟 0.2

**② p 和 q**：p = 老师的表，q = 学生的表。**同一个字，两张表给的分数常常差很远**——批改要处理的
就是这种偏差：学生可能"高估"某个字（自己给 0.6、老师只给 0.2），也可能"低估"某个字
（自己给 0.2、老师给 0.5）。

**③ 为什么不能"学生写啥都收"**：那最终输出的分布就变成学生的水平了。验证规则要在
"多收学生的字（快）"与"分布仍是老师的（准）"之间取平衡。下面的两种规则都能做到这一点，
只是"怎么做到"完全不同。

### A.2 标准验证（standard）：逐字打卡，一票否决

#### A.2.1 规则

对学生的第 i 个字（学生写的是 `d_i`）：

1. 算一个**比值** `老师给 d_i 的概率 ÷ 学生给 d_i 的概率`；
2. 掷一个 0~1 的均匀随机数 u；
3. `u` 小于这个比值（比值超过 1 就按 1 算）→ **这个字算数**；否则 → **不算数，而且从这里截断**：
   它后面学生写的字全部作废（因为上下文已经断了）。

一句话：**逐字打卡，第一个不过关就整段作废**（逻辑上叫 AND）。

#### A.2.2 手算表（本附录的例子 ①）

    老师的表 p：  猫 0.5   狗 0.3   鸟 0.2
    学生的表 q：  猫 0.2   狗 0.6   鸟 0.2

| 学生写的字 | 比值 p/q | 接受概率 min(1, p/q) | 白话 |
|---|---|---|---|
| 猫 | 0.5 / 0.2 = 2.5 | 1.0 | 学生**低估**了猫 → 必收 |
| 狗 | 0.3 / 0.6 = 0.5 | 0.5 | 学生**高估**了狗 → 一半概率收 |
| 鸟 | 0.2 / 0.2 = 1.0 | 1.0 | 两边一样 → 必收 |

直觉：**学生谦虚（给得比老师低）的字几乎必收；学生吹牛（给得比老师高）的字有概率被拒。**
（这也是"K=1 时两种规则完全一样"的原因：只有一个字时，块验证的合格线恰好等于这个比值）。

#### A.2.3 被拒之后怎么办：老师补一个字

被拒的位置不会保留学生写的字，"老师"会从**"老师比学生多出来的那部分"**里补一个自己的字：

    max(p − q, 0) = max(0.5−0.2, 0.3−0.6, 0.2−0.2) = (0.3, 0, 0)  →  归一化 (1, 0, 0)

也就是：**被拒了就把这个字改成"猫"，而且必然改成猫**（实测 `residual_probs` 给的就是
`[1.0, 0.0, 0.0]`）。为什么这么补？看下一段的加法。

#### A.2.4 "无损"的手算证明（三行加法）

把学生按自己的表 q 随机抽字这件事算进去，看最终输出的第 1 个字是什么分布：

| 学生抽到 | 概率 | 经过验证后输出什么 | 贡献 |
|---|---|---|---|
| 猫 | 0.2 | 必收 → 猫 | 猫 +0.2 |
| 狗 | 0.6 | 一半收（狗），一半被拒后补成猫 | 狗 +0.3、猫 +0.3 |
| 鸟 | 0.2 | 必收 → 鸟 | 鸟 +0.2 |

加起来：猫 `0.2+0.3 = 0.5`、狗 `0.3`、鸟 `0.2` —— **正好等于老师的表 p**。
这就是"无损"：虽然中间走了"学生猜、老师挑"的弯路，最终每个字的分布和老师自己写一模一样。
（标准验证的这套规则来自 [Leviathan et al. 2023](https://arxiv.org/abs/2211.17192)。）

#### A.2.5 它的痛点：一票否决太浪费

学生一口气写了 3 个字，如果第 1 个就被拒，那第 2、3 个**哪怕写得再好也白写**。
下面这个例子（本附录的例子 ②）就会看到这个浪费有多严重。

### A.3 块验证（block）：给"整段"抽奖，取最长的中奖段

#### A.3.1 规则（和标准验证只差"怎么用随机数"）

对每一位 i，准备两个东西：

* 一个**整段合格线** `h_i`（不是"这一个字的 p/q"，怎么算见 A.3.3）；
* 一个独立的骰子 `u_i`。

规则：`u_i ≤ h_i` 就记下"**前 i+1 个字整段合格**"；最后

    接受长度 τ = 所有"整段合格"里最长的那个长度（一个都没中就 0）

注意这跟标准验证**不是**同一件事：

| | 怎么判 | 逻辑 | 第 1 个字被拒会怎样 |
|---|---|---|---|
| 标准验证 | 逐位看 p/q 比值，**连着**过 | AND（一票否决） | 后面全作废 |
| 块验证 | 给每个前缀长度一次"整段合格"的抽奖，**取最长的中奖段** | OR（可以"后面中奖"） | 后面仍有机会被算数 |

"整段"两个字是关键：**第 i 位的中奖条件不是"第 i 个字本身好不好"，而是"前 i+1 个字整体
像不像老师会写的"。** 所以块验证允许"某一位单看很悬、但整段看还行"的情况被收下。

#### A.3.2 完全能手算的例子 ②（K=3）

    老师的表（三个位置都一样）p：  猫 0.6   狗 0.2   鸟 0.2
    学生的表     第 1 位 q：        猫 0.15  狗 0.60  鸟 0.25     ← 学生最看好"狗"
                 第 2、3 位 q：     猫 1/3   狗 1/3   鸟 1/3      ← 三个字并列，取第一个"猫"
    学生这一次抢答的字：            狗、猫、猫
    （真实运行时学生是按自己的表 q 抽的；这里固定成"狗猫猫"是为了能手算）

**第一步：每一位的原始比值 p(学生写的字) / q(学生写的字)**

| 位 | 学生写 | 老师给 | 学生给 | 原始比值 |
|---|---|---|---|---|
| 1 | 狗 | 0.2 | 0.6 | 1/3 ≈ 0.333（学生吹牛） |
| 2 | 猫 | 0.6 | 1/3 | 1.8（学生谦虚） |
| 3 | 猫 | 0.6 | 1/3 | 1.8（学生谦虚） |

**第二步：累计出"前缀联合比" p_i（这才是块验证的量，注意上限）**

    p_0 = 1/3                      （只看第 1 个字）
    p_1 = 1/3 × 1.8 = 0.6          （前 2 个字的累计）
    p_2 = 0.6 × 1.8 = 1.08 → 截到 1.0   （前 3 个字的累计）

    （"累计比值不超过 1"在代码里写成对数形式：`log_p = min(log_p + Δ, 0)`，
      等价于"p_i 不许超过 1"，但**上限加在累计值上**，不是每一步各自截断——见 A.3.3 的警告）

学生吹牛造成的亏空（1/3），被后面两处谦虚的富余（×1.8、×1.8）**补了回来**。

**第三步：每一位的"整段合格线" h_i**

    残差质量 M_i = Σ_x max( p_i × 老师第 i+1 位的表 − 学生第 i+1 位的表 , 0 )
    h_i = M_i / (M_i + 1 − p_i)        （最后一个位置没有"下一位"，直接 h = p_i）

    （下表按"第几位"从 1 数；代码里从 0 数，位 1/2/3 对应 h_0/h_1/h_2）

| 位 | M_i（手算） | h_i |
|---|---|---|
| 1 | `max(1/3×0.6 − 1/3, 0) + max(1/3×0.2 − 1/3, 0) + max(1/3×0.2 − 1/3, 0) = 0` | `0/(0+1−1/3) = 0` |
| 2 | `max(0.6×0.6 − 1/3, 0) = 0.36 − 0.3333 = 0.0267`（另两字为负 → 记 0） | `0.0267/(0.0267+1−0.6) = 0.0625` |
| 3 | 最后一个位置，直接用前缀比 | `h_2 = p_2 = 1.0` |

**第四步：两种规则下的接受长度分布**

    标准验证：接受概率 min(1, 比值) = (1/3, 1, 1) → 要"连着"过三关：1/3 × 1 × 1 = 1/3
              τ = 3 的概率 1/3；否则第 1 位就被拒 → τ = 0（后面两个字全白写）
              τ 分布 = [2/3, 0, 0, 1/3]，平均接受长度 1.0
              实测（真内核跑 2000 次）：[0.653, 0, 0, 0.347]，平均 1.041
    块验证：  "前 3 个字整段合格"这条抽奖的合格线是 1.0 → 必过 → τ = 3 每次都成立
              τ 分布 = [0, 0, 0, 1]，平均接受长度 3.0
              实测（真内核跑 2000 次）：[0, 0, 0, 1]，平均 3.000

一句话解释这个差距：**学生第 2、3 个字明明写得很谦虚（老师其实很喜欢"猫"），标准验证却因为
第 1 个字吹牛而把它们全废了；块验证看的"前 3 个字整段"，累计比值已经补回来了，所以整段收下。**

#### A.3.3 合格线 h_i 为什么长这样（别被公式吓到）

* `p_i`（前缀联合比）：前 i+1 个字"学生给的分数"与"老师给的分数"的累计比值，
  **上限是 1**（超过 1 说明学生整体偏谦虚，没必要给额外奖励）。
* `M_i`（残差质量）：老师想把概率给某些字，但乘上 p_i 之后**还是比学生给的多**——
  这部分"老师多出来的质量"就是拒绝后要补字的地方。
* `h_i = M_i / (M_i + 1 − p_i)`：把"能补的质量"与"还没被覆盖的质量 1 − p_i"放一起做归一化，
  于是 h_i 天然落在 0~1 之间，并且**草稿越像老师，h_i 越接近 1**（p_i → 1 且 M_i → 0 时，
  代码里有个 `denom ≤ 0` 的兜底直接给 1.0）。

⚠️ **"上限加在累计值上"这个细节不能改**（075 §3.2 点名）：如果按"每一位各自
`min(1, p/q)` 再相乘"（相当于把上限加在每一步），上面这个例子的累计比会变成
`(1/3, 1/3, 1/3)`、合格线 `(0, 0, 1/3)`、平均接受长度 **1.0**（跟标准验证一样），
**与内核实测的 3.0 不符**。所以代码里是 `log_p = min(log_p + Δ, 0)`——上限加在**累计值**上。

#### A.3.4 被拒之后怎么补：多乘一个 p_tau

块验证拒绝时，补字的分布是 `max(p_tau × 老师 − 学生, 0)` 归一化，比标准验证多一个系数
`p_tau`（= 被拒位置**之前**的累计比值；第 1 位被拒时 p_tau = 1，于是两者一样）。
用 §6.2 那个"残差例"（K=2、8 个字；手算数字写在 `tests/step75/test_block_verification.py`
顶部）的实测数字看得最清楚——同样是"第 2 个字被拒"：

    块验证补字分布   [0.2728, 0.0137, 0, 0, 0, 0.0137, 0.6999, 0]
    标准验证补字分布 [0.2599, 0.1015, 0.0054, 0, 0.0054, 0.1015, 0.521, 0.0054]
    两者相差 TV = 0.19

如果块验证照抄标准验证的补法，上面这 0.19 的差异会被统计检验抓住（实测：匹配块残差时
卡方 1.90，匹配标准残差时 TV 0.19）。

#### A.3.5 那它是不是"稳赚"？不是

块验证不是白吃的午餐。草稿**差**的时候，`p_i` 小、残差质量也小，合格线被压低，反而不如
标准验证的"逐位打卡"。实测（K=3、随机生成 200 组设计取平均）：

| 学生与老师的接近程度 | 0.00（很差） | 0.25 | 0.50 | 0.75 | 0.90 | 1.00（完全一致） |
|---|---|---|---|---|---|---|
| 块验证的平均接受长度 | 0.303 | 0.531 | 1.294 | **2.540** | 2.906 | 3.000 |
| 标准验证的平均接受长度 | 0.333 | 0.557 | 1.212 | 2.381 | 2.840 | 3.000 |

论文（[Sun et al. 2024](https://arxiv.org/abs/2403.10444)）说的"每轮期望产出永不差于标准"，
是在**它的实验区间**（草稿相当接近老师）成立的；草稿很差时本实现实测更差。
本项目按需求 075 §5"以源码递推为实现基准"，**只如实记录、不改实现**。

### A.4 synthetic：不批改了，直接照"通过率表"掷骰子

#### A.4.1 为什么有人要这个东西

标准验证的接受长度取决于**学生有多像老师**。本机真实 EAGLE3 在几条 prompt 上实测只有
0.09~0.15 枚/轮（受 63 关已知差距影响）。这就让一类实验做不成：

* "接受长度 = 2 的时候，调度器/显存/图会怎样？"
* "接受长度掉到 0.3 系统扛不扛得住？"

因为这些数字被模型质量绑死，换台机器、换个 prompt 就复现不出来。
synthetic 的做法很直接：**不看学生的表，也不看老师的表**，直接配置一张"通过率表"，逐位掷骰子：
第 i 位掷出"通过"就把这个字算数，掷出"不通过"就截断。于是**接受长度变成输入参数**，
想要多少给多少，可复现、与模型无关。

#### A.4.2 配置怎么填：写"无条件"概率，内核用"条件"概率

配置里填的每个数，含义是"**前 i+1 个字全都通过**的概率"（这叫**无条件**概率）。
K=3、配置 `[0.8, 0.4, 0.2]` 的白话版，用"做 10 次实验"来理解：

| 位置 | 配置值（无条件） | 白话 | 内核实际用的**条件**率 |
|---|---|---|---|
| 第 1 个字 | 0.8 | 10 次里有 **8** 次第 1 个字通过 | 8/10 = **0.8** |
| 第 2 个字 | 0.4 | 10 次里有 **4** 次"前 2 个字都通过" | 4/8 = **0.5**（只在第 1 个已通过的 8 次里算） |
| 第 3 个字 | 0.2 | 10 次里有 **2** 次"前 3 个字都通过" | 2/4 = **0.5**（只在已经连过 2 个的 4 次里算） |

因为内核是"逐位早期终止"的循环（第 2 位只有在第 1 位通过时才会掷骰子），所以它必须用右侧那列
**条件率**。换算就一行代码：`spec_decode/utils.py::unconditional_to_conditional_rates`，
`[0.8, 0.4, 0.2] → [0.8, 0.5, 0.5]`。
平均接受长度 = 0.8 + 0.4 + 0.2 = **1.4 枚/轮**（实测：V1 1.256、V2 1.500，见 §6.4）。

⚠️ 如果**跳过换算**、直接拿 `[0.8, 0.4, 0.2]` 逐位抽签，"前 2 个字都通过"的概率会从 0.4 变成
`0.8 × 0.4 = 0.32`（实测差 0.08，是容差的 3 倍以上）——这就是需求 075 §3.4 专门点出这条的原因。

#### A.4.3 它**不**保证无损（本关要求写明的一条限制）

synthetic 完全绕过了"老师的表"，所以"输出分布 = 老师"这条保证没有了。可算的反例：

    老师给"猫" 99.77%、给"狗"只有 0.033%（8 个字的表）
    学生写"狗"
    配置 rates = [1.0, 1.0, 1.0]（每次都通过）
    → 内核照收"狗"（实测：把"狗"的概率改成恰好 0 也照收）

于是最终输出里出现了老师几乎不可能写的字。所以 synthetic 只能用来做
"接受长度 → 吞吐/调度/显存"的可复现实验，**不能**用它验收正确性（075 §4 要求测试报告写明这一点，
`tests/step75/test_synthetic.py::test_synthetic_is_not_distribution_preserving` 就是这条的证据）。

### A.5 三句话总结

1. **标准验证**：逐字"打卡"，第一个不过关就整段作废；被拒时从"老师比学生多出来的那部分"里补字。
2. **块验证**：给"前 1 个字""前 2 个字"…"前 K 个字"各抽一次"整段合格"的奖，取**最长**的中奖段；
   合格线由"前缀累计比值 + 下一位置的残差质量"算出，补字时要多乘一个 `p_tau`。
   它把"第 1 个字不合格"从"一票否决"变成了"后面还有机会"——这是它在草稿好时能多收字的原因，
   也是它在草稿差时可能更差的原因。
3. **synthetic**：不批改，直接按配置的通过率表掷骰子——接受长度可控、可复现，代价是失去
   "输出分布 = 老师"这条保证，只能用于实验。

> 对应的验收：
> `tests/step75/test_block_verification.py`（块验证：与冻结实现逐 kernel 逐值对照 + 与数学参考比分布）、
> `tests/step75/test_synthetic.py`（配置与换算、经验前缀接受率、V1/V2 一致、无损性反证）、
> `tests/step75/test_logit_chunks.py`（分块 logits 与陈旧 offsets 保护）、
> `benchmarks/check_step75_verify_modes.py`（20 项实测，本文所有"实测"数字都来自这里或 §6）。
