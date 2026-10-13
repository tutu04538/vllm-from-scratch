# 第七十五关：V2 块验证、Synthetic 模式与分块 Logits

> 需求：`learning_notes/14_vllm_from_scratch/投机解码完整需求/075_V2块验证Synthetic模式与分块Logits.md`
> 基线：本机 `vllm==0.28.0` 文件快照（2026-10-03）。执行路径：**V2 为主**（`minivllm/worker/gpu/`），
> synthetic **同时回归 V1**（`minivllm/sample/rejection_sampler.py`）。

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
