# 66 关对齐记录：Medusa 多头提议与 MLP 支持缺口

需求：[`066_Medusa多头提议与MLP支持缺口.md`](../../vllm-omni/learning_notes/14_vllm_from_scratch/投机解码完整需求/066_Medusa多头提议与MLP支持缺口.md)
基线：本机 `vllm==0.28.0` 文件快照。执行路径：**V1**（上游日志也这么说：
`Model Runner V2 does not yet support speculative method 'medusa'; using the V1 model runner instead`）。

> **状态：已实现并实跑**（`tests/step66` **51 项** + `benchmarks/check_step66_medusa.py` **28 项**全 PASS）。
> 与上游**真实 `Medusa` + `MedusaProposer`** 的数值对照：每个 head 的 blocks 与 logits
> 都是 **max|Δ| = 0.0**（逐位相同），候选**列顺序**也相同。**MLP speculator 按需求 §3 交的是
> "缺口证明"**（§5），生产路径保持不支持，也没有自创实现。

## 0. 一句话：这一关解决什么

"一次提多枚草稿"有两条不需要第二个大模型的路子：

| 路子 | 草稿怎么来 | 候选之间的关系 | 每条草稿的额外算力（7B target / hidden 4096） |
|---|---|---|---|
| 65 关的 MTP | target 自带的一层预测头 | **自回归**：第 2 枚条件在第 1 枚上 | 1 层 ≈ 0.5%（1.7B 上实测胶水 8.4M 参数） |
| **66 关的 Medusa** | N 个挂在 target hidden 上的小 head | **并行**：所有 head 读同一份 hidden，互相听不见 | 每个 head 一次 `4096²` GEMM ≈ 16.8M MAC，**依赖 target 的 1 次前向** |

Medusa 的算力账（用真实 checkpoint 的数字，`FasterDecoding/medusa-vicuna-7b-v1.3`：文件
1,478,537,775 字节，`hidden=4096`、5 个 head、每个 head 1 层）：

```text
target 每 token：≈ 7e9 参数 × 1 次前向            → 100%
Medusa head：    5 × 4096² ≈ 84M MAC               → ≈ 1.2%（相对 7B）
                  真正占地方的是 5 个 32000×4096 的 lm_head（草稿词表 GEMM）
一次 target 前向 + 5 个 head：产出 1（bonus）+ 最多 5 枚被接受的草稿
```

**它解决什么痛点**：EAGLE/MTP 要"多跑几层小模型"才能拿到第 2、3 枚草稿，Medusa 用**几个纯 MLP**
（没有 attention、没有 KV）换同样的候选数——代价是候选之间不互相条件，第 2 枚的接受率天然比
第 1 枚低（`num_accepted_tokens_per_pos` 会逐位衰减）。

**边界（需求 §2 明确的两条）**：

1. 本机 V1 的 Medusa 是**线性链**：每个 head 一个 `argmax`，`stack(dim=1)` 成 `[B, num_heads]`，
   **不是**论文里的 tree attention；配置里的 `max_paths=64` / `topk=10` 在 V1 里**没有任何读取点**。
2. 这个路径**不能**改成"每个 head 随机采样"：随机采样就必须提供正确的 `q`（每个 head 的分布），
   而上游只交回 token id（`draft_probs=None`）。argmax 的提议分布是**点质量**，`q` 天然正确。

## 1. 代码映射

| 本项目 | 上游参考 | 说明 |
|---|---|---|
| `spec_decode/medusa.py::MedusaProposer` | `v1/spec_decode/medusa.py:18-81` | 提议者本体（**不继承** `SpecDecodeBaseProposer`，上游也不继承）；`propose()` / `load_model()` / `dummy_run()` |
| `spec_decode/medusa.py::MedusaProposer.select_target_hidden_states` | `v1/worker/gpu_model_runner.py:5206-5225` | "取哪一行 hidden"的算式（本仓库把它从 Runner 收进提议者，见 §3.3） |
| `worker/gpu_model_runner.py::_propose_medusa` | 同上（Runner 里的 medusa 分支） | 每请求一行 hidden → `propose(K, hidden)` → `[B, K]` → 按 req_id 交回 |
| `worker/gpu_model_runner.py::_build_proposer` 的 medusa 分支 | `gpu_model_runner.py:691-694` | 分支位置一致（suffix 之后、extract 之前） |
| `models/medusa.py::Medusa`（注册名 `MedusaModel`） | `model_executor/models/medusa.py::Medusa` | `blocks[]`（残差 MLP）+ `lm_heads[]` + `compute_logits()` |
| `models/medusa.py::ResidualBlock` | 同名类 | `x = x + SiLU(Linear(x))` 叠 `num_hidden_layers` 次；`medusa_fc_bias` 决定要不要 bias |
| `models/medusa.py::Medusa.remap_old_checkpoint_key` | `Medusa._remap_old_checkpoint_key`（静态方法） | 旧 FasterDecoding 的 `{h}.{l}.linear.weight` / `{h}.{l}.weight` → `blocks.*` / `lm_heads.*` |
| `models/medusa.py::Medusa.load_weights` | 同名方法 | 名字归一、`token_map`、共享 lm_head、覆盖检查（严格度差异见 §3.4） |
| `layers/logits_processor.py::LogitsProcessor` | `model_executor/layers/logits_processor.py` 的子集 | 词表 GEMM 的收尾：`[..., :org_vocab_size]` 切片 + `scale`（TP=1、无量化、无 soft cap） |
| `config.py::medusa_hf_config` | `transformers_utils/configs/medusa.py::MedusaConfig` + `config/speculative.py:883-935` | 旧 checkpoint 的 key 改名、`model_type`/`architectures`、缺省值、**K → num_heads** |
| `config.py::SpeculativeConfig._resolve_medusa` / `derive_medusa_draft_config` | 同上 + `:923-935` 的词表对齐 | 本仓库的配置拿不到 target，所以"对齐词表"拆成派生方法（与 64/65 关同一个做法） |
| `config.py::MLP_SPECULATOR_MODEL_TYPE` + `__post_init__` 的版本缺口报错 | `config/speculative.py:958-961`（上游会认，然后建不起模型） | §5 |
| `testing/tiny_models.py::tiny_medusa_dir` | 无（测试用） | 旧格式 checkpoint（三种命名）+ `medusa_fc_bias` / `original_lm_head` / `token_map` 变体 |

## 2. 真实旧 checkpoint 的形态（本机实测，不是推测）

需求 §4 点名"旧格式 checkpoint"，所以先把真实文件看清楚（HF API + HTTP Range 只读了前 4 MB，
没有下载 1.4 GB 权重）：

```text
FasterDecoding/medusa-vicuna-7b-v1.3
  config.json        {"base_model_name_or_path": "lmsys/vicuna-7b-v1.3",
                      "medusa_num_heads": 2, "medusa_num_layers": 1,
                      "transformers_version": "4.31.0"}
  medusa_lm_head.pt  1,478,537,775 字节（torch pickle，不是 safetensors）
  文件里的 state_dict 键（实测）：
      0.0.linear.weight / 0.0.linear.bias / 0.1.weight
      1.0.linear.weight / 1.0.linear.bias / 1.1.weight
      … 一直到 4.1.weight（**文件里是 5 个 head**）
```

三件对实现有直接影响的事实：

1. **config.json 里没有 `model_type` / `vocab_size` / `hidden_size` / `architectures`**：
   `AutoConfig` 认不出它是 Medusa（上游因此要 `hf_overrides={"model_type": "medusa"}`），
   缺的字段全部落回 `MedusaConfig` 的默认值（`vocab_size=32001`、`hidden_size=4096`）。
   社区版（如 `siyuehuang/Qwen2-7B-Instruct-medusa`）则是**基座 config + `medusa_num_heads/_layers`**，
   并把基座的 `architectures`（`Qwen2ForCausalLM`）一起抄了进去——这两种形态本关都处理（§3.7）。
2. **`vocab_size` 必须与 target 对齐**：旧 config 缺它 → 默认 32001，而 `lm_heads.*.weight` 的真实
   宽度是 target 的词表（151936）。上游为此有一段"不等就两个都改成 target 的"（§3.6）。
3. **config 说 2 个 head、文件里有 5 个**：因为上游把 `num_heads` 直接改写成 **K**
   （`num_lookahead_tokens` 的 setter），checkpoint 自己的声明反而被覆盖——**Medusa 的 head 数
   就是 K**，多出来的 head 静默丢掉。本关把这条写成配置期归一 + 加载期记账（§3.4）。

顺带一条边界：真实旧 checkpoint 是 `.pt`，本仓库加载器**只读 safetensors**，所以会在加载期
明确报 `NotImplementedError`（不是模糊失败）——`tests/step66` 有一条用例钉这个行为。

## 3. 与上游的差异账本（逐条）

1. **`MedusaProposer.__init__(vllm_config, device)`** 与上游逐字一致；差别只在
   `__init__` 里多做一步 `derive_medusa_draft_config()`（"与 target 对齐词表"）。上游的
   `SpeculativeConfig` 自己持有 `target_model_config`，本仓库的配置拿不到，所以这一步由提议者在
   构造时做——与 64 关 `derive_extract_hidden_states_config()` 的位置相同。
2. **`assert` → 明确报错**：上游用 `assert num_speculative_tokens == self.num_speculative_tokens`、
   `assert (truncated == orig) or token_map is not None`；本仓库换成 `RuntimeError` / `ValueError`。
   理由：`python -O` 会关掉断言，而这两条都决定"草稿列数与验证行数对不对得上"。
   另外多一条上游没有的检查：`propose()` 返回的列数必须等于 K。
3. **行选择的位置与 stride**：
   - 位置：上游把算式写在 Runner 里（`gpu_model_runner.py:5206-5225`），本仓库收进
     `MedusaProposer.select_target_hidden_states()`——那段算式的唯一消费者就是本提议者。
   - stride：上游第二条分支用 `offset += num_draft + 1`（`num_draft` = 本轮**采用的**草稿数）。
     它对"每请求恰好 K_i+1 行"的批是对的（上游会用 `pad_spec_decode` 把 decode 请求补成满宽），
     但**一批里混进中间 prefill 块**（本轮排了 n>1 行、0 枚草稿）时只前进 1 行，后面所有请求的
     行号整体错位。本仓库用调度快照的 `num_scheduled_tokens`，没有 prefill 块时与上游**逐值相同**
     （无草稿批退化成 `arange(B)`，等于上游第一条分支）。
     `benchmarks/check_step66_medusa.py::D2` 就是这个错位的最小反证（上游行号 `[0, 3, 4]` 对上
     本仓库的 `[7, 8]`）。
   - 非 ready 行：上游会给中间 prefill 块算出 `offset - 1`（第一个请求时就是 `-1`）并把结果一起
     交回去，靠 Scheduler 事后丢弃；本仓库在 Runner 里跳过它们（不提没有意义的草稿）。
4. **加载的严格度**：上游对"检查点里有、模型里没有"的名字**一律静默丢弃**。本仓库只丢
   三类**登记过**的名字，并记进 `dropped_weights` 供对账：
   `blocks.{h≥K}.*` / `lm_heads.{h≥K}.*`（K 改写了 head 数）、`original_lm_head=True` 时
   `lm_heads.{h≥1}.weight`（所有 head 共享第 0 份）、以及本配置不建的 bias
   （`medusa_fc_bias` 未开时的 `blocks.*.layers.*.bias`、本模型没有的 `lm_heads.*.bias`）。
   其余认不出的名字**当场报错**——那是"这个家族的块没实现"的唯一信号（AGENTS §8）。
   反向的缺失（K > 检查点里的 head 数）由覆盖检查报错，**不会**留下 `torch.empty` 的随机参数。
5. **`dummy_run()`**：上游由显存 profiling 触发（69 关的 CUDA Graph 基建）。本仓库没有 profiling
   路径，所以在 `load_model()` 末尾调一次做**形状/设备自检 + 预热**——与 60 关
   `NgramProposerGPU._dummy_run()` 同一个用法（也让这个方法有真实调用方，而不是留一份没人走的代码）。
6. **词表对齐的副作用照抄**：上游 `if draft_hf.vocab_size != target_vocab:` 时把 `vocab_size` 与
   `truncated_vocab_size` **都**改成 target 的，于是显式声明的截断词表在这个前提下也会被冲掉。
   本仓库不"顺手修好"：想保留截断，就得让配置里的 `vocab_size` 与 target 一致
   （社区版 config 抄了基座的 `vocab_size`，正是这个原因）。两条分支都有用例。
7. **社区版 `architectures` 明确拒绝**：上游沿用配置里写的基座架构名（`Qwen2ForCausalLM`），
   于是会去建基座模型——权重名对不上（上游静默丢）或者 `compute_logits(blocks)` 直接把 hidden
   当 logits 用，属于静默跑错。本仓库在配置期报错并给出改法（把 draft 目录的 `architectures`
   改成 `["MedusaModel"]`）。旧文件（没有这一项）与显式写对的情况都正常。
8. **`MedusaConfig.from_pretrained` 的改名循环收窄**：上游是"key 里同时含 `num` 与
   `heads`/`layers` 就改名"，于是 `num_attention_heads` / `num_key_value_heads` 也会被改写成
   `num_heads`（社区版 config 里两者都在）。因为它随后立刻被 K 覆盖，**观测不到**；本仓库只认
   `medusa_*` 前缀，不做这个有副作用的宽匹配。
9. **`hidden_size` 必须在配置期与 target 一致**：上游会在第一次前向炸形状错；本仓库提前到
   `derive_medusa_draft_config()`。
10. **EPLB 组合**：上游 `assert not (is_mixture_of_experts(model) and enable_eplb)`。本仓库既没有
    MoE 模型也没有 EPLB（没有并行配置），这条在当前代码里不可能成立；仍按上游写成显式检查
    （`_reject_unsupported_eplb()`），将来谁加了 `parallel_config.enable_eplb` 会立刻报错，
    测试用"人为造出组合"的方式覆盖它。
11. **`LogitsProcessor` 是子集**：没有 TP gather/all-gather、`head_dtype`、`soft_cap`
    （本仓库 TP=1、不量化、没有 Gemma2 那类模型）。
12. **`max_paths` / `topk` / `max_seq_len`** 照抄进配置，但 V1 不读它们（线性链，没有树）。

## 4. 设计要点（改动时不要破坏）

1. **head 数 = K，只有一个来源**：`medusa_hf_config()` 写 `num_heads = K`；模型按它建 `blocks`；
   `propose()` 返回 `[B, num_heads]`；Runner 再断言列数 == K。任何一处"自己再推一次 head 数"
   都会让草稿列数与验证行数错位（**不报错的静默错**）。
2. **取哪一行 hidden 的语义**：一轮验证的 query 是 `[b][d1]…[dK]`，采样后序列的最后一个 token
   是 bonus，而**本轮没有算过 bonus 的 hidden**，所以要用"产出 bonus 的那一行" = 块内第
   `采样数 - 1` 行（首拒时就是第 0 行 = b 自己的 hidden）。写成"最后一行"或"第一个采样行"都会
   让草稿悄悄变差。
3. **行数口径用调度快照**（`num_scheduled_tokens`），不要用 `num_draft + 1`（§3.3 的错位）。
4. **`draft_probs=None` 是正确性的一部分**：argmax = 点质量 q，验证走 59 关的 `NO_DRAFT_PROBS`
   分支（greedy 下判据是"草稿 == target argmax"；random 行是接受概率 `p[d]`）。要改成随机采样
   head，就必须同时把每个 head 的分布交出来，否则采样分布不再精确等于 target 的分布。
5. **Medusa 不写 KV、不吃额外输入行**：`num_lookahead_tokens` 与 `max_num_new_slots_for_drafting`
   都是 **0**（上游 `VllmConfig.num_lookahead_tokens` 只给 `use_eagle()`/`uses_draft_model()` 留 K）。
   "顺手"给 Medusa 预留 lookahead 会让块分配与调度预算出现第二套口径。
6. **中间 prefill 块不提草稿**：没有"最后一个已算过的 token"可依据；上游算出 `-1` 行号再靠
   Scheduler 丢弃，本仓库在提议前就跳过。
7. **加载严格**：故意不加载的名字要**记账**（`dropped_weights`），认不出的名字要报错，
   缺参数由覆盖检查拦住。

## 5. MLP speculator：版本缺口（需求 §3 的交付物）

`mlp_speculator` 是 IBM 那条 MLP/token-embedding speculator（HF 上 `ibm-fms/*-mlp-speculator`）：
**一串纯 MLP**（没有 attention），每步吃"上一枚预测 token 的 embedding + 一个状态向量"，
状态按 `state_weight = 0.5 ** (0.5 / n_predict)` 混合——和 Medusa 的"N 个头并行"不同，它是在自己的
MLP 栈里**串行**的。

**本机 0.28.0 的事实是"配置层认得、模型层没有"**（四份证据，`tests/step66/test_mlp_support_boundary.py`
每条都有断言，都是 `pytest.raises` 的**预期失败**断言，不是 skip）：

| # | 证据 | 实测 |
|---|---|---|
| 1 | 上游配置层认得它 | `MLPSpeculatorConfig.from_pretrained(dir)` 正常解析：`model_type=mlp_speculator`、`n_predict=3`、`num_lookahead_tokens=3`；`config/speculative.py` 里有 `model_type == "mlp_speculator"` 的自动推断分支 |
| 2 | 注册表里那行是注释掉的 | `model_executor/models/registry.py:689` = `# "MLPSpeculatorPreTrainedModel": ("mlp_speculator", "MLPSpeculator"),`，同文件还有 `# Temporarily disabled.` / `# TODO(woosuk): Re-enable this once the MLP Speculator is supported in V1.`；`ModelRegistry._try_inspect_model_cls(...)` 返回 **None** |
| 3 | 拿配置建 `ModelConfig` 当场失败 | `ValidationError: Model architectures ['MLPSpeculatorPreTrainedModel'] are not supported for now.` ——**死在配置校验**，模型类没建、权重一个字节没读 |
| 4 | 执行层也没有它 | `model_executor/models/mlp_speculator.py` 里的 `MLPSpeculator` **没有 `forward`**（只有 `__init__` / `load_weights`，V0 遗留件）；`v1/worker/gpu_model_runner.py` 里 `mlp_speculator` 出现 **0 次**，分派会落到 `ValueError: Unknown speculative decoding method` |

本仓库的口径（需求 §3 三条逐条落实）：

1. **记录失败阶段**：上面四份证据 + `method="mlp_speculator"` 的**明确报错**（含"版本缺口"字样）；
2. **生产路径保持不支持**：**不写自创的 MLP 实现**去骗过枚举——
   `tests/step66/test_mlp_support_boundary.py::test_production_package_keeps_no_mlp_implementation`
   扫描 `minivllm/**/*.py`，只允许 `minivllm/config.py` 提到它（那条报错本身）；
3. **登记为版本缺口**：报错信息里写明"若日后要实现，必须先钉一个**真正支持它**的上游提交，
   单独增补需求；不得悄悄换参考版本"。

顺带：`get_model_class("MLPSpeculatorPreTrainedModel")` 在本仓库同样报"没有注册的模型结构"——
与上游的 `None` 对应；就算绕过配置校验把 `method` 改成 `mlp_speculator`，
`GPUModelRunner._build_proposer()` 也会落到"未知的投机方法"。

## 6. 实测

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/step66 -q                 # 51 passed
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmarks/check_step66_medusa.py         # 28 项 PASS / 0 FAIL
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/step58 ... tests/step66 -q # 412 passed
```

设备 `cuda:0`（RTX 5090 Laptop / WSL2），torch 2.13.0+cu130，tiny target `tiny_gqa`
（2 层、H=32、vocab 11、hidden 32），tiny Medusa head 由 `tiny_medusa_dir()` 现场生成。

| 项 | 实测 |
|---|---|
| 与上游 `Medusa` 的**每个 head 的 blocks / logits** | **max\|Δ\| = 0.0**（同一份权重、fp32，逐位相同） |
| 与上游 `MedusaProposer.propose()` 的**候选列顺序** | 相同（`[B, K]`，列 0 = head 0） |
| 配置归一 vs 上游 `MedusaConfig` | `hidden_size`/`vocab_size`/`truncated_vocab_size`/`num_heads`/`num_hidden_layers`/`max_paths`/`topk` 逐个相同；`architectures == ["MedusaModel"]` |
| 旧 checkpoint 的 key 改名 | `medusa_num_layers=1 → num_hidden_layers=1`；缺省值与上游 `MedusaConfig()` 逐个相同 |
| K 与 head 数 | 配置写 `medusa_num_heads=2`、K=5 → 建 5 个 head；检查点 5 个 head、K=2 → 裁掉 6 项并记账；K=6 → 覆盖检查报错 |
| 三种权重命名（旧格式 / `medusa_heads.` 前缀 / 本模型名字） | 加载的参数集合与数值逐位相同 |
| `original_lm_head` + `token_map` | 只建 1 个 `lm_head`；截断 5/11 时 `compute_logits` 的越界位置全为 `-inf`、argmax 全落在 `token_map` 内；逐 head 的 lm 权重按上游口径只用第 0 份 |
| 端到端 greedy（K=1/2/3 × 3 种命名） | 全部 == **非投机**（`spec=None`）逐 token 相同 |
| 草稿 | 每条 ready 请求正好 K 枚；`draft_probs is None`；两条请求的行不串；`propose()` 收到的行与"按该 hidden 重算 argmax"逐位相同 |
| 首拒 / 中拒 / 全接受 | 注入 oracle 草稿（greedy 验证是确定性的）后：`(3,3)` / `(3,0)` / `(3,1)`，三种模式的 greedy 输出都 == 非投机 |
| 行选择反证 | 把行整体错开一位后草稿必变（说明那份 hidden 真在用） |
| 混合 prefill 批次（budget=8、prompt 11） | greedy 一致；prefill 轮的所有草稿列表为空（不提无意义的草稿） |
| 小块数（`num_gpu_blocks=6`）抢占/恢复 | greedy 仍与非投机一致 |

## 7. 未做 / 待验（不要当成已覆盖）

1. **真实 Medusa checkpoint 的端到端**：本机没有（`models/` 只有 Qwen3-1.7B 与它的 eagle3 draft），
   且官方 checkpoint 是 **LLaMA/Vicuna 系**、权重是 `.pt`（本仓库加载器只读 safetensors，
   已实测会明确报错）。所以"真实权重下与 target 的接受率/输出对照"**未跑**；tiny + 上游类的
   逐值对照不能替代它。
2. **树形候选（论文的 tree attention）**：需求 §2 明确不做；`max_paths` / `topk` 在 V1 里也没有
   读取点。要做树得先有 72/75 关那套并行/块验证基建。
3. **`disable_padded_drafter_batch` / padded drafter batch**：本仓库没有这个开关（70 关的异步
   调度才有 padded drafter batch），所以上游那条 `sample_hidden_states.shape[0] == len(sampled_token_ids)`
   分支在本仓库合并成了"按调度快照取行"。
4. **CUDA Graph / 异步调度下的 Medusa**：69/70 关。上游日志这次也顺带印出
   `Async scheduling not supported with medusa-based speculative decoding and will be disabled`——
   即上游自己也是"Medusa 先关异步"。
5. **量化 head、MoE head + EPLB**：本仓库 TP=1、不量化、没有 MoE/EPLB。
6. **异构词表（67 关）**：Medusa 的 `token_map` 是"**草稿头**词表截断 + 散回原词表"，
   采样空间仍然是 target 的完整词表；EAGLE3 那种 `draft_vocab_size + d2t/t2d` 的 TLI 是另一件事。
7. **真实 checkpoint 的社区版形态**（基座 config + 抄来的 `architectures`）：本仓库选择在配置期
   报错并给出改法（§3.7），没有做"自动改写用户 config"。

## 8. 验证命令（复跑）

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/step66 -q
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmarks/check_step66_medusa.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/step58 tests/step59 tests/step60 \
    tests/step61 tests/step62 tests/step63 tests/step64 tests/step65 tests/step66 -q
```

`tests/step66` 里两条与上游交互的用例（`test_medusa.py::test_upstream_*`、
`test_mlp_support_boundary.py` 的四条证据）需要本机装有 `vllm==0.28.0`；它们实例化上游类做
数值对照/源码扫描，**生产路径不 import 上游**。
