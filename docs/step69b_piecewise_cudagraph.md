# 69 关补充：CUDA Graph 的 PIECEWISE 分段图（71 关的前置）

需求来源：`投机解码完整需求/071_动态投机长度与调度Graph兼容.md` §3.5 要求
"V1 full Graph 按 `_maybe_override_dynamic_sd_cudagraph_mode()` 降到 PIECEWISE"，
而 69 关当时只做了 FULL 那一半（PIECEWISE 在配置期明确拒绝）。本文件记录把另一半补齐的
实现、对照与差异。**这一份不新增关卡编号**：它是 69 关图轴的收尾，做完之后 71 关才有
合法的降级目标。

上游基线：本机 `vllm==0.28.0`。执行路径：V1（手工分段，不引入 torch.compile 轴）。

**一句话**：把每个 decoder 层切成"**注意力前段 / 注意力核心 / 后段**"，只把前/后两段录进图，
注意力留在图外——于是**混合 prefill+decode 批**（每请求 query 长度不同）也能用图，
因为图里没有任何"形状由每请求长度决定"的算子。

---

## 1. 交付物

| 文件 | 内容 |
|---|---|
| `minivllm/attention/backend.py`（新） | `AttentionCGSupport`（ALWAYS / UNIFORM_BATCH / UNIFORM_SINGLE_TOKEN_DECODE / NEVER）+ `min_cudagraph_support()`（多 KV group 取最保守） |
| `minivllm/attention/metadata.py`（改） | `AttentionMetadataBuilder._cudagraph_support`（默认 NEVER）+ `get_cudagraph_support()` |
| `minivllm/attention/backends/torch_sdpa.py`（改） | `TorchAttentionMetadataBuilder`（声明 **UNIFORM_BATCH**）、`TorchAttentionBackend`；`_graph_dispatched()` 收窄为"**只有 FULL** 才算图内"（PIECEWISE 的注意力是 eager，可用掩码写 KV、可走逐请求路径） |
| `minivllm/attention/backends/__init__.py`（改） | 导出上面三个名字 |
| `minivllm/models/qwen3.py`（改） | `Qwen3Attention.attention_pre()`（qkv+norm+RoPE）；`Qwen3DecoderLayer.attention_pre/attention_post/stage_attention_output/enable_piecewise_pieces()`；`Qwen3Model.enable_piecewise_pieces()` + embedding 输出静态化；`Qwen3ForCausalLM` 透传 |
| `minivllm/config.py`（改） | 放开 PIECEWISE / FULL_AND_PIECEWISE；默认回到上游的 `FULL_AND_PIECEWISE`；`splitting_ops` 不可配置（配了报错）；新增 `resolve_cudagraph_mode_and_sizes()`（按能力档位降级）与 `splitting_ops_contain_attention()` |
| `minivllm/cudagraph_dispatcher.py`（改） | 去掉"PIECEWISE 未实现"的拦截（键与分派逻辑 69 关就已按上游写好） |
| `minivllm/compilation/cuda_graph.py`（改） | 图显存池改成**全进程共享一个**（上游 `get_global_graph_pool()` 同款）：分段图是"每层两段 × 每档位"，一图一池会让显存按图数增长 |
| `minivllm/compilation/stats.py`（新） | `CUDAGraphStat` / `CUDAGraphLogging`（上游同名，聚合表逐段对齐） |
| `minivllm/worker/gpu_model_runner.py`（改） | 初始化时的**能力协商**；`_piecewise_attn_metadata()`（每轮新建、请求级不补齐）；`_prepare_inputs_padded()` 的 PIECEWISE 分支；`_dummy_run()` 的 PIECEWISE 分支；`execute_model()` 按模式选元数据；每轮的 `CUDAGraphStat` 记账 |
| `minivllm/spec_decode/draft_model.py`（改） | 草稿侧也用后端自己的 metadata builder（能力档位一致；草稿图仍属 74 关） |
| `tests/step69/test_piecewise_cudagraph.py`（新） | **9 项**：配置/切分点、能力协商与取 min、本后端档位、统计表（含 NONE）、`full` 降级、PIECEWISE 键无请求数+填充口径、分段图捕获一次/重放、四模式逐 token 一致（K=0 与 K=2）、未开分段图时不留任何包装器 |
| `tests/step69/test_spec_cudagraph.py`（改） | 3 处**期望值**更新（PIECEWISE 从"被拒绝"变"被接受/被降级"），断言强度不变；理由见 §6 |
| `docs/results.json → step69_piecewise` | 命令、设备、哈希与真实执行轨迹 |

包摘要：见 `docs/results.json` → `step69_piecewise.results.package.sha256`。

---

## 2. 路径映射与三态矩阵

| 本项目 | 上游（vllm 0.28.0） | 状态 |
|---|---|---|
| `CUDAGraphMode` 五值 + `decode_mode()/mixed_mode()` | `config/compilation.py:53-103` | ✅ 一致（69 关就已逐字对齐） |
| `CompilationConfig.resolve_cudagraph_mode_and_sizes()` | `config/compilation.py:1369-1470` | ✅ 三条降级规则一致；去掉上游的 SP/inductor 分支（本仓库没有那两轴），去掉 Mamba/块的校验（没有 Mamba） |
| `splitting_ops` / `splitting_ops_contain_attention()` | 同文件 `:1140-1210` / `:1300` 附近 | ⚠️ 语义一致、来源不同：上游是"编译期要拆的算子名"，本仓库是"模型结构里的注意力边界"，因此**不可配置** |
| `AttentionCGSupport` + 取 min | `v1/attention/backend.py:647` + `gpu_model_runner.py:7308` | ✅ 一致（本仓库只有一个后端组，取 min 的实现与上游同义） |
| PIECEWISE 的键放平 `num_reqs=None` | `v1/cudagraph_dispatcher.py:194-199` | ✅ 一致（69 关照抄，这一关才真的用上） |
| `CUDAGraphWrapper` 嵌套（外层 FULL / 内层 PIECEWISE） | `compilation/cuda_graph.py:145-290` | ✅ 一致（"模式不匹配就直通"那条正是嵌套分派的基础） |
| `CUDAGraphStat` / `CUDAGraphLogging` | 同文件 `:33-127` | ✅ 字段/表格式一致；差异：不做周期性自动打日志，由调用方触发 |
| —— 分段点怎么来 | `VllmBackend` 拆 fx 图 + `PiecewiseBackend` 每段编译捕获 | ⚠️ **机制不同**：本仓库手工分段（见 §6.1） |
| —— 段间张量的地址稳定性 | 依赖分配器确定性（`input_addresses` 只在 DEBUG 校验） | ⚠️ **主动收紧**：本仓库显式拷进静态缓冲（见 §6.2） |
| drafter 自己的分段图 | `proposer.initialize_cudagraph_keys(PIECEWISE)` | ❌ 仍未做（属 74 关），`cudagraph_mode` 记 NONE |
| breakable cudagraph / DP / ubatch / 多模态 encoder 图 | `breakable_cudagraph.py`、`dp_utils.py`、`gpu_ubatch_wrapper.py`、`encoder_cudagraph.py` | ❌ 未做（多卡/多模态/编译路径，均不在本仓库范围） |

**三态矩阵（这一关之后）**：

| 能力 | 上游 | 本项目 |
|---|---|---|
| FULL（decode 批全图） | 支持 | **支持** |
| FULL（混合批也全图） | 支持（要求 `AttentionCGSupport.ALWAYS`） | **拒绝**：本后端是 `UNIFORM_BATCH`，请求 `full` 会被降级成 `FULL_AND_PIECEWISE`（带 warning，不静默） |
| PIECEWISE（混合批分段图） | 支持（要求 `mode=VLLM_COMPILE`） | **支持**（手工分段，见 §6.1） |
| 两者的组合 `FULL_AND_PIECEWISE` | V1 默认 | **默认**（69 关时默认是 `FULL_DECODE_ONLY`，现已回到上游） |
| 非分段 `full` 字面执行 | 支持 | **不支持**（降级，不报错——与上游同一处理） |
| 编译一轴（dynamo/inductor/算子融合） | 支持 | **不支持**（`mode != none` 配置期报错） |
| drafter 图 / breakable / DP / ubatch / encoder 图 | 支持 | 不支持（各有归属关） |

---

## 3. 关键机制

### 3.1 分段点在哪里

```
每个 decoder 层：
    attention_pre(positions, hidden, residual)   ← 图（PIECEWISE）：融合 add-RMSNorm + qkv + q/k norm + RoPE
    ─────────────── 注意力核心（eager，写 KV + SDPA）───────────────
    attention_post(attn_out, residual)           ← 图（PIECEWISE）：o_proj + 融合 add-RMSNorm + MLP
```

这条边界的判据是"**形状由什么决定**"：两段的形状只取决于**行数**（图按档位定死），
注意力核心的形状还取决于**每请求 query 长度**（混合批里各不相同）→ 只能在图外。

上游的等价物是"`splitting_ops` = 注意力算子，编译期把 fx 图在这些算子处切开"：
同一件事的两个做法，图里"没有注意力"这个前提一致（§6.1 记机制差异）。

### 3.2 三种模式各走哪条路（同一个模型的同一份代码）

| 运行模式 | 外层全图包装器 | 内层分段包装器 | 注意力 |
|---|---|---|---|
| `FULL` | 捕获/重放整段前向 | **直通**（模式不匹配） | 图内 `_forward_padded()`（`clamp_min(0)` 写 KV） |
| `PIECEWISE` | **直通** | 捕获/重放每层两段 | eager（掩码写 KV + 逐请求注意力） |
| `NONE` | 直通 | 直通 | eager |

"模式不匹配就直通"是 `CUDAGraphWrapper.__call__` 的第一条规则（69 关就照上游实现了），
嵌套包装器因此各管各的——这一关才第一次真的用上它。

### 3.3 填充口径：PIECEWISE 只补 token 行

| | FULL | PIECEWISE |
|---|---|---|
| `num_tokens` | 补齐到档位 | 补齐到档位（同样） |
| `num_reqs` | 补齐后的请求数（图里的注意力按它定形状） | **真实请求数**（注意力在图外） |
| `query_start_loc` / `seq_lens` / 块表 | 按补齐后的请求数准备 | 按真实请求数**切片** |
| 元数据 | 建一次、缓存在图键下（图要读它的张量，必须是静态缓冲） | **每轮新建**（没有任何张量会被图读） |
| padding 行 | 槽位哨兵 + `seq_len=0` + 块表清零 | 槽位哨兵（token/position 归零）；请求级不补 |

### 3.4 静态边界缓冲（为什么必须有）

图里记的是**指针**。PIECEWISE 下每段的输入必须每次落在同一地址，所以：

```
embedding 输出   → 拷进 model._hidden_in_stage（第一段的输入）
注意力输出       → 拷进 layer._attn_out_stage（后段的输入；注意力是 eager，输出是新张量）
段与段之间的其它值（q/k/v、residual、每段输出）→ 本来就是**图自己的输出缓冲**，地址天然稳定
```

缓冲按 `max_num_batched_tokens`（输入工作区容量）开一次，**不随档位/轮次重建**。
（第一版按"图档位上限"开，实测第 6 行就写不下：档位上限只有 4，而 eager 回退的批可以更大。）

### 3.5 能力协商（为什么 `full` 会被降级）

`GPUModelRunner.initialize_cudagraph_capture()` 在**建键表之前**读一次注意力后端的能力档位：

```
TorchAttentionMetadataBuilder._cudagraph_support = UNIFORM_BATCH
   ↓ resolve_cudagraph_mode_and_sizes(min_cg_support, backend_name, uniform_decode_query_len)
mixed_mode()==FULL 且能力不是 ALWAYS → FULL_AND_PIECEWISE（切分点在注意力处）+ warning
decode_mode()==FULL 且能力是 NEVER   → PIECEWISE（或 NONE）
投机（1+K>1）且能力低于 UNIFORM_BATCH → 同上
```

这条链正是 71 关要复用的那一处：动态 K 打开时，上游用同一个函数把 full graph 降成 PIECEWISE。

### 3.6 图命中统计

每一轮（包括"这一轮没走图"）记一条 `CUDAGraphStat(num_unpadded_tokens, num_padded_tokens,
num_paddings, runtime_mode)`，`CUDAGraphLogging.generate_metric_table()` 聚合。实测样例：

```
| Unpadded Tokens | Padded Tokens | Num Paddings | Runtime Mode | Count |
| 1               | 1             | 0            | FULL         | 3     |
| 6               | 8             | 2            | PIECEWISE    | 1     |
| 4               | 4             | 0            | PIECEWISE    | 1     |
```

---

## 4. 验收对照（需求 069 §4 的图轴 + 071 §3.5 的前置）

| 条目 | 落点 |
|---|---|
| 混合 prefill/decode 批能用图 | `test_all_modes_agree_with_eager_token_by_token`（先 prefill 再加第二条请求 → 混合批；四种模式输出逐 token 相同，且分派记录里出现 `PIECEWISE`） |
| 分段键的正确性（任意请求数） | `test_piecewise_keys_drop_num_reqs_and_only_pad_tokens`：键 `num_reqs=None`；元数据按真实请求数切片 |
| 图与 eager 数值一致 | 同上（tiny，K=0 与 K=2）+ §5 的真实 1.7B 实测 |
| 稳定工作区不随档位重建 | `test_pieces_capture_once_per_key_and_replay_afterwards`：`captures == 键数`、`replays > 0`、`data_ptr()` 不变 |
| 未开分段图时零成本 | `test_piecewise_disabled_leaves_the_model_path_untouched`：`_piece_pre is None`、缓冲 `None` |
| 配置/能力边界不静默 | `test_piecewise_is_accepted_and_split_ops_are_recorded`、`test_attention_capability_min_is_the_most_conservative`、`test_plain_full_is_downgraded_by_capability_negotiation` |
| 统计口径 | `test_graph_stat_table_counts_every_mode_including_none` |

---

## 5. 实测

### 5.1 tiny 模型：四种模式逐 token 相同

```
none / piecewise / full_and_piecewise / full_decode_only × (K=None, K=2)
→ 6 组输出与 eager 完全一致；piecewise 组 9 步全部 PIECEWISE；
  full_and_piecewise 组 {PIECEWISE: 2, FULL: 7}
```

### 5.2 真实权重 Qwen3-1.7B（bf16，batch=1，档位 [1,2,4,8,16]）

| 模式 | 捕获 | 捕获耗时 | 图池 | 分派 | greedy（前 12 个 token） |
|---|---|---|---|---|---|
| `none` | — | — | — | `{NONE: 12}` | `[3764, 10, 4999, 1725, 15, 16, 17, 18, 19, 20, 21, 22]` |
| `full_decode_only` | 4 张 | 1.14 s | 3506 MiB | `{NONE: 1, FULL: 11}` | 同上（一致） |
| `full_and_piecewise` | 9 张 | 2.70 s | 3758 MiB | `{PIECEWISE: 1, FULL: 11}` | 同上（一致） |

混合批（A=8 token prompt 先跑，再入 B=5 token prompt）：

```
none:              {'A': [3764, 10, 4999, 1725, 15, 16], 'B': [16, 17, 18, 19, 20, 21]}  分派 {NONE: 8}
full_and_piecewise:{'A': same, 'B': same}                                                分派 {PIECEWISE: 2, FULL: 6}
→ 逐 token 一致
```

**代价**：分段图让捕获张数从 4 → 9（每层 2 段 × 5 个档位 = 56 张分段图共享同一个图池），
捕获时间 +1.6 s，池 +252 MiB（共享池，小图基本复用大图占下的显存）。
注意这不是"免费的默认值"：档位越多，分段图张数按 `2 × 层数 × 档位数` 增长。

### 5.3 每步 kernel 启动次数（tiny，decode 步，K=0，profiler 计数）

| 模式 | 每步 kernel 启动 | 图重放次数（8 步内） |
|---|---|---|
| `none`（eager） | **125.1** | 0 |
| `piecewise`（纯分段） | **49.9** | 28 |
| `full_and_piecewise` | **4.4** | 7 |

两条结论，都是这张表读出来的：分段图确实在省启动（125 → 50，约 −60%），但**纯 PIECEWISE 远不如
FULL（4.4）**——因为纯分段连 decode 步也要把注意力留在图外。这正是上游默认
`FULL_AND_PIECEWISE` 的理由：**FULL 用得上就用 FULL，只有混合批才落到分段图**。

### 5.4 回归

```
pytest tests/step58..70          → 556 passed（69 关 28 + 本关 9 + 其余基线 519）
benchmarks/check_step69_cudagraph.py 等全部 check 脚本 → 全绿（见 docs/results.json）
```

---

## 6. 差异账本

### 6.1 分段机制：手工分段 vs 编译期拆 fx 图（**最大的差异**）

上游 PIECEWISE 是 `torch.compile` 轴的产物：`splitting_ops = list(self._attention_ops)`
（`config/compilation.py:1155`）→ dynamo 抓图 → `VllmBackend.split_graph()` 在注意力算子处切开
→ `PiecewiseBackend` 对每段单独 inductor 编译并捕获（`compilation/piecewise_backend.py` 直接
import `torch._dynamo` / `torch._inductor`）。空 `splitting_ops` 时上游会警告并把 PIECEWISE
降成 NONE（"does not contain piecewise cudagraph"）。

本仓库没有编译轴，所以**手工**按模型结构切分。可对齐的部分：分派规则、键、填充口径、
"图里没有注意力"这个前提、以及"每段一张图 + 按档位捕获"的形状；**对不齐**的部分：
段内是未融合的 eager 算子序列（上游段内是 inductor 编译产物）、没有 `splitting_ops` 可配、
没有 compile cache 与 15 个 custom pass。要完全对齐必须把编译轴一起做出来（工作量大，
且其收益是算子融合，与投机主线关系不大）。

### 6.2 段间地址：显式静态缓冲 vs 依赖分配器确定性

上游 `CUDAGraphWrapper` 明确"不持有输入缓冲、不拷输入"，地址一致性靠分配器的确定性，
`input_addresses` 只在 `VLLM_LOGGING_LEVEL == DEBUG` 时校验。本仓库在**两处**显式拷贝
（embedding 输出、注意力输出），代价是每层每步一次 `[行数, hidden]` 的拷贝，
换来"地址必然稳定"（我们的地址校验是**始终开启**的，靠分配器赌不起）。

### 6.3 图显存池：共享 vs 每图一池

69 关的实现是"一张图一个池"（图少时无所谓）。分段图是 `2 × 层数 × 档位数` 张，
所以改成上游那样的**全进程共享池**。实测 1.7B：9 张图共 3758 MiB，其中分段图只多占 252 MiB。

### 6.4 `CUDAGraphStat` 不做周期性自动打日志

上游有 `observability_config.cudagraph_metrics` 开关 + 周期性 `log()`；本仓库没有 metrics
前端与日志节流配置，只提供 `generate_metric_table()`/`log()`，由探针/测试/demo 触发。

### 6.5 drafter 图仍未做（属 74 关）

上游 `SpecDecodeBaseProposer.initialize_cudagraph_keys()` 会在 `mixed_mode()` 是
PIECEWISE/FULL 时给草稿模型建 PIECEWISE 键；本仓库的 `proposer.initialize_cudagraph_keys()`
仍恒为 NONE（EAGLE 系的自回归提议循环每步形状不同，要接图得先做 74 关的多步融合图）。
**这不影响 target 侧的分段图**：两者是独立的图缓存。

### 6.6 仍然不支持的（三态矩阵里已列）

非分段 `full` 的字面执行（降级处理）、编译轴、breakable cudagraph、DP/ubatch/多模态 encoder 图。

---

## 7. 回归与后续

- 71 关接法：`resolve_cudagraph_mode_and_sizes()` 就是上游
  `_maybe_override_dynamic_sd_cudagraph_mode()` 要调的那个函数——动态 K 打开时，
  把 full graph 降成 PIECEWISE 的那条 warning 与降级动作已经就位，71 关只需在
  `CompilationConfig` 里按上游补 `_maybe_override_dynamic_sd_cudagraph_mode()`
  （`uses_dynamic_speculative_decoding()` 判定 + 强制 PIECEWISE）与 DP 关表两条规则。
- 69 关文档里"没有 PIECEWISE"的说法（§2 三态矩阵、§6.1/§6.6/§6.8、§7）已由本文件取代。
