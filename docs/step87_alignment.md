# 第八十七关：把新后端接入已有 Graph，而不是再造一套 Graph

> 需求：`单卡执行优化需求/087_Graph接入与GPU元数据边界.md`
> 前置：086（Gemma4 原生分页 Attention）
> 下一关：088（单卡执行公平对照与收尾）
>
> 状态：**实现完成（target eager/FULL、Gemma4 assistant Q-only 图、工作区生命周期、
> 有效长度与容量小对照均实测；完整性能矩阵留给 088）**

## 1. 这一关要解决什么

086 已经把 attention 数学换成 unified 分页 kernel，但“图里到底跑了什么”还没有收口：

1. 旧图仍然按块表**最大宽度**准备 gather/循环，真实历史很短也围着容量转；
2. 3D 工作区挂在 backend 的**进程级**缓存里，引擎 shutdown 不释放，也不属于任何引擎；
3. assistant 的 Impl 在 086 已切到 Triton，但 metadata 组仍是 Torch builder；
4. target/assistant 的图、元数据、有效长度、第一轮/后续草稿步没有放在同一组证据里。

087 不重写 kernel，也不新增图缓存；它做的是：

```text
已有图管理器（CudaGraphManager / ModelCudaGraphManager）
  → 接入 TRITON_ATTN
  → 固定 buffer / metadata / workspace 的所有权
  → 让 kernel 从设备上的 seq_lens 读实际长度
  → 补 target/assistant 的 eager vs Graph 证据
```

## 2. 文件与职责

| 文件 | 本关改动 |
|---|---|
| `minivllm/attention/backends/triton_attn.py` | 新增 `TritonAttentionWorkspacePool`；builder 持有池并挂到 metadata；Impl 不再按层缓存；shutdown 可 `release()` |
| `minivllm/attention/metadata.py` | `AttentionMetadata.workspace_pool`：metadata 只借引用，池归 builder |
| `minivllm/worker/gpu/model_runner.py` | target builder 池随 shutdown 释放；assistant 的 `attn_groups` builder 也显式 release |
| `minivllm/worker/gpu/spec_decode/autoregressive/speculator.py` | `draft_attention_backend` 接缝（默认 Torch）；草稿 metadata 组的 builder 可跟随 target 后端 |
| `minivllm/worker/gpu/spec_decode/gemma4/speculator.py` | assistant 覆写为 target 选定的后端；metadata/Impl/workspace 契约统一 |
| `minivllm/worker/gpu/cudagraph_utils.py` | 不改图管理器；本次只使用既有 `ModelCudaGraphManager` / `SpeculatorCudaGraphManager` |
| `minivllm/compilation/cuda_graph.py` | V1 wrapper 不动；V2 没有另造第二套图缓存 |

## 3. 上游映射

| 本项目 | 冻结上游 | 状态 |
|---|---|---|
| `attention/backends/triton_attn.py` | `v1/attention/backends/triton_attn.py` | 子集；新增的池是本地对 builder `__init__` 预分配工作区的等价落点 |
| `attention/metadata.py::AttentionMetadata.workspace_pool` | `TritonAttentionMetadata` 的三个 workspace 字段 | 本地用池对象承载；语义相同：同一 builder/每步稳定 |
| `worker/gpu/model_runner.py` | `v1/worker/gpu/model_runner.py` 的 capture/release 边界 | 子集；单卡、无 DCP/EPLB |
| `worker/gpu/cudagraph_utils.py` | 上游 V2 图管理器 | 已有实现，不在 087 重写 |
| `worker/gpu/spec_decode/autoregressive/speculator.py` | `v1/worker/gpu/spec_decode/autoregressive/speculator.py` | 增加草稿 metadata 后端选择；沿用既有 prefill/decode 图生命周期 |
| `worker/gpu/spec_decode/gemma4/speculator.py` | 上游 Gemma4 speculator | assistant 的 Q-only metadata 跟随 target 后端 |

## 4. Buffer 所有权表（需求 §3 的交付）

| buffer | owner | shape / dtype | 地址 | 谁更新内容 |
|---|---|---|---|---|
| `input_ids` | `InputBuffers`（Runner） | `[max_num_batched_tokens]` int32 | 常驻 | `prepare_prefill_inputs` / `combine_sampled_and_draft_tokens` 等内核 |
| `positions` | `InputBuffers` | `[max_num_batched_tokens]` int64 | 常驻 | 输入组装内核 |
| `query_start_loc` | `InputBuffers` | `[max_num_reqs + 1]` int32 | 常驻 | 输入组装内核；尾部按图补齐 |
| `seq_lens` | `InputBuffers` | `[max_num_reqs]` int32 | 常驻 | 输入组装 / `prepare_decode_inputs` / `update_draft_inputs`；kernel 每步从设备读 |
| `block_table` | `BlockTables.input_block_tables[group]` | `[max_num_reqs, max_num_blocks]` int32 | 常驻 | `gather_block_tables`；补齐行清零 |
| `slot_mapping` | `BlockTables.slot_mappings` | `[num_groups, max_num_batched_tokens]` int64 | 常驻 | `compute_slot_mappings`；padding 写 `-1` |
| target hidden | `ModelCudaGraphManager.hidden_states` | `[max_capture_tokens, hidden]` 模型 dtype | 捕获后固定 | 图重放写、`sample()` 前 clone/读取 |
| draft 输入 | speculator 自己的 `InputBuffers` | 同 target | 常驻 | `prepare_prefill_inputs` / `prepare_decode_inputs` |
| 共享 KV | target `Attention.kv_cache`（packed slab 视图） | `[2, blocks, block_size, kv_heads, head_dim]` 或页 stride 视图 | 常驻 | target 非共享层 `reshape_and_cache` 写；assistant Q-only 只读 |
| attention 输出 | `Attention._output_buffer` 或 unified 临时输出 | `[tokens, num_heads × head_dim]` | 捕获后的层内固定 | unified kernel 直接写；不再二次 copy |
| 2D/3D 工作区 | `TritonAttentionWorkspacePool`（builder 持有） | `[threshold, heads, 16, head_pad]` 等三块 FP32 | 每桶固定 | 3D kernel / `reduce_segments` 写；shutdown `release()` |
| CUDA Graph | 对应 `CudaGraphManager` | 不透明对象 | 捕获后固定 | Runner 只 `replay()` |

硬边界：

- 捕获后改的是**内容**，不换 tensor 地址；图键只表示形状 bucket/执行协议，不含 request
  ID，也不为每个实际 `seq_len` 建图；
- 图里没有第二份 KV 镜像，也没有第二套 graph cache；
- `2D/3D` 在**启动前**按 host 侧 `max_query_len`、`num_seqs`、配置决定；kernel 在设备上读
  `seq_lens` 决定循环到哪一 tile。不能在重放时读 GPU 再切 Python 分支；
- 容量只是块表/寻址的安全上界，不决定每层扫描长度。

## 5. 实际执行边界

### 5.1 target

- 统一 decode：`max_query_len = 1 + K`；K=2 是 3 行，走 **2D**。
- K=0 单 token decode：`max_seqlen_q=1` 且 `num_seqs <= 128 // num_kv_heads`，走 **3D** +
  `reduce_segments`。
- `FULL` 模式可以录混合 prefill/decode 图；`FULL_DECODE_ONLY` 只录 target decode 满图，
  prefill 仍 eager。两者都是现有明确策略，不把“prefill eager”误报成“decode Graph 不支持”。
- `2D/3D` 是每个 graph bucket 自己选的：K=2 的 3 行 decode 是 2D；捕获时录的 1 行 mixed
  bucket 仍可能走 3D，所以“target builder 有 workspace”不等于“K=2 decode 用了 3D”。这一点
  在真实测试里用“实际选中的 bucket 全是 3 行”区分。

### 5.2 assistant

- Gemma4 assistant 的 4 个 Q-only 层继续读 target 的同一份 KV，不分配第二份 cache。
- 086 让 assistant 的 Impl 走 Triton；087 让它的 metadata 组也使用
  `TritonAttentionMetadataBuilder`，从而拿到同一套 `workspace_pool` 和
  `update_draft_decode_metadata()` 契约。
- assistant 有 sliding/full 两个 KV group，所以是两个 builder、各自一个池；K=2 的 1 行
  decode 使用 3D，两个池各一个桶。
- 首轮真正 prefill 时 `has_prefill=True`，`uniform_token_count` 为 `None`，草稿 prefill 图
  按现有策略走 eager；后续 decode 轮才重放草稿 prefill 图（位置 0）与 decode 图（K−1 步）。
  这是策略边界，不是 Graph 能力缺口。

### 5.3 metadata 与 CPU/GPU 边界

- `_build_attn_metadata()` / `prepare_inputs_to_capture()` / `TritonAttentionImpl` 入口没有
  `.item()`、`.cpu()`、`.tolist()`；`seq_lens.numel()` 只读 host 形状。
- 采样提交结果时的 D2H 保留；087 不要求全系统零同步，也不打开异步调度。
- 草稿步的 `seq_lens/slot_mapping` 是常驻缓冲视图，`update_draft_decode_metadata()` 是空操作
  （与上游 Triton 后端同一条理由）；图重放不执行 Python，但状态已由输入内核原地更新。

## 6. 工作区生命周期

087 之前：

```text
_WORKSPACE_CACHE[(device, heads, head_dim, threshold, segments)]
  ↑ 模块级字典
  ↑ 引擎 shutdown 不清理
```

087 之后：

```text
Runner
  └─ TritonAttentionMetadataBuilder
       └─ TritonAttentionWorkspacePool（每个 builder 一个）
            └─ 桶（按 device/heads/head_dim/threshold/segments）
```

- `build()` 把池挂到每份 `AttentionMetadata`；
- `TritonAttentionImpl` 只从 metadata 借池，不再有 `_workspace` 字段；
- target builder 在 Runner shutdown 释放；
- assistant 的每个 `AttentionGroup.metadata_builders` 也显式 release；
- 弱引用测试证明 shutdown 后工作区 tensor 真被回收，不只是“从字典里删掉”。

## 7. 验收与实测

### 7.1 测试

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/step87 -q
# 13 passed

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest tests/step87 -q
# 4 passed, 9 skipped（CPU 单测 + CUDA/真实模型用例按设备跳过）
```

覆盖：

| 文件 | 内容 |
|---|---|
| `test_workspace_ownership.py` | builder 同一池；地址复用；release 弱引用；无进程级缓存；2D/3D CUDA Graph 重放读取更新后的 `seq_lens` |
| `test_target_graph_boundary.py` | 同 bucket 换 prompt/seq_len 不重捕；捕获 metadata 全是常驻缓冲视图；容量 32/64 `max_model_len`（块表 8/16 列）输出/图键相同；K=4；草稿逐轮状态 vs eager；静态检查 metadata 路径无 D2H token |
| `test_real_gemma4_graph.py` | 真实 target K=0 FULL、target+assistant K=2 FULL：hidden/logits/输出对照；assistant 两个 Triton builder；target/assistant 3D 桶；随机采样触发拒绝；shutdown 释放工作区 |

### 7.2 真实 kernel trace

`python benchmarks/check_step87_graph.py --trace`，真实 Gemma4 target+assistant、B=1、K=2、
`FULL_DECODE_ONLY`，前 1 轮 prefill 在 profiler 外，后 8 个 decode step：

| 指标 | 实测 |
|---|---:|
| `kernel_unified_attention` | 344 = 8×43（35 target + 8 assistant） |
| `reduce_segments` | 32 = 8×4（assistant 4 个 3D 层） |
| `reshape_and_cache_kernel` | 120 = 8×15（target 15 个非共享层） |
| legacy paged/fixed | 0 |
| CPU kernel/graph launch events | 14/step |
| GPU kernel events | 1012.5/step |
| 捕获图 | 3 张（target 1、draft prefill 1、draft decode 1） |
| graph pool | 88,080,384 bytes |
| target 3D workspace | 0（K=2 decode 2D） |
| assistant 3D workspace | 8,454,144 + 16,842,752 = 25,296,896 bytes |

同一配置下 086 的 eager trace 是 CPU launch 323/step、unified 344、reduce 32、reshape 120；
087 的 kernel 数没有变，CPU launch 降到 14/step，说明图确实把 target/assistant 的主干录进去
了。两次 trace 的采集边界相同（首轮 prefill 外，8 个 decode），但本关仍不把它当公平性能表，
公平对照留 088。

### 7.3 容量小对照

`python benchmarks/check_step87_graph.py --capacity`：同一 prompt、同一有效长度、同一
K=2，只改物理块容量 512 → 2048：

| blocks | 输出 | target 图键 | target workspace | 8 step 用时 |
|---:|---|---|---:|---:|
| 512 | 相同 | `(3,1,3,None)` | 0 | 82.26 ms |
| 2048 | 相同 | `(3,1,3,None)` | 0 | 85.36 ms |

耗时比 1.038（2048 这一轮反而略慢，单次噪声大于差异）；单次运行、包含完整
`engine.step`，只用于确认“容量变大没有让图键/输出变化”，不作为加速/变慢结论。结构证据是：kernel 的 `_tile_loop_bounds()` 只从 `seq_len` 算
`num_tiles/tile_end`，块表宽度只参与 `seq_offset // BLOCK_SIZE` 寻址。

## 8. 差异与边界

| 编号 | 差异 / 边界 | 影响 |
|---|---|---|
| D87-1 | 工作区从进程级缓存改为 builder 池；同一模型同一 shape 仍共享 | shutdown 真释放；不再跨引擎泄漏；图地址仍稳定 |
| D87-2 | Gemma4 assistant 的 metadata 组跟随 target 后端；EAGLE/MTP 等草稿层仍 Torch | 087 只扩大 Gemma4 assistant，不声称所有草稿层上 Triton |
| D87-3 | target 捕获仍用 `make_dummy` 的 query 长度，不调用会填 `seq_lens=1` 的基类 `build_for_cudagraph_capture()` | 捕获仍安全、无 NaN；重放前输入内核会把真实 `seq_lens` 写回同一块缓冲；与上游“填 1 提速”有差异 |
| D87-4 | 没有给 `seq_threshold_3D` 做 capture-size 吸附 | 仍是 `128 // num_kv_heads`；小 batch 3D 选择不随捕获档位改变 |
| D87-5 | 首轮 assistant prefill 走 eager；后续 decode 轮才重放草稿 prefill/decode 图 | 符合现有 prefill eager 策略；不能解读为“assistant 完全不进图” |
| D87-6 | K=2 target 的 3 行 decode 固定 2D；full D=512 的小 grid 性能边界仍在 | 087 不解决 086 的 ~0.94× micro 结论，留 088 |
| D87-7 | 采样提交仍有 D2H；没有打开异步调度 | 是合法边界，不是本关遗漏 |
| D87-8 | 未做整个 `Engine.step` 一张图、未合并 target/draft launch | 需求明确不要求；CPU 调度/采样边界保留 |

## 9. 复现命令

```bash
PY=/home/user/anaconda3/envs/vllm-omni-dev/bin/python

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY -m pytest tests/step87 -q
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  $PY -m pytest tests/step87 -q

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY -m pytest \
  tests/step58 … tests/step87 -q
# 1267 passed, 148 warnings, 202.28s

$PY benchmarks/check_step87_graph.py --trace
$PY benchmarks/check_step87_graph.py --capacity

# 与本关相关的既有回归
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY -m pytest \
  tests/step74/test_target_graph.py tests/step74/test_graph_manager.py \
  tests/step80/test_gemma4_real_integration.py tests/step83 -q
```

真实 trace/容量原始 JSON 在 ignored 的 `benchmarks/results/step87/`，不要提交。

## 10. 一句话

**087 把 086 的 unified 分页 attention 真正接进已有 target/assistant CUDA Graph：
3D 工作区从进程级缓存改成 builder 持有并在 shutdown 释放，assistant 的 metadata 组跟随
Triton，kernel 每步从设备上的 `seq_lens` 决定循环长度；真实 K=0/2 eager 与 Graph 的
hidden/logits/输出一致，K=2 trace 中 CPU launch 从 323/step 降到 14/step，容量 512→2048
不改变图键或输出。**
