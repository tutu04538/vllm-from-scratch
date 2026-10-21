# 第七十四关补充 Phase 3：breakable PIECEWISE 与混合批

> 接 Phase 1/2：Phase 1 只做统一 decode 的 FULL，Phase 2 做 varlen decode。Phase 3 把
> **混合 prefill/decode 批**接到 target 图：不是 `torch.compile`，而是上游
> `vllm/compilation/breakable_cudagraph.py` 的 **breakable CUDA Graph** 思路。
> 参考实现：`BreakableCUDAGraphCapture`、`BreakableCUDAGraphWrapper`、
> `eager_break_during_capture`，以及 `unified_attention_with_output` 上的断点装饰器。

## 1. 为什么不能直接把混合批录成一张 FULL 图

FULL 图要求先 padding 到固定 `(num_tokens, num_reqs)`，但混合批里：

```text
一条 decode 请求：query_len = 1（或 1+K）
一条 prefill 请求： query_len = 5、7、……（每轮不同）
```

`query_len` 是每请求的，不能用一个 `uniform_token_count` 描述。上游的 PIECEWISE 做法是
**把模型切成若干段**：

```text
graph 段 0：embedding / qkv / 前段 MLP
eager 段 ：attention（读 runtime metadata，写固定 output buffer）
graph 段 1：o_proj / 后段 MLP
...
```

这样图的形状只由 token 行数定，attention 的每请求长度在 eager 段里用真实
`query_start_loc/seq_lens` 算。

## 2. 本仓库实现

### 2.1 文件

| 文件 | 作用 |
|---|---|
| `minivllm/compilation/breakable_cudagraph.py` | `BreakableCUDAGraphCapture`（多段 capture/eager/replay）、`BreakableCUDAGraphWrapper`（按 `BatchDescriptor` 缓存）、`eager_break_during_capture` |
| `minivllm/attention/layer.py` | `Attention.forward` 加断点装饰器；固定 `_output_buffer`；`set_output_buffer(max_tokens)` |
| `minivllm/worker/gpu/cudagraph_utils.py` | `ModelCudaGraphManager.pw_model`、`capture_pw_graph()`、`run_pw_graph()`；PIECEWISE 描述符 |
| `minivllm/worker/gpu/model_runner.py` | target manager 用 `FULL_AND_PIECEWISE`；混合批 dispatch 到 PIECEWISE；首次执行先 capture 再 replay；纯 prefill 回退 eager |
| `tests/step74/test_target_graph.py` | 混合批端到端 equal + PIECEWISE 命中/重放断言 |

### 2.2 capture 流程

`BreakableCUDAGraphCapture`：

```text
__enter__：切到常驻 side stream
  capture_begin(pool=global_graph_pool)
  ...（graph 段 0）
add_eager(fn)：
  capture_end()          ← 结束当前段
  fn()                   ← 在 side stream 上 eager 跑 attention
  segments.append(fn)
  capture_begin()        ← 开始下一段
__exit__：
  capture_end()
  prev_stream.wait_stream(side stream)
  切回 prev_stream
```

`BreakableCUDAGraphWrapper.__call__` 在 forward context 为 PIECEWISE 时：

```text
key = (num_tokens, num_reqs=None)
entry 不存在 → 直接报错（Runner 必须先 capture()）
entry 存在   → capture.replay(); 返回 entry.output（figure 段写回同一静态缓冲）
```

### 2.3 断点：为什么 attention 必须写固定 output

`Attention.forward` 现在：

```python
result = self.impl.forward(...)
out = self._output_buffer[: result.shape[0]]
out.copy_(result)
return out
```

`_output_buffer` 在 `initialize_kv_cache()` 里按 `max_num_tokens × hidden` 分配一次。
图段 0 的输出（query/key/value）作为 eager attention 的输入；eager attention 的输出写回
**同一地址**，图段 1 才能读到。如果返回新张量，图段 1 读到的还是捕获时的旧内存
——不报错，结果错。

断点装饰器 `eager_break_during_capture` 只在：

```text
BreakableCUDAGraphCapture 活跃
且 forward context 的 cudagraph_runtime_mode != FULL
```

时把函数交给 `add_eager`；FULL 图仍把 attention 整段录进去。

### 2.4 运行时

```text
execute_model()
  dispatch(...) → desc = (PIECEWISE, num_tokens=8, num_reqs=None)
  prepare_inputs(desc) → 真实行 + padding 行
  prepare_attn() → 真实 slot + padding slot=-1
  if wrapper.entries[desc] is None:
      with set_forward_context(PIECEWISE, batch_descriptor(8, None)):
          capture_pw_graph(input_ids, positions)     # 用这一轮真实输入录段
  with set_forward_context(PIECEWISE, ...):
      model_output = run_pw_graph(input_ids, positions)   # replay
```

**为什么首次执行先 capture 再 replay**：capture 是“录一遍”而不是“算一遍”；录完之后
同一轮还要真正重放一次才能得到这一轮结果。把 capture 放在 `model()` 外，避免 eager
attention 和后续 replay 共用同一份半成品状态。

**纯 prefill 批**（`is_prefilling_np.all()`）不碰 breakable，仍 eager；混合批
（至少一条 prefill + 至少一条 decode）才走 PIECEWISE。这是本仓库的稳定性边界，不是
语义必需（见 §5）。

## 3. 图键与缓存

- `BatchExecutionDescriptor`：`(cg_mode=PIECEWISE, num_tokens=捕获档位, num_reqs=None)`；
  PIECEWISE 键**不约束请求数**（与上游一致），因为 attention 在 eager 段按真实 qsl 切片。
- 缓存只有两处：
  - `CudaGraphManager.graphs`：FULL 图；
  - `BreakableCUDAGraphWrapper.entries`：PIECEWISE 段序列。
  没有第三套图缓存。
- `record_selection` 对 PIECEWISE 记 `replayed=False`（它不在 `manager.graphs` 里）；
  验收改看 `pw_model.entries` / `pw_model.num_replays`。

## 4. 测试与实测

```text
pytest tests/step74/test_target_graph.py -q   → 6 passed
pytest tests/step79 -q                        → 45 passed
pytest tests/step58 … tests/step79 -q         → 956 passed
check_step74_fused_draft.py                   → 23/23（计时项有抖动，重跑通过）
check_step79_adaptive.py                      → 23/23
```

关键用例：

- `test_target_breakable_piecewise_mixed_batch_matches_eager`：
  先跑一条 decode，再插入一条新 prefill，制造 `a: q_len=1, b: q_len=4` 的混合批；
  `mode="full"` 与 `mode="none"` 的 greedy 输出逐 token 相同；
  `manager.pw_model.entries` 非空、`num_replays > 0`，selection 里出现
  `mode=PIECEWISE, desc_num_reqs=None`。
- `test_target_graph_dispatches_piecewise_for_mixed_batch`：混合批 dispatch 命中
  PIECEWISE，而不是 Phase 1 的 NONE。
- `test_target_full_decode_graph_matches_eager`：FULL decode 仍与 eager 相同
  （PIECEWISE 不抢 decode 批）。
- `test_target_graph_without_speculator`：非投机 decode 也走 FULL。

实测混合批例子：

```text
第一轮：a prefill 6 行 → eager（纯 prefill）
第二轮：a decode 1 行 + b prefill 4 行 = 5 行 → desc PIECEWISE 8 行
        段 0/1 图重放；attention eager；输出与 eager 相同
第三轮：a/b 都 decode → desc FULL 8 行；只 replay
```

## 5. 差异账本（Phase 3）

| 编号 | 差异 | 为什么 | 影响 |
|---|---|---|---|
| D-PW-1 | 没有 torch.compile/FX splitting；手工把 `Attention` 设为断点 | 本仓库没有 compile 一轴（69 关边界） | 段内是未融合 eager 算子；语义/图键对齐，性能不等价 |
| D-PW-2 | `BreakableCUDAGraphCapture` 只包 target 模型，不包 drafter | drafter 的 PIECEWISE 属 74 关原始范围之外；先做 target | 混合批 target 用图；草稿 prefill/decode 仍 FULL/eager |
| D-PW-3 | eager 断点用**强引用**保存参数，不用上游的 weakref | 本仓库图段输出没有“图池持有”的强生命周期，weakref 在 replay 时已失效（实测 `None.view`） | 多占一份临时张量引用；不改变语义 |
| D-PW-4 | PIECEWISE 惰性捕获：第一轮混合批先 capture 再 replay；init 只热身 | 本机 pinned/UVA 分配器在多次 init breakable capture 后会踩 `CachingHostAllocator` 断言；lazy 稳定 | 第一轮混合批多一次捕获开销；后续都是 replay |
| D-PW-5 | 纯 prefill 批不碰 PIECEWISE，仍 eager | 避免纯 prefill 首次捕获与已录 FULL 图共享图池的干扰；Phase 3 先保证混合批 | 纯 prefill 性能不变，正确性不变 |
| D-PW-6 | `Attention.forward` 对所有 target 层做一次 `copy_` 到静态缓冲 | 图段间必须固定地址；上游用自定义 op 输出参数 | 每层多一次拷贝；无静默错 |
| D-PW-7 | `num_graphs` 统计的是 capture 段数（不含 eager 段）；PIECEWISE 不进 `manager.graphs` | 与上游 stats 的“是否命中图”口径不同 | 验收用 `pw_model.entries/num_replays`，不拿 `manager.graphs` 冒充 |
| D-PW-8 | 没有 offloader / sequence parallel / DP 的 breakable 分支 | 本仓库没有这些能力 | 只覆盖单卡 target 混合批 |

## 6. 明确边界

- 本仓库的 PIECEWISE 是 **手工断点**，不是上游的 compile/FX 自动拆分；因此没有
  `splitting_ops`、没有 compile cache、段内未融合。
- target 的 PIECEWISE 只服务**混合批**；纯 prefill 仍 eager。上游如果对纯 prefill 也走
  PIECEWISE，本仓库没有对齐这一条（Phase 3 的稳定性取舍，见 D-PW-5）。
- 本机没有真实 DSpark 权重；混合批只验证 tiny 协议/数值/图，不给加速比。
- FULL 混合批（不是 PIECEWISE）仍不字面执行；本后端 capability 仍是 `UNIFORM_BATCH`。
- PIECEWISE 图段缓存在 `BreakableCUDAGraphWrapper.entries`，上限是捕获档位数 ×；
  没有跨引擎共享。
