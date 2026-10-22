# 第七十四关补充：Triton paged attention 后端与真正的 ALWAYS

> **状态更新（2026-10-22）**：`attention_backend="triton"` 现在优先映射到
> `docs/step74_triton_attn_upstream.md` 的上游 `TRITON_ATTN` 核心子集（block16/head16）；
> 本文件的逐行 kernel 保留为 `triton_paged` / 小 block/head 的 fallback。

> 目标：不再让 target 的 varlen/FULL 图受 `AttentionCGSupport.UNIFORM_BATCH` 限制。
> 新增一个 Triton paged attention 后端，声明 `AttentionCGSupport.ALWAYS`，支持
> **混合 prefill/decode 的 FULL 图**，从而让 DSpark 自适应验证走上游那条
> `require_varlen_cudagraph=True` 的硬门。

## 1. 为什么不是直接 import `flash_attn`

本机环境：

```text
flash_attn          : 未安装（ModuleNotFoundError）
flash_attn_2_cuda   : 未安装
flashinfer          : 0.6.16
triton              : 3.7.1
```

所以不能直接搬上游 `FlashAttentionImpl`。本项目之前（step22/37）已经有一版
Triton paged attention：

```text
_paged_attention_kernel       一个 program = 一行 query 的一个 head
tiled_paged_attention         一个 program = 一小块 query 的一个 head
```

它的语义是 paged causal attention + GQA，正好满足“任意 query 长度 + block table”；
本文件把它恢复成 V2 的一个后端。

## 2. 新增后端

```text
minivllm/attention/backends/triton_paged.py

TritonPagedAttentionMetadataBuilder
    _cudagraph_support = AttentionCGSupport.ALWAYS

TritonPagedAttentionImpl(TorchAttentionImpl)
    write_kv      继承 Torch（分页 KV 写法不变）
    forward:
        runtime_mode == FULL → _forward_paged()   # Triton kernel
        其它（eager / PIECEWISE）→ _forward_generic()（Torch 参考实现）

TritonPagedAttentionBackend
    get_name()      = "TRITON_PAGED"
    get_builder_cls = TritonPagedAttentionMetadataBuilder
    get_impl_cls    = TritonPagedAttentionImpl
```

配置入口：

```python
VllmConfig(..., attention_backend="triton")
# 默认仍是 "torch"
```

只允许 V2 Model Runner 使用；V1 配置期明确拒绝，不静默降级。

## 3. `_forward_paged()` 的输入

```text
query: [N, H*D]            按 query_start_loc 打包的扁平行
kv_cache: [2, num_blocks, block_size, Hk, D]
block_table: [B, max_blocks]
seq_lens: [B]
num_reqs: B                可能含 padding 请求
```

图内动态度量全部从 GPU 张量算：

```python
row = arange(N)                                  # 形状固定
token_to_req = (row[:, None] >= qsl[None, :-1]).sum(1) - 1
q_lens = qsl[1:] - qsl[:-1]
start = qsl[token_to_req]
query_pos = seq_lens[token_to_req] - q_lens[token_to_req] + (row - start)
```

没有 `.item()`、没有 Python 请求循环。`token_to_req` 对 padding 行会落到最后一个合法
请求，输出由上层忽略；不会越界。

kernel 里：

```text
一个 program = 一行 query 的一个 head
for blk in range(cdiv(seq_len, block_size)):
    phys = block_table[req, blk]
    K/V = kv_cache[phys, blk * block_size + offs]
    score = q · K
    causal: 只保留 key_pos <= query_pos
    online softmax
写回 out[row, head]
```

支持：

```text
GQA（q_head // group_size → kv_head）
causal / 非 causal
FP16 / FP32
padding 请求
混合 prefill/decode（每行自己查 token_to_req/query_pos）
```

不支持（明确边界）：

```text
量化 KV
sliding window
MLA / 多头潜变量
Triton tiled kernel（本文件先用逐行 kernel；性能不是本关目标）
```

## 4. ALWAYS 带来的行为变化

`resolve_cudagraph_mode_and_sizes()` 看到 `ALWAYS` 后不再把 `FULL` 降级：

```text
attention_backend="torch"   mode="full"
  → FULL_AND_PIECEWISE（混合批 PIECEWISE）
  → 混合批不能进 FULL 图

attention_backend="triton"  mode="full"
  → FULL（保持）
  → target manager 捕获混合批 FULL 图
```

`GPUModelRunner.initialize_cudagraph_capture()` 在 `mode == FULL` 时把
`target_mode` 设为 `FULL`（不是 `FULL_DECODE_ONLY`），并让 `_init_candidates()`
生成混合批 FULL 描述符：

```python
desc = (FULL, num_tokens=capture_size, num_reqs=min(num_tokens, max_num_reqs))
```

运行时 `execute_model()` 的 prefill gate 也相应放宽：

```text
decode FULL（uniform/max_query_len 非 None） + prefill → 继续回退 eager
mixed FULL（两个字段都 None） + prefill → 允许，交给 ALWAYS 后端
```

## 5. 自适应验证

Triton 后端时：

```text
_target_varlen_cudagraphs_enabled() → True
maybe_create_adaptive_verification_manager(require_varlen_cudagraph=True)
  → 检查 AttentionCGSupport.ALWAYS
  → Triton builder 满足
  → AdaptiveVerificationManager 正常创建
```

target manager：

```text
varlen_decode=True
mode=FULL
graphs 包含混合批 FULL + varlen decode FULL
```

`profile_adaptive_verification()` 仍然用 `captured_token_counts()` 生成尺寸，
`_cudagraph_limit = max(captured)`；`_dummy_run_adaptive()` 命中 target 图时
`run_fullgraph()`，记录 `full_cudagraph=True`。

## 6. 测试与实测

```text
pytest tests/step74/test_triton_paged.py -q       → 6 passed
pytest tests/step74/test_triton_backend_e2e.py -q → 2 passed
pytest tests/step79/test_adaptive_layout.py::test_adaptive_triton_backend_is_really_always -q
                                                   → 1 passed
pytest tests/step58 … tests/step79 -q             → 965 passed
```

覆盖：

- Triton kernel vs Torch `_forward_generic`：causal/非 causal、fp16/fp32，`rtol=1e-3`；
- padding 请求 / 尾部 padding 行：真实行相等、输出有限；
- 非投机混合批：`mode="full"` + Triton，输出与 Torch eager 逐 token 相同；
- EAGLE3 投机 target 图：输出与 Torch eager 相同；
- DSpark adaptive + Triton：`cudagraph_mode == FULL`、
  `_target_varlen_cudagraphs_enabled() == True`、`_cudagraph_limit == 8`、
  输出与 eager 相同。

## 7. 差异账本

| 编号 | 差异 | 影响 |
|---|---|---|
| D-TR-1 | 不是 `flash-attn` 包，而是本仓库历史 Triton paged kernel；逐行 kernel，未用 tiled/FA | 语义对齐 paged causal attention；性能不等价，长上下文比 FA 慢 |
| D-TR-2 | 只接 V2 target；草稿层仍是 Torch 参考后端 | target graph/adaptive 可用；草稿图不受影响 |
| D-TR-3 | 默认 `attention_backend="torch"`，Triton 必须显式开启 | 不改变现有默认行为/回归 |
| D-TR-4 | 不支持量化 KV / sliding window / MLA / tiled kernel | 这些配置下不能用 Triton 后端 |
| D-TR-5 | 没有真实 DSpark 权重；只验证 tiny 协议/数值/图 | 不给加速比，不声称生产性能 |
| D-TR-6 | `ALWAYS` 只在 target 的 FULL 图路径兑现；PIECEWISE 仍可走 breakable | 默认 `FULL_AND_PIECEWISE` 与 `FULL` 都能用；PIECEWISE 的 attention 仍是 eager |

## 8. 一句话

```text
Triton paged 后端 = 本仓库历史的 paged attention kernel + ALWAYS 能力声明
  → target 混合批可以进 FULL 图
  → adaptive 真正走上游的 ALWAYS 检查
  → 不再是“Torch 窄 varlen 图 + bypass 能力门”
```
