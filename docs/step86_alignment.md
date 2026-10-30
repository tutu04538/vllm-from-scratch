# 第八十六关：Gemma4 原生分页 Attention

> 需求：`单卡执行优化需求/086_Gemma4原生分页Attention.md`
> 前置：085（Norm/激活融合与 packed QKV）
> 状态：**实现完成（eager 主线 + Q-only + SWA + 3D；完整 Graph 元数据收口留给 087，
> 公平性能矩阵留给 088）**

## 1. 这一关改了什么

085 把 norm/激活/QKV 的小算子压下去了，但 attention 仍然有三类结构性开销：

1. eager 走 Torch `_forward_generic`：逐请求 Python 循环、全量 gather K/V、把 1 个 KV 头
   `repeat_interleave` 成 8 份；
2. sliding 层也按完整历史算，再 mask 掉窗口外 key；
3. Q-only 共享层（target 20 层 + assistant 4 层）根本不进 Triton。

086 把 target/assistant 的 attention 收敛为**一条 unified 分页 kernel**：

```text
Runner metadata
  → Gemma4 Attention
    → 有写权：triton_reshape_and_cache_flash
    → Q-only：不写，直接读源层 cache
    → TritonAttentionImpl
      → unified_attention（2D 或 3D）
```

eager、FULL 图、Q-only 都走同一条路径。旧的 `triton_paged.py`（逐行旧 kernel）和
`fixed_attention.py`（图内固定 kernel）已删除；`triton_paged` 配置名也删除。

## 2. 文件与职责

| 文件 | 职责 |
|---|---|
| `minivllm/attention/ops/triton_unified_attention.py` | `unified_attention` 薄入口、2D/3D 主 kernel、`reduce_segments`、3D 选择条件 |
| `minivllm/attention/ops/triton_reshape_and_cache_flash.py` | 按 slot 写 K/V；负 slot 跳过；接收真实 stride |
| `minivllm/attention/backends/triton_attn.py` | `TRITON_ATTN` backend / builder / impl；支持性检查、workspace、dispatch |
| `minivllm/attention/layer.py` | unified 直接写 output buffer 时跳过自拷贝 |
| `minivllm/worker/gpu/model_runner.py` | Gemma4 assistant 加载时也使用选定后端 |
| `minivllm/config.py` | 删除 `triton_paged` 配置名；删除 assistant 只能 Torch 的限制 |
| `minivllm/attention/backends/__init__.py` | `triton` / `triton_attn` 都映射到 `TRITON_ATTN` |

**已删除**：

```text
minivllm/attention/backends/triton_paged.py
minivllm/attention/ops/fixed_attention.py
tests/step74/test_triton_paged.py
```

`tests/step74/test_triton_backend_e2e.py`、`tests/step79/test_adaptive_layout.py` 的
`TRITON_ATTN_FIXED` 断言改为 `TRITON_ATTN`；`tests/step80` 原先“Triton 必须拒绝 Q-only”
的用例改为“Triton Q-only 输出与 Torch 一致，且所有 Q-only 层确实是 Triton”。

## 3. 上游映射

| 本项目 | 冻结上游 | 状态 |
|---|---|---|
| `attention/ops/triton_unified_attention.py` | `v1/attention/ops/triton_unified_attention.py` | 子集：2D/3D、GQA、causal/非 causal、SWA、paged KV、FP32 在线 softmax |
| 同上（helper 内联） | `v1/attention/ops/triton_attention_helpers.py` | 只保留 `find_seq_idx`、`resolve_seq`、`softmax_step`、tile bounds、mask |
| `attention/ops/triton_reshape_and_cache_flash.py` | `v1/attention/ops/triton_reshape_and_cache_flash.py` | 原生 dtype 子集；负 slot、strided cache |
| `attention/backends/triton_attn.py` | `v1/attention/backends/triton_attn.py` | 薄 backend + 能力检查；没有量化/cascade/MM/RSWA/TD 分支 |
| `attention/layer.py` | `Attention.forward` 的输出缓冲 | unified 直接写 `_output_buffer`；same-pointer 不再 copy |

本地 KV 布局仍是：

```text
kv_cache [2, num_blocks, block_size, num_kv_heads, head_size]
k = kv_cache[0], v = kv_cache[1]
```

kernel 接收两片 4D 视图和各自 block/slot/head/dim stride；不假设整个 cache contiguous。

## 4. 关键实现

### 4.1 KV 写入

`reshape_and_cache_kernel` 一个 program 处理一个 token 的 head×dim tile：

- 读 `slot_mapping[token]`；`slot < 0` 直接 `return`，不再像旧图内路径那样
  `clamp_min(0)` 写 0 号留白块；
- `block_idx = slot // block_size`、`offset = slot % block_size`；
- 源 K/V 按 `(token, head, dim)` 寻址，支持 packed QKV 的非连续 row stride；
- 目标 cache 按真实 `(block, slot, head, dim)` stride 写；
- Q-only 层不调用这个入口。

### 4.2 2D unified 主干

一个 program = **一个 query block × 一个 KV head**：

```text
BLOCK_M = 16（GQA group <= 16）或 next_pow2(group)
BLOCK_Q = BLOCK_M // group
BLOCK_M 行 = BLOCK_Q 个 query 位置 × group 个 query head
```

- grid `(total_q_blocks, num_kv_heads)`；
- `total_q_blocks = floor(num_tokens / BLOCK_Q) + num_seqs`，kernel 内二分
  `query_start_loc` 映射到 `(req, q_block)`；
- `q` 支持 packed QKV 的 `stride_q0/stride_q1`；
- GQA 用 `query_head = kv_head * group + row % group`，**不复制 KV 头**；
- `tl.dot(Q, K^T)` + online softmax；FP32 时用 `input_precision="ieee"`，避免 TF32 误差；
- `P` cast 到 V dtype 后 `tl.dot`，累加器 FP32；
- `out` 直接写调用方提供的 buffer，不再先 `empty_like` 再拷贝。

### 4.3 SWA

`window_size=(w-1, 0)`，kernel 内 `SLIDING_WINDOW = w`：

```text
key 可见条件：q_abs - key_pos < w
```

tile 循环不是从 0 扫到 seq_len，而是先算：

```text
first_key = max(0, q_abs - w + 1)
tile_start = first_key // TILE_SIZE
tile_end   = last_allowed_key // TILE_SIZE + 1
```

再在 `[tile_start, tile_end)` 里做在线 softmax。窗口外 key 不参与计算，不是“先算完整
历史再 mask”。full 层 `SLIDING_WINDOW=0`，仍扫完整有效范围。

kernel 还会把当前 tile 中所有 query 都看不到的 V 清零，避免 `P=0 × NaN = NaN`。

### 4.4 3D 单 token 分段

低 batch、`max_seqlen_q == 1`、`num_seqs <= seq_threshold_3D` 时启用：

```text
seq_threshold_3D = max(1, 128 // num_kv_heads)
num_par_softmax_segments = 16
```

3D kernel 每条 KV 请求切 16 段，每段独立算 partial softmax，写：

```text
segm_output [threshold, num_heads, 16, head_padded]  FP32
segm_max    [threshold, num_heads, 16]
segm_expsum [threshold, num_heads, 16]
```

最后 `reduce_segments` 按 `max` 重新缩放并合并。

硬边界：

> **K=2 target 有 K+1 行 query，`max_seqlen_q > 1`，必须走 2D；不能因为 batch 小就
> 强行走只适用于单 query 的 3D。**

真实 K=2 trace 也验证了这一点：target 的 35 层只增加 `kernel_unified_attention`，没有
`reduce_segments`；assistant 的 4 层才走 3D。

### 4.5 Q-only 共享层

`TritonAttentionImpl.forward_q_only` 是独立入口：

```text
q = query.unflatten(...)
out = layer._output_buffer
unified_attention(q, source_k_cache, source_v_cache, out, metadata)
```

不调用 `reshape_and_cache`，所以源层 cache 逐位不变。target 最后 20 层和 assistant 4 层
全部走这个入口，不再继承 Torch 的 `_forward_padded`。

### 4.6 支持矩阵

| 能力 | 状态 |
|---|---|
| paged KV / GQA / causal / 非 causal | 支持 |
| sliding window | 支持 |
| head_dim 1..512（padded 至少 16） | 支持；真实 256/512、Qwen3 128、tiny 8/16 都测了 |
| block_size 任意正数 | 支持；本地 tiny 用 4，真实 Gemma4 用 16 |
| fp16 / bf16 / fp32 | 支持；FP32 dot 用 IEEE |
| 2D prefill/decode/混合批 | 支持 |
| 3D 单 token decode | 支持 |
| Q-only | 支持，只读 |
| 量化 KV / ALiBi / sinks / softcap / cascade / MM prefix / RSWA / chunked / TD | **不支持**，没有这些入口 |

本后端不假装是完整上游 `TRITON_ATTN`：没有量化、cascade、多模态 prefix、RSWA、tensor
descriptor；这些分支没有被静默降级到 Torch。

## 5. 测试

```text
tests/step86/test_unified_attention.py       17 passed
tests/step86/test_kv_write_and_qonly.py       8 passed
tests/step86/test_backend_integration.py      3 passed
tests/step86/test_real_gemma4_triton.py       2 passed
------------------------------------------------------
tests/step86                                 30 passed
```

CPU-only：

```text
CUDA_VISIBLE_DEVICES='' pytest tests/step86 -q
2 passed, 28 skipped
```

覆盖点：

- BF16/FP16/FP32、GQA（8/1、4/1、7/1）、真实 head_dim 256/512、tiny 8/16；
- 变长 prefill、单 token decode、K+1 多行、混合批、空请求；
- SWA `q_len=1/3/5`、左边界、full/sliding 混合批；
- packed QKV row stride、非连续 KV block stride、padded output 不越界；
- 负 slot 不写、packed cache stride、`do_kv_cache_update`；
- Q-only cache bitwise 不变、直接写 output buffer；
- 3D vs 2D vs Torch、空序列/tail segment；
- `can_use_3d(max_seqlen_q=3)` 必须为 False（K=2 target 留 2D）；
- 旧模块已删除、`triton_paged` 配置名被拒绝；
- tiny V2 eager / FULL 图 / 投机目标图；
- 真实 Gemma4 target K=0、target+assistant K=2，与 Torch 输出逐 token 相同；
- 真实 K=2 随机采样至少触发一次拒绝，请求结束后再次请求能复用 slot；
- 真实 K=2 运行期间把 `TorchAttentionImpl._forward_generic` 替换成抛异常，仍能完成，
  证明热路径没有回 Torch。

全量回归：

```text
pytest tests/step58 … tests/step86 -q
1254 passed, 0 failed, 0 skipped, 148 warnings in 169.82s
```

## 6. 真实 K=2 eager trace

固定 14-token prompt、真实 target 4.63B + assistant 77.7M、BF16、K=2、eager；
首轮 prefill 在 profiler 外，后面 8 个 decode step：

```text
build_s                   4.54
profiled_decode_steps     8
tokens                    [9259, 236888, 1030, …, 106]
GPU kernel events         8092（1011.5/step）
CPU kernel launch events  2584（323/step）
distinct GPU kernels      56
kernel_unified_attention  344（43/step）
reduce_segments            32（4/step）
reshape_and_cache_kernel  120（15/step）
legacy paged/fixed          0
allocated                 9,849,767,936 B
reserved                 10,454,302,720 B
peak allocated           10,331,055,104 B
```

解释：

- **43 = 35 target + 8 assistant**：assistant K=2 每步跑 2 个草稿步 × 4 层；
- **4/step `reduce_segments` = assistant 4 层**：target K=2 的 3 行 query 留在 2D；
- **15/step `reshape_and_cache` = target 15 个非共享层**：20 个 Q-only target 层和
  assistant 都不写 cache；
- `legacy paged/fixed = 0`：旧 kernel 没有在热路径复活。

## 7. 小规模 attention 测量

`benchmarks/check_step86_attention.py --micro`，每个 case 独立子进程跑 3 次，每次
5 个 CUDA-event loop 取中位数；包含 Python launch 间隙，单机噪声明显，**不是**端到端
吞吐。

| case | unified 2D | unified 3D | Torch generic | 2D/Torch |
|---|---:|---:|---:|---:|
| decode SWA D=256, seq=4096, window=512 | 166.8 μs | 95.3 μs | 787.2 μs | 0.22 |
| decode full D=512, seq=2048 | 805.2 μs | 118.0 μs | 863.7 μs | 0.94 |
| prefill SWA D=256, q=128, seq=2048 | 165.6 μs | — | 929.3 μs | 0.18 |
| prefill full D=512, q=128, seq=2048 | 795.1 μs | — | 1157.6 μs | 0.70 |

结论按证据强度写：

- SWA decode/prefill 和 3D 单 token decode 明显低于 Torch 分解路径；
- **full D=512 的 K+1 多行 2D decode 只有约 0.94×**：B=1、3 行 query、1 个 KV head 时
  grid 只有约 2 个 program，GPU 填不满；3D 虽然快，但 K=2 target 有多行 query，不能用。
  这是 088 公平矩阵要单独看的项，不能宣称所有 decode 都加速；
- 微测包含 allocation/launch 差异（Torch 路径自带 `empty`，unified 用调用方 buffer），
  不能把表里的比值直接当端到端收益。

## 8. 差异与边界

| 编号 | 差异 | 影响 |
|---|---|---|
| D86-1 | 上游完整 unified 的量化、ALiBi、sinks、softcap、cascade、MM prefix、RSWA、chunked、TD 未实现 | 本地模型不用；入口没有这些参数，不静默降级 |
| D86-2 | 3D workspace 由 backend 的进程级 shape cache 持有，不在 metadata builder 里 | 同一模型 sliding/full 各共享一份；跨引擎会保留到进程结束（约 16–32 MiB/份） |
| D86-3 | target K=2 的 2D decode grid 小，full D=512 只有约 0.94× Torch | 不能用 3D；留给 088 公平矩阵/后续调优 |
| D86-4 | block_size 任意正数，上游声明 MultipleOf(16) | 本地 tiny 测试需要 block=4；真实模型是 16 |
| D86-5 | head_size 1..512，padded 至少 16 | tiny 8/16 可用；真实 256/512 是主路径 |
| D86-6 | FP32 用 IEEE dot；BF16/FP16 用 tensor core，累加 FP32 | FP32 与 Torch 约 1e-6；低精度是舍入差，不是逐位等价 |
| D86-7 | Q-only 的源层 cache 是 target 的 `[2,...]` 视图 | 只读，cache bitwise 不变；不复制第二份 |
| D86-8 | assistant 只在 Gemma4 MTP 加载时切到选定后端；EAGLE/DFlash 等草稿层仍 Torch | 086 只覆盖 Gemma4 assistant；草稿后端边界未扩大 |
| D86-9 | 3D 只在 `max_seqlen_q==1` 且 `num_seqs<=threshold` 时启用 | K=2 target 明确留 2D；threshold=128//kv_heads |
| D86-10 | 没有把旧 `triton_paged` / `fixed_attention` 保留为生产 fallback | 删除 7 个旧 step74 测试；新能力由 step86 覆盖；配置名 `triton_paged` 直接报错 |

## 9. 复现命令

```bash
PY=/home/user/anaconda3/envs/vllm-omni-dev/bin/python

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY -m pytest tests/step86 -q
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  $PY -m pytest tests/step86 -q

$PY benchmarks/check_step86_attention.py --micro
$PY benchmarks/check_step86_attention.py --trace
```

真实 trace 结果在 ignored 的 `benchmarks/results/step86/`；不要提交。

## 10. 一句话

**086 把 Gemma4 的 attention 从“eager 逐请求 gather + GQA 复制 + 图内固定 kernel”
改成“eager/图/Q-only 都走同一条 unified 分页 kernel”，并删除了 `triton_paged.py`、
`fixed_attention.py` 和 `triton_paged` 配置名；SWA/3D/Q-only 都有独立证据，K=2 target
明确留在 2D，full D=512 的 2D decode 性能边界也如实记录。**
