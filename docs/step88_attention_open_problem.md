# 开放问题：为什么 mini 的 paged K/V load 不被 Triton pipeline 选中？

> 目的：把当前 2D attention 的剩余差距写成可独立研究的问题。
> 约束：不直接大面积复制 vLLM 生产代码；允许读 vLLM/Triton 源码、做最小差分和实验。
> 当前生产提交：`644744e`；最新文档提交：`c1f86e4`。

## 1. 目标

mini 是一个从零实现的 vLLM 对齐引擎。当前要解决的目标场景：

```text
模型：Gemma4 target + assistant 投机
batch = 1
K = 2（每步 3 个 query token）
q_len = 3
context = 1024
num_query_heads = 8
num_kv_heads = 1（GQA group=8）
head_dim = 256
block_size = 16
causal = True
window = -1
dtype = bf16
GPU = RTX 5090 Laptop / SM120
Triton 3.7.1
vLLM 0.28.0（本机安装快照）
```

目标不是“把 vLLM 源码复制进来”，而是：

1. 解释 vLLM 的 2D attention 为什么快；
2. 在 mini 的 `minivllm/attention/ops/triton_unified_attention.py` 里用本地实现追平；
3. 每一步有可复现的 kernel 对照、TTGIR/PTX 证据和测试。

## 2. 当前结果

### mini 生产路径

```text
K/V 统一 [TILE, HEAD] load
K 在 tl.dot 前 tl.trans
2D num_warps=4
BLOCK_M=8, BLOCK_Q=1
head<=256: TILE64 load + 2×32 精确 subtile
head=512: TILE32
```

单 kernel，q3/ctx1024/hd256：

```text
mini: 0.0522 ms（另一轮 0.0553）
vLLM: 0.0492 ms（同轮；历史上 0.0486–0.0969 波动）
输出逐元素差: 0.0
```

22-case 对照：

```text
ctx128:  mini 0.0403 vs vLLM 0.0457 ms
ctx256:  mini 0.0447 vs vLLM 0.0575 ms
ctx512:  mini 0.0542 vs vLLM 0.0783 ms
ctx1024: mini 0.0553 vs vLLM 0.0492 ms（约 1.12×）
ctx2048: mini 0.1036 vs vLLM 0.0862 ms（约 1.20×）
hd512:  mini 0.1015 vs vLLM 0.0741 ms（约 1.37×）
```

真实 Graph（B1 K2 1024 context）：

```text
mini 150.87 tok/s
vLLM 168.42 tok/s
单轮 0.896×；vLLM 153–168 波动
```

### mini 编译结果

最终生产 kernel：

```text
regs 255 / spills 0
shared 20,480 B
TTGIR async_copy 0
PTX cp.async 0
K/V load layout: #blocked sizePerThread=[1,1], order=[1,0]
convert_layout 3 处：
  K  #blocked -> #linear
  V  #blocked -> #linear2
  output #mma -> #blocked1
```

### vLLM 编译结果

同一个 q3/ctx1024 输入，vLLM 的 2D unified attention：

```text
regs 158 / spills 0
shared 42,240 B
num_warps 4
num_stages 3
TILE_SIZE 32
BLOCK_M 16
BLOCK_Q 2
K_load: #blocked1 sizePerThread=[8,1], order=[0,1]
V_load: #blocked2 sizePerThread=[1,8], order=[1,0]
TTGIR async_copy 10
TTGIR async_wait 3
PTX cp.async 51
convert_layout 1 处（仅输出）
```

## 3. 核心问题

同样的 `tl.load` 源码形态，编译后：

```text
vLLM: K/V coalesced (#blocked1[8,1] / #blocked2[1,8])
      -> Triton pipeline 选中
      -> global -> shared async copy
      -> dot 从 shared 读

mini: K/V scalar (#blocked[1,1])
      -> 先 global -> register
      -> convert_layout 到 #linear 再 dot
      -> pipeline 不选
      -> async 0
```

想请更强模型回答/实现：

1. **Triton 3.7.1 里到底是哪个 pass / 哪条 cost model 决定 K/V load 用
   `#blocked[1,1]` 还是 `#blocked1[8,1]`？**
   - layout propagation？
   - coalesce pass？
   - remove-layout-conversions？
   - pipeline pass 的 load eligibility？
   - 需要具体到 pass 名、源码位置、判定条件。

2. **为什么 vLLM 的 kernel 得到 coalesced layout，而 mini 精确复刻它的
   `TILE32/BM16/w4/K [HEAD,TILE]/V [TILE,HEAD]/unmasked block-table` 后仍然是
   `#blocked1[1,1]`？**
   - 请用 TTGIR/MLIR 差分指出第一个分叉点；
   - 不要只说“kernel 结构不同”，要指出具体 IR 值/操作。

3. **有没有 Triton 3.7.1 支持的高层写法/编译选项，可以稳定让 mini 的 paged K/V
   load 进入 `ttg.async_copy_global_to_local`？**
   - 例如 block_ptr、tensor descriptor、layout hint、`tl.range` 参数、pass 选项；
   - 如果高层做不到，请明确说“不可能”，并给出最低成本替代方案
     （TMA / 手写 cp.async / 修改 Triton pass）。

4. **如果必须在 mini 里实现，请给出最小 patch 或伪代码**，并说明：
   - 是否保持输出逐元素 0.0 差；
   - 预计对 ctx1024/ctx2048/hd512 的收益；
   - 是否需要改 Triton 编译器或引入 CUDA 扩展。

## 4. 已排除项

这些都不是触发条件（均有本地实测/TTGIR 证据）：

```text
mask / dim_mask 去掉
block-table clamp 去掉，补一列避免越界
K 改 [HEAD,TILE] 或 [TILE,HEAD]
K 做/不做 tl.trans
TILE_SIZE 16/32/64
BLOCK_M 4/8/16/32
num_warps 4/8
num_stages 1..5
tl.range(num_stages=2..5)
tl.range(warp_specialize=True)（Triton 3.7.1 pipeline pass 直接失败）
load order K→V / V→K
cache_modifier .cg
eviction_policy evict_first/evict_last
把 K/V load 包进 @triton.jit 的 _cast_kv_tile 函数边界
精确复刻 vLLM 的 TILE32/BM16/w4/K [HEAD,TILE]/V [TILE,HEAD]/unmasked block-table
```

关键负结果：

```text
精确复刻 vLLM shape 后：
  mini K load 仍是 #blocked1[1,1]
  async 0
  ctx1024 0.155 ms（255 regs/2 spills）
```

## 5. 当前 workaround

没有硬碰 async，而是用数值精确的 tile 方案：

```text
一次 load 64 宽 K/V
-> reshape/permute/tl.split 成两个 32 宽 subtile
-> 按 TILE32 的顺序做两次 online softmax
-> BLOCK_M=8
```

效果：

```text
ctx1024 0.0522 ms
ctx2048 0.0990 ms
hd512   0.0975 ms
输出差 0.0
```

这解释了为什么“短/中上下文已追平，长上下文仍 1.2–1.37×”：  
**本地仍是 scalar K/V load，长上下文要为每次 load 付延迟；TILE64 subtile 只是摊薄了 loop/load 次数。**

## 6. 复现

仓库：`/home/user/proj/vllm-from-scratch`

```bash
PY=/home/user/anaconda3/envs/vllm-omni-dev/bin/python

# 同输入 mini vs vLLM，打印 regs/spills/shared、TTGIR async、PTX cp.async
$PY benchmarks/check_step88_attention_pipeline.py \
  --case qlen3_ctx1024 --warmup 50 --repeats 200 --rounds 10

# 22-case mini vs vLLM 2D kernel 对照
$PY benchmarks/check_step88_vllm_kernel_diff.py \
  --warmup 50 --repeats 200 --rounds 10

# 真实 Graph
$PY benchmarks/compare_step88_execution.py \
  --suite context --repeats 1
```

关键文件：

```text
minivllm/attention/ops/triton_unified_attention.py   # 本地生产 kernel
docs/step88_attention_pipeline.md                   # §10–§11.12
docs/results.json                                   # subtile_tile64_bm8 / async_root_cause
benchmarks/check_step88_attention_pipeline.py       # TTGIR/PTX 诊断
benchmarks/check_step88_vllm_kernel_diff.py         # 22-case 对照
```

当前生产 commit：

```text
644744e step88：TILE64 精确 subtile 提升 2D attention
```

最新文档 commit：

```text
c1f86e4 step88：记录去 dim_mask 负结果
```

## 7. 验收标准

任何候选方案必须同时满足：

1. `output` 与当前生产 kernel 逐元素 `max_abs == 0.0`（BF16/FP32 都检查）；
2. TTGIR 中出现 K/V 的 `ttg.async_copy_global_to_local`，而不只是 block-table；
3. q3/ctx1024 ≤ 0.0522 ms，ctx2048 明显优于 0.0990 ms，hd512 不退化；
4. `tests/step86`、`tests/step87`、`tests/step88` 全过；
5. `tests/step58`–`step88` 1297 passed；
6. 不 vendored vLLM 生产代码，不引入未解释的行为差异。

## 8. 建议调查顺序

1. 用 `MLIR_ENABLE_DUMP=1` / Triton pass dump，拿到 vLLM 和 mini 在
   `add_pipeline` 前后的 TTGIR，定位 K/V layout 第一次分叉的 pass；
2. 在 Triton `lib/Dialect/TritonGPU/Transforms/` 里找 layout propagation /
   coalesce / pipeline 的判定条件；
3. 做最小 kernel 差分：从 vLLM kernel 里逐段删除不影响 async 的部分，
   找到让 K/V 从 `#blocked[1,1]` 变成 `#blocked1[8,1]` 的最小 IR 条件；
4. 如果发现是某个高层写法可触发，给出 mini 的最小 patch；
5. 如果是编译器 pass 限制，明确说明并给 TMA/手写 cp.async 的替代方案。

## 9. 重要边界

- vLLM NVIDIA 正常路径**没有**手写 `cp.async`，也没有显式 layout hint；
  `cp.async` 是 Triton 自动 pipeline 生成的。
- vLLM 的 `tl.make_tensor_descriptor` 只在 `USE_TD`（Intel Xe2/Xe3）分支；
  NVIDIA 目标路径 `USE_TD=False`，是 dead code。
- 因此不要建议“照抄 vLLM 的 layout 写法”；vLLM 也没有。
- 本地最终工作不是 async，而是 layout + TILE64 + 2×32 subtile 的精确替代。
